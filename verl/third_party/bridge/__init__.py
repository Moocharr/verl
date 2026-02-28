import json
from functools import wraps
from importlib.metadata import version
from typing import Tuple

import packaging
import torch
import torch.nn as nn
from megatron.bridge import peft
from megatron.bridge.peft.utils import (TEColumnParallelLinear,
                                        TEColumnParallelGroupedLinear,
                                        TERowParallelLinear,
                                        TERowParallelGroupedLinear)
from megatron.bridge.peft.utils import HAVE_TE
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.utils.import_utils import safe_import_from
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.bridge.models.conversion.param_mapping import AutoMapping


from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import TELayerNormColumnParallelLinear
from mindspeed.te.pytorch.module import (TELinear,
                                         MindSpeedTEColumnParallelLinear,
                                         MindSpeedTEGroupedLinear,
                                         MindSpeedTEColumnParallelGroupedLinear,
                                         MindSpeedTERowParallelGroupedLinear)
TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
TERL = (TERowParallelLinear, TERowParallelGroupedLinear)


def get_adapter_attributes_from_linear(
    m: nn.Module, is_expert: bool = False
) -> Tuple[bool, int, int, bool, bool, bool]:
    """Returns attributes from the base layer.

    input_is_parallel, in_features, out_features, disable_tensor_parallel_comm, disable_sequence_parallel_comm, base_linear_is_parallel

    This function analyzes a linear module and extracts key attributes needed for adapter configuration,
    particularly for PEFT adapters in distributed training scenarios.

    Args:
        m: The linear module to analyze (should have a config attribute).

    Returns:
        A tuple containing:
            - input_is_parallel: Whether the input is already parallelized
            - in_features: Input feature dimension
            - out_features: Output feature dimension
            - disable_tensor_parallel_comm: Whether to disable tensor parallel communication
            - disable_sequence_parallel_comm: Whether to disable sequence parallel communication
            - base_linear_is_parallel: Whether the base linear layer uses parallelization

    Raises:
        NotImplementedError: If the layer type is not recognized for LoRA adaptation.
    """
    disable_sequence_parallel_comm = not m.config.sequence_parallel
    base_linear_is_parallel = True

    # In some modules (notably MoE shared_experts when moe_shared_expert_overlap is enabled),
    # Megatron disables TP-related communications on the base linear layer by
    # setting `parallel_mode=None` (TE) or `explicit_expert_comm=True` (legacy).
    # https://github.com/NVIDIA/Megatron-LM/blob/5b1ef0703184299fbf71f6131bf2f9a5331e7238/megatron/core/transformer/moe/shared_experts.py#L95-L104
    # The weights are still TP-sharded though, so we must keep using the real TP size
    disable_tensor_parallel_comm = getattr(m, "parallel_mode", "") is None or getattr(m, "explicit_expert_comm", False)
    if disable_tensor_parallel_comm:
        disable_sequence_parallel_comm = True

    if is_expert:
        tp_size = parallel_state.get_expert_tensor_parallel_world_size()
    else:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        in_features = m.in_features if hasattr(m, "in_features") else m.input_size
        if isinstance(m, MindSpeedTEColumnParallelGroupedLinear):
            out_features = m.out_features * tp_size if hasattr(m, "out_features") else m.output_size * tp_size
        else:
            out_features = m.out_features * tp_size if hasattr(m, "out_features") else m.output_size

        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
            # perf optimization for LoRA + SP
            if hasattr(m, "ub_overlap_ag"):
                ub_overlap_ag = m.ub_overlap_ag
            elif hasattr(m, "ub_overlap_ag_fprop"):
                ub_overlap_ag = m.ub_overlap_ag_fprop
            else:
                ub_overlap_ag = False
            if hasattr(m, "config") and m.config.sequence_parallel and not ub_overlap_ag:
                m.return_layernorm_output_gathered = True
                te_version = packaging.version.Version(version("transformer-engine"))
                if te_version >= packaging.version.Version("1.5.0dev") and (
                    not getattr(m.config, "tp_comm_overlap", False)
                    or getattr(m.config, "tp_comm_overlap_disable_qkv", False)
                ):
                    # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                    # in the forward method is not needed, so disable sp communications
                    # unless TP communication overlap is used
                    # disable_sequence_parallel_comm = True
                    pass
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        if isinstance(m, MindSpeedTERowParallelGroupedLinear):
            in_features = m.in_features * tp_size if hasattr(m, "in_features") else m.input_size * tp_size
        else:
            in_features = m.in_features * tp_size if hasattr(m, "in_features") else m.input_size
        out_features = m.out_features if hasattr(m, "out_features") else m.output_size
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features if hasattr(m, "in_features") else m.input_size
        out_features = m.out_features if hasattr(m, "out_features") else m.output_size
        base_linear_is_parallel = False
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return (
        input_is_parallel,
        in_features,
        out_features,
        disable_tensor_parallel_comm,
        disable_sequence_parallel_comm,
        base_linear_is_parallel,
    )

peft.utils.get_adapter_attributes_from_linear = get_adapter_attributes_from_linear


def key_to_filename_map_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        if self._key_to_filename_map_cache is None:
            fn(self)
            # MindSpeed doesn't save rotary_emb.inv_freq in checkpoints
            inv_freq_keys = []
            for key in self._key_to_filename_map:
                if key.endswith("rotary_emb.inv_freq"):
                    inv_freq_keys.append(key)
            for key in inv_freq_keys:
                self._key_to_filename_map_cache.pop(key)

        return self._key_to_filename_map_cache

    return wrapper

SafeTensorsStateSource.key_to_filename_map = key_to_filename_map_wrapper(SafeTensorsStateSource.key_to_filename_map)


def _detect_parallelism_type(self, module: nn.Module) -> str:
    """Detect parallelism type from module."""
    module_type = type(module).__name__
    if module_type.startswith("MindSpeed"):
        module_type = module_type.replace("MindSpeed", "")

    # Handle fused modules like TELayerNormColumnParallelLinear
    # These modules have both column-parallel weights (weight, bias)
    # and replicated layer norm weights (layer_norm_weight, layer_norm_bias)
    if module_type == "TELayerNormColumnParallelLinear":
        # Check the actual parameter name to determine the correct parallelism type
        if self.megatron_param and (
                self.megatron_param.endswith("layer_norm_weight") or self.megatron_param.endswith("layer_norm_bias")
        ):
            return "replicated"
        # All other parameters (weight, bias) are column-parallel
        return "column"

    # Check registry first
    for parallelism, types in self._MODULE_TYPE_REGISTRY.items():
        if module_type in types:
            return parallelism

    # Fallback to inspecting module attributes
    if hasattr(module, "tensor_model_parallel"):
        if not module.tensor_model_parallel:
            return "replicated"

        # Check partition dimension
        partition_dim = getattr(module, "partition_dim", None)
        if partition_dim == 0:
            return "column"
        elif partition_dim == 1:
            return "row"

    # Fallback for normalization layers
    if any(norm in module_type for norm in ["Norm", "Normalization"]):
        return "replicated"

    # Check parallel_mode for TELinear
    if module_type == "TELinear":
        if module.parallel_mode == "column":
            return "column"
        elif module.parallel_mode == "row":
            return "row"
        else:
            return "replicated"

    # Cannot determine - raise informative error
    known_types = {p: sorted(list(t)) for p, t in self._MODULE_TYPE_REGISTRY.items()}

    raise ValueError(
        f"Cannot determine parallelism type for module '{module_type}' "
        f"at weight '{self.megatron_param}'.\n"
        f"Please use an explicit mapping type (e.g., ColumnParallelMapping) "
        f"or register the module type using:\n"
        f"  AutoMapping.register_module_type('{module_type}', 'column|row|replicated')\n\n"
        f"Currently known module types:\n{json.dumps(known_types, indent=2)}"
    )

AutoMapping._detect_parallelism_type = _detect_parallelism_type
