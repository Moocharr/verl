#!/usr/bin/env bash
set -xeuo pipefail

# Need to install Megatron-Bridge
# NOTE: Make sure you use Megatron-Bridge later than 0.2.0
# (Recommend https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/83a7c1134c562d8c6decd10a1f0a6e6a7a8a3a44 or later)
# for proper MoE LoRA support.

# For Megatron communication/computation overlapping
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

TP=${TP:-8}
PP=${PP:-8}
CP=${CP:-4}
EP=${EP:-32}
ETP=${ETP:-1}
NNODES=${NNODES:-16}

infer_tp=4
infer_dp=32
infer_ep=128

ALL_OFFLOAD=${ALL_OFFLOAD:-True}


rollout_name="vllm"
project_name='verl_gspo_kimi-k2_megatron_lora'
exp_name='gspo_kimi-k2_megatron_lora'
adv_estimator=grpo
max_num_batched_tokens=1024
max_prompt_length=$((1024 * 32))
max_response_length=$((1024 * 2))
max_model_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((actor_ppo_max_token_len / CP))
infer_ppo_max_token_len=$((infer_ppo_max_token_len / CP))

loss_agg_mode="seq-mean-token-mean"
loss_mode=gspo
clip_ratio_low=0.0003
clip_ratio_high=0.0004

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
MODEL_PATH="unsloth/kimi-k2-instruct-0905-bf16"
CKPTS_DIR="ckpts"

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${gsm8k_train_path}
    data.val_files=${gsm8k_test_path}
    data.train_batch_size=32
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.shuffle=True
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.lora.rank=32
    actor_rollout_ref.model.lora.alpha=64
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    actor_rollout_ref.nccl_timeout=7200
    # # Optional: Use canonical LoRA
    # actor_rollout_ref.model.lora.type="canonical_lora"
    # actor_rollout_ref.model.lora.target_modules='["linear_q","linear_k","linear_v","linear_proj","linear_fc1_up","linear_fc1_gate","linear_fc2"]'

    # # Optional: Add dropout to LoRA layers
    # actor_rollout_ref.model.lora.dropout=0.05
    # actor_rollout_ref.model.lora.dropout_position=pre
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=3e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=6
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=7
    ++actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=16
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp}
    actor_rollout_ref.rollout.data_parallel_size=${infer_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${infer_ep}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.disable_log_stats=False
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    ++actor_rollout_ref.ref.megatron.override_transformer_config.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.sequence_parallel=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=6
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=7
    ++actor_rollout_ref.ref.megatron.override_transformer_config.multi_latent_attention=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.reset_position_ids=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=16
    trainer.nnodes=${NNODES}
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.device=npu
    trainer.val_before_train=False
    trainer.default_local_dir="${CKPTS_DIR}"
)

########################### Launch ###########################
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
