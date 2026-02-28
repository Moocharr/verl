# NPU Kimi-K2 GSPO LoRA 训练优化实践

Last updated: 03/05/2026.

本文章对应脚本地址：[kimi-k2_megatron_lora_npu](https://github.com/volcengine/verl/blob/main/examples/gspo_trainer/run_kimi-k2_megatron_lora_npu.sh)

---

## 目录

1. [简介](#1-简介)
2. [技术背景](#2-技术背景)
3. [环境准备](#3-环境准备)
4. [快速开始](#4-快速开始)
5. [性能调优](#5-性能调优)
6. [参考资源](#6-参考资源)

---

## 1. 简介

### 1.1 文档目标

本文档详细介绍了在 **昇腾 NPU** 上使用 **GSPO 算法** 对 **Kimi-K2** 大语言模型进行 **LoRA 微调** 的完整实践方案。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **算法** | GSPO (Group Relative Policy Optimization) - 序列级策略优化 |
| **微调方法** | LoRA (Low-Rank Adaptation) - 低秩适应 |
| **分布式框架** | Megatron-Bridge + MindSpeed |
| **推理引擎** | vLLM v1 + vLLM-Ascend |
| **硬件** | Atlas 800T A3 / Atlas 900 A3 SuperPoD |

### 1.3 硬件要求

| 配置项 | 要求 |
|--------|------|
| **节点数** | 16 台 Atlas 800T A3 |
| **单节点 NPU 数** | 16 颗昇腾 910B |
| **总 NPU 数** | 256 颗 |
| **单节点内存** | ≥ 1.5 TB |
| **互联网络** | RoCEv2 / HCCS |

---

## 2. 技术背景

### 2.1 GSPO 算法原理

GSPO (Group Relative Policy Optimization) 通过将优化颗粒度从 **token 级** 提升到 **sequence 级**，规避了 GRPO 会遇到的 **方差急剧增大** 导致训练不稳定的情况，增加了训练的稳定性，同时该算法也在一定程度上提升了算法的收敛速度。

**核心配置参数：**

```python
# 核心算法配置
algorithm.adv_estimator=grpo                          # 使用GRPO优势估计器
algorithm.use_kl_in_reward=False                      # 不在奖励中添加KL惩罚

# GSPO策略损失模式
actor_rollout_ref.actor.policy_loss.loss_mode=gspo    # 启用GSPO策略损失

# 极小裁剪范围（GSPO特色）
actor_rollout_ref.actor.clip_ratio_low=0.0003         # 裁剪下界，论文推荐值
actor_rollout_ref.actor.clip_ratio_high=0.0004        # 裁剪上界，论文推荐值

# 序列级损失聚合模式（GSPO核心）
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean  # 序列级平均

# 批次配置
actor_rollout_ref.rollout.n=16                        # 每个prompt生成16个响应
```

### 2.2 LoRA 技术介绍

**LoRA (Low-Rank Adaptation)** 是一种参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）技术，由 Hu et al. 在 2021 年提出。其核心思想是：**在保持预训练模型大部分参数不变的情况下，通过引入少量的可训练低秩矩阵来实现模型适应**。

#### LoRA 的核心原理

LoRA 基于这样一个观察：模型权重矩阵的更新具有**低内在维度（low intrinsic dimension）**。具体来说：

1. **低秩分解**：对于原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 不直接更新 $W_0$，而是引入低秩矩阵 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times k}$，其中 $r \ll \min(d, k)$。

2. **前向传播**：修改后的前向传播为：
   $$h = W_0 x + \Delta W x = W_0 x + BAx$$
   其中 $\Delta W = BA$ 是低秩更新矩阵。

3. **训练策略**：
   - 原始权重 $W_0$ **冻结**（不计算梯度，不参与更新）
   - 只有 $A$ 和 $B$ **可训练**
   - 初始化：$A$ 使用随机高斯初始化，$B$ 初始化为零（保证训练开始时 $\Delta W = 0$）

4. **缩放因子**：引入超参数 $\alpha$（通常设为 $2r$），实际缩放为 $\frac{\alpha}{r} \cdot BAx$，用于调节 LoRA 更新的幅度。

### 2.3 LoRA 在强化学习（RLHF/RLAIF）中的独特优势

将 LoRA 应用于强化学习微调（如 PPO、GRPO、GSPO 等算法）相比全参数微调具有**不可替代的优势**：

#### 1. 显著降低显存占用，支持更大批次

**问题**：RL 训练需要同时维护 Actor、Critic、Reference、Reward 等多个模型副本，全参数微调的显存消耗是 SFT 的 3-4 倍。

**LoRA 解决方案**：
- 仅优化低秩矩阵（通常不到原模型参数的 1%）
- 大幅减少优化器状态（Adam 需要 2 倍参数量的动量存储）和梯度存储
- 使得在相同硬件上可以：
  - 使用更大的批次大小（batch size），提高样本效率和训练稳定性
  - 支持更长的序列长度（context length）
  - 开启更大的生成数量（G，即每个 prompt 的响应数，对 GRPO/GSPO 至关重要）

#### 2. 缓解强化学习的灾难性遗忘（Catastrophic Forgetting）

**问题**：RL 微调的目标（如最大化奖励模型的评分）可能与预训练的知识分布发生偏离，全参数微调容易导致模型"跑偏"，遗忘预训练阶段获得的通用知识和指令遵循能力。

**LoRA 解决方案**：
- 原始预训练权重被**冻结**，作为知识锚点（knowledge anchor）保留
- 低秩更新 $\Delta W$ 专注于学习"策略改进"（policy improvement）所需的特定行为调整，而不是颠覆性地改变模型表示
- 这种"增量式"学习天然约束了策略更新的幅度，与 PPO 等算法的信任域（trust region）约束理念相契合

#### 3. 更快的训练速度和更低的计算成本

- **梯度计算加速**：只计算低秩矩阵的梯度，反向传播的计算量与可训练参数量成正比，通常可节省 50-90% 的反向传播时间。
- **检查点轻量化**：保存的 checkpoint 只包含低秩矩阵（几 MB 到几百 MB），相比全参数模型的数十 GB，极大节省存储空间和 IO 时间，便于频繁保存和实验迭代。
- **多实验并行**：单个 GPU 可以同时加载多个 LoRA 适配器进行并行实验，而全参数微调每个实验都需要独占 GPU。

#### 4. 更好的泛化性和模块化管理

- **任务特定适配器**：可以为不同的下游任务（如代码生成、数学推理、创意写作）训练独立的 LoRA 适配器，而共享同一个基础模型，实现"一个基础模型 + 多个任务头"的模块化架构。
- **组合适配**：研究表明多个 LoRA 适配器可以通过简单的权重插值或叠加进行组合，实现零样本或少样本的任务迁移。
- **A/B 测试和灰度发布**：在生产环境中，可以动态切换不同的 LoRA 适配器进行 A/B 测试，而无需重新部署整个模型服务。

#### 5. 强化学习特定的训练稳定性

- **KL 散度约束**：在 RLHF 中，需要约束当前策略与参考策略的 KL 散度，防止策略更新过大。LoRA 的低秩约束自然限制了策略更新的"表达能力"，使得策略在更新时更难"走极端"，有助于维持稳定的 KL 散度。
- **奖励黑客（Reward Hacking）缓解**：LoRA 有限的表达能力使得模型更难通过"作弊"方式（如生成重复无意义的高奖励 token）来最大化奖励，促使模型学习真正的、可泛化的策略改进。
- **样本效率**：在 RL 中，样本收集（如通过 vLLM 生成响应）通常是瓶颈。LoRA 更快的训练速度意味着可以更快地完成一次策略更新，从而可以更早地收集下一轮样本，提高整体的样本周转效率。

---

## 3. 环境准备

### 3.1 软件版本要求

| software          | version                                                    |
| ----------------- | ---------------------------------------------------------- |
| Python            | >= 3.10, <3.12                                             |
| CANN              | == 8.3.RC1                                                 |
| torch             | == 2.7.1                                                   |
| torch_npu         | == 2.7.1                                                   |
| verl              | main分支 commitId=252d76908b903ad8fb6969eb3a5e5f873c95ea2b |
| vllm              | v0.11.0                                                    |
| vllm-ascend       | v0.11.0-dev                                                |
| transformers      | 4.57.3                                                     |
| Megatron-Bridge   | >= 0.2.0                                                   |
| MindSpeed         | >= 0.3.0                                                   |

#### Megatron-Bridge 安装要求

Kimi-K2 模型使用了 MoE 架构和 LoRA 微调，需要特定版本的 Megatron-Bridge：

```bash
# 推荐使用以下 commit 或更新版本
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/83a7c1134c562d8c6decd10a1f0a6e6a7a8a3a44

pip install megatron-bridge>=0.2.0
```

在本实践中, 我们通过指定 verl 的commit id 以避免引入其他问题

```bash
cd verl
git checkout 252d76908b903ad8fb6969eb3a5e5f873c95ea2b
# 指定相应的recipe版本
git submodule update --init --recursive recipe
```

### 3.2 模型权重获取

从 Hugging Face 库下载对应的模型权重：[unsloth/kimi-k2-instruct-0905-bf16](https://huggingface.co/unsloth/kimi-k2-instruct-0905-bf16)

Kimi-K2 模型特点：
- **参数量**：32B（总参数量，激活参数量约为8B）
- **架构**：MoE（Mixture of Experts）
- **上下文长度**：支持 256K tokens
- **精度**：BF16

### 3.3 数据集准备

本实践使用 GSM8K 数据集进行数学推理能力训练：

```bash
# 下载 GSM8K 数据集
mkdir -p $HOME/data/gsm8k

# 训练集和测试集可以通过以下方式获取
# 数据集会被自动下载到指定路径
```

数据配置参数：
- 训练数据：`$HOME/data/gsm8k/train.parquet`
- 验证数据：`$HOME/data/gsm8k/test.parquet`
- 训练批次大小：32
- 最大提示长度：32768 tokens（32K）
- 最大响应长度：2048 tokens（2K）

### 3.4 jemalloc 安装

为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。

#### Ubuntu 操作系统

通过操作系统源安装 jemalloc（注意：要求 ubuntu 版本>=20.04）：

```shell
sudo apt install libjemalloc2
```

在启动任务前执行如下命令通过环境变量导入 jemalloc，需先通过 **find /usr -name libjemalloc.so.2** 确认文件是否存在：

```shell
# arm64 架构
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64 架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler 操作系统

执行如下命令从操作系统源安装 jemalloc：

```shell
yum install jemalloc
```

如果上述方法无法正常安装，可以通过源码编译安装。前往 jemalloc 官网下载最新稳定版本，官网地址: https://github.com/jemalloc/jemalloc/releases/

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

在启动任务前执行如下命令通过环境变量导入 jemalloc：

```shell
# 根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

---

## 4. 快速开始

### 4.1 单机测试（最小化验证）

在进行大规模训练前，建议先进行单机测试验证环境配置：

```bash
# 设置环境变量
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

# 启动 Ray
ray start --head --port=6766

# 运行测试（单节点，2卡）
TP=2 PP=1 CP=1 EP=1 NNODES=1 bash run_kimi-k2_megatron_lora_npu.sh
```

### 4.2 多机任务拉起（完整训练）

针对本实践提供的多机任务，可用下面的脚本拉起：

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

# 修改为当前需要跑的用例路径
DEFAULT_SH="./run_kimi-k2_megatron_lora_npu.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir -p logs

NNODES=16
NPUS_PER_NODE=16
# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

# 获取当前IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册 ray 直到成功
  while true; do
      # 尝试连接 ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```

**配置说明：**

- `DEFAULT_SH`: 修改为训练所用配置 sh 文件路径。在此案例中修改为 `run_kimi-k2_megatron_lora_npu.sh` 路径。
- `NNODES` 和 `NPUS_PER_NODE`: 修改为使用节点数量和每个节点 NPU 数量。在此案例中分别为16和16。
- `MASTER_ADDR`: 修改为对应主节点 IP。即所有节点的 MASTER_ADDR 应该相同。
- `SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`: 修改为对应通信网卡。

**获取通信网卡的方法：**

```bash
# 方法1：使用 hostname 获取默认 IP 对应的网卡
ip route get 8.8.8.8 | grep -oP 'dev \K\S+'

# 方法2：查看所有网卡及其 IP
ip addr show

# 方法3：使用 ifconfig（如果已安装）
ifconfig | grep -E "^[a-zA-Z0-9]+:" | awk -F: '{print $1}'
```

---

## 5. 性能调优

优化从训练、推理、调度和其他四个方面入手。

### 5.1 训练优化

#### 5.1.1 动态批次大小（Dynamic Batch Size）

```bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
```

**注意**：这两个参数调整过大可能会导致 OOM。

**主要调整** `actor_ppo_max_token_len`，调大了会降低训练的耗时。调整 `infer_ppo_max_token_len` 没有明显的收益，可以不动。

**参数说明**：

- **`actor_ppo_max_token_len`**: Actor模型在PPO更新(前向+反向传播)时每个GPU能处理的最大token数
- **`infer_ppo_max_token_len`**: 推理阶段(Reference policy和Rollout)计算log概率时每个GPU能处理的最大token数

#### 5.1.2 LoRA 配置优化

本实践使用 LoRA (Low-Rank Adaptation) 进行参数高效微调：

```bash
# LoRA 基础配置
actor_rollout_ref.model.lora.rank=32 \           # LoRA 秩，控制可训练参数量
actor_rollout_ref.model.lora.alpha=64 \          # LoRA 缩放系数，通常设为 2*rank
actor_rollout_ref.model.lora.lora_A_init_method=kaiming  # 初始化方法

# 可选：使用 Canonical LoRA（更灵活的 target_modules 配置）
# actor_rollout_ref.model.lora.type="canonical_lora"
# actor_rollout_ref.model.lora.target_modules='["linear_q","linear_k","linear_v","linear_proj","linear_fc1_up","linear_fc1_gate","linear_fc2"]'

# 可选：添加 LoRA dropout 防止过拟合
# actor_rollout_ref.model.lora.dropout=0.05
# actor_rollout_ref.model.lora.dropout_position=pre
```

**关键参数说明**：

- **rank**: 控制低秩矩阵的维度，越大表达能力越强，但计算量和显存占用也越大。对于 32B 级别的模型，rank=32 是一个平衡点。
- **alpha**: 缩放系数，通常设为 rank 的 2 倍（即 alpha/rank = 2），可以调节 LoRA 更新的幅度。
- **target_modules**: 指定哪些层添加 LoRA 适配器，默认通常是 attention 的 q、k、v、projection 和 MLP 层。
- **dropout**: 在 LoRA 路径上添加 dropout 可以防止过拟合，但通常保持为 0 以获得最佳性能。

#### 5.1.3 Megatron-Bridge 配置

本实践使用 Megatron-Bridge 进行大模型分布式训练：

```bash
# 启用 Megatron-Bridge
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False

# 并行配置
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8      # TP=8
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=8    # PP=8
actor_rollout_ref.actor.megatron.context_parallel_size=4           # CP=4
actor_rollout_ref.actor.megatron.expert_model_parallel_size=32     # EP=32

# 内存优化
actor_rollout_ref.actor.megatron.param_offload=True
actor_rollout_ref.actor.megatron.optimizer_offload=True
actor_rollout_ref.actor.megatron.grad_offload=True

# 重计算配置
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1

# Kimi-K2 特有配置
actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
actor_rollout_ref.actor.megatron.override_transformer_config.multi_latent_attention=True
actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=True
```

### 5.2 推理优化

#### 5.2.1 ACLgraph + FULL_DECODE_ONLY

推理算子下发方面的优化，平均能有 **15%~20%** 左右的性能收益。

```bash
# 开启ACLgraph+FULL_DECODE_ONLY（注意：当设置此参数为False时，TASK_QUEUE_ENABLE必须设置为1，不然会报错）
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \ 
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY' \
```

**`cudagraph_capture_sizes` 参数设置指南**：

- 设置的值对应的是批大小（单位为 **token**）
- 默认生成的算法如下，可做参考：

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

#### 5.2.2 推理后端切换

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

#### 5.2.3 使能 vLLM v1 版本

```bash
export VLLM_USE_V1=1
```
可以常开，一般都是正收益。

### 5.3 调度优化

#### 5.3.1 AIV（AI Vector Core）

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
```

HCCL_OP_EXPANSION_MODE 环境变量用于配置通信算法的编排展开位置：

- **AI_CPU**：通信算法的编排展开位置在 Device 侧的 AI CPU 计算单元。
- **AIV**：通信算法的编排展开位置在 Device 侧的 Vector Core 计算单元。（推荐）
- **HOST**：通信算法的编排展开位置为 Host 侧 CPU。
- **HOST_TS**：通信算法的编排展开位置为 Host 侧 CPU，Host 向 Device 的 Task Scheduler 下发任务。

#### 5.3.2 TASK_QUEUE_ENABLE

```bash
export TASK_QUEUE_ENABLE=2
```

- 图模式设置为 1
- 非图模式设置为 2

#### 5.3.3 绑核优化

```bash
export CPU_AFFINITY_CONF=1
```

### 5.4 其他优化

#### 5.4.1 使能 jemalloc

```bash
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

#### 5.4.2 多流复用

```bash
export MULTI_STREAM_MEMORY_REUSE=1
```

#### 5.4.3 vLLM Ascend 优化

```bash
# 启用昇腾 NPU 特有的 FLASHCOMM 高速通信优化技术
export VLLM_ASCEND_ENABLE_FLASHCOMM=1

# 启用昇腾 NPU 针对大模型推理的稠密计算优化
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1

# 启用 MLP 层的权重预取机制
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
```

---

## 6. 参考资源

### 6.1 官方文档

- [环境变量列表 - Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_001.html)
- [性能调优流程 - Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0001.html)
- [Megatron-Bridge MoE LoRA 支持](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/83a7c1134c562d8c6decd10a1f0a6e6a7a8a3a44)

### 6.2 模型与数据集

- [Kimi-K2 模型 - Hugging Face](https://huggingface.co/unsloth/kimi-k2-instruct-0905-bf16)
- [GSM8K 数据集](https://github.com/openai/grade-school-math)

### 6.3 社区资源

- [vLLM Ascend 文档](https://vllm-ascend.readthedocs.io/zh-cn/latest/)
- [MindSpeed 文档](https://gitee.com/ascend/MindSpeed)
- [VERL GitHub 仓库](https://github.com/volcengine/verl)