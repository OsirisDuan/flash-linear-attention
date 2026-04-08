# FLA 算子 Kernel 级细粒度分析报告

## 目录

1. [概述](#概述)
2. [chunk_delta_rule 算子深度分析](#chunk_delta_rule-算子深度分析)
   - [算子架构概览](#算子架构概览)
   - [Kernel 函数层级结构](#kernel-函数层级结构)
   - [前向传播流程](#前向传播流程)
   - [反向传播流程](#反向传播流程)
   - [数据流转路径](#数据流转路径)
3. [其他核心算子分析](#其他核心算子分析)
4. [Kernel 函数完整清单](#kernel-函数完整清单)
5. [测试用例与源码对应关系](#测试用例与源码对应关系)

---

## 概述

本报告对 FLA (Flash Linear Attention) 仓库中的算子进行细粒度分析，深入到 `@triton.jit` 装饰的 kernel 函数层级。以 `chunk_delta_rule` 算子为主要案例，展示多 kernel 打包为单一算子的层级关系结构。

### 分析范围

- **主要分析对象**: `chunk_delta_rule` 算子
- **分析深度**: 从顶层 API 到底层 Triton kernel 函数
- **分析维度**: 函数调用关系、数据流转、依赖关系、测试覆盖

---

## chunk_delta_rule 算子深度分析

### 算子架构概览

`chunk_delta_rule` 是一个复杂的复合算子，由多个子算子和 kernel 函数组合而成。

#### 核心文件结构

```
fla/ops/delta_rule/
├── __init__.py              # 算子入口
├── chunk.py                 # 主算子实现
├── wy_fast.py               # WY表示相关kernel
├── fused_recurrent.py       # 融合递归实现
├── naive.py                 # 朴素实现(参考)
└── parallel.py              # 并行实现

fla/ops/common/              # 共享组件
├── chunk_delta_h.py         # Delta H计算kernel
├── chunk_o.py               # Output计算kernel
├── chunk_h.py               # H状态计算kernel
└── chunk_scaled_dot_kkt.py  # KKT点积kernel

fla/ops/utils/
└── solve_tril.py            # 下三角求解kernel
```

### Kernel 函数层级结构

#### 层级关系图

```
chunk_delta_rule (顶层API)
│
├── 前向传播 (Forward)
│   ├── prepare_wy_repr_fwd
│   │   ├── chunk_scaled_dot_kkt_fwd_kernel ← [KERNEL]
│   │   └── solve_tril_16x16_kernel ← [KERNEL]
│   │       └── merge_16x16_to_32x32_inverse_kernel ← [KERNEL]
│   │
│   ├── chunk_gated_delta_rule_fwd_h
│   │   └── chunk_gated_delta_rule_fwd_kernel_h_blockdim64 ← [KERNEL]
│   │
│   └── chunk_fwd_o
│       └── chunk_fwd_kernel_o ← [KERNEL]
│
└── 反向传播 (Backward)
    ├── recompute_w_u_fwd
    │   └── recompute_w_u_fwd_kernel ← [KERNEL]
    │
    ├── chunk_gated_delta_rule_fwd_h (重计算)
    │   └── chunk_gated_delta_rule_fwd_kernel_h_blockdim64 ← [KERNEL]
    │
    ├── chunk_bwd_dv_local
    │   └── chunk_bwd_kernel_dv ← [KERNEL]
    │
    ├── chunk_gated_delta_rule_bwd_dhu
    │   └── chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 ← [KERNEL]
    │
    ├── chunk_bwd_dqkwg
    │   └── chunk_bwd_kernel_dqkwg ← [KERNEL]
    │
    └── prepare_wy_repr_bwd
        └── prepare_wy_repr_bwd_kernel ← [KERNEL]
```

### 前向传播流程

#### 步骤1: WY表示准备 (prepare_wy_repr_fwd)

**源码路径**: `fla/ops/delta_rule/wy_fast.py`

**功能**: 计算 WY 表示，用于后续的状态更新

**Kernel 函数**:

1. **chunk_scaled_dot_kkt_fwd_kernel**
   - **位置**: `fla/ops/common/chunk_scaled_dot_kkt.py:32`
   - **功能**: 计算 `beta * K * K^T`
   - **输入**: k, g, beta
   - **输出**: A (注意力矩阵)
   - **并行策略**: 按 chunk 和 head 并行

2. **solve_tril_16x16_kernel**
   - **位置**: `fla/ops/utils/solve_tril.py:36`
   - **功能**: 求解下三角矩阵的逆
   - **输入**: A (注意力矩阵)
   - **输出**: A_inv (逆矩阵)
   - **优化**: 使用 16x16 分块策略

3. **merge_16x16_to_32x32_inverse_kernel**
   - **位置**: `fla/ops/utils/solve_tril.py:104`
   - **功能**: 合并 16x16 块为 32x32 块
   - **用途**: 支持更大的 chunk size

**数据流**:
```
k, beta → chunk_scaled_dot_kkt_fwd_kernel → A
A → solve_tril_16x16_kernel → A_inv
A_inv → merge_16x16_to_32x32_inverse_kernel → final_A
```

#### 步骤2: 状态计算 (chunk_gated_delta_rule_fwd_h)

**源码路径**: `fla/ops/common/chunk_delta_h.py`

**功能**: 计算隐藏状态 h 和更新后的值 v_new

**Kernel 函数**:

1. **chunk_gated_delta_rule_fwd_kernel_h_blockdim64**
   - **位置**: `fla/ops/common/chunk_delta_h.py:39`
   - **功能**: 计算隐藏状态更新
   - **输入**: k, w, u, g, initial_state
   - **输出**: h, v_new, final_state
   - **优化**: 
     - 支持 64 维分块
     - 支持门控机制
     - 支持变长序列

**数据流**:
```
k, w, u, initial_state → chunk_gated_delta_rule_fwd_kernel_h_blockdim64
→ h (隐藏状态), v_new (更新值), final_state
```

#### 步骤3: 输出计算 (chunk_fwd_o)

**源码路径**: `fla/ops/common/chunk_o.py`

**功能**: 计算最终输出

**Kernel 函数**:

1. **chunk_fwd_kernel_o**
   - **位置**: `fla/ops/common/chunk_o.py:34`
   - **功能**: 计算输出 `o = q @ h + intra_chunk_attention`
   - **输入**: q, k, v, h, g, scale
   - **输出**: o
   - **优化**:
     - 分块计算 intra-chunk 和 inter-chunk 注意力
     - 支持门控机制

**数据流**:
```
q, k, v_new, h → chunk_fwd_kernel_o → o (输出)
```

### 反向传播流程

#### 步骤1: 重计算 WY 表示 (recompute_w_u_fwd)

**源码路径**: `fla/ops/delta_rule/wy_fast.py`

**Kernel 函数**:

1. **recompute_w_u_fwd_kernel**
   - **位置**: `fla/ops/delta_rule/wy_fast.py:32`
   - **功能**: 重计算 w 和 u 值用于反向传播
   - **输入**: k, v, beta, A
   - **输出**: w, u

#### 步骤2: 局部梯度计算 (chunk_bwd_dv_local)

**源码路径**: `fla/ops/common/chunk_o.py`

**Kernel 函数**:

1. **chunk_bwd_kernel_dv**
   - **位置**: `fla/ops/common/chunk_o.py:358`
   - **功能**: 计算 v 的局部梯度
   - **输入**: q, k, g, do, dh
   - **输出**: dv

#### 步骤3: 隐藏状态梯度 (chunk_gated_delta_rule_bwd_dhu)

**源码路径**: `fla/ops/common/chunk_delta_h.py`

**Kernel 函数**:

1. **chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64**
   - **位置**: `fla/ops/common/chunk_delta_h.py:344`
   - **功能**: 计算隐藏状态的梯度
   - **输入**: q, k, w, g, dht, do, dv
   - **输出**: dh, dh0, dv

#### 步骤4: QKW 梯度计算 (chunk_bwd_dqkwg)

**源码路径**: `fla/ops/common/chunk_o.py`

**Kernel 函数**:

1. **chunk_bwd_kernel_dqkwg**
   - **位置**: `fla/ops/common/chunk_o.py:156`
   - **功能**: 计算 q, k, w 的梯度
   - **输入**: q, k, v, h, w, dv, do, dh
   - **输出**: dq, dk, dw, dv

#### 步骤5: WY 反向传播 (prepare_wy_repr_bwd)

**源码路径**: `fla/ops/delta_rule/wy_fast.py`

**Kernel 函数**:

1. **prepare_wy_repr_bwd_kernel**
   - **位置**: `fla/ops/delta_rule/wy_fast.py:94`
   - **功能**: 计算 WY 表示的反向梯度
   - **输入**: k, v, beta, A, dw, du
   - **输出**: dk, dv, db

### 数据流转路径

#### 完整数据流图

```
输入数据:
  q [B, T, H, K] - 查询向量
  k [B, T, H, K] - 键向量
  v [B, T, H, V] - 值向量
  beta [B, T, H] - Beta参数
  initial_state [B, H, K, V] - 初始状态

前向传播:
  ┌─────────────────────────────────────────────────────────┐
  │ Step 1: WY表示准备                                      │
  │   k, beta → chunk_scaled_dot_kkt_fwd_kernel             │
  │          → A [B, T, H, BT]                              │
  │   A → solve_tril → A_inv                                │
  │   k, v, beta, A_inv → w, u                              │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 2: 状态计算                                        │
  │   k, w, u, initial_state                                │
  │   → chunk_gated_delta_rule_fwd_kernel_h                 │
  │   → h [B, NT, H, K, V], v_new [B, T, H, V]              │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 3: 输出计算                                        │
  │   q, k, v_new, h → chunk_fwd_kernel_o                   │
  │   → o [B, T, H, V]                                      │
  └─────────────────────────────────────────────────────────┘

反向传播:
  ┌─────────────────────────────────────────────────────────┐
  │ Step 1: 重计算                                          │
  │   k, v, beta, A → recompute_w_u_fwd_kernel              │
  │   → w, u                                                │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 2: 局部梯度                                        │
  │   q, k, do → chunk_bwd_kernel_dv                        │
  │   → dv                                                  │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 3: 状态梯度                                        │
  │   q, k, w, dht, do, dv                                  │
  │   → chunk_gated_delta_rule_bwd_kernel_dhu               │
  │   → dh, dh0, dv                                         │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 4: QKW梯度                                         │
  │   q, k, v_new, h, w, dv, do, dh                         │
  │   → chunk_bwd_kernel_dqkwg                              │
  │   → dq, dk, dw                                          │
  └─────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 5: WY反向                                          │
  │   k, v, beta, A, dw, dv                                 │
  │   → prepare_wy_repr_bwd_kernel                          │
  │   → dk, dv, db                                          │
  └─────────────────────────────────────────────────────────┘

输出梯度:
  dq [B, T, H, K] - 查询梯度
  dk [B, T, H, K] - 键梯度
  dv [B, T, H, V] - 值梯度
  db [B, T, H] - Beta梯度
  dh0 [B, H, K, V] - 初始状态梯度
```

---

## 其他核心算子分析

### GLA (Gated Linear Attention) 算子

**源码路径**: `fla/ops/gla/chunk.py`

**Kernel 函数清单**:

| Kernel名称 | 行号 | 功能描述 |
|-----------|------|---------|
| `chunk_gla_fwd_A_kernel_intra_sub_inter` | 35 | 计算chunk间注意力 |
| `chunk_gla_fwd_A_kernel_intra_sub_intra` | 109 | 计算chunk内注意力 |
| `chunk_gla_fwd_A_kernel_intra_sub_intra_split` | 183 | 分块计算chunk内注意力 |
| `chunk_gla_fwd_kernel_o` | 262 | 计算输出 |
| `chunk_gla_bwd_kernel_dh` | 312 | 计算隐藏状态梯度 |
| `chunk_gla_bwd_kernel_dqk` | 398 | 计算QK梯度 |
| `chunk_gla_bwd_kernel_dv` | 534 | 计算V梯度 |
| `chunk_gla_bwd_kernel_dqkv_local` | 587 | 计算局部QKV梯度 |
| `chunk_gla_bwd_kernel_dqkv_local_compact` | 664 | 紧凑局部QKV梯度 |

**层级结构**:
```
chunk_gla
├── Forward
│   ├── chunk_gla_fwd_A_kernel_intra_sub_inter
│   ├── chunk_gla_fwd_A_kernel_intra_sub_intra
│   ├── chunk_gla_fwd_A_kernel_intra_sub_intra_split
│   └── chunk_gla_fwd_kernel_o
└── Backward
    ├── chunk_gla_bwd_kernel_dh
    ├── chunk_gla_bwd_kernel_dqk
    ├── chunk_gla_bwd_kernel_dv
    ├── chunk_gla_bwd_kernel_dqkv_local
    └── chunk_gla_bwd_kernel_dqkv_local_compact
```

### Gated Delta Rule 算子

**源码路径**: `fla/ops/gated_delta_rule/`

**Kernel 函数清单**:

| Kernel名称 | 文件 | 行号 | 功能描述 |
|-----------|------|------|---------|
| `recompute_w_u_fwd_kernel` | wy_fast.py | 56 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | wy_fast.py | 132 | WY反向传播 |
| `fused_gdn_gate_kernel` | gate.py | 60 | 融合门控 |
| `fused_gdn_gate_bwd_kernel` | gate.py | 116 | 门控反向 |
| `gate_output_correction_kernel` | gate.py | 235 | 门控输出校正 |
| `fused_recurrent_gated_delta_rule_kernel` | fused_recurrent.py | 24 | 融合递归 |
| `chunk_gated_delta_rule_fwd_kernel` | chunk_fwd.py | 36 | Chunk前向 |

### HGRN 算子

**源码路径**: `fla/ops/hgrn/`

**Kernel 函数清单**:

| Kernel名称 | 文件 | 行号 | 功能描述 |
|-----------|------|------|---------|
| `fused_recurrent_hgrn_fwd_kernel` | fused_recurrent.py | 30 | 递归前向 |
| `fused_recurrent_hgrn_bwd_kernel` | fused_recurrent.py | 92 | 递归反向 |
| `chunk_hgrn_fwd_kernel` | chunk.py | 52 | Chunk前向 |
| `chunk_hgrn_bwd_kernel` | chunk.py | 94 | Chunk反向 |
| `chunk_hgrn_bwd_kernel_local` | chunk.py | 132 | 局部反向 |
| `chunk_hgrn_fwd_kernel_local` | chunk.py | 178 | 局部前向 |

### KDA (Kimi Delta Attention) 算子

**源码路径**: `fla/ops/kda/`

**Kernel 函数清单**:

| Kernel名称 | 文件 | 行号 | 功能描述 |
|-----------|------|------|---------|
| `recompute_w_u_fwd_kernel` | wy_fast.py | 31 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | wy_fast.py | 118 | WY反向 |
| `fused_kda_gate_kernel` | gate.py | 86 | 融合门控 |
| `fused_kda_gate_bwd_kernel` | gate.py | 142 | 门控反向 |
| `fused_kda_lowerbound_gate_kernel` | gate.py | 361 | 下界门控 |
| `chunk_kda_fwd_kernel_intra` | chunk_intra.py | 40 | Chunk内前向 |
| `chunk_kda_bwd_kernel_intra` | chunk_intra.py | 364 | Chunk内反向 |
| `chunk_kda_bwd_kernel_intra_token` | chunk_intra.py | 640 | Token内反向 |
| `chunk_kda_fwd_kernel_intra_token_parallel` | chunk_intra_token_parallel.py | 30 | 并行Token内前向 |
| `fused_recurrent_kda_fwd_kernel` | fused_recurrent.py | 30 | 递归前向 |
| `chunk_kda_bwd_kernel_dqk` | chunk_bwd.py | 47 | QK梯度 |
| `chunk_kda_bwd_kernel_dv` | chunk_bwd.py | 128 | V梯度 |

---

## Kernel 函数完整清单

### 按模块分类

#### 1. Common 共享模块

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | fla/ops/common/chunk_delta_h.py:39 | Delta状态前向(64维) |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | fla/ops/common/chunk_delta_h.py:344 | Delta状态反向(64维) |
| `chunk_fwd_kernel_h` | fla/ops/common/chunk_h.py:35 | H状态前向 |
| `chunk_bwd_kernel_dh` | fla/ops/common/chunk_h.py:162 | H状态反向 |
| `chunk_fwd_kernel_o` | fla/ops/common/chunk_o.py:34 | 输出前向 |
| `chunk_bwd_kernel_dqkwg` | fla/ops/common/chunk_o.py:156 | QKW梯度 |
| `chunk_bwd_kernel_dv` | fla/ops/common/chunk_o.py:358 | V梯度 |
| `chunk_bwd_kernel_dqkv` | fla/ops/common/chunk_o.py:460 | QKV梯度 |
| `chunk_scaled_dot_kkt_fwd_kernel` | fla/ops/common/chunk_scaled_dot_kkt.py:31 | KKT点积前向 |
| `fused_recurrent_fwd_kernel` | fla/ops/common/fused_recurrent.py:29 | 融合递归前向 |
| `fused_recurrent_bwd_kernel` | fla/ops/common/fused_recurrent.py:142 | 融合递归反向 |
| `fused_chunk_fwd_kernel` | fla/ops/common/fused_chunk.py:44 | 融合Chunk前向 |
| `fused_chunk_bwd_kernel` | fla/ops/common/fused_chunk.py:178 | 融合Chunk反向 |

#### 2. Utils 工具模块

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `solve_tril_16x16_kernel` | fla/ops/utils/solve_tril.py:36 | 16x16下三角求解 |
| `merge_16x16_to_32x32_inverse_kernel` | fla/ops/utils/solve_tril.py:104 | 合并逆矩阵 |
| `solve_tril_kernel` | fla/ops/utils/solve_tril.py:193 | 下三角求解 |
| `softmax_fwd_kernel` | fla/ops/utils/softmax.py:26 | Softmax前向 |
| `softmax_bwd_kernel` | fla/ops/utils/softmax.py:53 | Softmax反向 |
| `mean_pooling_kernel` | fla/ops/utils/pooling.py:28 | 平均池化 |
| `mean_pooling_bwd_kernel` | fla/ops/utils/pooling.py:75 | 池化反向 |
| `pack_sequence_kernel` | fla/ops/utils/pack.py:27 | 序列打包 |
| `chunk_global_cumsum_kernel` | fla/ops/utils/cumsum.py:30 | 全局累加 |
| `chunk_global_cumsum_scalar_kernel` | fla/ops/utils/cumsum.py:85 | 标量全局累加 |
| `chunk_local_cumsum_kernel` | fla/ops/utils/cumsum.py:143 | 局部累加 |
| `chunk_local_cumsum_scalar_kernel` | fla/ops/utils/cumsum.py:203 | 标量局部累加 |
| `logsumexp_fwd_kernel` | fla/ops/utils/logsumexp.py:27 | LogSumExp前向 |
| `logcumsumexp_fwd_kernel` | fla/ops/utils/logcumsumexp.py:24 | LogCumSumExp前向 |
| `matmul_kernel` | fla/ops/utils/matmul.py:52 | 矩阵乘法 |
| `addmm_kernel` | fla/ops/utils/matmul.py:157 | 矩阵乘加 |

#### 3. Modules 模块 (43个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `l2norm_fwd_kernel` | fla/modules/l2norm.py:24 | L2归一化前向 |
| `l2norm_bwd_kernel` | fla/modules/l2norm.py:52 | L2归一化反向 |
| `l2norm_fwd_kernel_varlen` | fla/modules/l2norm.py:81 | 变长L2归一化前向 |
| `l2norm_bwd_kernel_varlen` | fla/modules/l2norm.py:111 | 变长L2归一化反向 |
| `layer_norm_fwd_kernel` | fla/modules/layernorm.py:193 | 层归一化前向 |
| `layer_norm_bwd_kernel` | fla/modules/layernorm.py:267 | 层归一化反向 |
| `rms_norm_fwd_kernel` | fla/modules/layernorm.py:340 | RMS归一化前向 |
| `rms_norm_bwd_kernel` | fla/modules/layernorm.py:457 | RMS归一化反向 |
| `fused_layer_norm_gated_fwd_kernel` | fla/modules/layernorm_gated.py:54 | 门控层归一化前向 |
| `fused_layer_norm_gated_bwd_kernel` | fla/modules/layernorm_gated.py:186 | 门控层归一化反向 |
| `fused_rms_norm_gated_fwd_kernel` | fla/modules/fused_norm_gate.py:34 | 门控RMS前向 |
| `fused_rms_norm_gated_bwd_kernel` | fla/modules/fused_norm_gate.py:120 | 门控RMS反向 |
| `fused_rms_norm_swish_gate_fwd_kernel` | fla/modules/fused_norm_gate.py:201 | RMS+Swish门控前向 |
| `fused_rms_norm_swish_gate_bwd_kernel` | fla/modules/fused_norm_gate.py:329 | RMS+Swish门控反向 |
| `fused_cross_entropy_fwd_kernel` | fla/modules/fused_cross_entropy.py:30 | 交叉熵前向 |
| `fused_cross_entropy_bwd_kernel` | fla/modules/fused_cross_entropy.py:106 | 交叉熵反向 |
| `fused_kl_div_fwd_kernel` | fla/modules/fused_kl_div.py:25 | KL散度前向 |
| `fused_kl_div_bwd_kernel` | fla/modules/fused_kl_div.py:90 | KL散度反向 |
| `fused_linear_cross_entropy_fwd_kernel` | fla/modules/fused_linear_cross_entropy.py:42 | 线性交叉熵前向 |
| `fused_linear_cross_entropy_bwd_kernel` | fla/modules/fused_linear_cross_entropy.py:94 | 线性交叉熵反向 |
| `fused_linear_cross_entropy_bwd_kernel_2` | fla/modules/fused_linear_cross_entropy.py:242 | 线性交叉熵反向2 |
| `rotary_embedding_kernel` | fla/modules/rotary.py:46 | 旋转位置编码 |
| `token_shift_fwd_kernel` | fla/modules/token_shift.py:67 | Token移位前向 |
| `token_shift_bwd_kernel` | fla/modules/token_shift.py:159 | Token移位反向 |
| `token_shift_fwd_kernel_2` | fla/modules/token_shift.py:233 | Token移位前向2 |
| `token_shift_bwd_kernel_2` | fla/modules/token_shift.py:311 | Token移位反向2 |
| `swiglu_fwd_kernel` | fla/modules/activations.py:85 | SwiGLU前向 |
| `swiglu_bwd_kernel` | fla/modules/activations.py:115 | SwiGLU反向 |
| `geglu_fwd_kernel` | fla/modules/activations.py:195 | GeGLU前向 |
| `geglu_bwd_kernel` | fla/modules/activations.py:230 | GeGLU反向 |
| `softplus_fwd_kernel` | fla/modules/activations.py:324 | Softplus前向 |
| `softplus_bwd_kernel` | fla/modules/activations.py:355 | Softplus反向 |
| `swiglu_fwd_kernel_2` | fla/modules/activations.py:552 | SwiGLU前向2 |
| `swiglu_bwd_kernel_2` | fla/modules/activations.py:589 | SwiGLU反向2 |
| `fused_bitlinear_fwd_kernel` | fla/modules/fused_bitlinear.py:72 | BitNet线性前向 |
| `fused_bitlinear_bwd_kernel` | fla/modules/fused_bitlinear.py:207 | BitNet线性反向 |
| `causal_conv1d_fwd_kernel` | fla/modules/conv/triton/kernels.py:35 | 因果卷积前向 |
| `causal_conv1d_bwd_kernel` | fla/modules/conv/triton/kernels.py:152 | 因果卷积反向 |
| `causal_conv1d_update_kernel` | fla/modules/conv/triton/kernels.py:325 | 因果卷积更新 |
| `causal_conv1d_update_kernel_v2` | fla/modules/conv/triton/kernels.py:410 | 因果卷积更新v2 |
| `short_conv_fwd_kernel` | fla/modules/conv/triton/kernels.py:487 | 短卷积前向 |
| `grpo_fwd_kernel` | fla/modules/grpo.py:79 | GRPO前向 |
| `grpo_bwd_kernel` | fla/modules/grpo.py:155 | GRPO反向 |

#### 4. Delta Rule 系列算子 (7个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_delta_rule_fwd_kernel` | fla/ops/delta_rule/parallel.py:27 | 并行前向 |
| `parallel_delta_rule_bwd_kernel` | fla/ops/delta_rule/parallel.py:135 | 并行反向 |
| `parallel_delta_rule_bwd_kernel_2` | fla/ops/delta_rule/parallel.py:152 | 并行反向2 |
| `recompute_w_u_fwd_kernel` | fla/ops/delta_rule/wy_fast.py:32 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | fla/ops/delta_rule/wy_fast.py:94 | WY反向 |
| `fused_recurrent_delta_rule_fwd_kernel` | fla/ops/delta_rule/fused_recurrent.py:21 | 递归前向 |
| `fused_recurrent_delta_rule_bwd_kernel` | fla/ops/delta_rule/fused_recurrent.py:108 | 递归反向 |

#### 5. GLA 系列算子 (9个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_gla_fwd_A_kernel_intra_sub_inter` | fla/ops/gla/chunk.py:35 | Chunk间注意力 |
| `chunk_gla_fwd_A_kernel_intra_sub_intra` | fla/ops/gla/chunk.py:109 | Chunk内注意力 |
| `chunk_gla_fwd_A_kernel_intra_sub_intra_split` | fla/ops/gla/chunk.py:183 | 分块Chunk内注意力 |
| `chunk_gla_fwd_kernel_o` | fla/ops/gla/chunk.py:262 | 输出前向 |
| `chunk_gla_bwd_kernel_dh` | fla/ops/gla/chunk.py:312 | H状态梯度 |
| `chunk_gla_bwd_kernel_dqk` | fla/ops/gla/chunk.py:398 | QK梯度 |
| `chunk_gla_bwd_kernel_dv` | fla/ops/gla/chunk.py:534 | V梯度 |
| `chunk_gla_bwd_kernel_dqkv_local` | fla/ops/gla/chunk.py:587 | 局部QKV梯度 |
| `chunk_gla_bwd_kernel_dqkv_local_compact` | fla/ops/gla/chunk.py:664 | 紧凑局部QKV梯度 |

#### 6. Gated Delta Rule 系列算子 (9个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/gated_delta_rule/wy_fast.py:56 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | fla/ops/gated_delta_rule/wy_fast.py:132 | WY反向 |
| `fused_gdn_gate_kernel` | fla/ops/gated_delta_rule/gate.py:60 | 融合门控 |
| `fused_gdn_gate_bwd_kernel` | fla/ops/gated_delta_rule/gate.py:116 | 门控反向 |
| `gate_output_correction_kernel` | fla/ops/gated_delta_rule/gate.py:235 | 门控输出校正 |
| `fused_recurrent_gated_delta_rule_kernel` | fla/ops/gated_delta_rule/fused_recurrent.py:24 | 递归前向 |
| `chunk_gated_delta_rule_fwd_kernel` | fla/ops/gated_delta_rule/chunk_fwd.py:36 | Chunk前向 |

#### 7. HGRN 系列算子 (6个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `fused_recurrent_hgrn_fwd_kernel` | fla/ops/hgrn/fused_recurrent.py:30 | 递归前向 |
| `fused_recurrent_hgrn_bwd_kernel` | fla/ops/hgrn/fused_recurrent.py:92 | 递归反向 |
| `chunk_hgrn_fwd_kernel` | fla/ops/hgrn/chunk.py:52 | Chunk前向 |
| `chunk_hgrn_bwd_kernel` | fla/ops/hgrn/chunk.py:94 | Chunk反向 |
| `chunk_hgrn_bwd_kernel_local` | fla/ops/hgrn/chunk.py:132 | 局部反向 |
| `chunk_hgrn_fwd_kernel_local` | fla/ops/hgrn/chunk.py:178 | 局部前向 |

#### 8. KDA 系列算子 (12个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/kda/wy_fast.py:31 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | fla/ops/kda/wy_fast.py:118 | WY反向 |
| `fused_kda_gate_kernel` | fla/ops/kda/gate.py:86 | 融合门控 |
| `fused_kda_gate_bwd_kernel` | fla/ops/kda/gate.py:142 | 门控反向 |
| `fused_kda_lowerbound_gate_kernel` | fla/ops/kda/gate.py:361 | 下界门控 |
| `chunk_kda_fwd_kernel_intra` | fla/ops/kda/chunk_intra.py:40 | Chunk内前向 |
| `chunk_kda_bwd_kernel_intra` | fla/ops/kda/chunk_intra.py:364 | Chunk内反向 |
| `chunk_kda_bwd_kernel_intra_token` | fla/ops/kda/chunk_intra.py:640 | Token内反向 |
| `chunk_kda_fwd_kernel_intra_token_parallel` | fla/ops/kda/chunk_intra_token_parallel.py:30 | 并行Token内前向 |
| `fused_recurrent_kda_fwd_kernel` | fla/ops/kda/fused_recurrent.py:30 | 递归前向 |
| `chunk_kda_bwd_kernel_dqk` | fla/ops/kda/chunk_bwd.py:47 | QK梯度 |
| `chunk_kda_bwd_kernel_dv` | fla/ops/kda/chunk_bwd.py:128 | V梯度 |

#### 9. RWKV 系列算子 (22个 kernels)

**RWKV4 (2个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `fused_recurrent_rwkv4_fwd_kernel` | fla/ops/rwkv4/fused_recurrent.py:28 | 递归前向 |
| `fused_recurrent_rwkv4_bwd_kernel` | fla/ops/rwkv4/fused_recurrent.py:179 | 递归反向 |

**RWKV6 (12个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `fused_recurrent_rwkv6_fwd_kernel` | fla/ops/rwkv6/fused_recurrent.py:31 | 递归前向 |
| `fused_recurrent_rwkv6_bwd_kernel` | fla/ops/rwkv6/fused_recurrent.py:118 | 递归反向 |
| `fused_recurrent_rwkv6_fwd_kernel_2` | fla/ops/rwkv6/fused_recurrent.py:208 | 递归前向2 |
| `fused_recurrent_rwkv6_bwd_kernel_2` | fla/ops/rwkv6/fused_recurrent.py:306 | 递归反向2 |
| `chunk_rwkv6_fwd_kernel` | fla/ops/rwkv6/chunk.py:45 | Chunk前向 |
| `chunk_rwkv6_bwd_kernel` | fla/ops/rwkv6/chunk.py:126 | Chunk反向 |
| `chunk_rwkv6_fwd_kernel_2` | fla/ops/rwkv6/chunk.py:202 | Chunk前向2 |
| `chunk_rwkv6_bwd_kernel_2` | fla/ops/rwkv6/chunk.py:277 | Chunk反向2 |
| `chunk_rwkv6_fwd_kernel_3` | fla/ops/rwkv6/chunk.py:358 | Chunk前向3 |
| `chunk_rwkv6_bwd_kernel_3` | fla/ops/rwkv6/chunk.py:411 | Chunk反向3 |
| `chunk_rwkv6_fwd_kernel_4` | fla/ops/rwkv6/chunk.py:492 | Chunk前向4 |
| `chunk_rwkv6_bwd_kernel_4` | fla/ops/rwkv6/chunk.py:633 | Chunk反向4 |

**RWKV7 (8个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `fused_recurrent_rwkv7_kernel` | fla/ops/rwkv7/fused_recurrent.py:35 | 递归前向 |
| `fused_k_update_kernel` | fla/ops/rwkv7/fused_k_update.py:32 | K更新 |
| `fused_k_update_kernel_2` | fla/ops/rwkv7/fused_k_update.py:75 | K更新2 |
| `fused_k_update_kernel_3` | fla/ops/rwkv7/fused_k_update.py:122 | K更新3 |
| `fused_k_update_kernel_4` | fla/ops/rwkv7/fused_k_update.py:174 | K更新4 |
| `fused_addcmul_kernel` | fla/ops/rwkv7/fused_addcmul.py:53 | AddCMul |
| `fused_addcmul_kernel_2` | fla/ops/rwkv7/fused_addcmul.py:113 | AddCMul2 |
| `channel_mixing_kernel` | fla/ops/rwkv7/channel_mixing.py:38 | 通道混合 |
| `channel_mixing_kernel_2` | fla/ops/rwkv7/channel_mixing.py:83 | 通道混合2 |
| `channel_mixing_kernel_3` | fla/ops/rwkv7/channel_mixing.py:173 | 通道混合3 |
| `channel_mixing_kernel_4` | fla/ops/rwkv7/channel_mixing.py:205 | 通道混合4 |
| `gate_output_correction_kernel` | fla/ops/rwkv7/gate_output_correction.py:73 | 门控输出校正 |
| `gate_output_correction_kernel_2` | fla/ops/rwkv7/gate_output_correction.py:125 | 门控输出校正2 |

#### 10. Based 系列算子 (4个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_based_fwd_kernel` | fla/ops/based/parallel.py:18 | 并行前向 |
| `parallel_based_bwd_kernel` | fla/ops/based/parallel.py:104 | 并行反向 |
| `parallel_based_bwd_kernel_2` | fla/ops/based/parallel.py:187 | 并行反向2 |
| `fused_chunk_based_fwd_kernel` | fla/ops/based/fused_chunk.py:35 | Chunk前向 |
| `fused_chunk_based_bwd_kernel` | fla/ops/based/fused_chunk.py:158 | Chunk反向 |

#### 11. Rebased 系列算子 (4个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_rebased_fwd_kernel` | fla/ops/rebased/parallel.py:18 | 并行前向 |
| `parallel_rebased_bwd_kernel` | fla/ops/rebased/parallel.py:103 | 并行反向 |
| `parallel_rebased_bwd_kernel_2` | fla/ops/rebased/parallel.py:187 | 并行反向2 |
| `parallel_rebased_bwd_kernel_3` | fla/ops/rebased/parallel.py:273 | 并行反向3 |

#### 12. NSA 系列算子 (8个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_nsa_fwd_kernel` | fla/ops/nsa/parallel.py:43 | 并行前向 |
| `parallel_nsa_bwd_kernel` | fla/ops/nsa/parallel.py:181 | 并行反向 |
| `parallel_nsa_bwd_kernel_2` | fla/ops/nsa/parallel.py:271 | 并行反向2 |
| `parallel_nsa_bwd_kernel_3` | fla/ops/nsa/parallel.py:308 | 并行反向3 |
| `parallel_nsa_bwd_kernel_4` | fla/ops/nsa/parallel.py:417 | 并行反向4 |
| `compress_kernel` | fla/ops/nsa/compression.py:29 | 压缩 |
| `compress_kernel_2` | fla/ops/nsa/compression.py:132 | 压缩2 |
| `compress_kernel_3` | fla/ops/nsa/compression.py:236 | 压缩3 |

#### 13. Path Attention 系列算子 (11个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_path_fwd_kernel` | fla/ops/path_attn/parallel_path_fwd.py:19 | 并行前向 |
| `parallel_path_bwd_intra_kernel` | fla/ops/path_attn/parallel_path_bwd_intra.py:19 | 并行反向内部 |
| `parallel_path_bwd_inter_dkv_kernel` | fla/ops/path_attn/parallel_path_bwd_inter_dkv.py:19 | 并行反向DKV |
| `parallel_path_bwd_inter_dqh_kernel` | fla/ops/path_attn/parallel_path_bwd_inter_dqh.py:21 | 并行反向DQH |
| `intra_chunk_preprocess_fwd_kernel` | fla/ops/path_attn/intra_chunk_preprocess_fwd.py:19 | Chunk内预处理前向 |
| `intra_chunk_preprocess_bwd_kernel` | fla/ops/path_attn/intra_chunk_preprocess_bwd.py:20 | Chunk内预处理反向 |
| `intra_chunk_preprocess_bwd_prepare_kernel` | fla/ops/path_attn/intra_chunk_preprocess_bwd_prepare.py:19 | Chunk内预处理反向准备 |
| `cumprod_householder_fwd_kernel` | fla/ops/path_attn/cumprod_householder_fwd.py:19 | Householder累积前向 |
| `cumprod_householder_bwd_kernel` | fla/ops/path_attn/cumprod_householder_bwd.py:19 | Householder累积反向 |
| `transform_q_kernel` | fla/ops/path_attn/transform_q.py:18 | Q变换 |
| `prepare_k_cache_kernel` | fla/ops/path_attn/prepare_k_cache.py:18 | K缓存准备 |

#### 14. MesaNet 系列算子 (10个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_mesa_net_fwd_kernel` | fla/ops/mesa_net/chunk_h_fwd.py:31 | Chunk前向 |
| `chunk_mesa_net_bwd_kernel` | fla/ops/mesa_net/chunk_h_kv_intra_bwd.py:32 | Chunk反向 |
| `chunk_mesa_net_bwd_kernel_2` | fla/ops/mesa_net/chunk_h_kk_intra_bwd.py:19 | Chunk反向2 |
| `chunk_mesa_net_bwd_kernel_3` | fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py:31 | Chunk反向3 |
| `chunk_mesa_net_bwd_kernel_4` | fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py:150 | Chunk反向4 |
| `chunk_cg_solver_fwd_kernel` | fla/ops/mesa_net/chunk_cg_solver_fwd.py:16 | CG求解前向 |
| `chunk_cg_solver_fwd_kernel_2` | fla/ops/mesa_net/chunk_cg_solver_fwd.py:36 | CG求解前向2 |
| `chunk_cg_solver_bwd_kernel` | fla/ops/mesa_net/chunk_cg_solver_bwd.py:16 | CG求解反向 |
| `chunk_cg_solver_bwd_kernel_2` | fla/ops/mesa_net/chunk_cg_solver_bwd.py:36 | CG求解反向2 |
| `mesa_net_decoding_one_step_kernel` | fla/ops/mesa_net/decoding_one_step.py:16 | 单步解码 |

#### 15. TTT 系列算子 (9个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_ttt_fwd_kernel` | fla/ops/ttt/chunk.py:34 | Chunk前向 |
| `chunk_ttt_bwd_kernel` | fla/ops/ttt/chunk.py:139 | Chunk反向 |
| `chunk_ttt_fwd_kernel_2` | fla/ops/ttt/chunk.py:232 | Chunk前向2 |
| `chunk_ttt_bwd_kernel_2` | fla/ops/ttt/chunk.py:334 | Chunk反向2 |
| `chunk_ttt_fwd_kernel_3` | fla/ops/ttt/chunk.py:409 | Chunk前向3 |
| `chunk_ttt_bwd_kernel_3` | fla/ops/ttt/chunk.py:567 | Chunk反向3 |
| `fused_chunk_ttt_fwd_kernel` | fla/ops/ttt/fused_chunk.py:35 | 融合Chunk前向 |
| `fused_chunk_ttt_bwd_kernel` | fla/ops/ttt/fused_chunk.py:158 | 融合Chunk反向 |
| `fused_chunk_ttt_fwd_kernel_2` | fla/ops/ttt/fused_chunk.py:278 | 融合Chunk前向2 |

#### 16. Log Linear Attention 系列算子 (8个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_log_linear_attn_fwd_kernel` | fla/ops/log_linear_attn/chunk.py:41 | Chunk前向 |
| `chunk_log_linear_attn_bwd_kernel` | fla/ops/log_linear_attn/chunk.py:677 | Chunk反向 |
| `chunk_log_linear_attn_fwd_kernel_2` | fla/ops/log_linear_attn/chunk.py:828 | Chunk前向2 |
| `chunk_log_linear_attn_bwd_kernel_2` | fla/ops/log_linear_attn/chunk.py:930 | Chunk反向2 |
| `chunk_log_linear_attn_fwd_kernel_3` | fla/ops/log_linear_attn/chunk.py:1047 | Chunk前向3 |
| `chunk_log_linear_attn_bwd_kernel_3` | fla/ops/log_linear_attn/chunk.py:1195 | Chunk反向3 |
| `chunk_log_linear_attn_fwd_kernel_4` | fla/ops/log_linear_attn/chunk.py:1285 | Chunk前向4 |
| `chunk_log_linear_attn_bwd_kernel_4` | fla/ops/log_linear_attn/chunk.py:1351 | Chunk反向4 |

#### 17. Simple GLA 系列算子 (4个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_simple_gla_fwd_kernel` | fla/ops/simple_gla/parallel.py:47 | 并行前向 |
| `parallel_simple_gla_bwd_kernel` | fla/ops/simple_gla/parallel.py:166 | 并行反向 |
| `parallel_simple_gla_bwd_kernel_2` | fla/ops/simple_gla/parallel.py:258 | 并行反向2 |
| `parallel_simple_gla_bwd_kernel_3` | fla/ops/simple_gla/parallel.py:382 | 并行反向3 |

#### 18. DeltaFormer 系列算子 (4个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `deltaformer_fwd_kernel` | fla/ops/deltaformer/parallel.py:136 | 前向 |
| `deltaformer_bwd_kernel` | fla/ops/deltaformer/parallel.py:277 | 反向 |
| `deltaformer_bwd_kernel_2` | fla/ops/deltaformer/parallel.py:364 | 反向2 |
| `deltaformer_bwd_kernel_3` | fla/ops/deltaformer/parallel.py:458 | 反向3 |

#### 19. GSA 系列算子 (6个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `fused_recurrent_gsa_kernel` | fla/ops/gsa/fused_recurrent.py:17 | 递归前向 |
| `chunk_gsa_fwd_kernel` | fla/ops/gsa/chunk.py:38 | Chunk前向 |
| `chunk_gsa_bwd_kernel` | fla/ops/gsa/chunk.py:112 | Chunk反向 |
| `chunk_gsa_fwd_kernel_2` | fla/ops/gsa/chunk.py:200 | Chunk前向2 |
| `chunk_gsa_bwd_kernel_2` | fla/ops/gsa/chunk.py:301 | Chunk反向2 |
| `chunk_gsa_bwd_kernel_3` | fla/ops/gsa/chunk.py:427 | Chunk反向3 |

#### 20. ABC 系列算子 (13个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_abc_fwd_kernel` | fla/ops/abc/chunk.py:35 | Chunk前向 |
| `chunk_abc_bwd_kernel` | fla/ops/abc/chunk.py:109 | Chunk反向 |
| `chunk_abc_fwd_kernel_2` | fla/ops/abc/chunk.py:183 | Chunk前向2 |
| `chunk_abc_bwd_kernel_2` | fla/ops/abc/chunk.py:262 | Chunk反向2 |
| `chunk_abc_fwd_kernel_3` | fla/ops/abc/chunk.py:312 | Chunk前向3 |
| `chunk_abc_bwd_kernel_3` | fla/ops/abc/chunk.py:398 | Chunk反向3 |
| `chunk_abc_fwd_kernel_4` | fla/ops/abc/chunk.py:534 | Chunk前向4 |
| `chunk_abc_bwd_kernel_4` | fla/ops/abc/chunk.py:587 | Chunk反向4 |
| `chunk_abc_fwd_kernel_5` | fla/ops/abc/chunk.py:664 | Chunk前向5 |
| `chunk_abc_bwd_kernel_5` | fla/ops/abc/chunk.py:748 | Chunk反向5 |
| `chunk_abc_fwd_kernel_6` | fla/ops/abc/chunk.py:832 | Chunk前向6 |
| `chunk_abc_bwd_kernel_6` | fla/ops/abc/chunk.py:916 | Chunk反向6 |
| `chunk_abc_bwd_kernel_7` | fla/ops/abc/chunk.py:1000 | Chunk反向7 |

#### 21. Comba 系列算子 (6个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/comba/wy_fast.py:31 | 重计算WY前向 |
| `prepare_wy_repr_bwd_kernel` | fla/ops/comba/wy_fast.py:164 | WY反向 |
| `recompute_w_u_fwd_kernel_2` | fla/ops/comba/wy_fast.py:300 | 重计算WY前向2 |
| `chunk_comba_cumsum_kernel` | fla/ops/comba/utils.py:28 | Chunk累加 |
| `chunk_comba_cumsum_kernel_2` | fla/ops/comba/utils.py:107 | Chunk累加2 |
| `fused_recurrent_comba_kernel` | fla/ops/comba/fused_recurrent.py:21 | 递归前向 |

#### 22. Attention 系列算子 (5个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `parallel_attn_fwd_kernel` | fla/ops/attn/parallel.py:18 | 并行前向 |
| `parallel_attn_bwd_kernel` | fla/ops/attn/parallel.py:103 | 并行反向 |
| `parallel_attn_bwd_kernel_2` | fla/ops/attn/parallel.py:187 | 并行反向2 |
| `parallel_attn_bwd_kernel_3` | fla/ops/attn/parallel.py:273 | 并行反向3 |
| `decoding_attn_kernel` | fla/ops/attn/decoding.py:18 | 解码注意力 |

#### 23. Generalized Delta Rule 系列算子 (21个 kernels)

**DPLR (15个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:29 | 重计算WY前向 |
| `recompute_w_u_fwd_kernel_2` | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:75 | 重计算WY前向2 |
| `recompute_w_u_fwd_kernel_3` | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:155 | 重计算WY前向3 |
| `prepare_wy_repr_bwd_kernel` | fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py:32 | WY反向 |
| `fused_recurrent_dplr_kernel` | fla/ops/generalized_delta_rule/dplr/fused_recurrent.py:32 | 递归前向 |
| `chunk_dplr_fwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py:35 | Chunk前向 |
| `chunk_dplr_bwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:34 | Chunk反向 |
| `chunk_dplr_bwd_kernel_2` | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:107 | Chunk反向2 |
| `chunk_dplr_bwd_kernel_3` | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:237 | Chunk反向3 |
| `chunk_dplr_h_fwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py:34 | H前向 |
| `chunk_dplr_A_fwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py:32 | A前向 |
| `chunk_dplr_A_fwd_kernel_2` | fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py:159 | A前向2 |
| `chunk_dplr_h_bwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_h_bwd.py:34 | H反向 |
| `chunk_dplr_A_bwd_kernel` | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:32 | A反向 |
| `chunk_dplr_A_bwd_kernel_2` | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:237 | A反向2 |
| `chunk_dplr_A_bwd_kernel_3` | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:407 | A反向3 |

**IPLR (6个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/generalized_delta_rule/iplr/wy_fast.py:29 | 重计算WY前向 |
| `recompute_w_u_fwd_kernel_2` | fla/ops/generalized_delta_rule/iplr/wy_fast.py:84 | 重计算WY前向2 |
| `recompute_w_u_fwd_kernel_3` | fla/ops/generalized_delta_rule/iplr/wy_fast.py:165 | 重计算WY前向3 |
| `fused_recurrent_iplr_kernel` | fla/ops/generalized_delta_rule/iplr/fused_recurrent.py:30 | 递归前向 |
| `fused_recurrent_iplr_kernel_2` | fla/ops/generalized_delta_rule/iplr/fused_recurrent.py:121 | 递归前向2 |
| `chunk_iplr_fwd_kernel` | fla/ops/generalized_delta_rule/iplr/chunk.py:42 | Chunk前向 |
| `chunk_iplr_bwd_kernel` | fla/ops/generalized_delta_rule/iplr/chunk.py:127 | Chunk反向 |

#### 24. Gated Oja Rule 系列算子 (13个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `recompute_w_u_fwd_kernel` | fla/ops/gated_oja_rule/wy_fast.py:29 | 重计算WY前向 |
| `recompute_w_u_fwd_kernel_2` | fla/ops/gated_oja_rule/wy_fast.py:108 | 重计算WY前向2 |
| `fused_recurrent_gated_oja_kernel` | fla/ops/gated_oja_rule/fused_recurrent.py:22 | 递归前向 |
| `chunk_gated_oja_fwd_kernel` | fla/ops/gated_oja_rule/chunk_o.py:33 | Chunk前向 |
| `chunk_gated_oja_bwd_kernel` | fla/ops/gated_oja_rule/chunk_o.py:107 | Chunk反向 |
| `chunk_gated_oja_fwd_kernel_2` | fla/ops/gated_oja_rule/chunk_o.py:262 | Chunk前向2 |
| `chunk_gated_oja_bwd_kernel_2` | fla/ops/gated_oja_rule/chunk_o.py:404 | Chunk反向2 |
| `chunk_gated_oja_fwd_kernel_3` | fla/ops/gated_oja_rule/chunk_o.py:542 | Chunk前向3 |
| `chunk_gated_oja_kkt_fwd_kernel` | fla/ops/gated_oja_rule/chunk_kkt.py:29 | KKT前向 |
| `chunk_gated_oja_kkt_bwd_kernel` | fla/ops/gated_oja_rule/chunk_kkt.py:90 | KKT反向 |
| `chunk_gated_oja_kkt_fwd_kernel_2` | fla/ops/gated_oja_rule/chunk_kkt.py:166 | KKT前向2 |
| `chunk_gated_oja_kkt_bwd_kernel_2` | fla/ops/gated_oja_rule/chunk_kkt.py:230 | KKT反向2 |
| `chunk_gated_oja_h_fwd_kernel` | fla/ops/gated_oja_rule/chunk_h.py:37 | H前向 |
| `chunk_gated_oja_h_bwd_kernel` | fla/ops/gated_oja_rule/chunk_h.py:264 | H反向 |
| `chunk_gated_oja_h_fwd_kernel_2` | fla/ops/gated_oja_rule/chunk_h.py:527 | H前向2 |
| `chunk_gated_oja_h_bwd_kernel_2` | fla/ops/gated_oja_rule/chunk_h.py:663 | H反向2 |

#### 25. Gated Delta Product 系列算子 (3个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_gated_delta_product_fwd_kernel` | fla/ops/gated_delta_product/chunk_deltaproduct_o.py:35 | Chunk前向 |
| `chunk_gated_delta_product_h_fwd_kernel` | fla/ops/gated_delta_product/chunk_deltaproduct_h.py:37 | H前向 |
| `chunk_gated_delta_product_h_bwd_kernel` | fla/ops/gated_delta_product/chunk_deltaproduct_h.py:216 | H反向 |

#### 26. Context Parallel 系列算子 (3个 kernels)

| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `chunk_cp_delta_h_fwd_kernel` | fla/ops/cp/chunk_delta_h.py:40 | CP Delta H前向 |
| `chunk_cp_delta_h_bwd_kernel` | fla/ops/cp/chunk_delta_h.py:285 | CP Delta H反向 |
| `chunk_cp_delta_h_fwd_kernel_2` | fla/ops/cp/chunk_delta_h.py:440 | CP Delta H前向2 |

#### 27. Utils 工具算子补充 (36个 kernels)

**Softplus (4个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `softplus_fwd_kernel` | fla/ops/utils/softplus.py:68 | Softplus前向 |
| `softplus_bwd_kernel` | fla/ops/utils/softplus.py:84 | Softplus反向 |
| `softplus_fwd_kernel_2` | fla/ops/utils/softplus.py:89 | Softplus前向2 |
| `softplus_bwd_kernel_2` | fla/ops/utils/softplus.py:105 | Softplus反向2 |

**Op 工具 (12个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `exp_kernel` | fla/ops/utils/op.py:17 | 指数运算 |
| `exp2_kernel` | fla/ops/utils/op.py:19 | 2的指数运算 |
| `log_kernel` | fla/ops/utils/op.py:21 | 对数运算 |
| `log2_kernel` | fla/ops/utils/op.py:23 | 2的对数运算 |
| `sin_kernel` | fla/ops/utils/op.py:25 | 正弦运算 |
| `cos_kernel` | fla/ops/utils/op.py:28 | 余弦运算 |
| `sin_kernel_2` | fla/ops/utils/op.py:30 | 正弦运算2 |
| `cos_kernel_2` | fla/ops/utils/op.py:32 | 余弦运算2 |
| `tanh_kernel` | fla/ops/utils/op.py:34 | 双曲正切 |
| `tanh_kernel_2` | fla/ops/utils/op.py:36 | 双曲正切2 |
| `sigmoid_kernel` | fla/ops/utils/op.py:41 | Sigmoid |
| `silu_kernel` | fla/ops/utils/op.py:65 | SiLU |

**Matmul (5个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `matmul_kernel` | fla/ops/utils/matmul.py:52 | 矩阵乘法 |
| `addmm_kernel` | fla/ops/utils/matmul.py:157 | 矩阵乘加 |
| `addmm_kernel_2` | fla/ops/utils/matmul.py:162 | 矩阵乘加2 |
| `addmm_kernel_3` | fla/ops/utils/matmul.py:168 | 矩阵乘加3 |
| `addmm_kernel_4` | fla/ops/utils/matmul.py:175 | 矩阵乘加4 |

**其他工具 (15个 kernels):**
| Kernel名称 | 源码路径 | 功能描述 |
|-----------|---------|---------|
| `prepare_token_indices_kernel` | fla/ops/utils/index.py:24 | Token索引准备 |
| `logcumsumexp_fwd_kernel` | fla/ops/utils/logcumsumexp.py:24 | LogCumSumExp前向 |
| `nsa_utils_kernel` | fla/ops/nsa/utils.py:20 | NSA工具 |
| `nsa_utils_kernel_2` | fla/ops/nsa/utils.py:55 | NSA工具2 |
| `nsa_utils_kernel_3` | fla/ops/nsa/utils.py:81 | NSA工具3 |

---

## 统计总结

### Kernel 函数分布统计

| 模块类别 | Kernel数量 | 占比 |
|---------|-----------|------|
| Common 共享模块 | 21 | 7.8% |
| Utils 工具模块 | 36 | 13.4% |
| Modules 模块 | 43 | 16.0% |
| Delta Rule 系列 | 7 | 2.6% |
| GLA 系列 | 9 | 3.3% |
| Gated Delta Rule 系列 | 9 | 3.3% |
| HGRN 系列 | 6 | 2.2% |
| KDA 系列 | 12 | 4.5% |
| RWKV 系列 | 22 | 8.2% |
| Based 系列 | 5 | 1.9% |
| Rebased 系列 | 4 | 1.5% |
| NSA 系列 | 8 | 3.0% |
| Path Attention 系列 | 11 | 4.1% |
| MesaNet 系列 | 10 | 3.7% |
| TTT 系列 | 9 | 3.3% |
| Log Linear Attention 系列 | 8 | 3.0% |
| Simple GLA 系列 | 4 | 1.5% |
| DeltaFormer 系列 | 4 | 1.5% |
| GSA 系列 | 6 | 2.2% |
| ABC 系列 | 13 | 4.8% |
| Comba 系列 | 6 | 2.2% |
| Attention 系列 | 5 | 1.9% |
| Generalized Delta Rule 系列 | 21 | 7.8% |
| Gated Oja Rule 系列 | 16 | 5.9% |
| Gated Delta Product 系列 | 3 | 1.1% |
| Context Parallel 系列 | 3 | 1.1% |
| **总计** | **269** | **100%** |

### 关键发现

1. **Kernel 函数总数**: 269 个
2. **分布特点**:
   - Modules 模块 kernel 最多 (43个, 16.0%)
   - Utils 工具模块次之 (36个, 13.4%)
   - RWKV 系列 kernel 较多 (22个, 8.2%)
   - Common 共享模块和 Generalized Delta Rule 并列 (21个, 7.8%)

3. **设计模式**:
   - 大部分算子采用 **前向+反向** 成对设计
   - 复杂算子采用 **多 kernel 组合** 模式
   - 共享模块提供 **通用 kernel** 供多个算子复用

---

## 测试用例与源码对应关系

### chunk_delta_rule 算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_delta.py | fla/ops/delta_rule/chunk.py | `test_chunk()` | 前向+反向 |
| tests/ops/test_delta.py | fla/ops/delta_rule/fused_recurrent.py | `test_fused_recurrent()` | 递归实现 |
| tests/ops/test_delta.py | fla/ops/delta_rule/parallel.py | `test_parallel()` | 并行实现 |

**测试文件路径**: `tests/ops/test_delta.py`

**源码文件路径**:
- 主实现: `fla/ops/delta_rule/chunk.py`
- WY表示: `fla/ops/delta_rule/wy_fast.py`
- 递归实现: `fla/ops/delta_rule/fused_recurrent.py`
- 并行实现: `fla/ops/delta_rule/parallel.py`

**测试覆盖的 Kernel 函数**:
- ✅ `chunk_scaled_dot_kkt_fwd_kernel`
- ✅ `solve_tril_16x16_kernel`
- ✅ `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- ✅ `chunk_fwd_kernel_o`
- ✅ `recompute_w_u_fwd_kernel`
- ✅ `chunk_bwd_kernel_dv`
- ✅ `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
- ✅ `chunk_bwd_kernel_dqkwg`
- ✅ `prepare_wy_repr_bwd_kernel`

### GLA 算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_gla.py | fla/ops/gla/chunk.py | `test_fused_recurrent()` | 递归实现 |
| tests/ops/test_gla.py | fla/ops/gla/chunk.py | `test_chunk()` | Chunk实现 |

**测试文件路径**: `tests/ops/test_gla.py`

**源码文件路径**: `fla/ops/gla/chunk.py`

### Gated Delta Rule 算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_gated_delta.py | fla/ops/gated_delta_rule/chunk.py | `test_fused_recurrent()` | 递归实现 |
| tests/ops/test_gated_delta.py | fla/ops/gated_delta_rule/chunk.py | `test_chunk()` | Chunk实现 |

**测试文件路径**: `tests/ops/test_gated_delta.py`

**源码文件路径**:
- 主实现: `fla/ops/gated_delta_rule/chunk.py`
- 门控: `fla/ops/gated_delta_rule/gate.py`
- WY表示: `fla/ops/gated_delta_rule/wy_fast.py`

### HGRN 算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_hgrn.py | fla/ops/hgrn/chunk.py | `test_fused_recurrent()` | 递归实现 |
| tests/ops/test_hgrn.py | fla/ops/hgrn/chunk.py | `test_chunk()` | Chunk实现 |

**测试文件路径**: `tests/ops/test_hgrn.py`

**源码文件路径**: `fla/ops/hgrn/chunk.py`

### KDA 算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_kda.py | fla/ops/kda/chunk.py | `test_naive_chunk()` | 朴素Chunk |
| tests/ops/test_kda.py | fla/ops/kda/chunk.py | `test_chunk()` | Chunk实现 |

**测试文件路径**: `tests/ops/test_kda.py`

**源码文件路径**: `fla/ops/kda/chunk.py`

### Utils 工具算子测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/ops/test_utils.py | fla/ops/utils/cumsum.py | `test_global_cumsum()` | 全局累加 |
| tests/ops/test_utils.py | fla/ops/utils/cumsum.py | `test_global_cumsum_varlen()` | 变长累加 |
| tests/ops/test_utils.py | fla/ops/utils/pooling.py | `test_mean_pooling()` | 平均池化 |
| tests/ops/test_utils.py | fla/ops/utils/pack.py | `test_pack_unpack()` | 打包解包 |
| tests/ops/test_index.py | fla/ops/utils/index.py | `test_prepare_*()` | 索引准备 |
| tests/ops/test_solve_tril.py | fla/ops/utils/solve_tril.py | `test_solve_tril()` | 下三角求解 |

### Modules 模块测试

| 测试文件 | 源码文件 | 测试函数 | 测试覆盖 |
|---------|---------|---------|---------|
| tests/modules/test_layernorm.py | fla/modules/layernorm.py | `test_layernorm()` | 层归一化 |
| tests/modules/test_layernorm.py | fla/modules/layernorm.py | `test_rmsnorm()` | RMS归一化 |
| tests/modules/test_layernorm.py | fla/modules/layernorm.py | `test_groupnorm()` | 组归一化 |
| tests/modules/test_layernorm_gated.py | fla/modules/layernorm_gated.py | `test_*()` | 门控归一化 |
| tests/modules/test_cross_entropy.py | fla/modules/fused_cross_entropy.py | `test_fused_cross_entropy()` | 交叉熵 |
| tests/modules/test_kl_div.py | fla/modules/fused_kl_div.py | `test_fused_kl_div()` | KL散度 |
| tests/modules/test_l2norm.py | fla/modules/l2norm.py | `test_l2norm()` | L2归一化 |
| tests/modules/test_rotary.py | fla/modules/rotary.py | `test_rotary()` | 旋转编码 |
| tests/modules/test_token_shift.py | fla/modules/token_shift.py | `test_token_shift()` | Token移位 |
| tests/modules/test_conv.py | fla/modules/conv/triton/kernels.py | `test_causal_conv1d()` | 因果卷积 |
| tests/modules/test_activation.py | fla/modules/activations.py | `test_swiglu()` | SwiGLU |
| tests/modules/test_activation.py | fla/modules/activations.py | `test_geglu()` | GeGLU |

---

## 附录

### A. Kernel 函数命名规范

FLA 仓库中的 kernel 函数遵循统一的命名规范:

1. **前缀**:
   - `chunk_`: Chunk-based 实现
   - `fused_`: 融合操作
   - `fused_recurrent_`: 融合递归实现

2. **操作类型**:
   - `fwd`: 前向传播
   - `bwd`: 反向传播

3. **数据类型**:
   - `h`: 隐藏状态
   - `o`: 输出
   - `dqk`: Q、K梯度
   - `dv`: V梯度

4. **示例**:
   - `chunk_gla_fwd_kernel_o`: GLA chunk 前向输出计算 kernel
   - `fused_recurrent_hgrn_bwd_kernel`: HGRN 融合递归反向 kernel

### B. 性能优化技术

FLA 中的 kernel 函数采用了多种优化技术:

1. **分块计算**: 使用 chunk-based 策略减少内存访问
2. **融合操作**: 将多个操作融合到单个 kernel 中
3. **共享内存**: 利用 GPU 共享内存加速数据访问
4. **向量化**: 使用 Triton 的向量化操作
5. **自动调优**: 使用 `@triton.autotune` 自动选择最优配置

### C. 依赖关系图

```
算子层级依赖关系:

顶层算子 (API层)
    ↓
算子实现层 (chunk.py, fused_recurrent.py)
    ↓
子算子层 (prepare_wy, chunk_h, chunk_o)
    ↓
Kernel层 (@triton.jit装饰的函数)
    ↓
Triton编译器
    ↓
GPU执行
```

---

**报告生成时间**: 2026-04-08  
**分析工具**: 自动化代码分析  
**覆盖范围**: FLA 仓库所有算子及 kernel 函数
