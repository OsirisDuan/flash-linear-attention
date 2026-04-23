# T3 任务报告：Kernel复用度统计

## 1. 任务概述

| 项目 | 内容 |
|------|------|
| 任务ID | T3 |
| 任务名称 | Kernel复用度统计 |
| 执行时间 | 2026-04-22 |
| 执行状态 | ✅ 已完成（含评分规则优化） |
| 输入文件 | agent_file/tmp_file/t2_kernel_operator_model_mapping.csv |
| 输出文件 | agent_file/tmp_file/t3_kernel_reuse_stats.csv |
| 执行脚本 | agent_file/script/t3_kernel_reuse_stats.py |

## 2. 任务目标

统计每个 kernel 被多少个算子/模块共享使用，回答"修改该 kernel 的影响范围有多大？"。采用双轨制策略：

- **全局排名视角**：核心算子 kernel 正常评分，共享模块 kernel 复用度降权（上限 2 分），防止霸榜
- **共享模块内部视角**：共享模块 kernel 使用原始复用度得分，用于模块内优先级比较

## 3. 执行过程

### 3.1 执行步骤

1. **步骤 3.1：读取 T2 数据**
   - 解析 `t2_kernel_operator_model_mapping.csv`，获取 312 条 kernel 记录

2. **步骤 3.2：扫描 import 链**
   - 扫描 `fla/ops/` 下算子对 `common` 和 `utils` 的 import 引用
   - 扫描算子之间的跨算子引用
   - 扫描 `fla/models/` 和 `fla/layers/` 对 `fla/modules/` 的引用

3. **步骤 3.3：计算每个 kernel 的复用度**
   - 核心算子 kernel：自身算子 + 跨算子引用方
   - `fla/ops/common/` kernel：import 该 common 模块的算子
   - `fla/ops/utils/` kernel：import 该 utils 模块的算子
   - `fla/modules/` kernel：模块自身 + 使用该模块的模型 + 引用该模块的 Layer

4. **步骤 3.4：双轨制评分**
   - 全局排名得分：核心算子=原始得分，共享模块=min(原始得分, 2)
   - 模块内排名得分：均=原始得分（不降权）

### 3.2 关键技术决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| kernel 唯一标识 | Kernel源码路径 + Kernel行号 | 存在 24 组同名 kernel，需用路径+行号区分 |
| 复用度定义 | 被多少个算子/模块/模型/Layer 使用 | 统一维度，反映修改影响范围 |
| `fla/modules/` 复用度 | 模块自身 + 使用模型 + 引用 Layer | 模块被模型直接使用和被 Layer 间接使用都算复用 |
| 降权上限 | min(原始得分, 2) | 按 TASK_PLAN 规定，共享模块 kernel 全局排名得分不超过 2 |
| 复用度分级 | ≥4=极高(5分), 3=高(4分), 2=中(3分), 1+≥2模型=低(2分), 1+≤1模型=极低(1分) | SKILL.md 规则，count=1 时按模型数区分，提升区分度 |

### 3.3 评分规则优化

初始版本使用简单的 4 级分级（≥4=5分, 3=4分, 2=3分, 1=1分），导致两极分化严重：

| 来源 | 1分占比 | 5分占比 | 信息熵 |
|------|---------|---------|--------|
| 核心算子 | 86.3% | 6.6% | 0.764 bits |
| 共享模块 | 36.0% | 43.0% | 1.734 bits |

优化后采用 SKILL.md 规则，count=1 时按模型数区分：

| 来源 | 1分占比 | 2分占比 | 5分占比 | 信息熵 |
|------|---------|---------|---------|--------|
| 核心算子 | 76.9% | 9.4% | 6.6% | 1.193 bits (+56%) |
| 共享模块 | 26.0% | 10.0% | 43.0% | 2.041 bits (+18%) |

核心算子信息熵提升 56%，20 个"1算子+多模型"的 kernel 从 1 分提升到 2 分，语义合理——"1个算子但被多个模型使用"确实比"1个算子1个模型"影响范围更大。

### 3.4 遇到的问题与解决

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| 同名 kernel 区分 | 24 组同名 kernel 需区分 | 使用 `Kernel源码路径:Kernel行号` 作为唯一标识 |
| `utils/op.py` 重复记录 | `exp/exp2/log/log2/tanh` 各出现 2 次（if/else 分支） | 保留 T2 原始数据，两条记录复用度相同 |
| `safe_dot` 重复记录 | 同一路径同一行号出现 2 次 | 保留 T2 原始数据，两条记录复用度相同 |
| `fla/modules/` 复用度计算 | 初始版本混入了文件名（如 `modeling_bitnet`） | 改为统计模型名称和 Layer 名称，避免文件名污染 |
| 评分两极分化 | 86% 核心算子 kernel 都是 1 分 | 采用 SKILL.md 规则，count=1 时按模型数区分 1 分 vs 2 分 |

## 4. 执行结果

### 4.1 总体数据

| 指标 | 数量 |
|------|------|
| 总 kernel 数 | 312 |
| 核心算子 kernel | 212 |
| 共享模块 kernel | 100 |

### 4.2 复用度分布

| 复用度级别 | 总数 | 核心算子 | 共享模块 |
|-----------|------|---------|---------|
| 极高 (≥4) | 57 | 14 | 43 |
| 高 (3) | 12 | 3 | 9 |
| 中 (2) | 24 | 12 | 12 |
| 低 (1算子+≥2模型) | 30 | 20 | 10 |
| 极低 (1算子+≤1模型) | 189 | 163 | 26 |

### 4.3 高复用度 kernel (≥3个算子/模块)

#### 核心算子 kernel

| Kernel | 算子/模块数 | 使用者列表 |
|--------|-----------|-----------|
| chunk_fwd_kernel_h | 5 | gla, gsa, mesa_net, rwkv6, simple_gla |
| chunk_bwd_kernel_dh | 5 | gla, gsa, mesa_net, rwkv6, simple_gla |
| chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4 | comba, delta_rule, gated_delta_rule, kda |
| chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | 4 | comba, delta_rule, gated_delta_rule, kda |
| chunk_fwd_kernel_o | 4 | comba, delta_rule, gated_delta_rule, simple_gla |
| chunk_bwd_kernel_dqkwg | 4 | comba, delta_rule, gated_delta_rule, simple_gla |
| chunk_bwd_kernel_dv | 4 | comba, delta_rule, gated_delta_rule, simple_gla |
| chunk_bwd_kernel_dv_local | 4 | comba, delta_rule, gated_delta_rule, simple_gla |
| chunk_scaled_dot_kkt_fwd_kernel | 3 | delta_rule, gated_delta_product, path_attn |
| fused_recurrent_fwd_kernel | 3 | gla, gsa, simple_gla |
| fused_recurrent_bwd_kernel | 3 | gla, gsa, simple_gla |
| chunk_global_cumsum_kernel | 5 | attn, gla, gsa, path_attn, simple_gla |
| chunk_global_cumsum_scalar_kernel | 5 | attn, gla, gsa, path_attn, simple_gla |
| chunk_global_cumsum_vector_kernel | 5 | attn, gla, gsa, path_attn, simple_gla |
| prepare_position_ids_kernel | 6 | comba, delta_rule, gated_delta_product, gated_delta_rule, gated_oja_rule, kda |

#### 共享模块 kernel (top 10)

| Kernel | 算子/模块/模型数 | 来源 |
|--------|----------------|------|
| exp/exp2/log/log2/tanh | 20 | utils/op |
| gather/make_tensor_descriptor | 20 | utils/op |
| sigmoid_fwd/bwd_kernel | 12 | fla/modules/activations |
| swish_fwd/bwd_kernel | 12 | fla/modules/activations |
| swiglu_fwd/fwdbwd_kernel | 12 | fla/modules/activations |
| logsigmoid_fwd/bwd_kernel | 12 | fla/modules/activations |
| layer_norm_fwd/bwd_kernel | 3-4 | fla/modules/layernorm* |
| rms_norm_fwd/bwd_kernel | 2 | fla/modules/layernorm |

### 4.4 低复用度但多模型的 kernel（SKILL.md 规则新增 2 分）

| Kernel | 算子 | 模型数 | 复用度级别 |
|--------|------|--------|-----------|
| chunk_rwkv6_fwd_cumsum_kernel | RWKV6 | 2 | 低(2分) |
| chunk_rwkv6_fwd_A_kernel_* | RWKV6 | 2 | 低(2分) |
| fused_recurrent_rwkv6_*_kernel | RWKV6 | 2 | 低(2分) |
| naive_attn_decoding_kernel | Attn | 4 | 极高(5分) |
| parallel_attn_fwd_kernel | Attn | 4 | 极高(5分) |

### 4.5 双轨制得分验证

| 验证项 | 结果 |
|--------|------|
| 共享模块 kernel 全局得分 ≤ 2 | ✅ 全部通过 |
| 共享模块 kernel 模块内得分 = 原始得分 | ✅ 全部通过 |
| 核心算子 kernel 全局得分 = 原始得分 | ✅ 全部通过 |

## 5. 质量验证

### 5.1 已知共享 kernel 验证

| 验证项 | 期望 | 实际 | 结果 |
|--------|------|------|------|
| `chunk_fwd_kernel_h` | ≥4 个算子 | 5 个 (gla,gsa,mesa_net,rwkv6,simple_gla) | ✅ |
| `fused_recurrent_fwd_kernel` | ≥3 个算子 | 3 个 (gla,gsa,simple_gla) | ✅ |

### 5.2 import 链完整性

| 来源 | 被引用模块数 | 验证 |
|------|------------|------|
| common 模块 | 7 个 | ✅ |
| utils 模块 | 10 个 | ✅ |
| 跨算子引用 | 7 个 | ✅ |
| fla.modules 引用 | 43 个 | ✅ |

### 5.3 双轨制验证

| 验证项 | 结果 |
|--------|------|
| 全局排名：所有共享模块 kernel 复用度得分 ≤ 2 | ✅ 通过 |
| 模块内排名：共享模块 kernel 使用原始得分（无降权） | ✅ 通过 |

### 5.4 评分规则区分度验证

| 来源 | 旧信息熵 | 新信息熵 | 提升 |
|------|---------|---------|------|
| 全部 | 1.272 bits | 1.676 bits | +32% |
| 核心算子 | 0.764 bits | 1.193 bits | +56% |
| 共享模块 | 1.734 bits | 2.041 bits | +18% |

## 6. 对后续任务的影响

### 6.1 输出数据对 T5 的影响

- T5 综合评分需要使用 `全局排名复用度得分`（权重 30%）
- T5 模块内排名需要使用 `模块内排名复用度得分`
- 复用度权重 30% 在全局排名中，共享模块 kernel 的复用度得分被限制在 ≤2，不会霸榜
- 新增 `使用模型数量` 列，可用于交叉验证

### 6.2 需注意的异常情况

1. **同名 kernel**：24 组同名 kernel 已通过 `Kernel源码路径:Kernel行号` 区分，T5 需使用同样的唯一标识

2. **`utils/op.py` 中的极简工具函数**：`exp/exp2/log/log2/tanh/gather/make_tensor_descriptor` 被几乎所有算子使用（20个），复用度极高(5分)。但这些是极简的 pointwise 操作或条件编译辅助函数，实际修改风险很低。T5 评分时需注意这些 kernel 的复用度得分虽然高，但综合优先级可能不高

3. **`fla/modules/` kernel 的复用度维度**：统计的是"模块+模型+Layer"的混合维度，与核心算子的"算子"维度不完全一致。这是合理的，因为模块的消费者是模型和 Layer

4. **无复用（count=0）的 kernel**：`common/chunk_h_parallel`(4条)、`common/chunk_h_split`(4条)、`common/fused_chunk`(2条)、`utils/logsumexp`(1条)、`utils/matmul`(5条)、`utils/pack`(1条) 没有被任何算子 import，复用度为 0。这些可能是废弃代码或仅通过其他方式调用

5. **`fla/modules/` 中 count=1 的 kernel**：`fused_kl_div`(2条)、`grpo`(2条) 只被自身模块引用，没有模型/Layer 使用。这些是训练辅助模块，不影响推理

6. **评分规则变更影响**：count=1 的 kernel 从统一 1 分改为按模型数区分 1 分/2 分，163 个保持 1 分，20 个提升到 2 分。T5 综合评分时需注意此变更

## 7. 输出文件示例

```csv
Kernel函数名称,kernel来源,共享算子/模块数量,共享算子/模块列表,使用模型数量,复用度级别,原始复用度得分,全局排名复用度得分,模块内排名复用度得分,是否为common目录kernel,Kernel源码路径,Kernel行号
chunk_abc_fwd_kernel_h,核心算子,1,ABC,0,极低,1,1,1,否,fla/ops/abc/chunk.py,18
chunk_fwd_kernel_h,共享模块,5,gla,gsa,mesa_net,rwkv6,simple_gla,5,极高,5,2,5,否,fla/ops/common/chunk_h.py,34
sigmoid_fwd_kernel,共享模块,12,activations,bitnet,...,12,极高,5,2,5,否,fla/modules/activations.py,85
```
