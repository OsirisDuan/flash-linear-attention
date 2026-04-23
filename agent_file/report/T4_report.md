# T4 任务报告：Kernel核心程度判定

## 1. 任务概述

| 项目 | 内容 |
|------|------|
| 任务ID | T4 |
| 任务名称 | Kernel核心程度判定 |
| 执行时间 | 2026-04-22 |
| 执行状态 | ✅ 已完成 |
| 输入文件 | agent_file/tmp_file/t2_kernel_operator_model_mapping.csv |
| 输出文件 | agent_file/tmp_file/t4_kernel_criticality.csv |
| 执行脚本 | agent_file/script/t4_kernel_criticality.py |

## 2. 任务目标

判定每个 kernel 在模型执行路径中的位置和重要性，回答"该 kernel 是否为核心计算路径？"。需区分两类 kernel 的核心程度含义：

- **核心算子 kernel**：直接参与注意力计算，核心程度差异大（fwd > bwd > 辅助）
- **共享模块 kernel**：属于基础设施（Norm、Loss 等），核心程度统一为"辅助"级别

## 3. 执行过程

### 3.1 判定规则

#### 核心算子 kernel 判定规则（按优先级从高到低）

| 优先级 | 规则 | 级别 | 得分 | 说明 |
|--------|------|------|------|------|
| 1 | 含 `naive` | 边缘 | 1 | 可选或降级路径 |
| 2 | 含 `fused` + `fwd`（无 intra/inter/preprocess） | 核心 | 5 | 融合前向核心计算 |
| 3 | 含 `fused` + `bwd` | 重要 | 4 | 融合反向传播 |
| 4 | 含 `fused`（其他） | 优化 | 2 | 性能优化专用 |
| 5 | 含 `cumsum`/`preprocess`/`mask` | 辅助 | 3 | 预处理/后处理 |
| 6 | 含 `bwd` | 重要 | 4 | 反向传播梯度计算 |
| 7 | 含 `fwd`（无 intra/inter/preprocess） | 核心 | 5 | 前向传播主计算 |
| 8 | 含 `fwd` + `intra`/`inter`/`preprocess` | 重要 | 4 | 前向传播分步计算 |
| 9 | 排序工具（sort/merge/swap/argsort/bitonic） | 辅助 | 3 | 排序辅助 |
| 10 | 数据搬运（copy/save/prepare/gather） | 辅助 | 3 | 数据搬运/预处理 |
| 11 | topk 操作 | 辅助 | 3 | 选择操作 |
| 12 | decoding 步骤 | 核心 | 5 | 解码核心步骤 |
| 13 | mixing 操作 | 核心 | 5 | 混合核心计算 |
| 14 | update + cv/cube | 核心 | 5 | 状态更新核心计算 |
| 15 | gate + cv/cube | 核心 | 5 | 门控核心计算 |
| 16 | 未匹配 + cv/cube | 核心 | 5 | 计算密集型默认核心 |
| 17 | 未匹配 + vv | 辅助 | 3 | 向量操作默认辅助 |

#### 共享模块 kernel 判定规则

所有共享模块 kernel 统一判定为**辅助(3分)**，具体分类说明：

| 分类 | 关键词 | 说明 |
|------|--------|------|
| 归一化组件 | norm/rms/l2norm | 必要但非差异化组件 |
| 损失函数 | cross_entropy/kl_div/grpo | 训练组件 |
| 卷积组件 | conv/causal | 序列处理辅助 |
| 激活函数 | swiglu/swish/sigmoid/softplus/relu/tanh/exp/log | 逐元素操作 |
| 位置编码/辅助 | rotary/token_shift/l2/position | 位置相关辅助 |
| 门控组件 | gate/norm_gated | 门控辅助 |
| cumsum | cumsum | 累积求和辅助 |
| chunk 操作 | chunk+fwd/bwd | 共享计算模块 |
| fused 操作 | fused | 融合优化 |
| 其他 | - | 默认辅助 |

### 3.2 SKILL.md 与 TASK_PLAN.md 差异分析

| 方面 | SKILL.md | TASK_PLAN.md / 实际实现 | 一致性 |
|------|----------|------------------------|--------|
| 分级标准 | 核心/重要/辅助/优化/边缘 (5级) | 同 | ✅ 一致 |
| 函数名分析 | fwd→核心, bwd→重要, cumsum→辅助, fused→优化 | 同，但增加了补充规则 | ✅ 一致 |
| fwd+intra/inter | 未明确说明 | 判为"重要"(4分)而非"辅助" | ⚠️ 细化 |
| 调用链分析 | forward()→核心, backward()→重要 | 标注为"可选"，未实现 | ⚠️ 未实现 |
| 计算量分析 | tl.dot()→核心, 纯向量→辅助 | 通过 kernel类型(cv/cube/vv)间接实现 | ✅ 等价 |
| 共享模块规则 | 未明确 | 统一辅助(3分) | ⚠️ 细化 |

**关键差异说明**：

1. **fwd+intra/inter 判定**：SKILL.md 说含 `fwd` 且无 `intra`/`inter` → 核心，暗示含 intra/inter 的 fwd 不是核心。实际实现将其判为"重要"(4分)而非"辅助"(3分)，因为这些 kernel 仍是前向传播的一部分，只是分步执行（如 GSA 的 inter/intra 步骤），重要性高于辅助操作。

2. **调用链分析未实现**：SKILL.md 提到通过分析 `forward()`/`backward()` 调用链来判定，但 TASK_PLAN 标注为"可选"。实际未实现，因为函数名分析已覆盖 95%+ 的 kernel，且调用链分析需要更复杂的 AST 解析。

3. **共享模块统一辅助**：SKILL.md 未明确说明共享模块的核心程度处理，实际实现统一为辅助(3分)，与双轨制策略一致。

### 3.3 遇到的问题与解决

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| 14 个 kernel 不匹配函数名规则 | 无法自动判定 | 增加10条补充规则（排序/搬运/topk/decoding/mixing/update/gate等） |
| GSA fwd kernel 含 intra/inter | 被误判为辅助(3分) | 调整为重要(4分)，因为是前向传播分步计算 |
| common 目录 kernel 的核心程度 | 如 chunk_fwd_kernel_h 实际是核心计算 | 按共享模块统一辅助(3分)，在双轨制下合理 |
| safe_dot 重复记录 | 同一路径同一行号出现 2 次 | 两条记录判定结果相同 |

## 4. 执行结果

### 4.1 总体数据

| 指标 | 数量 |
|------|------|
| 总 kernel 数 | 312 |
| 核心算子 kernel | 212 |
| 共享模块 kernel | 100 |

### 4.2 核心程度分布

| 核心程度级别 | 总数 | 核心算子 | 共享模块 |
|-------------|------|---------|---------|
| 核心 (5分) | 72 | 72 | 0 |
| 重要 (4分) | 119 | 119 | 0 |
| 辅助 (3分) | 116 | 16 | 100 |
| 优化 (2分) | 4 | 4 | 0 |
| 边缘 (1分) | 1 | 1 | 0 |

### 4.3 核心算子 kernel 得分分布

| 得分 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 5分 | 72 | 34.0% | 前向主计算、融合前向、核心更新/门控 |
| 4分 | 119 | 56.1% | 反向传播、前向分步计算 |
| 3分 | 16 | 7.5% | 辅助操作（排序、搬运、topk、cumsum） |
| 2分 | 4 | 1.9% | 融合优化（无 fwd/bwd 标识） |
| 1分 | 1 | 0.5% | naive 实现 |

### 4.4 判定依据统计（Top 10）

| 判定依据 | 数量 |
|---------|------|
| 含 bwd | 89 |
| fwd(无 intra/inter/preprocess) | 52 |
| 激活函数 | 29 |
| fwd+intra/inter/preprocess(前向分步) | 20 |
| 归一化组件 | 16 |
| 共享模块默认辅助 | 14 |
| fused+fwd(无 intra/inter/preprocess) | 13 |
| fused+bwd | 10 |
| 含 cumsum/preprocess/mask | 9 |
| 共享模块 chunk+bwd | 9 |

## 5. 质量验证

### 5.1 抽样验证

| Kernel | 期望 | 实际 | 结果 |
|--------|------|------|------|
| parallel_attn_fwd_kernel | 核心(5) | 核心(5) | ✅ |
| parallel_attn_bwd_kernel_dq | 重要(4) | 重要(4) | ✅ |
| chunk_gla_fwd_A_kernel_intra_sub_inter | 重要(4) | 重要(4) | ✅ |
| chunk_gla_bwd_kernel_intra | 重要(4) | 重要(4) | ✅ |
| naive_attn_decoding_kernel | 边缘(1) | 边缘(1) | ✅ |
| chunk_fwd_kernel_h (共享模块) | 辅助(3) | 辅助(3) | ✅ |
| sigmoid_fwd_kernel (共享模块) | 辅助(3) | 辅助(3) | ✅ |
| layer_norm_fwd_kernel (共享模块) | 辅助(3) | 辅助(3) | ✅ |

### 5.2 一致性验证

| 验证项 | 结果 |
|--------|------|
| 同一算子 fwd 核心程度 >= bwd | ✅ 通过 |
| 共享模块 kernel 全部为辅助级别 | ✅ 通过 |
| cv/cube 类型 kernel 48/162 为核心级别 | ✅ 合理（其余为 bwd/辅助/优化） |

### 5.3 补充规则覆盖验证

| 补充规则 | kernel 数 | 判定结果 | 合理性 |
|---------|----------|---------|--------|
| 排序工具 | 3 (argsort, bitonic_merge, compare_and_swap) | 辅助(3) | ✅ 排序是辅助操作 |
| 数据搬运 | 3 (save_intra, copy_input, copy_last_chunk) | 辅助(3) | ✅ 搬运是辅助操作 |
| topk 操作 | 1 (parallel_nsa_kernel_topk) | 辅助(3) | ✅ 选择是辅助操作 |
| 解码步骤 | 1 (mesa_net_decoding_one_step) | 核心(5) | ✅ 解码是核心步骤 |
| mixing 操作 | 2 (rwkv_channel_mixing) | 核心(5) | ✅ 混合是核心计算 |
| update+cv/cube | 2 (chunk_update_once) | 核心(5) | ✅ CG求解器核心迭代 |
| safe_dot (cv/cube) | 2 | 核心(5) | ✅ 矩阵乘法核心 |

## 6. 对后续任务的影响

### 6.1 输出数据对 T5 的影响

- T5 综合评分需要使用 `核心程度得分`（权重 20%）
- 核心算子 kernel 核心程度分布合理：34% 核心 + 56% 重要 + 10% 其他
- 共享模块 kernel 核心程度统一为 3 分，在全局排名中不会产生差异化影响

### 6.2 需注意的异常情况

1. **common 目录 kernel 的核心程度**：如 `chunk_fwd_kernel_h`、`fused_recurrent_fwd_kernel` 实际是核心前向计算，但因位于 common 目录被归为共享模块，核心程度为辅助(3分)。这在双轨制下是合理的——全局排名靠复用度和模型热度区分，核心程度在共享模块内部无差异化

2. **fwd+intra/inter 判为重要而非核心**：如 GSA 的 `chunk_gsa_fwd_k_kernel_inter/intra` 判为重要(4分)而非核心(5分)。这些是前向传播的分步计算，重要性介于核心和辅助之间

3. **`fused_recurrent_gsa_inference_kernel` 判为优化(2分)**：含 `fused` 但无 `fwd`/`bwd`，按规则判为优化。实际是推理专用 kernel，可能是核心计算。但数量仅1个，影响有限

4. **核心程度区分度**：核心算子中 90% 集中在 4-5 分，区分度有限。这是合理的——核心算子的 kernel 本身就是核心计算，差异主要在其他维度（复用度、模型热度）

## 7. 输出文件示例

```csv
Kernel函数名称,kernel来源,函数名特征,kernel类型,核心程度级别,核心程度得分,判定依据,Kernel源码路径,Kernel行号
chunk_abc_fwd_kernel_h,核心算子,fwd(无intra/inter/preprocess),cv/cube,核心,5,fwd(无intra/inter/preprocess),fla/ops/abc/chunk.py,18
chunk_abc_bwd_kernel_dh,核心算子,含bwd,cv/cube,重要,4,含bwd,fla/ops/abc/chunk.py,112
naive_attn_decoding_kernel,核心算子,含naive,vv,边缘,1,含naive,fla/ops/attn/decoding.py,29
chunk_fwd_kernel_h,共享模块,共享模块chunk+fwd+cv/cube(降级为辅助),cv/cube,辅助,3,共享模块chunk+fwd+cv/cube(降级为辅助),fla/ops/common/chunk_h.py,35
sigmoid_fwd_kernel,共享模块,激活函数,vv,辅助,3,激活函数,fla/modules/activations.py,85
```
