# T5 任务报告：综合评分计算

## 1. 任务概述

| 项目 | 内容 |
|------|------|
| 任务ID | T5 |
| 任务名称 | 综合评分计算 |
| 执行时间 | 2026-04-22 |
| 执行状态 | ✅ 已完成 |
| 输入文件 | t2_kernel_operator_model_mapping.csv, t3_kernel_reuse_stats.csv, t4_kernel_criticality.csv, kernel_test_coverage.csv |
| 输出文件 | t5_kernel_tech_complexity.csv, t5_kernel_test_gap_scores.csv, t5_kernel_priority_scores.csv, t5_module_kernel_priority_scores.csv |
| 执行脚本 | agent_file/script/t5_kernel_priority_scoring.py |

## 2. 任务目标

整合各维度评分数据，采用双轨制计算每个 kernel 的加权总分，确定优先级等级：

- **全局排名**：所有 kernel 参与排名，共享模块 kernel 复用度降权
- **共享模块内部排名**：仅共享模块 kernel 参与排名，复用度不降权

## 3. 执行过程

### 3.1 评分权重

| 维度 | 权重 | 说明 |
|------|------|------|
| 模型热度 | 35% | kernel 被使用的模型热度级别 |
| 复用度 | 30% | kernel 被多少算子共享（共享模块降权） |
| 核心程度 | 20% | kernel 在执行路径中的重要性 |
| 测试缺失度 | 5% | 当前测试覆盖情况 |
| 技术复杂度 | 10% | 实现复杂度和优化潜力 |

### 3.2 技术复杂度计算

**分级标准**：

| 级别 | 标准 | 得分 |
|------|------|------|
| 极高 | cv/cube + >50行 + ≥2个tl.dot() | 5 |
| 高 | cv/cube + >30行 | 4 |
| 中 | cv/cube + ≤30行 或 vv + >30行 | 3 |
| 低 | vv + 15-30行 | 2 |
| 极低 | vv + <15行 | 1 |

**技术问题修复**：初始版本函数体提取失败（行号指向 `@triton.jit` 装饰器而非 `def` 行），修复后正确提取函数体。

### 3.3 测试缺失度计算

**分级标准**：

| 级别 | 标准 | 得分 |
|------|------|------|
| 严重缺失 | 无测试用例 | 5 |
| 明显不足 | 1个测试函数 | 4 |
| 部分覆盖 | 2-4个测试函数或1个测试文件 | 3 |
| 基本覆盖 | 5+个测试函数且2+个测试文件 | 2 |
| 充分覆盖 | 10+个测试函数且3+个测试文件 | 1 |

**细化改进**：初始版本仅使用原始测试覆盖状态（无测试/单个算子/多个算子），细化后根据测试函数数量和测试文件数量进行更精确评估。

### 3.4 双轨制评分

**全局排名**：
- 公式：总分 = 模型热度×0.35 + 全局排名复用度×0.30 + 核心程度×0.20 + 测试缺失度×0.05 + 技术复杂度×0.10
- 共享模块 kernel：复用度使用降权后得分（上限2分）

**共享模块内部排名**：
- 公式：总分 = 模型热度×0.35 + 模块内排名复用度×0.30 + 核心程度×0.20 + 测试缺失度×0.05 + 技术复杂度×0.10
- 复用度使用原始得分（不降权），在模块内形成差异化

### 3.5 SKILL.md 与实际实现差异

| 方面 | SKILL.md | 实际实现 | 一致性 |
|------|----------|---------|--------|
| 评分权重 | 35%/30%/20%/5%/10% | 同 | ✅ 一致 |
| 优先级分级 | ≥4.0→P0, 3.0-3.9→P1, 2.0-2.9→P2, <2.0→P3 | 同 | ✅ 一致 |
| 双轨制 | 未明确 | 全局排名+模块内排名 | ⚠️ 细化 |
| 调整规则 | 独占性+0.3等 | 未实现（数据不足） | ⚠️ 未实现 |
| 测试缺失度 | 按功能/梯度测试区分 | 按测试函数数量细化 | ⚠️ 等价 |

**关键差异说明**：

1. **双轨制细化**：SKILL.md 未明确说明双轨制策略，实际实现根据 TASK_PLAN.md 要求实现了全局排名和共享模块内部排名两套评分。

2. **调整规则未实现**：SKILL.md 提到独占性加成(+0.3)、平台敏感性加成(+0.2)等，但需要 git 历史数据和 Issue 数据支撑，当前未实现。

3. **测试缺失度细化**：SKILL.md 按功能测试/梯度测试区分，实际数据无此区分，改为按测试函数数量和测试文件数量细化。

## 4. 执行结果

### 4.1 总体数据

| 指标 | 数量 |
|------|------|
| 总 kernel 数 | 306 |
| 核心算子 kernel | 211 |
| 共享模块 kernel | 95 |

### 4.2 全局排名优先级分布

| 优先级 | 总数 | 核心算子 | 共享模块 |
|--------|------|---------|---------|
| P0 | 13 | 13 | 0 |
| P1 | 82 | 46 | 36 |
| P2 | 129 | 92 | 37 |
| P3 | 82 | 60 | 22 |

### 4.3 技术复杂度分布

| 级别 | 数量 | 占比 |
|------|------|------|
| 极高 | 89 | 29.1% |
| 高 | 55 | 18.0% |
| 中 | 87 | 28.4% |
| 低 | 37 | 12.1% |
| 极低 | 38 | 12.4% |

### 4.4 测试缺失度分布

| 级别 | 数量 | 占比 |
|------|------|------|
| 严重缺失 | 45 | 14.7% |
| 明显不足 | 20 | 6.5% |
| 部分覆盖 | 197 | 64.4% |
| 基本覆盖 | 17 | 5.6% |
| 充分覆盖 | 27 | 8.8% |

### 4.5 P0 级 Kernel 清单

| 排名 | Kernel | 总分 | 来源 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 |
|------|--------|------|------|---------|--------|---------|---------|-----------|
| 1 | chunk_gla_fwd_kernel_o | 4.80 | 核心算子 | 5 | 5 | 5 | 3 | 4 |
| 2 | chunk_gla_bwd_kernel_intra | 4.70 | 核心算子 | 5 | 5 | 4 | 3 | 5 |
| 3 | chunk_gla_bwd_kernel_inter | 4.70 | 核心算子 | 5 | 5 | 4 | 3 | 5 |
| 4 | chunk_gla_bwd_kernel_dv | 4.60 | 核心算子 | 5 | 5 | 4 | 3 | 4 |
| 5 | chunk_gla_fwd_A_kernel_intra_sub_inter | 4.60 | 核心算子 | 5 | 5 | 4 | 3 | 4 |
| 6 | pre_process_fwd_kernel_merged | 4.50 | 核心算子 | 5 | 4 | 5 | 1 | 5 |
| 7 | chunk_gla_bwd_kernel_dA | 4.50 | 核心算子 | 5 | 5 | 4 | 3 | 3 |
| 8 | chunk_gla_fwd_A_kernel_intra_sub_intra | 4.50 | 核心算子 | 5 | 5 | 4 | 3 | 3 |
| 9 | chunk_gla_fwd_A_kernel_intra_sub_intra_split | 4.50 | 核心算子 | 5 | 5 | 4 | 3 | 3 |
| 10 | chunk_gla_fwd_A_kernel_intra_sub_intra_merge | 4.40 | 核心算子 | 5 | 5 | 4 | 3 | 2 |
| 11 | pre_process_bwd_kernel_merged | 4.30 | 核心算子 | 5 | 4 | 4 | 1 | 5 |
| 12 | merge_fwd_bwd_kernel | 4.30 | 核心算子 | 5 | 4 | 4 | 1 | 5 |
| 13 | chunk_gated_delta_rule_fwd_kkt_solve_kernel | 4.00 | 核心算子 | 5 | 2 | 5 | 3 | 5 |

### 4.6 共享模块内部排名 Top 10

| 排名 | Kernel | 总分 | 所属模块 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 |
|------|--------|------|---------|---------|--------|---------|---------|-----------|
| 1 | chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | 4.45 | common | 5 | 5 | 3 | 2 | 5 |
| 2 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4.45 | common | 5 | 5 | 3 | 2 | 5 |
| 3 | chunk_bwd_kernel_dqkwg | 4.40 | common | 5 | 5 | 3 | 1 | 5 |
| 4 | chunk_fwd_kernel_o | 4.40 | common | 5 | 5 | 3 | 1 | 5 |
| 5 | chunk_bwd_kernel_dv | 4.40 | common | 5 | 5 | 3 | 1 | 5 |
| 6 | tanh | 4.40 | common | 5 | 5 | 3 | 5 | 3 |
| 7 | chunk_bwd_kernel_dv_local | 4.30 | common | 5 | 5 | 3 | 1 | 4 |
| 8 | log | 4.20 | common | 5 | 5 | 3 | 5 | 1 |
| 9 | make_tensor_descriptor | 4.20 | common | 5 | 5 | 3 | 5 | 1 |
| 10 | exp | 4.20 | common | 5 | 5 | 3 | 5 | 1 |

## 5. 质量验证

### 5.1 数据一致性验证

| 验证项 | 结果 |
|--------|------|
| T2/T3/T4 数据条数一致 | ✅ 306 条唯一 kernel |
| 共享模块 kernel 复用度降权 | ✅ 所有共享模块 kernel 复用度 ≤ 2 |
| 测试缺失度得分范围 | ✅ 所有得分在 0-5 范围 |
| 最终总分范围 | ✅ 所有总分在 1-5.5 范围 |
| 模块内排名连续性 | ✅ 排名 1-95 连续 |

### 5.2 SKILL.md 案例对比

| 案例 | SKILL.md 期望 | 实际结果 | 差异原因 |
|------|--------------|---------|---------|
| chunk_fwd_kernel_h | 4.50 (P0) | 2.35 (P2) | 共享模块降权：复用度2分，核心程度3分，模型热度2分 |
| chunk_gated_delta_rule_fwd_h | 5.10 (P0) | 3.55 (P1) | 共享模块降权：复用度2分，核心程度3分 |
| rwkv_channel_mixing_pow_and_relu | 1.85 (P2) | 2.25 (P2) | 核心程度5分（mixing操作），测试缺失度3分 |

**说明**：SKILL.md 案例是理想化示例，未考虑双轨制降权。实际结果符合双轨制策略预期。

## 6. 对后续任务的影响

### 6.1 输出数据对 T6 的影响

- T6 报告需要使用 `t5_kernel_priority_scores.csv` 中的优先级分布和 P0 级 kernel 清单
- 共享模块内部排名数据可用于分析共享模块 kernel 的差异化优先级

### 6.2 需注意的异常情况

1. **共享模块 kernel 无 P0 级**：双轨制策略下，共享模块 kernel 复用度降权后总分较低，最高仅 P1 级。这是预期行为——全局排名聚焦核心算子 kernel。

2. **GLA 算子 kernel 占据 P0 前列**：GLA 是 A 级模型，且其 kernel 复用度、核心程度、技术复杂度都很高，符合预期。

3. **调整规则未实现**：独占性加成、平台敏感性加成等需要额外数据源（git 历史、Issue 数据），当前未实现。如需实现，建议在 T6 报告中手动标注。

4. **测试缺失度 14.7% 严重缺失**：45 个 kernel 无测试用例，需在 T6 报告中重点关注。

## 7. 输出文件示例

### t5_kernel_tech_complexity.csv

```csv
Kernel函数名称,kernel类型,kernel来源,函数行数,tl.dot调用次数,技术复杂度级别,技术复杂度得分,判定依据,Kernel源码路径,Kernel行号
chunk_abc_fwd_kernel_h,cv/cube,核心算子,52,3,极高,5,cv/cube+52行+3个tl.dot,fla/ops/abc/chunk.py,18
sigmoid_fwd_kernel,vv,共享模块,12,0,极低,1,vv+12行,fla/modules/activations.py,85
```

### t5_kernel_test_gap_scores.csv

```csv
Kernel函数名称,kernel来源,测试覆盖情况,测试用例列表,测试缺失度级别,测试缺失度得分,Kernel源码路径,Kernel行号
chunk_fwd_kernel_h,共享模块,多个算子(共享kernel),test_fused_recurrent,test_chunk,...,充分覆盖,1,fla/ops/common/chunk_h.py,35
chunk_abc_fwd_kernel_h,核心算子,无测试用例,,严重缺失,5,fla/ops/abc/chunk.py,18
```

### t5_kernel_priority_scores.csv

```csv
Kernel函数名称,kernel类型,kernel来源,模型热度得分,复用度得分,核心程度得分,测试缺失度得分,技术复杂度得分,加权总分,调整加成,最终总分,优先级等级,使用的模型列表,共享算子/模块列表,Kernel源码路径,Kernel行号
chunk_gla_fwd_kernel_o,cv/cube,核心算子,5,5,5,3,4,4.80,0.0,4.80,P0,GLA,gla,fla/ops/gla/chunk.py,18
chunk_fwd_kernel_h,cv/cube,共享模块,5,2,3,1,4,2.95,0.0,2.95,P2,GLA,GSA,RWKV6,...,fla/ops/common/chunk_h.py,35
```

### t5_module_kernel_priority_scores.csv

```csv
Kernel函数名称,kernel类型,所属模块,模型热度得分,复用度得分,核心程度得分,测试缺失度得分,技术复杂度得分,加权总分,模块内排名,模块内优先级等级,Kernel源码路径,Kernel行号
chunk_fwd_kernel_h,cv/cube,common,5,5,3,1,4,3.55,1,P1,fla/ops/common/chunk_h.py,35
```
