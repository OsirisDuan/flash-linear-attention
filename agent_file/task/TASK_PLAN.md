# Kernel 优先级评估任务执行方案

## 1. 方案概述

### 1.1 总体目标

基于已构建的 Kernel 优先级评估框架（KPEF），对 flash-linear-attention 仓库中的所有 Triton kernel 函数进行系统性优先级评估，生成量化的优先级排名报告，为后续的测试覆盖优化和性能优化提供数据驱动的决策依据。

### 1.2 已有工作基础

| 文件 | 路径 | 内容说明 |
|------|------|---------|
| operators_kernels_mapping.csv | agent_file/tmp_file/ | 算子与 kernel 映射关系，含 kernel 类型（vv/cv/cube） |
| kernel_test_coverage.csv | agent_file/tmp_file/ | Kernel 测试覆盖情况，含所属算子 |
| test_cases_inventory.csv | agent_file/tmp_file/ | 测试用例清单 |
| special_test_scenarios.csv | agent_file/tmp_file/ | 特殊测试场景分析 |
| SKILL.md | agent_file/skill/ | 评估框架完整定义 |

### 1.3 评估维度与输入需求对照

| 评估维度 | 权重 | 所需输入数据 | 数据来源 |
|---------|------|-------------|---------|
| 模型热度 | 35% | kernel → 算子/模块 → 模型映射 + 模型分级表 | **需新建** |
| 复用度 | 30% | kernel 被多少算子/模块共享 | **需新建** |
| 核心程度 | 20% | kernel 在执行路径中的位置 | **需新建** |
| 测试缺失度 | 5% | 测试覆盖情况 | 已有（kernel_test_coverage.csv） |
| 技术复杂度 | 10% | kernel 类型 + 函数行数 | 部分已有，需补充行数 |

### 1.4 核心算子与共享模块的区分

本仓库中 Triton kernel 分布在两类代码中，需区别对待：

| 类别 | 位置 | 特征 | 示例 | 评估特点 |
|------|------|------|------|---------|
| **核心算子** | `fla/ops/*` | 模型特有的注意力/序列处理算子，决定模型差异化能力 | `gated_delta_rule`, `gla`, `kda` | 模型热度差异大，复用度差异大，是评估重点 |
| **共享模块** | `fla/modules/*` | 所有模型共用的基础组件（Norm、Loss、MLP 等） | `RMSNorm`, `FusedCrossEntropyLoss`, `GatedMLP` | 天然被所有模型使用，复用度极高但差异化低 |

**关键发现**：每个模型实际使用的 kernel 远不止 1 个核心算子。以 GatedDeltaNet 为例，除核心算子 `gated_delta_rule` 外，还依赖 `fla.modules` 中的 `RMSNorm`、`FusedCrossEntropyLoss`、`GatedMLP` 等共享模块，这些模块同样包含 Triton kernel。

### 1.5 双轨制评估策略

由于核心算子 kernel 和共享模块 kernel 的特征差异显著，采用**双轨制评估**：

| 评估轨道 | 比较范围 | 复用度处理 | 目的 |
|---------|---------|-----------|------|
| **全局排名** | 所有 kernel（核心算子 + 共享模块） | 共享模块 kernel 复用度降权（上限 2 分） | 防止共享模块 kernel 因天然高复用霸榜，确保核心算子 kernel 的差异化价值被正确体现 |
| **共享模块内部排名** | 仅共享模块 kernel | 不降权，原始复用度得分 | 在共享模块内部识别相对更重要的 kernel，为共享模块的测试/优化提供优先级参考 |

**双轨制输出**：
- `t5_kernel_priority_scores.csv`：全局排名（共享模块降权），用于整体优先级决策
- `t5_module_kernel_priority_scores.csv`：共享模块内部排名（不降权），用于模块内优先级参考

---

## 2. 任务分解与依赖关系

### 2.1 任务依赖图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         任务依赖关系图                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [T1] 模型-算子/模块映射分析                                          │
│    │  ├─ 核心算子映射 (fla/ops)                                       │
│    │  └─ 共享模块映射 (fla/modules)                                   │
│    │                                                                  │
│    ├──→ [T2] 算子/模块-Kernel映射整合                                 │
│    │         │                                                        │
│    │         ├──→ [T3] Kernel复用度统计                               │
│    │         │         │                                              │
│    │         │         └──→ [T5] 综合评分计算                         │
│    │         │                   │                                    │
│    ├──→ [T4] Kernel核心程度判定  ──┘                                   │
│    │                                                                  │
│    └──→ [T6] 优先级报告生成                                            │
│                                                                      │
│  [已有数据] ──→ 直接用于评分                                           │
│    • kernel_test_coverage.csv → 测试缺失度                            │
│    • operators_kernels_mapping.csv → kernel类型                       │
│                                                                      │
│  [评估策略] 双轨制                                                   │
│    • 全局排名：核心算子正常评分 + 共享模块复用度降权                    │
│    • 模块内排名：仅共享模块，不降权，内部比较优先级                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 任务清单

| 任务ID | 任务名称 | 前置依赖 | 输出文件 | 预计工作量 |
|--------|---------|---------|---------|-----------|
| T1 | 模型-算子/模块映射分析 | 无 | t1_model_operator_mapping.csv + t1_model_module_mapping.csv | 中 |
| T2 | 算子/模块-Kernel映射整合 | T1 | t2_kernel_operator_model_mapping.csv | 中 |
| T3 | Kernel复用度统计 | T2 | t3_kernel_reuse_stats.csv | 中 |
| T4 | Kernel核心程度判定 | T2 | t4_kernel_criticality.csv | 中 |
| T5 | 综合评分计算 | T1, T2, T3, T4 | t5_kernel_tech_complexity.csv + t5_kernel_test_gap_scores.csv + t5_kernel_priority_scores.csv + t5_module_kernel_priority_scores.csv | 中 |
| T6 | 优先级报告生成 | T5 | t6_kernel_priority_report.md | 低 |

### 2.3 任务要求

1. 你的子任务的输出文件均需保存在 `agent_file/tmp_file/` 目录下
2. 如果你的子任务需要编写脚本，脚本需保存在 `agent_file/script/` 目录下
3. 现在环境中的python位于D:\program\python314\python.exe，直接使用"python tmp.py"可能无法运行python脚本，如果你需要运行的话需要改为"D:\program\python314\python.exe tmp.py"。
4. 每个子任务完成后，必须在 `agent_file/report/` 目录下编写任务报告，文件命名为 `{任务ID}_report.md`（如 `T1_report.md`）。报告需包含以下内容：
   - **任务概述**：任务ID、名称、执行时间、执行状态、输出文件路径、脚本路径
   - **执行过程**：实际执行的步骤、关键技术决策、遇到的问题与解决方案
   - **执行结果**：总体数据统计、输出文件字段说明（如有变更需说明）、分类统计
   - **质量验证**：按 TASK_PLAN 中定义的验证方法逐项检查并记录结果
   - **对后续任务的影响**：输出数据对下游任务的输入影响、需注意的异常情况

---

## 3. 子任务详细设计

### 3.1 任务 T1：模型-算子/模块映射分析

#### 3.1.1 任务目标

建立"模型 → Layer/Module → 算子/模块"的完整映射关系，回答"该 kernel 被哪些模型使用？"这一核心问题。映射范围覆盖两类代码：

- **核心算子**（`fla/ops/*`）：模型特有的注意力/序列处理算子
- **共享模块**（`fla/modules/*`）：所有模型共用的基础组件（Norm、Loss、MLP 等）

#### 3.1.2 执行步骤

**步骤 1.1：遍历模型目录**

```
输入：fla/models/ 目录结构
操作：遍历所有子目录，识别模型名称
输出：模型列表
```

**步骤 1.2：解析模型定义文件 — 核心算子链**

```
输入：fla/models/*/modeling_*.py
操作：提取 import 语句中的 layer 引用
模式1：from fla.layers import Xxx → 通过 LAYER_CLASS_TO_MODULE 映射表转换
模式2：from fla.layers.xxx import Yyy → 直接提取模块名
输出：模型 → Layer 映射
```

**步骤 1.3：解析 Layer 文件 — 核心算子**

```
输入：fla/layers/*.py
操作：提取 layer 中使用的算子 import
模式：from fla.ops.xxx import yyy
输出：Layer → 核心算子 映射
```

**步骤 1.4：解析模型定义文件 — 共享模块链**

```
输入：fla/models/*/modeling_*.py
操作：提取 import 语句中的 modules 引用
模式1：from fla.modules import Xxx, Yyy
模式2：from fla.modules.xxx import yyy
输出：模型 → 共享模块 映射
```

**步骤 1.5：解析共享模块文件 — kernel 依赖**

```
输入：fla/modules/*.py 及子目录
操作：提取模块中使用的 fla.ops 引用
模式：from fla.ops.xxx import yyy
输出：共享模块 → 算子 映射
注意：部分模块（如 RMSNorm）自身包含 @triton.jit kernel，不依赖 fla.ops
```

**步骤 1.6：关联模型与算子/模块**

```
输入：模型 → Layer → 核心算子 映射 + 模型 → 共享模块 映射
操作：传递性关联，得到 模型 → 算子/模块 完整映射
输出：t1_model_operator_mapping.csv + t1_model_module_mapping.csv
```

#### 3.1.3 输出文件规范

**文件1**：`t1_model_operator_mapping.csv`（核心算子映射）

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| 模型名称 | string | 模型标识符 | GatedDeltaNet |
| 模型热度级别 | string | S/A/B/C/D | S |
| Layer名称 | string | 使用的 Layer | GatedDeltaNet |
| Layer模块 | string | Layer 对应的模块文件名 | gated_deltanet |
| Layer类型 | string | fla核心算子/标准注意力/外部库 | fla核心算子 |
| 算子名称 | string | 使用的算子 | gated_delta_rule |
| 算子源码路径 | string | 算子代码位置 | fla/ops/gated_delta_rule |
| 算子函数 | string | 具体调用的函数列表 | chunk_gated_delta_rule, fused_recurrent_gated_delta_rule |

**文件2**：`t1_model_module_mapping.csv`（共享模块映射）

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| 模型名称 | string | 模型标识符 | GatedDeltaNet |
| 模型热度级别 | string | S/A/B/C/D | S |
| 模块名称 | string | 使用的共享模块类名 | GatedMLP |
| 模块源码路径 | string | 模块代码位置 | fla/modules |
| 含TritonKernel | bool | 该模块是否包含 @triton.jit kernel | True |
| Kernel数量 | int | 模块中 @triton.jit kernel 的数量 | 4 |

**示例数据**：

```csv
# t1_model_operator_mapping.csv
模型名称,模型热度级别,Layer名称,Layer模块,Layer类型,算子名称,算子源码路径,算子函数
GatedDeltaNet,S,GatedDeltaNet,gated_deltanet,fla核心算子,gated_delta_rule,fla/ops/gated_delta_rule,chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
GatedDeltaNet,S,Attention,attn,标准注意力,,,

# t1_model_module_mapping.csv
模型名称,模型热度级别,模块名称,模块源码路径,含TritonKernel,Kernel数量
GatedDeltaNet,S,FusedCrossEntropyLoss,fla/modules,True,2
GatedDeltaNet,S,FusedLinearCrossEntropyLoss,fla/modules,True,3
GatedDeltaNet,S,RMSNorm,fla/modules,True,4
GatedDeltaNet,S,GatedMLP,fla/modules,True,4
GatedDeltaNet,S,l2warp,fla/modules/l2warp,True,1
```

#### 3.1.4 质量验证方法

1. **完整性检查**：确保所有 29 个模型目录都被分析
2. **一致性检查**：Layer 名称与 `fla/layers/` 目录下的文件名对应
3. **交叉验证**：对比 README.md 中的模型列表，确认无遗漏
4. **共享模块覆盖率**：确认 `fla/modules/` 中含 `@triton.jit` 的模块均被映射
5. **模块映射完整性**：每个模型至少应包含 RMSNorm + FusedCrossEntropyLoss + GatedMLP 三个共享模块

---

### 3.2 任务 T2：算子/模块-Kernel映射整合

#### 3.2.1 任务目标

整合已有数据，建立"Kernel → 算子/模块 → 模型"的完整映射链，为后续各维度评分提供基础数据。需区分两类 kernel 来源：

- **核心算子 kernel**：来自 `fla/ops/*`，通过 `operators_kernels_mapping.csv` 获取
- **共享模块 kernel**：来自 `fla/modules/*`，需扫描 `@triton.jit` 装饰器获取

#### 3.2.2 执行步骤

**步骤 2.1：读取已有核心算子映射数据**

```
输入：agent_file/tmp_file/operators_kernels_mapping.csv
操作：解析算子与 kernel 的对应关系
注意：处理"一对多"映射关系
```

**步骤 2.2：扫描共享模块 kernel**

```
输入：fla/modules/*.py 及子目录
操作：扫描所有 @triton.jit 装饰器，提取 kernel 函数名
输出：共享模块 kernel 列表（含函数名、源码路径、行号）
```

**步骤 2.3：整合模型信息 — 核心算子 kernel**

```
输入：T1 输出的 t1_model_operator_mapping.csv
操作：通过算子名称关联，得到 kernel → 模型 映射
```

**步骤 2.4：整合模型信息 — 共享模块 kernel**

```
输入：T1 输出的 t1_model_module_mapping.csv
操作：通过模块名称关联，得到 kernel → 模型 映射
注意：共享模块被所有使用它的模型共享，需展开所有模型
```

**步骤 2.5：合并输出**

```
操作：合并核心算子 kernel 和共享模块 kernel 的映射
增加字段：kernel来源（核心算子/共享模块）
输出：t2_kernel_operator_model_mapping.csv
```

#### 3.2.3 输出文件规范

**文件名**：`t2_kernel_operator_model_mapping.csv`

**字段定义**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel类型 | string | vv 或 cv/cube |
| kernel来源 | string | 核心算子 或 共享模块 |
| 所属算子/模块 | string | 算子或模块名称 |
| 使用的模型列表 | string | 逗号分隔的模型名称 |
| 各模型热度级别 | string | 逗号分隔，与模型列表一一对应，如 S,A,B |
| 最高模型热度级别 | string | 使用的模型中最高的热度级别（S/A/B/C/D） |
| 模型热度得分 | int | 1-5 分（S=5, A=4, B=3, C=2, D=1，取最高级别对应分值） |
| Kernel源码路径 | string | 源码位置 |
| Kernel行号 | int | 函数起始行号 |

#### 3.2.4 质量验证方法

1. **覆盖率检查**：确保所有 kernel（含共享模块 kernel）都有对应的模型信息
2. **热度级别验证**：手动抽查 S 级模型的 kernel 映射是否正确
3. **来源区分验证**：确认 `fla/modules/` 下的 kernel 被标记为"共享模块"
4. **共享模块 kernel 数量**：预期 `fla/modules/` 下约有 43 个 `@triton.jit` kernel

---

### 3.3 任务 T3：Kernel复用度统计 ✅ 已完成

#### 3.3.1 任务目标

统计每个 kernel 被多少个算子/模块共享使用，回答"修改该 kernel 的影响范围有多大？"。采用双轨制策略：

- **全局排名视角**：核心算子 kernel 正常评分，共享模块 kernel 复用度降权（上限 2 分），防止霸榜
- **共享模块内部视角**：共享模块 kernel 使用原始复用度得分，用于模块内优先级比较

#### 3.3.2 执行步骤

**步骤 3.1：识别共享 Kernel**

```
输入：fla/ops/common/ 目录
操作：列出所有共享 kernel 文件
关键文件：chunk_h.py, chunk_o.py, fused_recurrent.py, chunk_delta_h.py 等
```

**步骤 3.2：追踪 import 链**

```
输入：fla/ops/**/*.py
操作：搜索 from fla.ops.common import 和 from fla.ops.xxx import
统计：每个 kernel 被引用的次数和来源算子
```

**步骤 3.3：统计复用度**

```
操作：汇总每个 kernel 被使用的算子/模块数量，并结合使用模型数量细分
分级规则（SKILL.md 统一适用）：
  ≥4个算子/模块=极高(5分), 3个=高(4分), 2个=中(3分),
  1个且≥2个模型=低(2分), 1个且≤1个模型=极低(1分)
```

**步骤 3.4：双轨制复用度得分**

```
操作：为每个 kernel 计算两个复用度得分
全局排名得分（用于跨类别比较）：
  - 核心算子 kernel：原始得分
  - 共享模块 kernel：min(原始得分, 2)（降权，防止霸榜）
模块内排名得分（用于共享模块内部比较）：
  - 核心算子 kernel：原始得分（不参与模块内排名）
  - 共享模块 kernel：原始得分（不降权）
```

#### 3.3.3 输出文件规范

**文件名**：`t3_kernel_reuse_stats.csv`

**字段定义**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel来源 | string | 核心算子 或 共享模块 |
| 共享算子/模块数量 | int | 被多少算子/模块使用 |
| 共享算子/模块列表 | string | 逗号分隔的算子/模块名称 |
| 使用模型数量 | int | 该 kernel 所属算子/模块被多少模型使用 |
| 复用度级别 | string | 极高/高/中/低/极低 |
| 原始复用度得分 | int | 1-5 分（未降权，count=1时按模型数区分1分vs2分） |
| 全局排名复用度得分 | int | 1-5 分（共享模块降权后，核心算子=原始得分） |
| 模块内排名复用度得分 | int | 1-5 分（均不降权，仅共享模块 kernel 使用） |
| 是否为common目录kernel | bool | 是否位于 fla/ops/common/ |

#### 3.3.4 质量验证方法

1. **已知共享 kernel 验证**：
   - `chunk_fwd_h` 应被 ≥4 个算子使用
   - `fused_recurrent` 应被 ≥3 个算子使用
2. **import 链完整性**：确保所有 import 语句都被解析
3. **双轨制验证**：
   - 全局排名：确认所有共享模块 kernel 的复用度得分 ≤ 2
   - 模块内排名：确认共享模块 kernel 使用原始得分（无降权）

---

### 3.4 任务 T4：Kernel核心程度判定 ✅ 已完成

#### 3.4.1 任务目标

判定每个 kernel 在模型执行路径中的位置和重要性，回答"该 kernel 是否为核心计算路径？"。需区分两类 kernel 的核心程度含义：

- **核心算子 kernel**：直接参与注意力计算，核心程度差异大（fwd > bwd > 辅助）
- **共享模块 kernel**：属于基础设施（Norm、Loss 等），核心程度普遍为"辅助"级别

#### 3.4.2 执行步骤

**步骤 4.1：函数名模式分析（核心算子 kernel）**

```
规则：
- 含 "fwd" 且无 "intra"/"inter"/"preprocess" → 核心(5分)
- 含 "bwd" → 重要(4分)
- 含 "cumsum"/"preprocess"/"mask" → 辅助(3分)
- 含 "fused" → 优化(2分)
- 含 "naive" → 边缘(1分)
```

**步骤 4.2：共享模块 kernel 核心程度判定**

```
规则：
- 含 "norm"/"rms" → 辅助(3分)（归一化是必要但非差异化组件）
- 含 "cross_entropy"/"kl_div"/"linear_cross_entropy" → 辅助(3分)（损失函数）
- 含 "conv"/"causal" → 辅助(3分)（卷积组件）
- 含 "activation"/"swiglu" → 辅助(3分)（激活函数）
- 含 "rotary"/"token_shift"/"l2" → 辅助(3分)（位置编码/辅助操作）
- 含 "gate"/"norm_gated" → 辅助(3分)（门控组件）
- 其他共享模块 kernel → 辅助(3分)
```

**步骤 4.3：kernel 类型辅助判断**

```
规则：
- cv/cube 类型 → 更可能是核心计算
- vv 类型 → 可能是辅助或优化
```

**步骤 4.4：调用链分析（可选）**

```
输入：fla/layers/*.py 中的 forward() 方法
操作：识别直接调用的 kernel
判定：forward() 中直接调用 → 核心
```

#### 3.4.3 输出文件规范

**文件名**：`t4_kernel_criticality.csv`

**字段定义**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel来源 | string | 核心算子 或 共享模块 |
| 函数名特征 | string | fwd/bwd/preprocess/fused/naive/norm/loss 等 |
| kernel类型 | string | vv 或 cv/cube |
| 核心程度级别 | string | 核心/重要/辅助/优化/边缘 |
| 核心程度得分 | int | 1-5 分 |
| 判定依据 | string | 简要说明判定理由 |

#### 3.4.4 质量验证方法

1. **抽样验证**：手动检查 10 个 kernel 的判定结果
2. **一致性检查**：同一算子的 fwd kernel 应比 bwd kernel 核心程度更高

---

### 3.5 任务 T5：综合评分计算 ✅ 已完成

#### 3.5.1 任务目标

整合各维度评分数据，采用双轨制计算每个 kernel 的加权总分，确定优先级等级。

- **全局排名**：所有 kernel 参与排名，共享模块 kernel 复用度降权
- **共享模块内部排名**：仅共享模块 kernel 参与排名，复用度不降权

#### 3.5.2 执行步骤

**步骤 5.1：收集各维度数据**

```
输入：
- T2: t2_kernel_operator_model_mapping.csv → 模型热度得分（含 kernel来源 标记）
- T3: t3_kernel_reuse_stats.csv → 复用度得分（含全局排名和模块内排名两套得分）
- T4: t4_kernel_criticality.csv → 核心程度得分（共享模块默认辅助级）
- 已有: kernel_test_coverage.csv → 测试缺失度得分
- 已有: operators_kernels_mapping.csv → 技术复杂度得分（需补充行数）
- T2 新增: fla/modules/ 下的 kernel（需扫描 @triton.jit 获取）
```

**评分可追溯性对照**：每个维度的得分必须有对应的支撑数据，确保评分结果可追溯。

| 维度 | 得分来源 | 支撑数据文件 | 支撑数据内容 |
|------|---------|-------------|-------------|
| 模型热度 | T2 输出 | t2_kernel_operator_model_mapping.csv | 各模型热度级别、最高级别、得分 |
| 复用度 | T3 输出 | t3_kernel_reuse_stats.csv | 共享算子/模块数量、列表、原始得分、降权后得分 |
| 核心程度 | T4 输出 | t4_kernel_criticality.csv | 函数名特征、kernel类型、判定依据 |
| 技术复杂度 | T5 步骤 5.2 计算 | t5_kernel_tech_complexity.csv | 函数行数、tl.dot调用次数、判定依据 |
| 测试缺失度 | T5 步骤 5.3 计算 | t5_kernel_test_gap_scores.csv | 测试覆盖情况、测试用例列表、级别映射 |

**步骤 5.2：计算技术复杂度得分**

```
输入：kernel 源码文件
操作：
  1. 读取函数体
  2. 统计行数
  3. 统计 tl.dot() 调用次数
分级：
  - cv/cube + >50行 + 多个tl.dot() → 极高(5分)
  - cv/cube + >30行 → 高(4分)
  - cv/cube + ≤30行 或 vv + >30行 → 中(3分)
  - vv + 15-30行 → 低(2分)
  - vv + <15行 → 极低(1分)
输出：t5_kernel_tech_complexity.csv（保存原始数据，供评分追溯）
```

**步骤 5.3：计算测试缺失度得分**

```
输入：kernel_test_coverage.csv 的"测试覆盖情况"列
分级：
  - 无测试用例 → 严重缺失(5分)
  - 有测试但单一场景 → 明显不足(4分)
  - 基本功能测试 → 部分覆盖(3分)
  - 功能+梯度测试 → 基本覆盖(2分)
  - 全面覆盖 → 充分覆盖(1分)
注意：共享模块 kernel 可能不在 kernel_test_coverage.csv 中，需标记为"未统计"
输出：t5_kernel_test_gap_scores.csv（保存覆盖状态与得分映射，供评分追溯）
```

**步骤 5.4：计算加权总分 — 全局排名**

```
公式：
总分 = 模型热度分 × 0.35
     + 全局排名复用度分 × 0.30
     + 核心程度分 × 0.20
     + 测试缺失度分 × 0.05
     + 技术复杂度分 × 0.10

说明：
- 核心算子 kernel：各维度正常评分
- 共享模块 kernel：复用度使用降权后得分（上限2分），核心程度默认辅助(3分)，
  模型热度天然为S级（被所有模型使用），因此总分通常不会过高
```

**步骤 5.4b：计算加权总分 — 共享模块内部排名**

```
公式（仅对 kernel来源="共享模块" 的 kernel 计算）：
总分 = 模型热度分 × 0.35
     + 模块内排名复用度分 × 0.30
     + 核心程度分 × 0.20
     + 测试缺失度分 × 0.05
     + 技术复杂度分 × 0.10

说明：
- 复用度使用原始得分（不降权），在共享模块内部形成差异化
- 核心程度、技术复杂度等维度保持不变，这些维度在模块内仍有区分度
```

**步骤 5.5：应用调整规则**

```
加成项：
- 独占性加成：+0.3（fla 独有实现，仅适用于核心算子 kernel）
- 平台敏感性加成：+0.2
- 近期变更加成：+0.2（近3月≥5次提交）
- 社区问题加成：+0.3（≥3个相关Issue）
- 加成上限：+0.5
注意：共享模块 kernel 不适用独占性加成（它们是通用组件，非 fla 独占）
```

**步骤 5.6：确定优先级等级**

```
分级：
- ≥4.0 → P0 紧急
- 3.0-3.9 → P1 高
- 2.0-2.9 → P2 中
- 1.0-1.9 → P3 低
```

#### 3.5.3 输出文件规范

**文件1**：`t5_kernel_tech_complexity.csv`（技术复杂度支撑数据）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel类型 | string | vv 或 cv/cube |
| kernel来源 | string | 核心算子 或 共享模块 |
| 函数行数 | int | 函数体行数 |
| tl.dot调用次数 | int | tl.dot() 调用次数 |
| 技术复杂度级别 | string | 极高/高/中/低/极低 |
| 技术复杂度得分 | int | 1-5 分 |
| 判定依据 | string | 简要说明分级理由 |

**文件2**：`t5_kernel_test_gap_scores.csv`（测试缺失度支撑数据）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel来源 | string | 核心算子 或 共享模块 |
| 测试覆盖情况 | string | 原始覆盖状态（来自 kernel_test_coverage.csv） |
| 测试用例列表 | string | 关联的测试用例名称 |
| 测试缺失度级别 | string | 严重缺失/明显不足/部分覆盖/基本覆盖/充分覆盖/未统计 |
| 测试缺失度得分 | int | 1-5 分（未统计时记为 0） |

**文件3**：`t5_kernel_priority_scores.csv`（全局排名）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel类型 | string | vv 或 cv/cube |
| kernel来源 | string | 核心算子 或 共享模块 |
| 模型热度得分 | int | 1-5 |
| 复用度得分 | int | 1-5（共享模块已降权） |
| 核心程度得分 | int | 1-5 |
| 测试缺失度得分 | int | 1-5 |
| 技术复杂度得分 | int | 1-5 |
| 加权总分 | float | 计算结果 |
| 调整加成 | float | 加成值 |
| 最终总分 | float | 加权总分 + 调整加成 |
| 优先级等级 | string | P0/P1/P2/P3 |
| 使用的模型列表 | string | 逗号分隔 |
| 共享算子/模块列表 | string | 逗号分隔 |

**文件4**：`t5_module_kernel_priority_scores.csv`（共享模块内部排名）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Kernel函数名称 | string | kernel 标识符 |
| kernel类型 | string | vv 或 cv/cube |
| 所属模块 | string | 共享模块名称 |
| 模型热度得分 | int | 1-5 |
| 复用度得分 | int | 1-5（原始得分，不降权） |
| 核心程度得分 | int | 1-5 |
| 测试缺失度得分 | int | 1-5 |
| 技术复杂度得分 | int | 1-5 |
| 加权总分 | float | 计算结果 |
| 模块内排名 | int | 在共享模块 kernel 中的排名 |
| 模块内优先级等级 | string | P0/P1/P2/P3 |

---

### 3.6 任务 T6：优先级报告生成 ✅ 已完成

#### 3.6.1 任务目标

生成人类可读的优先级评估报告，包含关键发现、高分 kernel 清单、行动建议等。报告需区分核心算子 kernel 和共享模块 kernel 的评估结果。

#### 3.6.2 报告结构

```markdown
# Kernel 优先级评估报告

## 1. 执行摘要
- 评估 kernel 总数（核心算子 + 共享模块）
- 全局排名各优先级分布
- 核心算子 vs 共享模块 kernel 分布对比
- 双轨制评估策略说明
- 关键发现

## 2. 全局排名 — P0 级 Kernel 清单
- 详细列表及评分依据
- 行动建议

## 3. 核心算子 Kernel 分析（全局排名视角）
- 模型热度分布
- 复用度分布
- 核心程度分布
- 测试覆盖现状

## 4. 共享模块 Kernel 分析
### 4.1 全局排名视角
- 共享模块 kernel 在全局排名中的位置
- 复用度降权效果说明
- 与核心算子 kernel 的优先级对比

### 4.2 模块内排名视角（不降权）
- 共享模块 kernel 内部优先级排名
- 各模块 kernel 数量与复杂度
- 模块内 Top kernel 识别
- 模块内排名与全局排名的差异分析

## 5. 高优先级 Kernel 详细分析
- Top 10 核心 kernel 深度分析（全局排名），总结这些算子高优先级的背后的统计数据支撑。
- Top 5 共享模块 kernel 分析（模块内排名），总结这些算子高优先级的背后的统计数据支撑。

## 6. 行动建议
- 短期行动项
- 中期规划
- 共享模块 kernel 的特殊建议（基于模块内排名）
```

#### 3.6.3 输出文件规范

**文件名**：`t6_kernel_priority_report.md`

---

## 4. 数据流转路径

### 4.1 数据流向图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            数据流转路径                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [源数据]                                                                │
│    fla/models/*/modeling_*.py ──┐                                        │
│    fla/layers/*.py ───────────┬─┼──→ T1: t1_model_operator_mapping.csv     │
│    fla/ops/**/*.py ───────────┘ │   (核心算子映射)                       │
│    fla/modules/*.py ────────────┼──→ T1: t1_model_module_mapping.csv       │
│                                 │   (共享模块映射)                        │
│  [已有数据]                      │                                       │
│    operators_kernels_mapping.csv ┼──→ T2: t2_kernel_operator_model_mapping.csv │
│    kernel_test_coverage.csv ────┼──→ (直接用于T5)                         │
│    fla/modules/ @triton.jit ────┼──→ T2: (扫描共享模块kernel)             │
│                                 │                                        │
│                                 ├──→ T3: t3_kernel_reuse_stats.csv          │
│                                 │        (含共享模块降权)                  │
│                                 │                                        │
│                                 └──→ T4: t4_kernel_criticality.csv          │
│                                          (含共享模块判定)                  │
│                                                                          │
│  [评分数据]                                                              │
│    T1 + T2 + T3 + T4 + 已有数据 ──→ T5: t5_kernel_priority_scores.csv      │
│                                     (区分核心算子/共享模块评分)            │
│                                                                          │
│  [最终报告]                                                              │
│    T5 ──→ T6: t6_kernel_priority_report.md                                  │
│            (含共享模块专项分析)                                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 中间成果存储规范

所有中间成果统一存储至 `agent_file/tmp_file/` 目录，命名规范如下：

| 文件类型 | 命名规范 | 示例 |
|---------|---------|------|
| 映射数据 | `{实体1}_{实体2}_mapping.csv` | t1_model_operator_mapping.csv, t1_model_module_mapping.csv |
| 统计数据 | `{统计对象}_stats.csv` | t3_kernel_reuse_stats.csv |
| 评分数据 | `{评分对象}_scores.csv` | t5_kernel_priority_scores.csv, t5_module_kernel_priority_scores.csv, t5_kernel_tech_complexity.csv, t5_kernel_test_gap_scores.csv |
| 分析报告 | `{报告主题}_report.md` | t6_kernel_priority_report.md |
| 执行日志 | `{任务ID}_execution.log` | T1_execution.log |

---

## 5. 执行计划

### 5.1 执行顺序

| 阶段 | 任务 | 预计耗时 | 并行可能性 |
|------|------|---------|-----------|
| 阶段1 | T1: 模型-算子/模块映射分析 | 30分钟 | 否（基础数据） |
| 阶段2 | T2: 算子/模块-Kernel映射整合 | 15分钟 | 否（依赖T1） |
| 阶段3 | T3 + T4 并行执行 | 25分钟 | 是（均依赖T2） |
| 阶段4 | T5: 综合评分计算 | 10分钟 | 否（依赖T1-T4） |
| 阶段5 | T6: 优先级报告生成 | 10分钟 | 否（依赖T5） |

### 5.2 关键里程碑

1. **M1**：完成 T1，产出 t1_model_operator_mapping.csv + t1_model_module_mapping.csv
2. **M2**：完成 T2，产出 t2_kernel_operator_model_mapping.csv（含共享模块 kernel）
3. **M3**：完成 T3 和 T4，产出复用度（含降权）和核心程度数据
4. **M4**：完成 T5，产出维度评分支撑数据 + 综合评分结果
5. **M5**：完成 T6，产出最终报告（含共享模块专项分析）

---

## 6. 质量保证

### 6.1 数据质量检查清单

| 检查项 | 检查方法 | 通过标准 |
|--------|---------|---------|
| 模型覆盖率 | 统计 T1 输出的模型数量 | ≥29 个模型 |
| 核心算子 Kernel 覆盖率 | 统计 T2 输出的核心算子 kernel 数量 | 与 operators_kernels_mapping.csv 一致 |
| 共享模块 Kernel 覆盖率 | 统计 T2 输出的共享模块 kernel 数量 | ≥43 个（fla/modules/ 下 @triton.jit 数量） |
| 映射完整性 | 检查空值比例 | 空值率 <5% |
| 得分合理性 | 检查得分分布 | 无异常值（如负分、超5分） |
| 降权验证 | 检查共享模块 kernel 复用度得分 | 全局排名 ≤ 2 分，模块内排名 = 原始得分 |
| 评分可追溯性 | 检查每个维度得分是否有对应支撑数据文件 | 5 个维度均有独立支撑文件或字段 |
| 技术复杂度原始数据 | 检查 t5_kernel_tech_complexity.csv | 函数行数和 tl.dot 调用次数无空值 |
| 测试缺失度映射 | 检查 t5_kernel_test_gap_scores.csv | 每个 kernel 均有覆盖状态和得分 |

### 6.2 交叉验证方法

1. **模型热度验证**：对比 README.md 中的模型列表
2. **复用度验证**：手动检查已知共享 kernel 的统计结果
3. **核心程度验证**：抽样检查 10 个 kernel 的判定依据
4. **总分验证**：手动计算 5 个 kernel 的总分，对比脚本结果
5. **共享模块验证**：确认每个模型至少映射了 RMSNorm + FusedCrossEntropyLoss + GatedMLP

---

## 7. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| import 语句格式不一致 | T1 解析失败 | 建立多种模式匹配规则 |
| 模型使用间接引用 | 映射不完整 | 增加二次解析逻辑 |
| kernel 函数名不规范 | 核心程度判定偏差 | 结合调用链分析辅助判断 |
| 数据文件版本不一致 | 结果不准确 | 使用 agent_file/tmp_file/ 下的最新数据 |
| 共享模块 kernel 未被现有 CSV 覆盖 | T2 整合遗漏 | T2 需独立扫描 fla/modules/ 的 @triton.jit |
| 共享模块降权策略不当 | 评分偏差 | 双轨制：全局排名降权 + 模块内排名不降权，可通过 T5 抽样验证两套排名的合理性 |
| 模型未显式 import 共享模块 | T1 模块映射遗漏 | 检查模型的 config 文件和 __init__.py 间接引用 |

---

## 8. 附录

### 8.1 模型热度分级参考表

| 模型 | 热度级别 | 依据 |
|------|---------|------|
| GatedDeltaNet (GDN) | S | 已被 Qwen3-Next 集成 |
| KDA | S | Kimi Delta Attention |
| MLA | S | DeepSeek-V2/V3 核心架构 |
| Mamba2 | A | ICML 2024 |
| GLA | A | ICML 2024 |
| DeltaNet | A | NeurIPS 2024 |
| RetNet | A | ACL 2024 |
| HGRN | A | NeurIPS 2023 |
| RWKV6 | B | COLM 2024 |
| RWKV7 | C | 预印本 |
| MesaNet | C | 预印本 |
| Comba | C | 预印本 |

### 8.2 已知共享 Kernel 列表

| 共享 Kernel | 被使用的算子 | 预期复用度级别 |
|------------|-------------|---------------|
| chunk_fwd_h | GLA, SimpleGLA, GSA, RWKV6, MesaNet | 极高 |
| chunk_fwd_o | SimpleGLA, DeltaRule, GatedDeltaRule, Comba | 极高 |
| fused_recurrent | GLA, SimpleGLA, GSA | 高 |
| chunk_gated_delta_rule_fwd_h | DeltaRule, GatedDeltaRule, KDA, Comba | 极高 |

### 8.3 共享模块参考列表

以下为 `fla/modules/` 中包含 `@triton.jit` kernel 的模块（预期约 43 个 kernel）：

| 模块名称 | 源码路径 | 预期 Kernel 数量 | 说明 |
|---------|---------|-----------------|------|
| activations | fla/modules/activations.py | ~8 | 激活函数（SwiGLU 等） |
| RMSNorm | fla/modules/rms_norm.py | ~4 | 归一化 |
| FusedCrossEntropyLoss | fla/modules/fused_cross_entropy.py | ~2 | 损失函数 |
| FusedLinearCrossEntropyLoss | fla/modules/fused_linear_cross_entropy.py | ~3 | 线性+损失融合 |
| GatedMLP | fla/modules/ | ~4 | 门控 MLP |
| conv | fla/modules/conv/triton/kernels.py | ~5 | 卷积 |
| l2warp | fla/modules/l2warp/ | ~1 | L2 正则化 |
| rotary | fla/modules/ | 待确认 | 旋转位置编码 |
| token_shift | fla/modules/ | 待确认 | Token 偏移 |

> **注意**：以上为预估数据，实际 kernel 数量以 T1/T2 执行时扫描 `@triton.jit` 装饰器的结果为准。
