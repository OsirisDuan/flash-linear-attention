# Kernel 优先级评估报告

## 1. 执行摘要

本次评估共覆盖 **306 个 Triton kernel**，其中核心算子 kernel 211 个，共享模块 kernel 95 个。

### 1.1 全局排名优先级分布

| 优先级 | 总数 | 核心算子 | 共享模块 | 占比 |
|--------|------|---------|---------|------|
| P0 | 13 | 13 | 0 | 4.2% |
| P1 | 82 | 46 | 36 | 26.8% |
| P2 | 129 | 92 | 37 | 42.2% |
| P3 | 82 | 60 | 22 | 26.8% |

### 1.2 双轨制评估策略说明

本评估采用**双轨制**策略：

- **全局排名**：所有 kernel 参与排名，共享模块 kernel 的复用度得分降权（上限2分），防止高复用的基础设施 kernel 占据排名前列
- **共享模块内部排名**：仅共享模块 kernel 参与排名，复用度使用原始得分（不降权），在模块内形成差异化

### 1.3 关键发现

1. **13 个 P0 级 kernel**：全部为核心算子 kernel，集中在 GLA 算子和 GatedDeltaRule 算子，被 S 级模型（KDA/Kimi）和 A 级模型（GLA）使用
2. **共享模块 kernel 无 P0 级**：双轨制降权后，共享模块 kernel 最高为 P1 级，符合预期——全局排名聚焦核心算子
3. **45 个 kernel 无测试用例**：其中核心算子 kernel 占比较大，需优先补充测试
4. **技术复杂度两极分化**：极高(89个) 和极低(38个) 占比大，cv/cube 类型 kernel 普遍复杂度高
5. **GLA 算子 kernel 占据 P0 前列**：GLA 的 chunk kernel 被 4+ 个算子共享，复用度极高

## 2. 全局排名 — P0 级 Kernel 清单

| 排名 | Kernel | 总分 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 | 使用的模型 | 共享算子 |
|------|--------|------|---------|--------|---------|---------|-----------|----------|---------|
| 1 | chunk_gla_fwd_kernel_o | 4.8 | 5 | 5 | 5 | 3 | 4 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 2 | chunk_gla_bwd_kernel_intra | 4.7 | 5 | 5 | 4 | 3 | 5 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 3 | chunk_gla_bwd_kernel_inter | 4.7 | 5 | 5 | 4 | 3 | 5 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 4 | chunk_gla_bwd_kernel_dv | 4.6 | 5 | 5 | 4 | 3 | 4 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 5 | chunk_gla_fwd_A_kernel_intra_sub_inter | 4.6 | 5 | 5 | 4 | 3 | 4 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 6 | pre_process_fwd_kernel_merged | 4.5 | 5 | 4 | 5 | 1 | 5 | GatedDeltaNet,GatedDeltaProduc... | CP,gated_delta_rule,... |
| 7 | chunk_gla_bwd_kernel_dA | 4.5 | 5 | 5 | 4 | 3 | 3 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 8 | chunk_gla_fwd_A_kernel_intra_sub_intra | 4.5 | 5 | 5 | 4 | 3 | 3 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 9 | chunk_gla_fwd_A_kernel_intra_sub_intra_split | 4.5 | 5 | 5 | 4 | 3 | 3 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 10 | chunk_gla_fwd_A_kernel_intra_sub_intra_merge | 4.4 | 5 | 5 | 4 | 3 | 2 | GLA,GSA,HGRN2,KDA,LightNet,RWK... | GLA,gsa,kda,rwkv6 |
| 11 | pre_process_bwd_kernel_merged | 4.3 | 5 | 4 | 4 | 1 | 5 | GatedDeltaNet,GatedDeltaProduc... | CP,gated_delta_rule,... |
| 12 | merge_fwd_bwd_kernel | 4.3 | 5 | 4 | 4 | 1 | 5 | GatedDeltaNet,GatedDeltaProduc... | CP,gated_delta_rule,... |
| 13 | chunk_gated_delta_rule_fwd_kkt_solve_kernel | 4.0 | 5 | 2 | 5 | 3 | 5 | GatedDeltaNet,GatedDeltaProduc... | GatedDeltaRule |

### P0 级行动建议

1. **立即补充测试**：P0 级 kernel 测试缺失度均为 3 分（部分覆盖），需补充独立单元测试
2. **GLA chunk kernel 优先**：GLA 的 fwd/bwd kernel 被 4+ 个算子共享，修改影响范围大
3. **GatedDeltaRule kkt_solve_kernel 关注**：虽复用度仅 2 分，但核心程度和技术复杂度均为 5 分，是算法核心

## 3. 核心算子 Kernel 分析（全局排名视角）

### 3.1 模型热度分布

| 得分 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 5 | 32 | 15.2% | S级模型 |
| 4 | 17 | 8.1% | A级模型 |
| 3 | 22 | 10.4% | B级模型 |
| 2 | 67 | 31.8% | C级模型 |
| 1 | 13 | 6.2% | D级模型 |

### 3.2 复用度分布

| 得分 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 5 | 14 | 6.6% | ≥4个算子共享 |
| 4 | 3 | 1.4% | 3个算子共享 |
| 3 | 12 | 5.7% | 2个算子共享 |
| 2 | 19 | 9.0% | 1算子+多模型 |
| 1 | 163 | 77.3% | 1算子+1模型 |

### 3.3 核心程度分布

| 得分 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 5 | 71 | 33.6% | 核心(前向主计算) |
| 4 | 119 | 56.4% | 重要(反向传播) |
| 3 | 16 | 7.6% | 辅助(预处理/后处理) |
| 2 | 4 | 1.9% | 优化(融合操作) |
| 1 | 1 | 0.5% | 边缘(naive实现) |

### 3.4 测试覆盖现状

| 得分 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 5 | 19 | 9.0% | 严重缺失(无测试) |
| 4 | 6 | 2.8% | 明显不足(1个测试) |
| 3 | 168 | 79.6% | 部分覆盖(2-4个测试) |
| 2 | 2 | 0.9% | 基本覆盖(5+测试) |
| 1 | 16 | 7.6% | 充分覆盖(10+测试) |

**19 个核心算子 kernel 无测试用例**，涉及算子：
- abc, rebased, rwkv4

## 4. 共享模块 Kernel 分析

### 4.1 全局排名视角

共享模块 kernel 在全局排名中的分布：

| 优先级 | 数量 | 占比 |
|--------|------|------|
| P0 | 0 | 0.0% |
| P1 | 36 | 37.9% |
| P2 | 37 | 38.9% |
| P3 | 22 | 23.2% |

**复用度降权效果**：共享模块 kernel 在全局排名中复用度得分上限为 2 分，而模块内排名使用原始得分。以 `chunk_fwd_kernel_h` 为例：
- 全局排名：复用度 2 分，总分 2.35，优先级 P2
- 模块内排名：复用度 5 分，总分 3.25，优先级 P1

### 4.2 模块内排名视角（不降权）

#### 各模块 kernel 数量与复杂度

| 模块 | kernel 数量 | P0 | P1 | P2 | P3 | 平均总分 |
|------|-----------|-----|-----|-----|-----|---------|
| activations | 8 | 8 | 0 | 0 | 0 | 4.15 |
| common | 52 | 14 | 10 | 10 | 18 | 2.79 |
| conv | 5 | 0 | 0 | 5 | 0 | 2.35 |
| fused_bitlinear | 2 | 0 | 2 | 0 | 0 | 3.33 |
| fused_cross_entropy | 2 | 0 | 2 | 0 | 0 | 3.35 |
| fused_kl_div | 2 | 0 | 0 | 0 | 2 | 1.32 |
| fused_linear_cross_entropy | 3 | 0 | 3 | 0 | 0 | 3.35 |
| fused_norm_gate | 4 | 0 | 4 | 0 | 0 | 3.10 |
| grpo | 2 | 0 | 0 | 0 | 2 | 1.40 |
| l2norm | 4 | 0 | 4 | 0 | 0 | 3.10 |
| layernorm | 4 | 2 | 2 | 0 | 0 | 3.98 |
| layernorm_gated | 2 | 0 | 2 | 0 | 0 | 3.20 |
| rotary | 1 | 0 | 1 | 0 | 0 | 3.35 |
| token_shift | 4 | 0 | 4 | 0 | 0 | 3.25 |

#### 模块内 Top 5 Kernel

| 排名 | Kernel | 总分 | 模块 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 |
|------|--------|------|------|---------|--------|---------|---------|-----------|
| 1 | chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | 4.45 | common | 5 | 5 | 3 | 2 | 5 |
| 2 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 4.45 | common | 5 | 5 | 3 | 2 | 5 |
| 3 | chunk_bwd_kernel_dqkwg | 4.4 | common | 5 | 5 | 3 | 1 | 5 |
| 4 | chunk_fwd_kernel_o | 4.4 | common | 5 | 5 | 3 | 1 | 5 |
| 5 | chunk_bwd_kernel_dv | 4.4 | common | 5 | 5 | 3 | 1 | 5 |

#### 模块内排名与全局排名的差异分析

共享模块 kernel 在模块内排名中优先级普遍高于全局排名，这是复用度降权的直接效果：

| Kernel | 模块内优先级 | 全局优先级 | 模块内复用度 | 全局复用度 |
|--------|------------|-----------|------------|-----------|
| chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | P0 | P1 | 5 | 2 |
| chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | P0 | P1 | 5 | 2 |
| chunk_bwd_kernel_dqkwg | P0 | P1 | 5 | 2 |
| chunk_fwd_kernel_o | P0 | P1 | 5 | 2 |
| chunk_bwd_kernel_dv | P0 | P1 | 5 | 2 |
| tanh | P0 | P1 | 5 | 2 |
| chunk_bwd_kernel_dv_local | P0 | P1 | 5 | 2 |
| log | P0 | P1 | 5 | 2 |
| make_tensor_descriptor | P0 | P1 | 5 | 2 |
| exp | P0 | P1 | 5 | 2 |

## 5. 高优先级 Kernel 详细分析

### 5.1 Top 10 核心 Kernel 深度分析（全局排名）

#### 1. chunk_gla_fwd_kernel_o

- **最终总分**: 4.8 (P0)
- **源码路径**: fla/ops/gla/chunk.py:312
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 5 | fwd(无intra/inter/preprocess) |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 4 | cv/cube+47行 |

#### 2. chunk_gla_bwd_kernel_intra

- **最终总分**: 4.7 (P0)
- **源码路径**: fla/ops/gla/chunk.py:398
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | 含bwd |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 5 | cv/cube+93行+2个tl.dot |

#### 3. chunk_gla_bwd_kernel_inter

- **最终总分**: 4.7 (P0)
- **源码路径**: fla/ops/gla/chunk.py:664
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | 含bwd |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 5 | cv/cube+74行+3个tl.dot |

#### 4. chunk_gla_bwd_kernel_dv

- **最终总分**: 4.6 (P0)
- **源码路径**: fla/ops/gla/chunk.py:587
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | 含bwd |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 4 | cv/cube+36行 |

#### 5. chunk_gla_fwd_A_kernel_intra_sub_inter

- **最终总分**: 4.6 (P0)
- **源码路径**: fla/ops/gla/chunk.py:35
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | fwd+intra/inter/preprocess(前向分步) |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 4 | cv/cube+36行 |

#### 6. pre_process_fwd_kernel_merged

- **最终总分**: 4.5 (P0)
- **源码路径**: fla/ops/cp/chunk_delta_h.py:40
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GatedDeltaNet,GatedDeltaProduct,KDA,MoM |
| 复用度(30%) | 4 | 共享算子: CP,gated_delta_rule,kda, 共享数量: 3 |
| 核心程度(20%) | 5 | fwd(无intra/inter/preprocess) |
| 测试缺失度(5%) | 1 | 多个算子 (测试函数: test_chunk_gdn_intracard_gqa,test_chunk_kda_intrac...) |
| 技术复杂度(10%) | 5 | cv/cube+188行+10个tl.dot |

#### 7. chunk_gla_bwd_kernel_dA

- **最终总分**: 4.5 (P0)
- **源码路径**: fla/ops/gla/chunk.py:534
- **kernel 类型**: cv/cube

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | 含bwd |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 3 | cv/cube+19行 |

#### 8. chunk_gla_fwd_A_kernel_intra_sub_intra

- **最终总分**: 4.5 (P0)
- **源码路径**: fla/ops/gla/chunk.py:109
- **kernel 类型**: vv

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | fwd+intra/inter/preprocess(前向分步) |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 3 | vv+36行 |

#### 9. chunk_gla_fwd_A_kernel_intra_sub_intra_split

- **最终总分**: 4.5 (P0)
- **源码路径**: fla/ops/gla/chunk.py:183
- **kernel 类型**: vv

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | fwd+intra/inter/preprocess(前向分步) |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 3 | vv+39行 |

#### 10. chunk_gla_fwd_A_kernel_intra_sub_intra_merge

- **最终总分**: 4.4 (P0)
- **源码路径**: fla/ops/gla/chunk.py:262
- **kernel 类型**: vv

**各维度得分及支撑数据**：

| 维度 | 得分 | 支撑数据 |
|------|------|---------|
| 模型热度(35%) | 5 | 使用的模型: GLA,GSA,HGRN2,KDA,LightNet,RWKV6,Rodimus |
| 复用度(30%) | 5 | 共享算子: GLA,gsa,kda,rwkv6, 共享数量: 4 |
| 核心程度(20%) | 4 | fwd+intra/inter/preprocess(前向分步) |
| 测试缺失度(5%) | 3 | 多个算子 (测试函数: test_chunk,test_chunk_varlen,test_fused_recurrent,...) |
| 技术复杂度(10%) | 2 | vv+18行 |

### 5.2 Top 5 共享模块 Kernel 分析（模块内排名）

#### 1. chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64

- **模块内总分**: 4.45 (P0)，全局总分: 3.55 (P1)
- **所属模块**: common
- **源码路径**: fla/ops/common/chunk_delta_h.py:344

**各维度得分及支撑数据**：

| 维度 | 模块内得分 | 全局得分 | 支撑数据 |
|------|-----------|---------|---------|
| 模型热度(35%) | 5 | 5 | 共享模块被所有模型使用 |
| 复用度(30%) | 5 | 2 | 原始复用度: 5, 共享数量: 4 |
| 核心程度(20%) | 3 | 3 | 门控组件 |
| 测试缺失度(5%) | 2 | 2 | 多个算子(共享kernel) |
| 技术复杂度(10%) | 5 | 5 | cv/cube+269行+24个tl.dot |

#### 2. chunk_gated_delta_rule_fwd_kernel_h_blockdim64

- **模块内总分**: 4.45 (P0)，全局总分: 3.55 (P1)
- **所属模块**: common
- **源码路径**: fla/ops/common/chunk_delta_h.py:39

**各维度得分及支撑数据**：

| 维度 | 模块内得分 | 全局得分 | 支撑数据 |
|------|-----------|---------|---------|
| 模型热度(35%) | 5 | 5 | 共享模块被所有模型使用 |
| 复用度(30%) | 5 | 2 | 原始复用度: 5, 共享数量: 4 |
| 核心程度(20%) | 3 | 3 | 门控组件 |
| 测试缺失度(5%) | 2 | 2 | 多个算子(共享kernel) |
| 技术复杂度(10%) | 5 | 5 | cv/cube+244行+16个tl.dot |

#### 3. chunk_bwd_kernel_dqkwg

- **模块内总分**: 4.4 (P0)，全局总分: 3.5 (P1)
- **所属模块**: common
- **源码路径**: fla/ops/common/chunk_o.py:156

**各维度得分及支撑数据**：

| 维度 | 模块内得分 | 全局得分 | 支撑数据 |
|------|-----------|---------|---------|
| 模型热度(35%) | 5 | 5 | 共享模块被所有模型使用 |
| 复用度(30%) | 5 | 2 | 原始复用度: 5, 共享数量: 4 |
| 核心程度(20%) | 3 | 3 | 共享模块chunk+bwd |
| 测试缺失度(5%) | 1 | 1 | 多个算子(共享kernel) |
| 技术复杂度(10%) | 5 | 5 | cv/cube+138行+12个tl.dot |

#### 4. chunk_fwd_kernel_o

- **模块内总分**: 4.4 (P0)，全局总分: 3.5 (P1)
- **所属模块**: common
- **源码路径**: fla/ops/common/chunk_o.py:34

**各维度得分及支撑数据**：

| 维度 | 模块内得分 | 全局得分 | 支撑数据 |
|------|-----------|---------|---------|
| 模型热度(35%) | 5 | 5 | 共享模块被所有模型使用 |
| 复用度(30%) | 5 | 2 | 原始复用度: 5, 共享数量: 4 |
| 核心程度(20%) | 3 | 3 | 共享模块chunk+fwd+cv/cube(降级为辅助) |
| 测试缺失度(5%) | 1 | 1 | 多个算子 |
| 技术复杂度(10%) | 5 | 5 | cv/cube+69行+4个tl.dot |

#### 5. chunk_bwd_kernel_dv

- **模块内总分**: 4.4 (P0)，全局总分: 3.5 (P1)
- **所属模块**: common
- **源码路径**: fla/ops/common/chunk_o.py:358

**各维度得分及支撑数据**：

| 维度 | 模块内得分 | 全局得分 | 支撑数据 |
|------|-----------|---------|---------|
| 模型热度(35%) | 5 | 5 | 共享模块被所有模型使用 |
| 复用度(30%) | 5 | 2 | 原始复用度: 5, 共享数量: 4 |
| 核心程度(20%) | 3 | 3 | 共享模块chunk+bwd |
| 测试缺失度(5%) | 1 | 1 | 多个算子(共享kernel) |
| 技术复杂度(10%) | 5 | 5 | cv/cube+55行+3个tl.dot |

## 6. 行动建议

### 6.1 短期行动项（P0 级 kernel）

1. **补充 GLA chunk kernel 测试**：`chunk_gla_fwd_kernel_o`、`chunk_gla_bwd_kernel_*` 等 9 个 P0 级 kernel 需补充独立单元测试，当前仅有间接测试覆盖
2. **补充 GatedDeltaRule kernel 测试**：`pre_process_fwd/bwd_kernel_merged`、`merge_fwd_bwd_kernel` 测试缺失度仅 1 分（充分覆盖），但 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 需关注
3. **建立共享 kernel 回归测试**：`chunk_fwd_kernel_o`、`chunk_bwd_kernel_dqkwg` 等被多算子共享的 kernel，修改前需确认对所有使用算子的影响

### 6.2 中期规划（P1 级 kernel）

1. **覆盖 P1 级核心算子 kernel**：46 个 P1 级核心算子 kernel 需逐步补充测试
2. **优化高复用 kernel 性能**：复用度 5 分的 kernel（如 `chunk_fwd_kernel_o`）性能优化收益最大
3. **关注无测试 kernel**：19 个核心算子 kernel 无任何测试用例，需优先补充

### 6.3 共享模块 kernel 的特殊建议（基于模块内排名）

1. **chunk_delta_h 模块**：`chunk_gated_delta_rule_fwd/bwd_kernel_h_blockdim64` 在模块内排名 Top 2，被 S 级模型（KDA/Kimi）使用，需确保稳定性
2. **chunk_o 模块**：`chunk_fwd_kernel_o`、`chunk_bwd_kernel_dqkwg`、`chunk_bwd_kernel_dv` 被 4+ 算子共享，修改需谨慎
3. **utils/op.py 简单函数**：`tanh`、`log`、`exp` 等在模块内排名靠前（因高复用+高测试缺失），但实际是简单的逐元素操作，修改风险低，可降低优先级
4. **activations 模块**：`swiglu_fwdbwd_kernel` 在模块内排名较高，需关注梯度正确性测试
