# Flash Linear Attention 模型百科

本文档基于任务T1的输出结果，对Flash Linear Attention库中支持的所有模型进行详细介绍，包括模型原理、热度级别、核心特点等信息。

---

## 目录

- [S级模型（高热度）](#s级模型高热度)
  - [GatedDeltaNet](#gateddeltanet)
  - [KDA (Kimi Delta Attention)](#kda-kimi-delta-attention)
  - [MLA (Multi-head Latent Attention)](#mla-multi-head-latent-attention)
- [A级模型（中高热度）](#a级模型中高热度)
  - [DeltaNet](#deltanet)
  - [ForgettingTransformer](#forgettingtransformer)
  - [GLA (Gated Linear Attention)](#gla-gated-linear-attention)
  - [HGRN (Hierarchical Gated Recurrent Network)](#hgrn-hierarchical-gated-recurrent-network)
  - [Mamba2](#mamba2)
  - [NSA (Native Sparse Attention)](#nsa-native-sparse-attention)
  - [RetNet](#retnet)
  - [Samba](#samba)
- [B级模型（中等热度）](#b级模型中等热度)
  - [BitNet](#bitnet)
  - [GSA (Gated Slot Attention)](#gsa-gated-slot-attention)
  - [HGRN2](#hgrn2)
  - [LightNet](#lightnet)
  - [LinearAttention](#linearattention)
  - [RWKV6](#rwkv6)
- [C级模型（中低热度）](#c级模型中低热度)
  - [Comba](#comba)
  - [DeltaFormer](#deltaformer)
  - [GatedDeltaProduct](#gateddeltaproduct)
  - [LogLinearMamba2](#loglinearmamba2)
  - [Mamba](#mamba)
  - [MesaNet](#mesanet)
  - [MoM (Mixture-of-Memories)](#mom-mixture-of-memories)
  - [PaTHAttention](#pathattention)
  - [Rodimus](#rodimus)
  - [RWKV7](#rwkv7)
- [D级模型（低热度）](#d级模型低热度)
  - [ABC (Attention with Bilinear Correlation)](#abc-attention-with-bilinear-correlation)
  - [Transformer](#transformer)

---

## S级模型（高热度）

### GatedDeltaNet

**热度级别**: S级

**核心原理**:  
GatedDeltaNet 是在 DeltaNet 基础上引入门控机制的线性注意力模型。其核心创新在于使用 **通道级（channel-wise）门控**，为记忆的每个维度配备独立的遗忘旋钮，使模型能够动态、精细地控制哪些维度的信息需要保留，哪些可以快速遗忘。

**技术特点**:
- 采用改进的 Delta Rule（增量学习规则）进行状态更新
- 通道级细粒度门控机制，优于传统的标量门控
- 使用 Diagonal-Plus-Low-Rank (DPLR) 结构实现硬件高效的并行计算
- 支持分块并行计算和 kernel fusion 优化

**应用场景**:
- 已被阿里 Qwen3.5 系列模型采用（75%的注意力层使用 Gated DeltaNet）
- Kimi Linear 模型的 KDA 模块基于 GatedDeltaNet 扩展

**算子支持**:
- 核心算子: `gated_delta_rule`
- 实现路径: `fla/ops/gated_delta_rule`
- 支持函数: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`

---

### KDA (Kimi Delta Attention)

**热度级别**: S级

**核心原理**:  
KDA 是月之暗面（Moonshot AI）提出的 Kimi Linear 架构的核心模块。它在 Gated DeltaNet (GDN) 基础上加入了更细粒度的门控机制，实现了 **通道级独立遗忘率**，使模型能够更精确地调控有限的 RNN 式记忆。

**技术特点**:
- 通道级门控：每个特征维度都有独立的遗忘率
- 改进的 Delta Rule：数学上保证稳定性，即使百万级 token 序列也不会梯度爆炸
- DPLR 转移矩阵特化变体：计算量比通用 DPLR 减少约 50%
- 3:1 混合层设计：每 3 层 KDA 后加 1 层全注意力

**性能表现**:
- 在相同训练规模下首次超越全注意力模型
- KV 缓存减少 75%
- 长上下文解码速度提升最高 6 倍
- 100 万上下文长度下的解码吞吐量显著提升

**应用场景**:
- Kimi Linear 48B-A3B 模型（48B 总参数，3B 激活参数）
- 支持智能体推理和测试时扩展

**算子支持**:
- 核心算子: `kda`
- 实现路径: `fla/ops/kda`
- 支持函数: `chunk_kda`, `fused_recurrent_kda`

---

### MLA (Multi-head Latent Attention)

**热度级别**: S级

**核心原理**:  
MLA 是 DeepSeek 提出的多头潜在注意力机制，通过 **低秩 KV 压缩** 显著减少 KV Cache 的内存占用。其核心思想是将 Key 和 Value 压缩到低维潜在空间，在推理时只需缓存压缩后的潜在向量。

**技术特点**:
- 低秩 KV 压缩：将 KV Cache 大小压缩到原来的 1/4 甚至更小
- 兼容分组查询注意力 (GQA)：支持多查询并行处理
- 保持与标准注意力相当的表达能力
- 适合长上下文场景，显著降低显存需求

**应用场景**:
- DeepSeek-V2、DeepSeek-V3 等旗舰模型
- Kimi Linear 的全注意力层采用 MLA

**算子支持**:
- 核心算子: 外部库 (flash_attn)
- Layer: `MultiheadLatentAttention`

---

## A级模型（中高热度）

### DeltaNet

**热度级别**: A级

**核心原理**:  
DeltaNet 将线性注意力的状态更新过程重新解释为 **在线学习**，使用经典的 Delta Rule 来更新记忆状态。核心思想是：状态 S 像一个可学习的记忆，会根据新信息不断自我修正。

**技术特点**:
- Delta Rule 更新：从二次损失目标的单步在线梯度下降推导
- 秩 1 更新：结构优美，支持硬件高效并行计算
- 解决了线性注意力"只会累加、从不遗忘"的问题
- 提出了硬件高效的并行算法，支持序列长度维度的并行

**性能表现**:
- 在关联回忆任务上显著优于标准线性注意力
- 1.3B 模型在 100B tokens 训练后优于 Mamba 和 GLA

**算子支持**:
- 核心算子: `delta_rule`
- 实现路径: `fla/ops/delta_rule`
- 支持函数: `chunk_delta_rule`, `fused_recurrent_delta_rule`

---

### ForgettingTransformer

**热度级别**: A级

**核心原理**:  
ForgettingTransformer (FoX) 在标准 softmax 注意力中引入 **遗忘门机制**，允许模型选择性地"遗忘"不重要的历史信息。结合 PaTH Attention 使用时，可进一步提升长上下文理解能力。

**技术特点**:
- 在 softmax 注意力输出后引入数据依赖的门控
- 以数据依赖方式降低早期、相关性较低信息的权重
- 改善长序列处理能力
- 与 PaTH Attention 结合形成 PaTH-FoX 系统

**算子支持**:
- 核心算子: `forgetting_attn`, `attn`
- 实现路径: `fla/ops/forgetting_attn`, `fla/ops/attn`
- 支持函数: `parallel_forgetting_attn`, `attn_decoding_one_step`

---

### GLA (Gated Linear Attention)

**热度级别**: A级

**核心原理**:  
GLA 是门控线性注意力的代表模型，通过引入 **数据依赖的门控机制** 来控制信息流。与 RetNet 的标量门控不同，GLA 采用向量级门控，提供更细粒度的控制。

**技术特点**:
- 向量级门控：每个维度有独立的门控值
- 硬件高效训练算法：支持分块并行计算
- 线性时间复杂度：O(n) 而非 O(n²)
- 良好的长上下文外推能力

**应用场景**:
- 多个开源模型采用 GLA 作为核心注意力机制
- HGRN2、LightNet、Rodimus 等模型复用 GLA 算子

**算子支持**:
- 核心算子: `gla`
- 实现路径: `fla/ops/gla`
- 支持函数: `chunk_gla`, `fused_chunk_gla`, `fused_recurrent_gla`

---

### HGRN (Hierarchical Gated Recurrent Network)

**热度级别**: A级

**核心原理**:  
HGRN 是分层门控循环网络，通过 **分层遗忘门** 建模不同时间尺度的依赖关系。不同层学习不同时间尺度的模式，底层捕捉短期依赖，高层捕捉长期依赖。

**技术特点**:
- 分层遗忘门：每层有独立的遗忘率
- 多时间尺度建模：自动学习不同层的时间尺度
- 线性推理复杂度
- 适合需要多层次时间建模的任务

**算子支持**:
- 核心算子: `hgrn`
- 实现路径: `fla/ops/hgrn`
- 支持函数: `chunk_hgrn`, `fused_recurrent_hgrn`

---

### Mamba2

**热度级别**: A级

**核心原理**:  
Mamba2 是 Mamba 的升级版本，提出了 **SSD (State Space Duality) 框架**，揭示了状态空间模型与半可分离矩阵之间的等价性。核心创新是将选择性 SSM 改进为矩阵乘法形式，训练速度提升 2-8 倍。

**技术特点**:
- SSD 框架：统一 SSM 和线性注意力的理论
- 更大的状态维度：从 16 提升到 256
- 硬件高效算法：利用半可分离矩阵的块分解
- 与 FlashAttention-2 竞争：序列长度 16K 时快 6 倍

**性能表现**:
- 在 MQAR 任务上显著优于 Mamba-1
- 混合模型（Mamba-2 + 注意力层）优于纯 Transformer

**算子支持**:
- 核心算子: 外部库 (mamba_ssm)
- Layer: `Mamba2`

---

### NSA (Native Sparse Attention)

**热度级别**: A级

**核心原理**:  
NSA 是原生稀疏注意力机制，结合 **滑动窗口、压缩和选择性注意力** 三种机制，在保持计算效率的同时捕捉长距离依赖。

**技术特点**:
- 滑动窗口注意力：捕捉局部依赖
- 压缩注意力：对历史信息进行压缩表示
- 选择性注意力：动态选择重要的历史 token
- 线性计算复杂度

**算子支持**:
- 核心算子: `nsa`
- 实现路径: `fla/ops/nsa`
- 支持函数: `parallel_nsa`

---

### RetNet

**热度级别**: A级

**核心原理**:  
RetNet (Retention Network) 使用 **Retention 机制** 替代标准注意力，支持并行训练和 O(1) 推理复杂度。核心是多尺度保留机制，通过不同的衰减率捕捉不同时间尺度的依赖。

**技术特点**:
- 三种计算形式：并行、循环、分块
- 多尺度衰减：不同 head 有不同的衰减率
- O(1) 推理复杂度：推理时只需维护固定大小的状态
- 训练时并行计算：支持序列长度维度的并行

**算子支持**:
- 核心算子: `retention`
- 实现路径: `fla/ops/retention`
- 支持函数: `chunk_retention`, `fused_chunk_retention`, `fused_recurrent_retention`, `parallel_retention`

---

### Samba

**热度级别**: A级

**核心原理**:  
Samba 是微软提出的混合架构，结合 **Mamba 和滑动窗口注意力 (SWA)**。Mamba 层捕捉时间依赖语义，SWA 层精确检索近期记忆，两者分工明确、功能互补。

**技术特点**:
- 逐层混合架构：Mamba 与 SWA 分层结合
- 无限上下文窗口：理论上可处理任意长度文本
- 线性计算复杂度：避免 Transformer 的二次复杂度
- 高效长度外推：预训练 4K 可零样本外推至 100 万 token

**性能表现**:
- 38 亿参数模型在短上下文基准上优于 Llama-3 8B
- 128K 长度吞吐量比分组查询注意力高 3.73 倍
- Passkey Retrieval 任务中 256K 长度完美召回

**算子支持**:
- 核心算子: 外部库 (mamba_ssm)
- Layer: `Mamba` + 标准注意力

---

## B级模型（中等热度）

### BitNet

**热度级别**: B级

**核心原理**:  
BitNet 是微软提出的 **1.58-bit 量化** 大语言模型架构，使用三值权重 {-1, 0, 1} 表示，显著降低计算资源需求，同时保持与全精度模型相当的性能。

**技术特点**:
- 1.58-bit 量化：权重仅用三值表示
- FusedBitLinear：融合量化的线性层
- SwiGLU 激活函数：支持量化训练
- 显著降低内存和计算需求

**算子支持**:
- 核心算子: 外部库 (flash_attn)
- 特殊模块: `FusedBitLinear`, `swiglu`

---

### GSA (Gated Slot Attention)

**热度级别**: B级

**核心原理**:  
GSA 是门控槽注意力，结合 **GLA 和 softmax 操作**。通过门控机制控制信息流，同时使用 softmax 保证注意力的归一化特性。

**技术特点**:
- 门控机制：数据依赖的信息过滤
- Slot Attention：槽位注意力机制
- 结合线性注意力和 softmax 注意力的优点
- 支持分块并行计算

**算子支持**:
- 核心算子: `gsa`
- 实现路径: `fla/ops/gsa`
- 支持函数: `chunk_gsa`, `fused_recurrent_gsa`

---

### HGRN2

**热度级别**: B级

**核心原理**:  
HGRN2 是 HGRN 的第二代版本，复用 GLA 算子实现更高效的分层门控循环网络。

**技术特点**:
- 复用 GLA 算子：继承 GLA 的硬件高效性
- 分层门控：多时间尺度建模
- 改进的遗忘门机制

**算子支持**:
- 核心算子: `gla` (复用)
- 实现路径: `fla/ops/gla`

---

### LightNet

**热度级别**: B级

**核心原理**:  
LightNet 是轻量级线性注意力网络，复用 GLA 算子实现高效的序列建模。

**技术特点**:
- 轻量级设计：适合资源受限场景
- 复用 GLA 算子：硬件高效
- 良好的性能-效率平衡

**算子支持**:
- 核心算子: `gla` (复用)
- 实现路径: `fla/ops/gla`

---

### LinearAttention

**热度级别**: B级

**核心原理**:  
LinearAttention 是基础的线性注意力实现，通过核函数技巧将注意力计算从 O(n²) 降低到 O(n)。

**技术特点**:
- 核函数近似：使用特征映射替代 softmax
- 循环形式：支持 O(1) 推理
- 分块并行：支持高效训练
- 线性时间和空间复杂度

**算子支持**:
- 核心算子: `linear_attn`
- 实现路径: `fla/ops/linear_attn`
- 支持函数: `chunk_linear_attn`, `fused_chunk_linear_attn`, `fused_recurrent_linear_attn`

---

### RWKV6

**热度级别**: B级

**核心原理**:  
RWKV6 是纯 RNN 架构的大语言模型，无需注意力机制即可实现与 Transformer 相当的性能。通过 **时间混合和通道混合** 实现序列建模。

**技术特点**:
- 纯 RNN 架构：无需注意力
- 线性推理复杂度
- Token Shift：时间信息传递
- 支持并行训练

**算子支持**:
- 核心算子: `rwkv6`
- 实现路径: `fla/ops/rwkv6`
- 支持函数: `chunk_rwkv6`, `fused_recurrent_rwkv6`

---

## C级模型（中低热度）

### Comba

**热度级别**: C级

**核心原理**:  
Comba 基于 **双线性 RNN 和闭环控制理论**，采用标量加低秩状态转换实现高效的序列建模。

**技术特点**:
- 双线性 RNN：高效的状态转换
- 闭环控制理论：稳定的训练过程
- 标量加低秩：减少参数量
- 线性计算复杂度

**算子支持**:
- 核心算子: `comba`
- 实现路径: `fla/ops/comba`
- 支持函数: `chunk_comba`, `fused_recurrent_comba`

---

### DeltaFormer

**热度级别**: C级

**核心原理**:  
DeltaFormer 结合 **Delta Rule 和 Transformer 架构**，在标准注意力框架中引入增量更新机制。

**技术特点**:
- Delta Rule：增量状态更新
- Transformer 架构：保持注意力机制的表达能力
- 改进的长距离依赖建模

**算子支持**:
- 核心算子: `deltaformer`
- 实现路径: `fla/ops/deltaformer`
- 支持函数: `deltaformer_attn`

---

### GatedDeltaProduct

**热度级别**: C级

**核心原理**:  
GatedDeltaProduct 是门控 Delta 乘积注意力，结合门控机制和 Delta Rule 的乘积形式。

**技术特点**:
- 门控 Delta 乘积：改进的信息流控制
- 复用 gated_delta_rule 算子
- 支持分块并行计算

**算子支持**:
- 核心算子: `gated_delta_product`, `gated_delta_rule` (复用)
- 实现路径: `fla/ops/gated_delta_product`, `fla/ops/gated_delta_rule`

---

### LogLinearMamba2

**热度级别**: C级

**核心原理**:  
LogLinearMamba2 是对数线性版本的 Mamba2，通过 **对数线性注意力** 实现更高效的序列建模。

**技术特点**:
- 对数线性注意力：改进的计算效率
- 复用 log_linear_attn 算子
- 适合长序列处理

**算子支持**:
- 核心算子: `log_linear_attn`
- 实现路径: `fla/ops/log_linear_attn`
- 支持函数: `LogLinearAttentionState`, `chunk_log_linear_attn`

---

### Mamba

**热度级别**: C级

**核心原理**:  
Mamba 是第一代选择性状态空间模型，通过 **输入依赖的选择机制** 实现高效的序列建模。虽然已被 Mamba2 取代，但仍是重要的基础模型。

**技术特点**:
- 选择性 SSM：输入依赖的状态转换
- 硬件感知并行扫描：高效训练
- 线性时间复杂度
- 适合长序列建模

**算子支持**:
- 核心算子: 外部库 (mamba_ssm)
- Layer: `Mamba`

---

### MesaNet

**热度级别**: C级

**核心原理**:  
MesaNet 通过 **局部最优测试时训练** 进行序列建模，使用共轭梯度求解器优化状态更新。

**技术特点**:
- 测试时训练：动态适应输入
- 共轭梯度求解器：高效优化
- 局部最优：平衡精度和效率
- 支持解码和并行计算

**算子支持**:
- 核心算子: `mesa_net`
- 实现路径: `fla/ops/mesa_net`
- 支持函数: `chunk_mesa_net`, `mesa_net_decoding_one_step`

---

### MoM (Mixture-of-Memories)

**热度级别**: C级

**核心原理**:  
MoM 是混合记忆架构，维护 **多个独立的记忆状态**，通过路由网络将 token 分配到不同的记忆模块。

**技术特点**:
- 多记忆状态：增加记忆容量
- 路由网络：动态分配 token
- 复用 gated_delta_rule 算子
- 适合需要大量记忆的任务

**算子支持**:
- 核心算子: `gated_delta_rule` (复用)
- 实现路径: `fla/ops/gated_delta_rule`

---

### PaTHAttention

**热度级别**: C级

**核心原理**:  
PaTHAttention 是 MIT 提出的路径注意力机制，将两个 token 之间的关系视为由中间 token 构成的"路径"，通过 **累积 Householder 变换** 实现数据依赖的位置编码。

**技术特点**:
- 数据依赖位置编码：替代静态的 RoPE
- Householder 变换：累积路径信息
- 内容感知：根据路径内容调整位置表示
- 改善状态跟踪和序列推理能力

**性能表现**:
- 在状态跟踪任务上优于 RoPE
- 改善长上下文理解
- 与 ForgettingTransformer 结合效果更佳

**算子支持**:
- 核心算子: `path_attn`, `attn` (复用)
- 实现路径: `fla/ops/path_attn`, `fla/ops/attn`
- 支持函数: `parallel_path_attn`, `attn_decoding_one_step`

---

### Rodimus

**热度级别**: C级

**核心原理**:  
Rodimus 是蚂蚁集团提出的高效大语言模型架构，结合 **线性注意力和滑动窗口共享键注意力 (SW-SKA)**，实现性能与效率的平衡。

**技术特点**:
- DDTS 机制：数据依赖的调节选择
- SW-SKA：滑动窗口共享键注意力
- 语义、token、head 三重压缩
- 训练复杂度亚二次，推理复杂度线性

**性能表现**:
- 1.6B 模型优于同等大小的 Mamba2 和 Qwen2.5-Coder
- 在代码和数学任务上表现突出
- 推理速度快，资源占用低

**算子支持**:
- 核心算子: `gla` (复用)
- 实现路径: `fla/ops/gla`

---

### RWKV7

**热度级别**: C级

**核心原理**:  
RWKV7 是 RWKV 系列的第七代版本，继续优化纯 RNN 架构，支持更高效的序列建模。

**技术特点**:
- 改进的 RNN 架构
- 支持多种循环模式
- 线性推理复杂度
- 复用 LayerNorm 和激活函数模块

**算子支持**:
- 核心算子: `rwkv7`
- 实现路径: `fla/ops/rwkv7`
- 支持函数: `chunk_rwkv7`, `fused_mul_recurrent_rwkv7`

---

## D级模型（低热度）

### ABC (Attention with Bilinear Correlation)

**热度级别**: D级

**核心原理**:  
ABC 是 CVPR 2024 提出的双线性相关注意力，主要用于红外小目标检测。深度融合 CNN 的局部归纳偏置与 Transformer 的全局相关性建模能力。

**技术特点**:
- 双线性注意力模块 (BAM)：计算全局空间相关性
- CLFT 模块：卷积线性融合 Transformer
- UCDC 模块：U 型卷积-空洞卷积
- 专为视觉任务设计

**算子支持**:
- 核心算子: `abc`
- 实现路径: `fla/ops/abc`
- 支持函数: `chunk_abc`

---

### Transformer

**热度级别**: D级

**核心原理**:  
Transformer 是标准的注意力架构，作为基线模型存在。使用标准的 softmax 自注意力机制，计算复杂度为 O(n²)。

**技术特点**:
- 标准 softmax 注意力
- 二次计算复杂度
- 最强的表达能力
- 作为对比基线

**算子支持**:
- 核心算子: 标准注意力 (attn)
- Layer: `Attention`

---

## 总结

### 热度分布

| 热度级别 | 模型数量 | 代表模型 |
|---------|---------|---------|
| S级 | 3 | GatedDeltaNet, KDA, MLA |
| A级 | 9 | DeltaNet, GLA, Mamba2, RetNet, Samba 等 |
| B级 | 6 | BitNet, GSA, HGRN2, LightNet, LinearAttention, RWKV6 |
| C级 | 12 | Comba, DeltaFormer, MoM, PaTHAttention, Rodimus 等 |
| D级 | 2 | ABC, Transformer |

### 技术演进趋势

1. **从标准注意力到线性注意力**: Transformer → LinearAttention → GLA → DeltaNet → GatedDeltaNet
2. **从标量门控到通道级门控**: RetNet → GLA → GatedDeltaNet → KDA
3. **从单一架构到混合架构**: 纯 Transformer → Mamba/SSM → Samba (Mamba+SWA) → Kimi Linear (KDA+MLA)
4. **从静态位置编码到动态位置编码**: RoPE → PaTHAttention

### 核心技术方向

1. **线性注意力**: 降低计算复杂度从 O(n²) 到 O(n)
2. **门控机制**: 精细控制信息流，改善长序列建模
3. **Delta Rule**: 在线学习视角，改进状态更新
4. **混合架构**: 结合不同机制的优势
5. **硬件优化**: 分块并行、kernel fusion、IO 感知算法

---

*本文档基于 Flash Linear Attention 库的任务 T1 输出结果生成，模型热度级别基于行业应用和学术影响力综合评估。*
