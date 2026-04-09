# Kernel函数测试覆盖分析报告

**生成时间**: 2026-04-09 14:33:49

## 统计摘要

| 指标 | 数值 |
|------|------|
| 算子总数 | 54 |
| Kernel函数总数 | 312 |
| 有测试覆盖的Kernel | 239 |
| 无测试覆盖的Kernel | 73 |
| Kernel测试覆盖率 | 76.6% |
| 测试用例总数 | 204 |

## 无测试用例的算子

- ABC
- RWKV4
- Rebased
- logcumsumexp
- logsumexp
- matmul
- norm_gate
- op
- softmax
- softplus

## 详细表格

| 算子名称 | Kernel函数 | Kernel定义路径 | 所属模块 | 功能描述 | 测试用例路径 | 测试覆盖情况 | 备注 |
|----------|------------|----------------|----------|----------|--------------|--------------|------|
| ABC | chunk_abc_fwd_kernel_h | fla/ops/abc/chunk.py:18 | abc | Attention-based Chunk-wise Linear Attention分块线性注意力 | 无测试用例 | 无测试用例 | |
| | chunk_abc_fwd_kernel_intra_K | fla/ops/abc/chunk.py:82 | | | | | |
| | chunk_abc_fwd_kernel_K | fla/ops/abc/chunk.py:132 | | | | | |
| | chunk_abc_fwd_kernel_intra_V | fla/ops/abc/chunk.py:190 | | | | | |
| | chunk_abc_fwd_kernel_V | fla/ops/abc/chunk.py:248 | | | | | |
| | chunk_abc_bwd_kernel_dh | fla/ops/abc/chunk.py:300 | | | | | |
| | chunk_abc_bwd_kernel_V | fla/ops/abc/chunk.py:359 | | | | | |
| | chunk_abc_bwd_kernel_intra_V | fla/ops/abc/chunk.py:453 | | | | | |
| | chunk_abc_bwd_kernel_intra_K | fla/ops/abc/chunk.py:546 | | | | | |
| | chunk_abc_bwd_kernel_K | fla/ops/abc/chunk.py:604 | | | | | |
| | chunk_abc_bwd_kernel_intra_KV | fla/ops/abc/chunk.py:694 | | | | | |
| | chunk_abc_bwd_kernel_rcum_inter | fla/ops/abc/chunk.py:748 | | | | | |
| | chunk_abc_bwd_kernel_rcum_intra | fla/ops/abc/chunk.py:784 | | | | | |
| Attn | naive_attn_decoding_kernel | fla/ops/attn/decoding.py:29 | attn | 标准Flash Attention并行和解码模式 | tests\ops\test_attn.py | 多个算子 | |
| | parallel_attn_fwd_kernel | fla/ops/attn/parallel.py:25 | | | | | |
| | parallel_attn_bwd_kernel_preprocess | fla/ops/attn/parallel.py:149 | | | | | |
| | parallel_attn_bwd_kernel_dq | fla/ops/attn/parallel.py:172 | | | | | |
| | parallel_attn_bwd_kernel_dkv | fla/ops/attn/parallel.py:306 | | | | | |
| Based | fused_chunk_based_fwd_kernel | fla/ops/based/fused_chunk.py:15 | based | Based线性注意力使用泰勒展开 | tests\ops\test_based.py | 多个算子 | |
| | fused_chunk_based_bwd_kernel | fla/ops/based/fused_chunk.py:113 | | | | | |
| | parallel_based_fwd_kernel | fla/ops/based/parallel.py:18 | | | | | |
| | _parallel_based_bwd_dq | fla/ops/based/parallel.py:104 | | | | | |
| | _parallel_based_bwd_dkv | fla/ops/based/parallel.py:187 | | | | | |
| | parallel_based_bwd_kernel | fla/ops/based/parallel.py:268 | | | | | |
| CP | pre_process_fwd_kernel_merged | fla/ops/cp/chunk_delta_h.py:40 | cp | 分布式线性注意力的上下文并行 | tests\context_parallel\test_cp_bwd_gk_offset.py|tests\context_parallel\test_cp_conv.py|tests\context_parallel\test_cp_gdn.py|tests\context_parallel\test_cp_kda.py|tests\ops\test_intracard_cache.py | 多个算子 | |
| | merge_fwd_bwd_kernel | fla/ops/cp/chunk_delta_h.py:285 | | | | | |
| | pre_process_bwd_kernel_merged | fla/ops/cp/chunk_delta_h.py:440 | | | | | |
| Comba | fused_recurrent_comba_fwd_kernel | fla/ops/comba/fused_recurrent.py:21 | comba | 压缩线性注意力带门控 | tests\ops\test_comba.py | 多个算子 | |
| | chunk_comba_cumsum_scalar_fwd_kernel | fla/ops/comba/utils.py:28 | | | | | |
| | chunk_comba_cumsum_scalar_bwd_kernel | fla/ops/comba/utils.py:107 | | | | | |
| | chunk_scaled_dot_comba_pkt_fwd_kernel | fla/ops/comba/wy_fast.py:31 | | | | | |
| | prepare_wy_repr_bwd_kernel | fla/ops/comba/wy_fast.py:164 | | | | | |
| | recompute_w_u_fwd_kernel | fla/ops/comba/wy_fast.py:300 | | | | | |
| Common | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | fla/ops/common/chunk_delta_h.py:39 | common | 分块线性注意力共享kernel(多算子共享) | tests\ops\test_gated_delta.py|tests\ops\test_gla.py|tests\ops\test_gsa.py|tests\ops\test_hgrn.py|tests\ops\test_kda.py|tests\ops\test_linear_attn.py|tests\ops\test_retention.py|tests\ops\test_rwkv6.py|tests\ops\test_simple_gla.py | 多个算子(共享kernel) | |
| | chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 | fla/ops/common/chunk_delta_h.py:344 | | | | | |
| | chunk_fwd_kernel_h | fla/ops/common/chunk_h.py:35 | | | | | |
| | chunk_bwd_kernel_dh | fla/ops/common/chunk_h.py:162 | | | | | |
| | chunk_fwd_kernel_h_parallel | fla/ops/common/chunk_h_parallel.py:38 | | | | | |
| | chunk_fwd_kernel_h_reduction | fla/ops/common/chunk_h_parallel.py:153 | | | | | |
| | chunk_bwd_kernel_dh_parallel | fla/ops/common/chunk_h_parallel.py:239 | | | | | |
| | chunk_bwd_kernel_dh_reduction | fla/ops/common/chunk_h_parallel.py:343 | | | | | |
| | chunk_fwd_kernel_h_split | fla/ops/common/chunk_h_split.py:32 | | | | | |
| | chunk_fwd_kernel_h_reduction | fla/ops/common/chunk_h_split.py:152 | | | | | |
| | chunk_bwd_kernel_dh_split | fla/ops/common/chunk_h_split.py:239 | | | | | |
| | chunk_bwd_kernel_dh_reduction | fla/ops/common/chunk_h_split.py:358 | | | | | |
| | chunk_fwd_kernel_o | fla/ops/common/chunk_o.py:34 | | | | | |
| | chunk_bwd_kernel_dqkwg | fla/ops/common/chunk_o.py:156 | | | | | |
| | chunk_bwd_kernel_dv | fla/ops/common/chunk_o.py:358 | | | | | |
| | chunk_bwd_kernel_dv_local | fla/ops/common/chunk_o.py:460 | | | | | |
| | chunk_scaled_dot_kkt_fwd_kernel | fla/ops/common/chunk_scaled_dot_kkt.py:31 | | | | | |
| | fused_chunk_fwd_kernel | fla/ops/common/fused_chunk.py:44 | | | | | |
| | fused_chunk_bwd_kernel | fla/ops/common/fused_chunk.py:178 | | | | | |
| | fused_recurrent_fwd_kernel | fla/ops/common/fused_recurrent.py:29 | | | | | |
| | fused_recurrent_bwd_kernel | fla/ops/common/fused_recurrent.py:142 | | | | | |
| DPLR | chunk_dplr_bwd_kernel_intra | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:32 | generalized_delta_rule | Generalized Delta Rule (DPLR变体) | tests\ops\test_dplr_delta.py | 多个算子 | |
| | chunk_dplr_bwd_kernel_intra_tensorcore | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:237 | | | | | |
| | chunk_dplr_bwd_dgk_kernel | fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py:407 | | | | | |
| | chunk_dplr_fwd_A_kernel_intra_sub_intra | fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py:32 | | | | | |
| | chunk_dplr_fwd_A_kernel_intra_tensorcore | fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py:159 | | | | | |
| | chunk_dplr_bwd_kernel_dhu | fla/ops/generalized_delta_rule/dplr/chunk_h_bwd.py:34 | | | | | |
| | chunk_dplr_fwd_kernel_h | fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py:34 | | | | | |
| | chunk_dplr_bwd_kernel_dAu | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:34 | | | | | |
| | chunk_dplr_bwd_o_kernel | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:107 | | | | | |
| | chunk_dplr_bwd_kernel_dv | fla/ops/generalized_delta_rule/dplr/chunk_o_bwd.py:237 | | | | | |
| | chunk_dplr_fwd_kernel_o | fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py:35 | | | | | |
| | fused_recurrent_dplr_delta_rule_fwd_kernel | fla/ops/generalized_delta_rule/dplr/fused_recurrent.py:32 | | | | | |
| | prepare_wy_repr_bwd_kernel | fla/ops/generalized_delta_rule/dplr/wy_fast_bwd.py:32 | | | | | |
| | prepare_wy_repr_fwd_kernel_chunk32 | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:29 | | | | | |
| | prepare_wy_repr_fwd_kernel_chunk64 | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:75 | | | | | |
| | wu_fwd_kernel | fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py:155 | | | | | |
| DeltaRule | fused_recurrent_delta_rule_fwd_kernel | fla/ops/delta_rule/fused_recurrent.py:21 | delta_rule | Delta Rule线性注意力 | tests\ops\test_delta.py | 多个算子 | |
| | fused_recurrent_delta_rule_bwd_kernel | fla/ops/delta_rule/fused_recurrent.py:108 | | | | | |
| | chunk_transform_qk_fwd_kernel | fla/ops/delta_rule/parallel.py:27 | | | | | |
| | save_intra_chunk_attn | fla/ops/delta_rule/parallel.py:135 | | | | | |
| | parallel_delta_rule_fwd_kernel | fla/ops/delta_rule/parallel.py:152 | | | | | |
| | recompute_w_u_fwd_kernel | fla/ops/delta_rule/wy_fast.py:32 | | | | | |
| | prepare_wy_repr_bwd_kernel | fla/ops/delta_rule/wy_fast.py:94 | | | | | |
| Deltaformer | parallel_deltaformer_fwd_kernel | fla/ops/deltaformer/parallel.py:136 | deltaformer | Deltaformer并行线性注意力 | tests\ops\test_deltaformer.py | 多个算子 | |
| | parallel_deltaformer_bwd_kernel_u | fla/ops/deltaformer/parallel.py:277 | | | | | |
| | parallel_deltaformer_bwd_kernel_row_sum | fla/ops/deltaformer/parallel.py:364 | | | | | |
| | parallel_deltaformer_bwd_kernel_qk | fla/ops/deltaformer/parallel.py:458 | | | | | |
| GLA | chunk_gla_fwd_A_kernel_intra_sub_inter | fla/ops/gla/chunk.py:35 | gla | Gated Linear Attention门控线性注意力 | tests\ops\test_gla.py | 多个算子 | |
| | chunk_gla_fwd_A_kernel_intra_sub_intra | fla/ops/gla/chunk.py:109 | | | | | |
| | chunk_gla_fwd_A_kernel_intra_sub_intra_split | fla/ops/gla/chunk.py:183 | | | | | |
| | chunk_gla_fwd_A_kernel_intra_sub_intra_merge | fla/ops/gla/chunk.py:262 | | | | | |
| | chunk_gla_fwd_kernel_o | fla/ops/gla/chunk.py:312 | | | | | |
| | chunk_gla_bwd_kernel_intra | fla/ops/gla/chunk.py:398 | | | | | |
| | chunk_gla_bwd_kernel_dA | fla/ops/gla/chunk.py:534 | | | | | |
| | chunk_gla_bwd_kernel_dv | fla/ops/gla/chunk.py:587 | | | | | |
| | chunk_gla_bwd_kernel_inter | fla/ops/gla/chunk.py:664 | | | | | |
| GSA | chunk_gsa_fwd_k_kernel_inter | fla/ops/gsa/chunk.py:38 | gsa | Gated State Space Attention门控状态空间注意力 | tests\ops\test_gsa.py | 多个算子 | |
| | chunk_gsa_fwd_k_kernel_intra | fla/ops/gsa/chunk.py:112 | | | | | |
| | chunk_gsa_bwd_k_kernel_dA | fla/ops/gsa/chunk.py:200 | | | | | |
| | chunk_gsa_bwd_k_kernel_dqkvg | fla/ops/gsa/chunk.py:301 | | | | | |
| | chunk_gsa_bwd_k_kernel_intra_dvg | fla/ops/gsa/chunk.py:427 | | | | | |
| | fused_recurrent_gsa_inference_kernel | fla/ops/gsa/fused_recurrent.py:17 | | | | | |
| GatedDeltaProduct | chunk_gated_delta_product_fwd_kernel_h_blockdim64 | fla/ops/gated_delta_product/chunk_deltaproduct_h.py:37 | gated_delta_product | Gated Delta Product线性注意力 | tests\ops\test_gated_delta.py|tests\ops\test_gated_delta_product.py|tests\ops\test_gla.py|tests\ops\test_kda.py|tests\ops\test_linear_attn.py|tests\ops\test_retention.py|tests\ops\test_rwkv6.py|tests\ops\test_simple_gla.py | 多个算子 | |
| | chunk_gated_delta_product_bwd_kernel_dhu_blockdim64 | fla/ops/gated_delta_product/chunk_deltaproduct_h.py:216 | | | | | |
| | chunk_fwd_kernel_o | fla/ops/gated_delta_product/chunk_deltaproduct_o.py:35 | | | | | |
| GatedDeltaRule | chunk_gated_delta_rule_fwd_kkt_solve_kernel | fla/ops/gated_delta_rule/chunk_fwd.py:36 | gated_delta_rule | Gated Delta Rule线性注意力 | tests\ops\test_gated_delta.py | 多个算子 | |
| | fused_recurrent_gated_delta_rule_fwd_kernel | fla/ops/gated_delta_rule/fused_recurrent.py:24 | | | | | |
| | gdn_gate_chunk_cumsum_scalar_kernel | fla/ops/gated_delta_rule/gate.py:60 | | | | | |
| | gdn_gate_bwd_kernel | fla/ops/gated_delta_rule/gate.py:116 | | | | | |
| | gdn_gate_fwd_kernel | fla/ops/gated_delta_rule/gate.py:235 | | | | | |
| | safe_dot | fla/ops/gated_delta_rule/wy_fast.py:27 | | | | | |
| | safe_dot | fla/ops/gated_delta_rule/wy_fast.py:38 | | | | | |
| | recompute_w_u_fwd_kernel | fla/ops/gated_delta_rule/wy_fast.py:56 | | | | | |
| | prepare_wy_repr_bwd_kernel | fla/ops/gated_delta_rule/wy_fast.py:132 | | | | | |
| GatedOjaRule | chunk_oja_fwd_kernel_h_blockdim64 | fla/ops/gated_oja_rule/chunk_h.py:37 | gated_oja_rule | Gated Oja Rule线性注意力 | tests\ops\test_gated_delta.py|tests\ops\test_gla.py|tests\ops\test_kda.py|tests\ops\test_linear_attn.py|tests\ops\test_oja.py|tests\ops\test_retention.py|tests\ops\test_rwkv6.py|tests\ops\test_simple_gla.py | 多个算子 | |
| | chunk_oja_bwd_kernel_dhu_blockdim64 | fla/ops/gated_oja_rule/chunk_h.py:264 | | | | | |
| | chunk_gsa_bwd_k_kernel_dqkvg | fla/ops/gated_oja_rule/chunk_h.py:527 | | | | | |
| | chunk_oja_bwd_kernel_dvwg_h | fla/ops/gated_oja_rule/chunk_h.py:663 | | | | | |
| | chunk_scaled_dot_kkt_fwd_kernel | fla/ops/gated_oja_rule/chunk_kkt.py:29 | | | | | |
| | chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter | fla/ops/gated_oja_rule/chunk_kkt.py:90 | | | | | |
| | chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra | fla/ops/gated_oja_rule/chunk_kkt.py:166 | | | | | |
| | chunk_scaled_dot_kkt_bwd_kernel_gk | fla/ops/gated_oja_rule/chunk_kkt.py:230 | | | | | |
| | chunk_oja_fwd_inter | fla/ops/gated_oja_rule/chunk_o.py:33 | | | | | |
| | chunk_oja_fwd_intra | fla/ops/gated_oja_rule/chunk_o.py:107 | | | | | |
| | chunk_oja_bwd_kernel_dA | fla/ops/gated_oja_rule/chunk_o.py:262 | | | | | |
| | chunk_oja_bwd_kernel_dqk | fla/ops/gated_oja_rule/chunk_o.py:404 | | | | | |
| | chunk_oja_bwd_kernel_dv_o | fla/ops/gated_oja_rule/chunk_o.py:542 | | | | | |
| | fused_recurrent_oja_fwd_kernel | fla/ops/gated_oja_rule/fused_recurrent.py:22 | | | | | |
| | recompute_w_u_fwd_kernel | fla/ops/gated_oja_rule/wy_fast.py:29 | | | | | |
| | prepare_wy_repr_bwd_kernel | fla/ops/gated_oja_rule/wy_fast.py:108 | | | | | |
| HGRN | chunk_hgrn_fwd_kernel_h | fla/ops/hgrn/chunk.py:52 | hgrn | Hierarchically Gated Recurrent Network分层门控递归网络 | tests\ops\test_hgrn.py | 多个算子 | |
| | chunk_hgrn_fwd_kernel_o | fla/ops/hgrn/chunk.py:94 | | | | | |
| | chunk_hgrn_bwd_kernel_h | fla/ops/hgrn/chunk.py:132 | | | | | |
| | chunk_hgrn_bwd_kernel_o | fla/ops/hgrn/chunk.py:178 | | | | | |
| | fused_recurrent_hgrn_fwd_kernel | fla/ops/hgrn/fused_recurrent.py:30 | | | | | |
| | fused_recurrent_hgrn_bwd_kernel | fla/ops/hgrn/fused_recurrent.py:92 | | | | | |
| IPLR | chunk_generalized_iplr_delta_rule_fwd_kernel_h | fla/ops/generalized_delta_rule/iplr/chunk.py:42 | generalized_delta_rule | Generalized Delta Rule (IPLR变体) | tests\ops\test_gated_delta.py|tests\ops\test_gla.py|tests\ops\test_gsa.py|tests\ops\test_hgrn.py|tests\ops\test_iplr_delta.py|tests\ops\test_kda.py|tests\ops\test_linear_attn.py|tests\ops\test_retention.py|tests\ops\test_rwkv6.py|tests\ops\test_simple_gla.py | 多个算子 | |
| | chunk_generalized_iplr_delta_rule_fwd_kernel_o | fla/ops/generalized_delta_rule/iplr/chunk.py:127 | | | | | |
| | fused_recurrent_fwd_kernel | fla/ops/generalized_delta_rule/iplr/fused_recurrent.py:30 | | | | | |
| | fused_recurrent_bwd_kernel | fla/ops/generalized_delta_rule/iplr/fused_recurrent.py:121 | | | | | |
| | prepare_wy_repr_fwd_kernel_chunk32 | fla/ops/generalized_delta_rule/iplr/wy_fast.py:29 | | | | | |
| | prepare_wy_repr_fwd_kernel_chunk64 | fla/ops/generalized_delta_rule/iplr/wy_fast.py:84 | | | | | |
| | wu_fwd_kernel | fla/ops/generalized_delta_rule/iplr/wy_fast.py:165 | | | | | |
| KDA | chunk_kda_bwd_kernel_dAv | fla/ops/kda/chunk_bwd.py:47 | kda | Kernelized Delta Attention核化Delta注意力 | tests\ops\test_kda.py | 多个算子 | |
| | chunk_kda_bwd_kernel_wy_dqkg_fused | fla/ops/kda/chunk_bwd.py:128 | | | | | |
| | chunk_kda_fwd_kernel_inter_solve_fused | fla/ops/kda/chunk_intra.py:40 | | | | | |
| | chunk_kda_bwd_kernel_intra | fla/ops/kda/chunk_intra.py:364 | | | | | |
| | chunk_kda_fwd_kernel_intra_sub_chunk | fla/ops/kda/chunk_intra.py:640 | | | | | |
| | chunk_kda_fwd_kernel_intra_token_parallel | fla/ops/kda/chunk_intra_token_parallel.py:30 | | | | | |
| | fused_recurrent_kda_fwd_kernel | fla/ops/kda/fused_recurrent.py:30 | | | | | |
| | kda_gate_fwd_kernel | fla/ops/kda/gate.py:86 | | | | | |
| | kda_gate_bwd_kernel | fla/ops/kda/gate.py:142 | | | | | |
| | kda_gate_chunk_cumsum_vector_kernel | fla/ops/kda/gate.py:361 | | | | | |
| | recompute_w_u_fwd_kda_kernel | fla/ops/kda/wy_fast.py:31 | | | | | |
| | prepare_wy_repr_bwd_kda_kernel | fla/ops/kda/wy_fast.py:118 | | | | | |
| LogLinearAttn | chunkwise_fwd_kernel | fla/ops/log_linear_attn/chunk.py:41 | log_linear_attn | Log线性注意力分块实现 | tests\ops\test_log_linear_attn.py | 多个算子 | |
| | copy_input_kernel | fla/ops/log_linear_attn/chunk.py:677 | | | | | |
| | copy_last_chunk_kernel | fla/ops/log_linear_attn/chunk.py:828 | | | | | |
| | chunkwise_bwd_kernel_dhg | fla/ops/log_linear_attn/chunk.py:930 | | | | | |
| | chunkwise_bwd_kernel_hdqgl | fla/ops/log_linear_attn/chunk.py:1047 | | | | | |
| | chunkwise_bwd_kernel_dkg | fla/ops/log_linear_attn/chunk.py:1195 | | | | | |
| | chunkwise_bwd_kernel_dv | fla/ops/log_linear_attn/chunk.py:1285 | | | | | |
| | chunkwise_bwd_kernel_diag | fla/ops/log_linear_attn/chunk.py:1351 | | | | | |
| MesaNet | chunk_update_once | fla/ops/mesa_net/chunk_cg_solver_bwd.py:16 | mesa_net | Mesa Network注意力带CG求解器 | tests\ops\test_mesa.py | 多个算子 | |
| | chunk_fwd_mesa_cg_dim64_kernel | fla/ops/mesa_net/chunk_cg_solver_bwd.py:36 | | | | | |
| | chunk_update_once | fla/ops/mesa_net/chunk_cg_solver_fwd.py:16 | | | | | |
| | chunk_fwd_mesa_cg_dim64_kernel | fla/ops/mesa_net/chunk_cg_solver_fwd.py:36 | | | | | |
| | chunk_mesa_net_fwd_kernel_h | fla/ops/mesa_net/chunk_h_fwd.py:31 | | | | | |
| | chunk_mesa_net_h_kk_bwd_intra_kernel | fla/ops/mesa_net/chunk_h_kk_intra_bwd.py:19 | | | | | |
| | chunk_mesa_net_h_kv_bwd_intra_kernel | fla/ops/mesa_net/chunk_h_kv_intra_bwd.py:32 | | | | | |
| | chunk_mesa_net_h_kv_bwd_intra_kernel_dkv | fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py:31 | | | | | |
| | chunk_mesa_net_h_kv_bwd_intra_kernel_dq | fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py:150 | | | | | |
| | mesa_net_decoding_one_step_kernel | fla/ops/mesa_net/decoding_one_step.py:16 | | | | | |
| NSA | parallel_nsa_compression_fwd_kernel | fla/ops/nsa/compression.py:29 | nsa | Native Sparse Attention原生稀疏注意力 | tests\ops\test_nsa.py | 多个算子 | |
| | parallel_nsa_compression_bwd_kernel_dq | fla/ops/nsa/compression.py:132 | | | | | |
| | parallel_nsa_compression_bwd_kernel_dkv | fla/ops/nsa/compression.py:236 | | | | | |
| | parallel_nsa_kernel_topk | fla/ops/nsa/parallel.py:43 | | | | | |
| | parallel_nsa_fwd_kernel | fla/ops/nsa/parallel.py:181 | | | | | |
| | parallel_nsa_kernel_mask | fla/ops/nsa/parallel.py:271 | | | | | |
| | parallel_nsa_bwd_kernel_dq | fla/ops/nsa/parallel.py:308 | | | | | |
| | parallel_nsa_bwd_kernel_dkv | fla/ops/nsa/parallel.py:417 | | | | | |
| | _compare_and_swap | fla/ops/nsa/utils.py:20 | | | | | |
| | _bitonic_merge | fla/ops/nsa/utils.py:55 | | | | | |
| | argsort | fla/ops/nsa/utils.py:81 | | | | | |
| PathAttn | chunk_cumprod_householder_bwd_kernel | fla/ops/path_attn/cumprod_householder_bwd.py:19 | path_attn | Path Attention带Householder变换 | tests\ops\test_path_attn.py | 多个算子 | |
| | chunk_cumprod_householder_fwd_kernel | fla/ops/path_attn/cumprod_householder_fwd.py:19 | | | | | |
| | intra_chunk_preprocess_bwd_kernel | fla/ops/path_attn/intra_chunk_preprocess_bwd.py:20 | | | | | |
| | chunk_transform_qk_bwd_kernel_prepare | fla/ops/path_attn/intra_chunk_preprocess_bwd_prepare.py:19 | | | | | |
| | intra_chunk_preprocess_fwd_kernel | fla/ops/path_attn/intra_chunk_preprocess_fwd.py:19 | | | | | |
| | parallel_path_bwd_dkv_kernel | fla/ops/path_attn/parallel_path_bwd_inter_dkv.py:19 | | | | | |
| | parallel_path_bwd_dq_kernel | fla/ops/path_attn/parallel_path_bwd_inter_dqh.py:21 | | | | | |
| | parallel_path_bwd_intra_chunk_kernel | fla/ops/path_attn/parallel_path_bwd_intra.py:19 | | | | | |
| | parallel_path_fwd_kernel | fla/ops/path_attn/parallel_path_fwd.py:19 | | | | | |
| | parallel_path_fwd_kernel_prepare_k_cache | fla/ops/path_attn/prepare_k_cache.py:18 | | | | | |
| | transform_q_fwd_kernel | fla/ops/path_attn/transform_q.py:18 | | | | | |
| RWKV4 | fused_recurrent_rwkv4_forward_kernel | fla/ops/rwkv4/fused_recurrent.py:28 | rwkv4 | RWKV-4注意力机制 | 无测试用例 | 无测试用例 | |
| | fused_recurrent_rwkv4_backward_kernel | fla/ops/rwkv4/fused_recurrent.py:179 | | | | | |
| RWKV6 | fused_recurrent_rwkv6_fwd_kernel | fla/ops/rwkv6/fused_recurrent.py:31 | rwkv6 | RWKV-6注意力机制分块实现 | tests\ops\test_rwkv6.py | 多个算子 | |
| | fused_recurrent_rwkv6_bwd_kernel_dq | fla/ops/rwkv6/fused_recurrent.py:118 | | | | | |
| | fused_recurrent_rwkv6_bwd_kernel_dkv | fla/ops/rwkv6/fused_recurrent.py:208 | | | | | |
| | fused_recurrent_rwkv6_bwd_kernel_dw | fla/ops/rwkv6/fused_recurrent.py:306 | | | | | |
| | chunk_rwkv6_fwd_cumsum_kernel | fla/ops/rwkv6/chunk.py:45 | | | | | |
| | chunk_rwkv6_fwd_A_kernel_intra_sub_inter | fla/ops/rwkv6/chunk.py:126 | | | | | |
| | chunk_rwkv6_fwd_A_kernel_intra_sub_intra | fla/ops/rwkv6/chunk.py:202 | | | | | |
| | chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split | fla/ops/rwkv6/chunk.py:277 | | | | | |
| | chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge | fla/ops/rwkv6/chunk.py:358 | | | | | |
| | chunk_rwkv6_bwd_kernel_dh | fla/ops/rwkv6/chunk.py:411 | | | | | |
| | chunk_rwkv6_bwd_kernel_intra | fla/ops/rwkv6/chunk.py:492 | | | | | |
| | chunk_rwkv6_bwd_kernel_inter | fla/ops/rwkv6/chunk.py:633 | | | | | |
| RWKV7 | rwkv_seq_mix_kernel | fla/ops/rwkv7/channel_mixing.py:38 | rwkv7 | RWKV-7注意力机制 | tests\ops\test_rwkv7.py | 多个算子 | |
| | rwkv_channel_mixing_pow_and_relu | fla/ops/rwkv7/channel_mixing.py:83 | | | | | |
| | relu_square_bwd_kernel | fla/ops/rwkv7/channel_mixing.py:173 | | | | | |
| | rwkv_mix_bwd_kenel | fla/ops/rwkv7/channel_mixing.py:205 | | | | | |
| | fused_addcmul_fwd_kernel | fla/ops/rwkv7/fused_addcmul.py:53 | | | | | |
| | addcmul_bwd_kernel1 | fla/ops/rwkv7/fused_addcmul.py:113 | | | | | |
| | k_update_fwd_kernel_short | fla/ops/rwkv7/fused_k_update.py:32 | | | | | |
| | k_update_fwd_kernel_long | fla/ops/rwkv7/fused_k_update.py:75 | | | | | |
| | k_update_bwd_kernel_short | fla/ops/rwkv7/fused_k_update.py:122 | | | | | |
| | k_update_bwd_kernel_long | fla/ops/rwkv7/fused_k_update.py:174 | | | | | |
| | fused_recurrent_rwkv7_fwd_kernel | fla/ops/rwkv7/fused_recurrent.py:35 | | | | | |
| | gate_output_correction_fwd_kernel | fla/ops/rwkv7/gate_output_correction.py:73 | | | | | |
| | gate_output_correction_bwd_kernel | fla/ops/rwkv7/gate_output_correction.py:125 | | | | | |
| Rebased | parallel_rebased_fwd_kernel | fla/ops/rebased/parallel.py:18 | rebased | Rebased线性注意力带可学习核 | 无测试用例 | 无测试用例 | |
| | _parallel_rebased_bwd_dq | fla/ops/rebased/parallel.py:103 | | | | | |
| | _parallel_rebased_bwd_dkv | fla/ops/rebased/parallel.py:187 | | | | | |
| | parallel_rebased_bwd_kernel | fla/ops/rebased/parallel.py:273 | | | | | |
| SimpleGLA | parallel_simple_gla_fwd_kernel | fla/ops/simple_gla/parallel.py:47 | simple_gla | Simple Gated Linear Attention简化门控线性注意力 | tests\ops\test_simple_gla.py | 多个算子 | |
| | parallel_simple_gla_bwd_kernel_dq | fla/ops/simple_gla/parallel.py:166 | | | | | |
| | parallel_simple_gla_bwd_kernel_dkv | fla/ops/simple_gla/parallel.py:258 | | | | | |
| | parallel_simple_gla_bwd_kernel | fla/ops/simple_gla/parallel.py:382 | | | | | |
| TTT | chunk_ttt_linear_fwd_kernel_h | fla/ops/ttt/chunk.py:34 | ttt | Test-Time Training线性注意力 | tests\ops\test_ttt.py | 多个算子 | |
| | chunk_ttt_linear_fwd_kernel_o | fla/ops/ttt/chunk.py:139 | | | | | |
| | chunk_ttt_linear_bwd_kernel_h | fla/ops/ttt/chunk.py:232 | | | | | |
| | chunk_ttt_linear_bwd_kernel_dv_local | fla/ops/ttt/chunk.py:334 | | | | | |
| | chunk_ttt_linear_bwd_kernel_norm | fla/ops/ttt/chunk.py:409 | | | | | |
| | chunk_bwd_kernel_dqke | fla/ops/ttt/chunk.py:567 | | | | | |
| | fused_chunk_ttt_linear_fwd_kernel | fla/ops/ttt/fused_chunk.py:35 | | | | | |
| | fused_chunk_ttt_linear_bwd_kernel_h | fla/ops/ttt/fused_chunk.py:158 | | | | | |
| | fused_chunk_ttt_linear_bwd_kernel_dh | fla/ops/ttt/fused_chunk.py:278 | | | | | |
| bitlinear | layer_norm_fwd_kernel_quant | fla/modules/fused_bitlinear.py:72 | fused_bitlinear | 融合量化LayerNorm | tests\modules\test_layernorm.py | 多个算子 | |
| | layer_norm_bwd_kernel | fla/modules/fused_bitlinear.py:207 | | | | | |
| causal_conv1d | causal_conv1d_fwd_kernel | fla/modules/conv/triton/kernels.py:35 | conv | 因果一维卷积 | tests\modules\test_conv.py | 多个算子 | |
| | causal_conv1d_bwd_kernel | fla/modules/conv/triton/kernels.py:152 | | | | | |
| | causal_conv1d_update_kernel | fla/modules/conv/triton/kernels.py:325 | | | | | |
| | compute_dh0_kernel | fla/modules/conv/triton/kernels.py:410 | | | | | |
| | causal_conv1d_states_fwd_kernel | fla/modules/conv/triton/kernels.py:487 | | | | | |
| cross_entropy | cross_entropy_fwd_kernel | fla/modules/fused_cross_entropy.py:30 | fused_cross_entropy | 融合交叉熵损失 | tests\modules\test_cross_entropy.py | 多个算子 | |
| | cross_entropy_bwd_kernel | fla/modules/fused_cross_entropy.py:106 | | | | | |
| cumsum | chunk_local_cumsum_scalar_kernel | fla/ops/utils/cumsum.py:30 | utils | 分块累加和工具函数 | tests\ops\test_utils.py | 多个算子 | |
| | chunk_local_cumsum_vector_kernel | fla/ops/utils/cumsum.py:85 | | | | | |
| | chunk_global_cumsum_scalar_kernel | fla/ops/utils/cumsum.py:143 | | | | | |
| | chunk_global_cumsum_vector_kernel | fla/ops/utils/cumsum.py:203 | | | | | |
| grpo | grpo_fwd_kernel | fla/modules/grpo.py:79 | grpo | GRPO损失函数 | tests\modules\test_grpo.py | 多个算子 | |
| | grpo_bwd_kernel | fla/modules/grpo.py:155 | | | | | |
| index | prepare_position_ids_kernel | fla/ops/utils/index.py:24 | utils | 位置ID准备工具函数 | tests\ops\test_index.py | 单个算子 | |
| kl_div | kl_div_kernel | fla/modules/fused_kl_div.py:25 | fused_kl_div | 融合KL散度损失 | tests\modules\test_kl_div.py | 多个算子 | |
| | elementwise_mul_kernel | fla/modules/fused_kl_div.py:90 | | | | | |
| l2norm | l2norm_fwd_kernel1 | fla/modules/l2norm.py:24 | l2norm | L2归一化 | tests\modules\test_l2norm.py | 多个算子 | |
| | l2norm_bwd_kernel1 | fla/modules/l2norm.py:52 | | | | | |
| | l2norm_fwd_kernel | fla/modules/l2norm.py:81 | | | | | |
| | l2norm_bwd_kernel | fla/modules/l2norm.py:111 | | | | | |
| layernorm | layer_norm_fwd_kernel | fla/modules/layernorm.py:193 | layernorm | LayerNorm/RMSNorm | tests\modules\test_layernorm.py | 多个算子 | |
| | layer_norm_fwd_kernel1 | fla/modules/layernorm.py:267 | | | | | |
| | layer_norm_bwd_kernel | fla/modules/layernorm.py:340 | | | | | |
| | layer_norm_bwd_kernel1 | fla/modules/layernorm.py:457 | | | | | |
| layernorm_gated | layer_norm_fwd_kernel | fla/modules/layernorm_gated.py:54 | layernorm_gated | 门控LayerNorm | tests\modules\test_layernorm.py|tests\modules\test_layernorm_gated.py | 多个算子 | |
| | layer_norm_bwd_kernel | fla/modules/layernorm_gated.py:186 | | | | | |
| linear_cross_entropy | logsumexp_fwd_kernel | fla/modules/fused_linear_cross_entropy.py:42 | fused_linear_cross_entropy | 融合线性交叉熵损失 | tests\modules\test_kl_div.py|tests\modules\test_l2warp.py | 多个算子 | |
| | cross_entropy_kernel | fla/modules/fused_linear_cross_entropy.py:94 | | | | | |
| | elementwise_mul_kernel | fla/modules/fused_linear_cross_entropy.py:242 | | | | | |
| logcumsumexp | logcumsumexp_fwd_kernel | fla/ops/utils/logcumsumexp.py:24 | utils | 对数累加指数工具函数 | 无测试用例 | 无测试用例 | |
| logsigmoid | logsigmoid_fwd_kernel | fla/modules/activations.py:195 | activations | LogSigmoid激活函数 | tests\modules\test_activations_noncontiguous.py | 多个算子 | |
| | logsigmoid_bwd_kernel | fla/modules/activations.py:230 | | | | | |
| logsumexp | logsumexp_fwd_kernel | fla/ops/utils/logsumexp.py:27 | utils | 对数求和指数工具函数 | 无测试用例 | 无测试用例 | |
| matmul | matmul_kernel | fla/ops/utils/matmul.py:52 | utils | 激活函数工具 | 无测试用例 | 无测试用例 | |
| | leaky_relu | fla/ops/utils/matmul.py:157 | | | | | |
| | sigmoid | fla/ops/utils/matmul.py:162 | | | | | |
| | tanh | fla/ops/utils/matmul.py:168 | | | | | |
| | relu | fla/ops/utils/matmul.py:175 | | | | | |
| norm_gate | layer_norm_gated_fwd_kernel | fla/modules/fused_norm_gate.py:34 | fused_norm_gate | 融合门控LayerNorm | 无测试用例 | 无测试用例 | |
| | layer_norm_gated_fwd_kernel1 | fla/modules/fused_norm_gate.py:120 | | | | | |
| | layer_norm_gated_bwd_kernel | fla/modules/fused_norm_gate.py:201 | | | | | |
| | layer_norm_gated_bwd_kernel1 | fla/modules/fused_norm_gate.py:329 | | | | | |
| op | exp | fla/ops/utils/op.py:17 | utils | 张量描述符工具 | 无测试用例 | 无测试用例 | |
| | exp2 | fla/ops/utils/op.py:19 | | | | | |
| | log | fla/ops/utils/op.py:21 | | | | | |
| | log2 | fla/ops/utils/op.py:23 | | | | | |
| | tanh | fla/ops/utils/op.py:25 | | | | | |
| | exp | fla/ops/utils/op.py:28 | | | | | |
| | exp2 | fla/ops/utils/op.py:30 | | | | | |
| | log | fla/ops/utils/op.py:32 | | | | | |
| | log2 | fla/ops/utils/op.py:34 | | | | | |
| | tanh | fla/ops/utils/op.py:36 | | | | | |
| | gather | fla/ops/utils/op.py:41 | | | | | |
| | make_tensor_descriptor | fla/ops/utils/op.py:65 | | | | | |
| pack | packunpack_sequence_kernel | fla/ops/utils/pack.py:27 | utils | 序列打包解包工具 | tests\ops\test_utils.py | 单个算子 | |
| pooling | mean_pooling_fwd_kernel | fla/ops/utils/pooling.py:28 | utils | 平均池化工具 | tests\ops\test_utils.py | 多个算子 | |
| | mean_pooling_bwd_kernel | fla/ops/utils/pooling.py:75 | | | | | |
| rotary | rotary_embedding_kernel | fla/modules/rotary.py:46 | rotary | 旋转位置编码 | tests\modules\test_rotary.py | 单个算子 | |
| sigmoid | sigmoid_fwd_kernel | fla/modules/activations.py:85 | activations | Sigmoid激活函数 | tests\modules\test_activations_noncontiguous.py | 多个算子 | |
| | sigmoid_bwd_kernel | fla/modules/activations.py:115 | | | | | |
| softmax | softmax_fwd_kernel | fla/ops/utils/softmax.py:26 | utils | Softmax工具 | 无测试用例 | 无测试用例 | |
| | softmax_bwd_kernel | fla/ops/utils/softmax.py:53 | | | | | |
| softplus | softplus_nv | fla/ops/utils/softplus.py:68 | utils | Softplus2工具 | 无测试用例 | 无测试用例 | |
| | softplus_triton | fla/ops/utils/softplus.py:84 | | | | | |
| | softplus2_nv | fla/ops/utils/softplus.py:89 | | | | | |
| | softplus2_triton | fla/ops/utils/softplus.py:105 | | | | | |
| solve_tril | solve_tril_16x16_kernel | fla/ops/utils/solve_tril.py:36 | utils | 下三角矩阵逆求解工具 | tests\ops\test_solve_tril.py | 多个算子 | |
| | merge_16x16_to_32x32_inverse_kernel | fla/ops/utils/solve_tril.py:104 | | | | | |
| | merge_16x16_to_64x64_inverse_kernel | fla/ops/utils/solve_tril.py:193 | | | | | |
| swiglu | swiglu_fwd_kernel | fla/modules/activations.py:552 | activations | SwiGLU激活函数 | tests\modules\test_activation.py|tests\modules\test_activations_noncontiguous.py | 多个算子 | |
| | swiglu_fwdbwd_kernel | fla/modules/activations.py:589 | | | | | |
| swish | swish_fwd_kernel | fla/modules/activations.py:324 | activations | Swish/SiLU激活函数 | tests\modules\test_activations_noncontiguous.py | 多个算子 | |
| | swish_bwd_kernel | fla/modules/activations.py:355 | | | | | |
| token_shift | token_shift_fwd_kernel_short | fla/modules/token_shift.py:67 | token_shift | Token移位 | tests\modules\test_token_shift.py | 多个算子 | |
| | token_shift_fwd_kernel_long | fla/modules/token_shift.py:159 | | | | | |
| | token_shift_bwd_kernel_short | fla/modules/token_shift.py:233 | | | | | |
| | token_shift_bwd_kernel_long | fla/modules/token_shift.py:311 | | | | | |

## 特殊测试场景分析

### 多kernel合并测试

单个算子的多个kernel共享同一测试用例，测试该算子的完整功能。

### 多算子多kernel综合测试

测试用例覆盖多个算子的多个kernel，验证相关功能的正确性。主要原因：
1. 共享kernel机制：Common算子提供的kernel被多个算子共享使用
2. 相似架构：GatedDeltaRule、KDA、GLA等算子具有相似的架构设计
3. 代码复用：通过共享kernel减少代码重复，提高维护效率

## 关键发现

1. **算子级测试为主**：大多数测试用例针对整个算子的功能进行测试，而非单独测试每个kernel
2. **共享kernel间接测试**：共享kernel通过使用它们的算子的测试用例间接验证
3. **变体测试丰富**：同一算子通常有多个测试变体（varlen、gqa、transpose_state等）
4. **部分算子缺少测试**：ABC、RWKV4、Rebased、bitlinear、norm_gate等算子暂未实现测试用例

---

*本报告由自动化脚本生成*