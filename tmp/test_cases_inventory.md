# 测试用例清单

## 统计摘要
- **测试用例总数**: 204个
- **测试文件总数**: 48个
- **单元测试**: 173个 (84.8%)
- **集成测试**: 31个 (15.2%)
- **算子测试**: 153个 (75.0%)
- **模块测试**: 49个 (24.0%)
- **层测试**: 2个 (1.0%)

---

## 测试用例详细清单

### 1. 算子测试 (tests/ops/) - 116个测试用例

| 测试文件 | 测试函数 | 测试类型 | 测试对象 | 对象类型 | 备注 |
|---------|---------|---------|---------|---------|------|
| tests/ops/test_attn.py | test_parallel | 单元测试 | parallel_attention_parallel | 算子 | |
| tests/ops/test_attn.py | test_parallel_varlen | 单元测试 | parallel_attention_parallel | 算子 | |
| tests/ops/test_based.py | test_based | 单元测试 | based | 算子 | |
| tests/ops/test_comba.py | test_cumsum_local_scalar_fwd | 单元测试 | comba | 算子 | |
| tests/ops/test_comba.py | test_fused_recurrent | 单元测试 | comba_fused_recurrent | 算子 | |
| tests/ops/test_comba.py | test_chunk | 单元测试 | comba_chunk | 算子 | |
| tests/ops/test_comba.py | test_chunk_varlen | 单元测试 | comba_chunk | 算子 | |
| tests/ops/test_delta.py | test_chunk | 单元测试 | delta_rule_chunk | 算子 | |
| tests/ops/test_delta.py | test_chunk_varlen | 单元测试 | delta_rule_chunk | 算子 | |
| tests/ops/test_deltaformer.py | test_deltaformer_attn | 单元测试 | deltaformer | 算子 | Test DeltaFormer pre-attention by comparing fused implementation with naive reference. |
| tests/ops/test_deltaformer.py | test_deltaformer_attn_varlen | 单元测试 | deltaformer_varlen | 算子 | |
| tests/ops/test_delta_product.py | test_chunk | 单元测试 | delta_product_chunk | 算子 | |
| tests/ops/test_delta_product.py | test_chunk_varlen | 单元测试 | delta_product_chunk | 算子 | |
| tests/ops/test_dplr_delta.py | test_recurrent_fwd | 单元测试 | dplr_delta | 算子 | |
| tests/ops/test_dplr_delta.py | test_fused_recurrent | 单元测试 | dplr_delta_fused_recurrent | 算子 | |
| tests/ops/test_dplr_delta.py | test_chunk | 单元测试 | dplr_delta_chunk | 算子 | |
| tests/ops/test_dplr_delta.py | test_chunk_varlen | 单元测试 | dplr_delta_chunk | 算子 | |
| tests/ops/test_forgetting_attn.py | test_parallel | 单元测试 | forgetting_attention_parallel | 算子 | |
| tests/ops/test_forgetting_attn.py | test_parallel_varlen | 单元测试 | forgetting_attention_parallel | 算子 | |
| tests/ops/test_gated_delta.py | test_fused_recurrent | 单元测试 | gated_delta_rule_fused_recurrent | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk | 单元测试 | gated_delta_rule_chunk | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk_gqa | 单元测试 | gated_delta_rule_chunk | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk_transpose_state | 单元测试 | gated_delta_rule_chunk | 算子 | |
| tests/ops/test_gated_delta.py | test_fused_recurrent_transpose_state | 单元测试 | gated_delta_rule_fused_recurrent | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk_varlen | 单元测试 | gated_delta_rule_chunk | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk_varlen_prefill | 单元测试 | gated_delta_rule_chunk | 算子 | |
| tests/ops/test_gated_delta.py | test_chunk_gate_in_kernel | 单元测试 | gated_delta_rule_chunk | 算子 | Test use_gate_in_kernel=True path: fused gate activation + chunk cumsum inside kernel. |
| tests/ops/test_gated_delta.py | test_chunk_gate_in_kernel_gqa | 单元测试 | gated_delta_rule_chunk | 算子 | Test use_gate_in_kernel=True with grouped value attention (HV > H). |
| tests/ops/test_gated_delta.py | test_chunk_gate_in_kernel_varlen | 单元测试 | gated_delta_rule_chunk | 算子 | Test use_gate_in_kernel=True with variable-length sequences. |
| tests/ops/test_gated_delta.py | test_gate | 单元测试 | gated_delta_rule_gate | 算子 | |
| tests/ops/test_gated_delta_product.py | test_chunk | 单元测试 | gated_delta_product_chunk | 算子 | |
| tests/ops/test_gated_delta_product.py | test_chunk_varlen | 单元测试 | gated_delta_product_chunk | 算子 | |
| tests/ops/test_gla.py | test_fused_recurrent | 单元测试 | gla_fused_recurrent | 算子 | |
| tests/ops/test_gla.py | test_fused_recurrent_varlen | 单元测试 | gla_fused_recurrent | 算子 | |
| tests/ops/test_gla.py | test_chunk | 单元测试 | gla_chunk | 算子 | |
| tests/ops/test_gla.py | test_chunk_varlen | 单元测试 | gla_chunk | 算子 | |
| tests/ops/test_gsa.py | test_fused_recurrent | 单元测试 | gsa_fused_recurrent | 算子 | |
| tests/ops/test_gsa.py | test_fused_recurrent_varlen | 单元测试 | gsa_fused_recurrent | 算子 | |
| tests/ops/test_gsa.py | test_chunk | 单元测试 | gsa_chunk | 算子 | |
| tests/ops/test_gsa.py | test_chunk_varlen | 单元测试 | gsa_chunk | 算子 | |
| tests/ops/test_gsa.py | test_inference | 单元测试 | gsa_inference | 算子 | |
| tests/ops/test_hgrn.py | test_fused_recurrent | 单元测试 | hgrn_fused_recurrent | 算子 | |
| tests/ops/test_hgrn.py | test_fused_recurrent_varlen | 单元测试 | hgrn_fused_recurrent | 算子 | |
| tests/ops/test_hgrn.py | test_chunk | 单元测试 | hgrn_chunk | 算子 | |
| tests/ops/test_index.py | test_prepare_ids_correctness | 单元测试 | index | 算子 | |
| tests/ops/test_index.py | test_chunk_utils_correctness | 单元测试 | index_chunk | 算子 | |
| tests/ops/test_index.py | test_split_cu_seqlens_correctness | 单元测试 | index | 算子 | |
| tests/ops/test_index.py | test_edge_cases | 单元测试 | index | 算子 | |
| tests/ops/test_intracard_cache.py | test_chunk_kda_intracard_cache_hit_same_cu_seqlens_object | 单元测试 | intracard_cache_chunk | 算子 | E2E: chunk_kda should reuse intracard precompute cache on second call. |
| tests/ops/test_intracard_cache.py | test_intracard_backend_disabled_by_default | 单元测试 | intracard_cache_intracard | 算子 | Verify that IntraCardCPBackend is disabled by default. |
| tests/ops/test_intracard_cache.py | test_intracard_backend_disabled_when_env_var_is_zero | 单元测试 | intracard_cache_intracard | 算子 | Verify that IntraCardCPBackend is disabled when FLA_INTRACARD_CP=0. |
| tests/ops/test_intracard_cache.py | test_intracard_backend_enabled_when_env_var_is_one | 单元测试 | intracard_cache_intracard | 算子 | Verify that IntraCardCPBackend is enabled when FLA_INTRACARD_CP=1. |
| tests/ops/test_intracard_cache.py | test_chunk_gdn_intracard_gqa | 单元测试 | intracard_cache_chunk | 算子 | E2E: chunk_gated_delta_rule intracard path produces correct results with GQA. |
| tests/ops/test_iplr_delta.py | test_fused_recurrent | 单元测试 | iplr_delta_fused_recurrent | 算子 | |
| tests/ops/test_iplr_delta.py | test_chunk | 单元测试 | iplr_delta_chunk | 算子 | |
| tests/ops/test_kda.py | test_naive_chunk | 单元测试 | kda_chunk | 算子 | |
| tests/ops/test_kda.py | test_fused_recurrent | 单元测试 | kda_fused_recurrent | 算子 | |
| tests/ops/test_kda.py | test_fused_recurrent_transpose_state | 单元测试 | kda_fused_recurrent | 算子 | |
| tests/ops/test_kda.py | test_fused_recurrent_vllm_decode | 单元测试 | kda_fused_recurrent | 算子 | Test vLLM-style decoding with continuous batching and paged state storage. |
| tests/ops/test_kda.py | test_chunk | 单元测试 | kda_chunk | 算子 | |
| tests/ops/test_kda.py | test_chunk_transpose_state | 单元测试 | kda_chunk | 算子 | |
| tests/ops/test_kda.py | test_chunk_varlen | 单元测试 | kda_chunk | 算子 | |
| tests/ops/test_kda.py | test_chunk_varlen_prefill | 单元测试 | kda_chunk | 算子 | |
| tests/ops/test_kda.py | test_gate | 单元测试 | kda_gate | 算子 | |
| tests/ops/test_kda.py | test_chunk_return_intermediate_states | 单元测试 | kda_chunk | 算子 | Test that return_intermediate_states=True works in inference mode. |
| tests/ops/test_linear_attn.py | test_fused_recurrent | 单元测试 | linear_attention_fused_recurrent | 算子 | |
| tests/ops/test_linear_attn.py | test_chunk | 单元测试 | linear_attention_chunk | 算子 | |
| tests/ops/test_linear_attn.py | test_fused_chunk | 单元测试 | linear_attention_chunk | 算子 | |
| tests/ops/test_log_linear_attn.py | test_chunk | 单元测试 | log_linear_attention_chunk | 算子 | |
| tests/ops/test_log_linear_attn.py | test_chunk_bwd | 单元测试 | log_linear_attention_chunk | 算子 | |
| tests/ops/test_log_linear_attn.py | test_chunk_varlen | 单元测试 | log_linear_attention_chunk | 算子 | |
| tests/ops/test_mesa.py | test_chunk | 单元测试 | mesa_chunk | 算子 | |
| tests/ops/test_mesa.py | test_chunk_varlen | 单元测试 | mesa_chunk | 算子 | |
| tests/ops/test_mesa.py | test_decoding_one_step | 单元测试 | mesa_decoding | 算子 | |
| tests/ops/test_nsa.py | test_parallel | 单元测试 | nsa_parallel | 算子 | |
| tests/ops/test_nsa.py | test_parallel_varlen | 单元测试 | nsa_parallel | 算子 | |
| tests/ops/test_oja.py | test_naive_chunk_oja | 单元测试 | oja_chunk | 算子 | |
| tests/ops/test_oja.py | test_fused_recurrent | 单元测试 | oja_fused_recurrent | 算子 | |
| tests/ops/test_oja.py | test_chunk_forward | 单元测试 | oja_chunk | 算子 | |
| tests/ops/test_oja.py | test_chunk | 单元测试 | oja_chunk | 算子 | |
| tests/ops/test_oja.py | test_chunk_varlen | 单元测试 | oja_chunk | 算子 | |
| tests/ops/test_path_attn.py | test_parallel | 单元测试 | path_attention_parallel | 算子 | |
| tests/ops/test_path_attn.py | test_parallel_varlen | 单元测试 | path_attention_parallel | 算子 | |
| tests/ops/test_retention.py | test_chunk | 单元测试 | retention_chunk | 算子 | |
| tests/ops/test_retention.py | test_chunk_varlen | 单元测试 | retention_chunk | 算子 | |
| tests/ops/test_retention.py | test_fused_chunk | 单元测试 | retention_chunk | 算子 | |
| tests/ops/test_retention.py | test_fused_chunk_varlen | 单元测试 | retention_chunk | 算子 | |
| tests/ops/test_retention.py | test_parallel | 单元测试 | retention_parallel | 算子 | |
| tests/ops/test_rwkv6.py | test_chunk | 单元测试 | rwkv6_chunk | 算子 | |
| tests/ops/test_rwkv6.py | test_chunk_varlen | 单元测试 | rwkv6_chunk | 算子 | |
| tests/ops/test_rwkv7.py | test_channel_mixing_gradients | 单元测试 | rwkv7 | 算子 | |
| tests/ops/test_rwkv7.py | test_fused_mul_recurrent_fwd | 单元测试 | rwkv7 | 算子 | |
| tests/ops/test_rwkv7.py | test_fused_rwkv7_addcmul | 单元测试 | rwkv7 | 算子 | |
| tests/ops/test_rwkv7.py | test_fused_k_update | 单元测试 | rwkv7 | 算子 | |
| tests/ops/test_rwkv7.py | test_gate_output_correction | 单元测试 | rwkv7_gate | 算子 | |
| tests/ops/test_simple_gla.py | test_fused_recurrent | 单元测试 | simple_gla_fused_recurrent | 算子 | |
| tests/ops/test_simple_gla.py | test_fused_recurrent_varlen | 单元测试 | simple_gla_fused_recurrent | 算子 | |
| tests/ops/test_simple_gla.py | test_chunk | 单元测试 | simple_gla_chunk | 算子 | |
| tests/ops/test_simple_gla.py | test_chunk_varlen | 单元测试 | simple_gla_chunk | 算子 | |
| tests/ops/test_simple_gla.py | test_fused_chunk | 单元测试 | simple_gla_chunk | 算子 | |
| tests/ops/test_simple_gla.py | test_fused_chunk_varlen | 单元测试 | simple_gla_chunk | 算子 | |
| tests/ops/test_simple_gla.py | test_parallel | 单元测试 | simple_gla_parallel | 算子 | |
| tests/ops/test_simple_gla.py | test_parallel_varlen | 单元测试 | simple_gla_parallel | 算子 | |
| tests/ops/test_simple_gla.py | test_simple_gla_to_mamba2 | 单元测试 | simple_gla | 算子 | |
| tests/ops/test_solve_tril.py | test_solve_tril | 单元测试 | solve_tril | 算子 | |
| tests/ops/test_solve_tril.py | test_solve_tril_varlen | 单元测试 | solve_tril_varlen | 算子 | |
| tests/ops/test_titans.py | test_naive_chunk | 单元测试 | titans_chunk | 算子 | |
| tests/ops/test_ttt.py | test_chunk | 单元测试 | ttt_chunk | 算子 | |
| tests/ops/test_ttt.py | test_fused_chunk | 单元测试 | ttt_chunk | 算子 | |
| tests/ops/test_ttt.py | test_chunk_varlen | 单元测试 | ttt_chunk | 算子 | |
| tests/ops/test_utils.py | test_global_cumsum | 单元测试 | utils | 算子 | |
| tests/ops/test_utils.py | test_global_cumsum_varlen | 单元测试 | utils_varlen | 算子 | |
| tests/ops/test_utils.py | test_global_reversed_cumsum | 单元测试 | utils | 算子 | |
| tests/ops/test_utils.py | test_global_reversed_cumsum_varlen | 单元测试 | utils_varlen | 算子 | |
| tests/ops/test_utils.py | test_local_cumsum | 单元测试 | utils | 算子 | |
| tests/ops/test_utils.py | test_local_cumsum_varlen | 单元测试 | utils_varlen | 算子 | |
| tests/ops/test_utils.py | test_mean_pooling | 单元测试 | utils | 算子 | |
| tests/ops/test_utils.py | test_mean_pooling_varlen | 单元测试 | utils_varlen | 算子 | |
| tests/ops/test_utils.py | test_pack_sequence | 单元测试 | utils | 算子 | |
| tests/ops/test_utils.py | test_unpack_sequence | 单元测试 | utils | 算子 | |

---

### 2. 模块测试 (tests/modules/) - 49个测试用例

| 测试文件 | 测试函数 | 测试类型 | 测试对象 | 对象类型 | 备注 |
|---------|---------|---------|---------|---------|------|
| tests/modules/test_activation.py | test_sigmoid | 单元测试 | activation | 模块 | |
| tests/modules/test_activation.py | test_logsigmoid | 单元测试 | activation | 模块 | |
| tests/modules/test_activation.py | test_swish | 单元测试 | activation | 模块 | |
| tests/modules/test_activation.py | test_swiglu | 单元测试 | activation | 模块 | |
| tests/modules/test_activation.py | test_swiglu_linear | 单元测试 | activation | 模块 | |
| tests/modules/test_activations_noncontiguous.py | test_sigmoid | 单元测试 | activation | 模块 | |
| tests/modules/test_activations_noncontiguous.py | test_logsigmoid | 单元测试 | activation | 模块 | |
| tests/modules/test_activations_noncontiguous.py | test_swish | 单元测试 | activation | 模块 | |
| tests/modules/test_activations_noncontiguous.py | test_swiglu | 单元测试 | activation | 模块 | |
| tests/modules/test_activations_noncontiguous.py | test_swiglu_contiguous | 单元测试 | activation | 模块 | Test that contiguous inputs still work correctly. |
| tests/modules/test_activations_noncontiguous.py | test_is_inner_contiguous | 单元测试 | activation | 模块 | Test _is_inner_contiguous correctly classifies tensor layouts. |
| tests/modules/test_conv.py | test_conv | 单元测试 | convolution | 模块 | |
| tests/modules/test_conv.py | test_conv_varlen | 单元测试 | convolution_varlen | 模块 | |
| tests/modules/test_conv.py | test_conv_decoding | 单元测试 | convolution_decoding | 模块 | |
| tests/modules/test_conv.py | test_conv_with_cache_prefill_fwd | 单元测试 | convolution_prefill | 模块 | |
| tests/modules/test_conv.py | test_conv_varlen_with_cache_prefill_fwd | 单元测试 | convolution_varlen | 模块 | |
| tests/modules/test_conv.py | test_conv_decoding_with_cache | 单元测试 | convolution_decoding | 模块 | |
| tests/modules/test_conv.py | test_conv_varlen_decoding | 单元测试 | convolution_varlen | 模块 | Test varlen mode decoding with causal_conv1d_update. |
| tests/modules/test_conv.py | test_conv_decoding_non_contiguous_x | 单元测试 | convolution_decoding | 模块 | Test decoding with non-contiguous input x. |
| tests/modules/test_conv.py | test_fast_conv_varlen | 单元测试 | convolution_varlen | 模块 | |
| tests/modules/test_conv.py | test_conv_cache_backward | 单元测试 | convolution_backward | 模块 | |
| tests/modules/test_conv.py | test_conv_varlen_initial_state_backward_random | 单元测试 | convolution_varlen | 模块 | |
| tests/modules/test_conv.py | test_conv_cache_backward_no_final_state | 单元测试 | convolution_backward | 模块 | Test backward with initial_state but WITHOUT output_final_state. |
| tests/modules/test_conv.py | test_conv_non_contiguous_qkv | 单元测试 | convolution | 模块 | Test non-contiguous input from QKV concatenated tensor. |
| tests/modules/test_conv.py | test_conv_varlen_non_contiguous_qkv | 单元测试 | convolution_varlen | 模块 | Test non-contiguous input from QKV concatenated tensor. |
| tests/modules/test_cross_entropy.py | test_fused_cross_entropy | 单元测试 | cross_entropy | 模块 | |
| tests/modules/test_cross_entropy.py | test_fused_cross_entropy_softcap | 单元测试 | cross_entropy | 模块 | |
| tests/modules/test_cross_entropy.py | test_fused_linear_cross_entropy | 单元测试 | cross_entropy | 模块 | |
| tests/modules/test_cross_entropy.py | test_fused_linear_cross_entropy_softcap | 单元测试 | cross_entropy | 模块 | |
| tests/modules/test_grpo.py | test_fused_grpos | 单元测试 | grpo | 模块 | |
| tests/modules/test_kl_div.py | test_fused | 单元测试 | kl_div | 模块 | |
| tests/modules/test_l2norm.py | test_l2norm | 单元测试 | l2norm | 模块 | |
| tests/modules/test_l2warp.py | test_fused_linear_cross_entropy_l2_warp | 单元测试 | l2_warp | 模块 | |
| tests/modules/test_layernorm.py | test_layernorm | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_groupnorm | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_rmsnorm | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_layernorm_linear | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_groupnorm_linear | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_rmsnorm_linear | 单元测试 | layer_norm | 模块 | |
| tests/modules/test_layernorm.py | test_rmsnorm_small_t | 单元测试 | layer_norm | 模块 | RMSNorm backward must handle T < SM_count without illegal memory access. |
| tests/modules/test_layernorm.py | test_layernorm_small_t | 单元测试 | layer_norm | 模块 | LayerNorm backward must handle T < SM_count without illegal memory access. |
| tests/modules/test_layernorm.py | test_groupnorm_small_t | 单元测试 | layer_norm | 模块 | GroupNorm backward must handle T < SM_count without illegal memory access. |
| tests/modules/test_layernorm_gated.py | test_layernorm_gated | 单元测试 | layer_norm_gated_gate | 模块 | |
| tests/modules/test_layernorm_gated.py | test_rmsnorm_gated | 单元测试 | layer_norm_gated_gate | 模块 | |
| tests/modules/test_rotary.py | test_rotary | 单元测试 | rotary | 模块 | |
| tests/modules/test_rotary.py | test_rotary_with_offsets | 单元测试 | rotary | 模块 | |
| tests/modules/test_rotary.py | test_rotary_varlen | 单元测试 | rotary_varlen | 模块 | |
| tests/modules/test_token_shift.py | test_token_shift | 单元测试 | token_shift | 模块 | |
| tests/modules/test_token_shift.py | test_all_with_and_without_varlen | 单元测试 | token_shift_varlen | 模块 | |

---

### 3. 上下文并行测试 (tests/context_parallel/) - 37个测试用例

| 测试文件 | 测试函数 | 测试类型 | 测试对象 | 对象类型 | 备注 |
|---------|---------|---------|---------|---------|------|
| tests/context_parallel/test_cp_bwd_gk_offset.py | test_cp2_kda_backward | 单元测试 | context_parallel_backward | 算子 | End-to-end test: CP2 KDA backward matches non-CP reference. |
| tests/context_parallel/test_cp_bwd_gk_offset.py | test_stage1_gk_per_head_sensitivity | 单元测试 | context_parallel | 算子 | Verify stage 1 (dh computation) is sensitive to per-head gk values. |
| tests/context_parallel/test_cp_conv.py | test_cp2_sequence_cut | 集成测试 | context_parallel_convolution | 算子 | CP2 with sequences cut in the middle. |
| tests/context_parallel/test_cp_conv.py | test_cp2_boundary_aligned | 集成测试 | context_parallel_convolution | 算子 | CP2 with sequence boundaries aligned with rank boundaries. |
| tests/context_parallel/test_cp_conv.py | test_cp4_complex | 集成测试 | context_parallel_convolution | 算子 | CP4 with complex sequence distribution. |
| tests/context_parallel/test_cp_conv.py | test_cp4_single_sequence | 集成测试 | context_parallel_convolution | 算子 | CP4 with a single long sequence spanning all ranks. |
| tests/context_parallel/test_cp_conv.py | test_cp2_many_short_sequences | 集成测试 | context_parallel_convolution | 算子 | CP2 with many short sequences. |
| tests/context_parallel/test_cp_conv.py | test_cp2_short_tail_len1 | 集成测试 | context_parallel_convolution | 算子 | CP2: seq0 has 513 tokens, rank 1 gets length-1 tail. |
| tests/context_parallel/test_cp_conv.py | test_cp2_short_tail_len2 | 集成测试 | context_parallel_convolution | 算子 | CP2: seq0 has 514 tokens, rank 1 gets length-2 tail. |
| tests/context_parallel/test_cp_conv.py | test_cp4_every_rank_gets_short_tail | 集成测试 | context_parallel_convolution | 算子 | CP4: every non-first rank gets length-1 local sequence tail. |
| tests/context_parallel/test_cp_conv.py | test_cp2_multiple_short_tails | 集成测试 | context_parallel_convolution | 算子 | CP2: multiple sequences each end 1 token into rank 1. |
| tests/context_parallel/test_cp_conv.py | test_cp2_global_len1_sequence | 集成测试 | context_parallel_convolution | 算子 | CP2: globally length-1 sequence at rank boundary. |
| tests/context_parallel/test_cp_conv.py | test_cp4_multiple_short_tails | 集成测试 | context_parallel_convolution | 算子 | CP4: multiple sequences with short tails across ranks. |
| tests/context_parallel/test_cp_conv.py | test_cp4_worst_case_many_len1 | 集成测试 | context_parallel_convolution | 算子 | CP4: globally length-1 sequences + short tails. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_sequence_cut | 集成测试 | context_parallel_gdn | 算子 | CP2: sequences cut across rank boundary. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_boundary_aligned | 集成测试 | context_parallel_gdn | 算子 | CP2: sequence boundaries aligned with rank boundaries. |
| tests/context_parallel/test_cp_gdn.py | test_cp4_complex | 集成测试 | context_parallel_gdn | 算子 | CP4: complex sequence distribution. |
| tests/context_parallel/test_cp_gdn.py | test_cp4_single_sequence | 集成测试 | context_parallel_gdn | 算子 | CP4: single long sequence spanning all ranks. |
| tests/context_parallel/test_cp_gdn.py | test_cp8_single_sequence | 集成测试 | context_parallel_gdn | 算子 | CP8: single long sequence spanning all ranks. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_many_short_sequences | 集成测试 | context_parallel_gdn | 算子 | CP2: many short sequences. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_gqa_sequence_cut | 集成测试 | context_parallel_gdn_gqa | 算子 | CP2 GQA: sequences cut across rank boundary, Hq < H. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_gqa_single_sequence | 集成测试 | context_parallel_gdn_gqa | 算子 | CP2 GQA: single long sequence with Hq < H. |
| tests/context_parallel/test_cp_gdn.py | test_cp2_transpose_state | 集成测试 | context_parallel_gdn_transpose_state | 算子 | CP2: transpose_state_layout=True with sequence cut. |
| tests/context_parallel/test_cp_gdn.py | test_cp4_transpose_state | 集成测试 | context_parallel_gdn_transpose_state | 算子 | CP4: transpose_state_layout=True with single sequence. |
| tests/context_parallel/test_cp_kda.py | test_cp2_sequence_cut | 集成测试 | context_parallel_kda | 算子 | CP2: sequences cut across rank boundary. |
| tests/context_parallel/test_cp_kda.py | test_cp2_boundary_aligned | 集成测试 | context_parallel_kda | 算子 | CP2: sequence boundaries aligned with rank boundaries. |
| tests/context_parallel/test_cp_kda.py | test_cp4_complex | 集成测试 | context_parallel_kda | 算子 | CP4: complex sequence distribution. |
| tests/context_parallel/test_cp_kda.py | test_cp4_single_sequence | 集成测试 | context_parallel_kda | 算子 | CP4: single long sequence spanning all ranks. |
| tests/context_parallel/test_cp_kda.py | test_cp8_single_sequence | 集成测试 | context_parallel_kda | 算子 | CP8: single long sequence spanning all ranks. |
| tests/context_parallel/test_cp_kda.py | test_cp2_many_short_sequences | 集成测试 | context_parallel_kda | 算子 | CP2: many short sequences. |
| tests/context_parallel/test_cp_kda.py | test_cp2_disable_recompute | 集成测试 | context_parallel_kda | 算子 | CP2: disable_recompute=True. |
| tests/context_parallel/test_cp_kda.py | test_cp2_transpose_state | 集成测试 | context_parallel_kda_transpose_state | 算子 | CP2: transpose_state_layout=True with sequence cut. |
| tests/context_parallel/test_cp_kda.py | test_cp4_transpose_state | 集成测试 | context_parallel_kda_transpose_state | 算子 | CP4: transpose_state_layout=True with single sequence. |

---

### 4. 层测试 (tests/layers/) - 2个测试用例

| 测试文件 | 测试函数 | 测试类型 | 测试对象 | 对象类型 | 备注 |
|---------|---------|---------|---------|---------|------|
| tests/layers/test_layer_cache_layer_idx.py | test_cache_requires_layer_idx | 单元测试 | layer_cache_cache | 层 | |
| tests/layers/test_layer_cache_layer_idx.py | test_layers_without_cache_still_work_with_layer_idx_none | 单元测试 | layer_cache_cache | 层 | |

---

## 测试覆盖分析

### 按算子/模块分类统计

#### 注意力机制类算子
- **parallel_attention**: 2个测试用例
- **forgetting_attention**: 2个测试用例
- **path_attention**: 2个测试用例
- **nsa**: 2个测试用例

#### 线性注意力类算子
- **gla**: 4个测试用例
- **simple_gla**: 9个测试用例
- **linear_attention**: 3个测试用例
- **log_linear_attention**: 3个测试用例

#### Delta规则类算子
- **delta_rule**: 2个测试用例
- **delta_product**: 2个测试用例
- **gated_delta_rule**: 11个测试用例
- **gated_delta_product**: 2个测试用例
- **dplr_delta**: 4个测试用例
- **iplr_delta**: 2个测试用例

#### 其他算子
- **based**: 1个测试用例
- **comba**: 4个测试用例
- **deltaformer**: 2个测试用例
- **gsa**: 5个测试用例
- **hgrn**: 3个测试用例
- **kda**: 11个测试用例
- **mesa**: 3个测试用例
- **oja**: 5个测试用例
- **retention**: 5个测试用例
- **rwkv6**: 2个测试用例
- **rwkv7**: 5个测试用例
- **solve_tril**: 2个测试用例
- **titans**: 1个测试用例
- **ttt**: 3个测试用例

#### 模块类
- **activation**: 11个测试用例
- **convolution**: 16个测试用例
- **layer_norm**: 11个测试用例
- **layer_norm_gated**: 2个测试用例
- **cross_entropy**: 4个测试用例
- **rotary**: 3个测试用例
- **token_shift**: 2个测试用例

#### 上下文并行
- **context_parallel_convolution**: 13个测试用例
- **context_parallel_gdn**: 11个测试用例
- **context_parallel_kda**: 11个测试用例
- **context_parallel**: 2个测试用例

---

## 测试模式总结

### 常见测试模式
1. **chunk**: 分块实现测试 - 最常见
2. **fused_recurrent**: 融合递归实现测试
3. **parallel**: 并行实现测试
4. **varlen**: 变长序列测试
5. **decoding**: 解码模式测试
6. **prefill**: 预填充模式测试
7. **gqa**: 分组查询注意力测试
8. **transpose_state**: 转置状态布局测试
9. **cache**: 缓存相关测试
10. **gate**: 门控机制测试

### 特殊测试场景
1. **集成测试**: 主要集中在 context_parallel 目录，测试多GPU分布式场景
2. **边界情况测试**: short_tail、small_t 等极端情况
3. **非连续内存测试**: 测试非连续内存布局的输入
4. **IntraCard缓存测试**: 测试IntraCard缓存机制

---

## 备注
- CSV格式的原始数据保存在: `tmp/test_cases_inventory.csv`
- 本Markdown表格版本更易于在IDE中查看和分析
