#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子任务3：匹配kernel函数与测试用例 (最终版)
输入：
  - tmp/operators_kernels_mapping.csv (算子与kernel函数映射表)
  - tmp/test_cases_inventory.csv (测试用例清单)
输出：
  - tmp/kernel_test_coverage.csv (kernel测试覆盖情况表)
"""

import csv
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_csv(filepath: str) -> List[Dict[str, str]]:
    """加载CSV文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_test_object_to_operator_mapping() -> Dict[str, str]:
    """构建测试对象名称到算子名称的映射"""
    mapping = {
        'parallel_attention': 'Attn',
        'attn': 'Attn',
        
        'based': 'Based',
        
        'comba': 'Comba',
        
        'delta_rule': 'DeltaRule',
        'delta': 'DeltaRule',
        
        'delta_product': 'DeltaProduct',
        
        'deltaformer': 'Deltaformer',
        
        'dplr_delta': 'DPLR',
        'dplr': 'DPLR',
        
        'iplr_delta': 'IPLR',
        'iplr': 'IPLR',
        
        'forgetting_attention': 'ForgettingAttn',
        'forgetting_attn': 'ForgettingAttn',
        
        'gated_delta_rule': 'GatedDeltaRule',
        'gated_delta': 'GatedDeltaRule',
        'gdn': 'GatedDeltaRule',
        
        'gated_delta_product': 'GatedDeltaProduct',
        
        'gla': 'GLA',
        
        'gsa': 'GSA',
        
        'hgrn': 'HGRN',
        
        'kda': 'KDA',
        
        'linear_attention': 'LinearAttn',
        'linear_attn': 'LinearAttn',
        
        'log_linear_attention': 'LogLinearAttn',
        'log_linear_attn': 'LogLinearAttn',
        
        'mesa': 'MesaNet',
        'mesa_net': 'MesaNet',
        
        'nsa': 'NSA',
        
        'oja': 'GatedOjaRule',
        'gated_oja_rule': 'GatedOjaRule',
        
        'path_attention': 'PathAttn',
        'path_attn': 'PathAttn',
        
        'retention': 'Retention',
        
        'rwkv6': 'RWKV6',
        
        'rwkv7': 'RWKV7',
        
        'simple_gla': 'SimpleGLA',
        
        'solve_tril': 'solve_tril',
        
        'titans': 'Titans',
        
        'ttt': 'TTT',
        
        'activation': 'activations',
        'sigmoid': 'sigmoid',
        'logsigmoid': 'logsigmoid',
        'swish': 'swish',
        'swiglu': 'swiglu',
        
        'convolution': 'causal_conv1d',
        'conv': 'causal_conv1d',
        
        'cross_entropy': 'cross_entropy',
        
        'kl_div': 'kl_div',
        
        'l2norm': 'l2norm',
        
        'l2_warp': 'linear_cross_entropy',
        
        'layer_norm': 'layernorm',
        'layernorm': 'layernorm',
        'rmsnorm': 'layernorm',
        'groupnorm': 'layernorm',
        
        'layer_norm_gated': 'layernorm_gated',
        'layernorm_gated': 'layernorm_gated',
        
        'rotary': 'rotary',
        
        'token_shift': 'token_shift',
        
        'grpo': 'grpo',
        
        'context_parallel': 'CP',
        'context_parallel_convolution': 'CP',
        'context_parallel_gdn': 'CP',
        'context_parallel_kda': 'CP',
        'context_parallel_backward': 'CP',
        
        'intracard_cache': 'CP',
        
        'layer_cache': 'layer_cache',
        
        'abc': 'ABC',
        
        'utils': 'utils',
        'utils_varlen': 'utils',
        'cumsum': 'cumsum',
        'index': 'index',
        'pooling': 'pooling',
        'pack': 'pack',
    }
    return mapping


def extract_operator_from_test_object(test_object: str) -> str:
    """从测试对象名称中提取算子名称"""
    test_object = test_object.lower().strip()
    
    suffixes_to_remove = [
        '_parallel', '_chunk', '_fused_recurrent', '_fused_chunk',
        '_varlen', '_decoding', '_inference', '_gate', '_chunk_varlen',
        '_parallel_varlen', '_fused_recurrent_varlen', '_fused_chunk_varlen',
        '_transpose_state', '_gqa', '_prefill', '_backward', '_intracard'
    ]
    
    result = test_object
    for suffix in sorted(suffixes_to_remove, key=len, reverse=True):
        if result.endswith(suffix):
            result = result[:-len(suffix)]
    
    return result


def match_test_object_to_operator(test_object: str, mapping: Dict[str, str]) -> str:
    """将测试对象匹配到算子名称"""
    test_object_lower = test_object.lower()
    
    if test_object_lower in mapping:
        return mapping[test_object_lower]
    
    extracted = extract_operator_from_test_object(test_object)
    if extracted in mapping:
        return mapping[extracted]
    
    for key, value in mapping.items():
        if key in test_object_lower or test_object_lower in key:
            return value
    
    return test_object


def build_shared_kernel_mapping() -> Dict[str, List[str]]:
    """构建共享kernel到使用它的算子的映射
    
    Common算子的kernel被多个算子共享使用
    """
    shared_kernels = {
        'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': ['GatedDeltaRule', 'KDA'],
        'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64': ['GatedDeltaRule', 'KDA'],
        'chunk_fwd_kernel_h': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_bwd_kernel_dh': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_fwd_kernel_h_parallel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_fwd_kernel_h_reduction': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_bwd_kernel_dh_parallel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_bwd_kernel_dh_reduction': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_fwd_kernel_h_split': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_bwd_kernel_dh_split': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn'],
        'chunk_fwd_kernel_o': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_bwd_kernel_dqkwg': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_bwd_kernel_dv': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_bwd_kernel_dv_local': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'chunk_scaled_dot_kkt_fwd_kernel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'fused_chunk_fwd_kernel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'fused_chunk_bwd_kernel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA'],
        'fused_recurrent_fwd_kernel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA', 'GSA', 'HGRN'],
        'fused_recurrent_bwd_kernel': ['GLA', 'SimpleGLA', 'RWKV6', 'Retention', 'LinearAttn', 'GatedDeltaRule', 'KDA', 'GSA', 'HGRN'],
    }
    return shared_kernels


def build_kernel_to_test_mapping() -> Dict[str, List[str]]:
    """构建kernel函数到测试函数的映射
    
    基于测试文件分析，建立kernel与测试函数的直接映射
    """
    mapping = {
        'sigmoid_fwd_kernel': ['test_sigmoid'],
        'sigmoid_bwd_kernel': ['test_sigmoid'],
        'logsigmoid_fwd_kernel': ['test_logsigmoid'],
        'logsigmoid_bwd_kernel': ['test_logsigmoid'],
        'swish_fwd_kernel': ['test_swish'],
        'swish_bwd_kernel': ['test_swish'],
        'swiglu_fwd_kernel': ['test_swiglu', 'test_swiglu_linear'],
        'swiglu_fwdbwd_kernel': ['test_swiglu', 'test_swiglu_linear'],
        
        'causal_conv1d_fwd_kernel': ['test_conv', 'test_conv_varlen'],
        'causal_conv1d_bwd_kernel': ['test_conv', 'test_conv_varlen', 'test_conv_cache_backward'],
        'causal_conv1d_update_kernel': ['test_conv_decoding', 'test_conv_decoding_with_cache'],
        'compute_dh0_kernel': ['test_conv_cache_backward'],
        'causal_conv1d_states_fwd_kernel': ['test_conv_with_cache_prefill_fwd'],
        
        'cross_entropy_fwd_kernel': ['test_fused_cross_entropy', 'test_fused_cross_entropy_softcap'],
        'cross_entropy_bwd_kernel': ['test_fused_cross_entropy', 'test_fused_cross_entropy_softcap'],
        
        'kl_div_kernel': ['test_fused'],
        'elementwise_mul_kernel': ['test_fused'],
        
        'l2norm_fwd_kernel': ['test_l2norm'],
        'l2norm_bwd_kernel': ['test_l2norm'],
        'l2norm_fwd_kernel1': ['test_l2norm'],
        'l2norm_bwd_kernel1': ['test_l2norm'],
        
        'layer_norm_fwd_kernel': ['test_layernorm', 'test_groupnorm', 'test_rmsnorm', 
                                   'test_layernorm_linear', 'test_groupnorm_linear', 'test_rmsnorm_linear',
                                   'test_rmsnorm_small_t', 'test_layernorm_small_t', 'test_groupnorm_small_t'],
        'layer_norm_fwd_kernel1': ['test_layernorm', 'test_groupnorm', 'test_rmsnorm'],
        'layer_norm_bwd_kernel': ['test_layernorm', 'test_groupnorm', 'test_rmsnorm'],
        'layer_norm_bwd_kernel1': ['test_layernorm', 'test_groupnorm', 'test_rmsnorm'],
        
        'rotary_embedding_kernel': ['test_rotary', 'test_rotary_with_offsets', 'test_rotary_varlen'],
        
        'token_shift_fwd_kernel_short': ['test_token_shift', 'test_all_with_and_without_varlen'],
        'token_shift_fwd_kernel_long': ['test_token_shift', 'test_all_with_and_without_varlen'],
        'token_shift_bwd_kernel_short': ['test_token_shift', 'test_all_with_and_without_varlen'],
        'token_shift_bwd_kernel_long': ['test_token_shift', 'test_all_with_and_without_varlen'],
        
        'grpo_fwd_kernel': ['test_fused_grpos'],
        'grpo_bwd_kernel': ['test_fused_grpos'],
        
        'chunk_local_cumsum_scalar_kernel': ['test_local_cumsum', 'test_local_cumsum_varlen'],
        'chunk_local_cumsum_vector_kernel': ['test_local_cumsum', 'test_local_cumsum_varlen'],
        'chunk_global_cumsum_scalar_kernel': ['test_global_cumsum', 'test_global_cumsum_varlen', 
                                               'test_global_reversed_cumsum', 'test_global_reversed_cumsum_varlen'],
        'chunk_global_cumsum_vector_kernel': ['test_global_cumsum', 'test_global_cumsum_varlen',
                                               'test_global_reversed_cumsum', 'test_global_reversed_cumsum_varlen'],
        'prepare_position_ids_kernel': ['test_prepare_ids_correctness', 'test_split_cu_seqlens_correctness', 'test_edge_cases'],
        'mean_pooling_fwd_kernel': ['test_mean_pooling', 'test_mean_pooling_varlen'],
        'mean_pooling_bwd_kernel': ['test_mean_pooling', 'test_mean_pooling_varlen'],
        'packunpack_sequence_kernel': ['test_pack_sequence', 'test_unpack_sequence'],
        
        'solve_tril_16x16_kernel': ['test_solve_tril', 'test_solve_tril_varlen'],
        'merge_16x16_to_32x32_inverse_kernel': ['test_solve_tril', 'test_solve_tril_varlen'],
        'merge_16x16_to_64x64_inverse_kernel': ['test_solve_tril', 'test_solve_tril_varlen'],
    }
    return mapping


def main():
    operators_kernels = load_csv('tmp/operators_kernels_mapping.csv')
    test_cases = load_csv('tmp/test_cases_inventory.csv')
    
    operator_to_kernels: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    kernel_to_operator: Dict[str, str] = {}
    for row in operators_kernels:
        operator_name = row['算子名称']
        kernel_name = row['Kernel函数名称']
        operator_to_kernels[operator_name].append(row)
        kernel_to_operator[kernel_name] = operator_name
    
    test_object_mapping = build_test_object_to_operator_mapping()
    shared_kernel_mapping = build_shared_kernel_mapping()
    kernel_test_mapping = build_kernel_to_test_mapping()
    
    test_func_to_test: Dict[str, Dict[str, str]] = {}
    for test in test_cases:
        test_func_name = test['测试函数名称']
        test_func_to_test[test_func_name] = test
    
    operator_to_tests: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    kernel_to_tests: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    
    for test in test_cases:
        test_object = test['测试对象']
        test_func_name = test['测试函数名称']
        
        operator_name = match_test_object_to_operator(test_object, test_object_mapping)
        operator_to_tests[operator_name].append(test)
    
    for kernel_name, test_funcs in kernel_test_mapping.items():
        for test_func in test_funcs:
            if test_func in test_func_to_test:
                kernel_to_tests[kernel_name].append(test_func_to_test[test_func])
    
    results = []
    covered_kernels = set()
    
    for row in operators_kernels:
        kernel_name = row['Kernel函数名称']
        operator_name = row['算子名称']
        kernel_path = row['Kernel源码路径']
        kernel_line = row['Kernel行号']
        
        tests_for_kernel = []
        
        if kernel_name in kernel_to_tests:
            tests_for_kernel.extend(kernel_to_tests[kernel_name])
        
        if operator_name in operator_to_tests:
            tests_for_kernel.extend(operator_to_tests[operator_name])
        
        if kernel_name in shared_kernel_mapping:
            for shared_op in shared_kernel_mapping[kernel_name]:
                if shared_op in operator_to_tests:
                    tests_for_kernel.extend(operator_to_tests[shared_op])
        
        tests_for_kernel = list({(t['测试文件路径'], t['测试函数名称']): t for t in tests_for_kernel}.values())
        
        if tests_for_kernel:
            covered_kernels.add(kernel_name)
            for test in tests_for_kernel:
                coverage_type = '多个算子' if len(operator_to_kernels.get(operator_name, [])) > 1 else '单个算子'
                if operator_name == 'Common':
                    coverage_type = '多个算子(共享kernel)'
                
                result = {
                    'Kernel函数名称': kernel_name,
                    '所属算子': operator_name,
                    '测试用例路径': test['测试文件路径'],
                    '测试函数名称': test['测试函数名称'],
                    '测试覆盖情况': coverage_type,
                    '无测试用例原因': '',
                    '备注': f"Kernel源码: {kernel_path}:{kernel_line}"
                }
                results.append(result)
        else:
            reason = '暂未实现测试用例'
            if operator_name == 'Common':
                reason = '共享kernel，通过使用它的算子间接测试'
            elif operator_name in ['RWKV4', 'Rebased', 'bitlinear', 'norm_gate', 'ABC']:
                reason = '算子暂未实现测试用例'
            elif operator_name in ['op', 'matmul', 'logcumsumexp', 'logsumexp', 'softmax', 'softplus']:
                reason = '工具函数，通常被其他算子间接测试'
            
            result = {
                'Kernel函数名称': kernel_name,
                '所属算子': operator_name,
                '测试用例路径': '',
                '测试函数名称': '',
                '测试覆盖情况': '无测试用例',
                '无测试用例原因': reason,
                '备注': f"Kernel源码: {kernel_path}:{kernel_line}"
            }
            results.append(result)
    
    output_file = 'tmp/kernel_test_coverage.csv'
    fieldnames = ['Kernel函数名称', '所属算子', '测试用例路径', '测试函数名称', '测试覆盖情况', '无测试用例原因', '备注']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    total_kernels = len(operators_kernels)
    covered_count = len(covered_kernels)
    uncovered_count = total_kernels - covered_count
    
    print(f"Kernel测试覆盖统计:")
    print(f"  - Kernel函数总数: {total_kernels}")
    print(f"  - 有测试覆盖的Kernel: {covered_count}")
    print(f"  - 无测试覆盖的Kernel: {uncovered_count}")
    print(f"  - 覆盖率: {covered_count/total_kernels*100:.1f}%")
    print(f"\n输出文件: {output_file}")
    
    uncovered_by_operator = defaultdict(list)
    for row in operators_kernels:
        if row['Kernel函数名称'] not in covered_kernels:
            uncovered_by_operator[row['算子名称']].append(row['Kernel函数名称'])
    
    if uncovered_by_operator:
        print(f"\n无测试用例的Kernel按算子分组 ({sum(len(v) for v in uncovered_by_operator.values())}个kernel, {len(uncovered_by_operator)}个算子):")
        for op in sorted(uncovered_by_operator.keys()):
            print(f"  - {op}: {len(uncovered_by_operator[op])}个kernel")


if __name__ == '__main__':
    main()
