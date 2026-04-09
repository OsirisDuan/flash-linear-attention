#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子任务3：匹配kernel函数与测试用例
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


def normalize_name(name: str) -> str:
    """标准化名称，用于匹配"""
    name = name.lower().strip()
    name = re.sub(r'[_\-\s]+', '', name)
    return name


def extract_operator_from_test_object(test_object: str) -> str:
    """从测试对象名称中提取算子名称
    
    例如：
    - parallel_attention_parallel -> attn
    - gla_fused_recurrent -> gla
    - based -> based
    - delta_rule_chunk -> delta_rule
    - gated_delta_rule_chunk -> gated_delta_rule
    """
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
        
        'utils': 'utils',
        'cumsum': 'cumsum',
        'index': 'index',
        'pooling': 'pooling',
        'pack': 'pack',
        
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
    }
    return mapping


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


def main():
    operators_kernels = load_csv('tmp/operators_kernels_mapping.csv')
    test_cases = load_csv('tmp/test_cases_inventory.csv')
    
    operator_to_kernels: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in operators_kernels:
        operator_name = row['算子名称']
        operator_to_kernels[operator_name].append(row)
    
    test_object_mapping = build_test_object_to_operator_mapping()
    
    operator_to_tests: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for test in test_cases:
        test_object = test['测试对象']
        operator_name = match_test_object_to_operator(test_object, test_object_mapping)
        operator_to_tests[operator_name].append(test)
    
    results = []
    
    kernel_test_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    
    for operator_name, kernels in operator_to_kernels.items():
        tests_for_operator = operator_to_tests.get(operator_name, [])
        
        for kernel in kernels:
            kernel_name = kernel['Kernel函数名称']
            
            if tests_for_operator:
                for test in tests_for_operator:
                    result = {
                        'Kernel函数名称': kernel_name,
                        '所属算子': operator_name,
                        '测试用例路径': test['测试文件路径'],
                        '测试函数名称': test['测试函数名称'],
                        '测试覆盖情况': '单个算子' if len(operator_to_kernels[operator_name]) == 1 else '多个算子',
                        '无测试用例原因': '',
                        '备注': f"Kernel源码: {kernel['Kernel源码路径']}:{kernel['Kernel行号']}"
                    }
                    results.append(result)
                    kernel_test_map[kernel_name].append(test)
            else:
                result = {
                    'Kernel函数名称': kernel_name,
                    '所属算子': operator_name,
                    '测试用例路径': '',
                    '测试函数名称': '',
                    '测试覆盖情况': '无测试用例',
                    '无测试用例原因': '暂未实现测试用例',
                    '备注': f"Kernel源码: {kernel['Kernel源码路径']}:{kernel['Kernel行号']}"
                }
                results.append(result)
    
    output_file = 'tmp/kernel_test_coverage.csv'
    fieldnames = ['Kernel函数名称', '所属算子', '测试用例路径', '测试函数名称', '测试覆盖情况', '无测试用例原因', '备注']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    total_kernels = len(operators_kernels)
    covered_kernels = sum(1 for k in kernel_test_map.keys() if kernel_test_map[k])
    uncovered_kernels = total_kernels - covered_kernels
    
    print(f"Kernel测试覆盖统计:")
    print(f"  - Kernel函数总数: {total_kernels}")
    print(f"  - 有测试覆盖的Kernel: {covered_kernels}")
    print(f"  - 无测试覆盖的Kernel: {uncovered_kernels}")
    print(f"  - 覆盖率: {covered_kernels/total_kernels*100:.1f}%")
    print(f"\n输出文件: {output_file}")
    
    uncovered_operators = set()
    for operator_name, kernels in operator_to_kernels.items():
        if operator_name not in operator_to_tests or not operator_to_tests[operator_name]:
            uncovered_operators.add(operator_name)
    
    if uncovered_operators:
        print(f"\n无测试用例的算子 ({len(uncovered_operators)}个):")
        for op in sorted(uncovered_operators):
            print(f"  - {op}")


if __name__ == '__main__':
    main()
