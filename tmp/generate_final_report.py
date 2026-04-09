#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子任务5：生成最终报告
输入：
  - tmp/operators_kernels_mapping.csv（子任务1输出）
  - tmp/test_cases_inventory.csv（子任务2输出）
  - tmp/kernel_test_coverage.csv（子任务3输出）
  - tmp/special_test_scenarios.csv（子任务4输出）
输出：
  - kernel_analysis_report.md（更新后的最终报告）
"""

import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set


def load_csv(filepath: str) -> List[Dict[str, str]]:
    """加载CSV文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_module_from_operator(operator_name: str) -> str:
    """根据算子名称推断所属模块"""
    ops_modules = {
        'ABC': 'ops/abc',
        'Attn': 'ops/attn',
        'Based': 'ops/based',
        'Comba': 'ops/comba',
        'Common': 'ops/common',
        'CP': 'ops/cp',
        'DeltaRule': 'ops/delta_rule',
        'DeltaProduct': 'ops/delta_product',
        'Deltaformer': 'ops/deltaformer',
        'DPLR': 'ops/generalized_delta_rule',
        'IPLR': 'ops/generalized_delta_rule',
        'ForgettingAttn': 'ops/forgetting_attn',
        'GatedDeltaRule': 'ops/gated_delta_rule',
        'GatedDeltaProduct': 'ops/gated_delta_product',
        'GatedOjaRule': 'ops/gated_oja_rule',
        'GLA': 'ops/gla',
        'GSA': 'ops/gsa',
        'HGRN': 'ops/hgrn',
        'KDA': 'ops/kda',
        'LinearAttn': 'ops/linear_attn',
        'LogLinearAttn': 'ops/log_linear_attn',
        'MesaNet': 'ops/mesa_net',
        'NSA': 'ops/nsa',
        'PathAttn': 'ops/path_attn',
        'Rebased': 'ops/rebased',
        'Retention': 'ops/retention',
        'RWKV4': 'ops/rwkv4',
        'RWKV6': 'ops/rwkv6',
        'RWKV7': 'ops/rwkv7',
        'SimpleGLA': 'ops/simple_gla',
        'solve_tril': 'ops/utils',
        'Titans': 'ops/titans',
        'TTT': 'ops/ttt',
        'cumsum': 'ops/utils',
        'index': 'ops/utils',
        'logcumsumexp': 'ops/utils',
        'logsumexp': 'ops/utils',
        'matmul': 'ops/utils',
        'op': 'ops/utils',
        'pack': 'ops/utils',
        'pooling': 'ops/utils',
        'softmax': 'ops/utils',
        'softplus': 'ops/utils',
        'sigmoid': 'modules/activations',
        'logsigmoid': 'modules/activations',
        'swish': 'modules/activations',
        'swiglu': 'modules/activations',
        'activations': 'modules/activations',
        'causal_conv1d': 'modules/conv',
        'bitlinear': 'modules/fused_bitlinear',
        'cross_entropy': 'modules/fused_cross_entropy',
        'linear_cross_entropy': 'modules/fused_linear_cross_entropy',
        'kl_div': 'modules/fused_kl_div',
        'norm_gate': 'modules/fused_norm_gate',
        'grpo': 'modules/grpo',
        'l2norm': 'modules/l2norm',
        'layernorm': 'modules/layernorm',
        'layernorm_gated': 'modules/layernorm_gated',
        'rotary': 'modules/rotary',
        'token_shift': 'modules/token_shift',
    }
    return ops_modules.get(operator_name, 'unknown')


def build_operator_info(mapping_rows: List[Dict], coverage_rows: List[Dict]) -> Dict:
    """构建算子信息字典"""
    operator_info = defaultdict(lambda: {
        'kernels': [],
        'module': '',
        'description': '',
        'test_paths': set(),
        'test_funcs': set(),
        'coverage_status': '无测试用例',
        'remarks': set()
    })
    
    for row in mapping_rows:
        op_name = row['算子名称']
        operator_info[op_name]['module'] = row['算子所属模块']
        operator_info[op_name]['description'] = row['算子功能描述']
        operator_info[op_name]['kernels'].append({
            'name': row['Kernel函数名称'],
            'path': row['Kernel源码路径'],
            'line': row['Kernel行号']
        })
    
    for row in coverage_rows:
        if row['测试用例路径']:
            op_name = row['所属算子']
            operator_info[op_name]['test_paths'].add(row['测试用例路径'])
            operator_info[op_name]['test_funcs'].add(row['测试函数名称'])
            operator_info[op_name]['coverage_status'] = row['测试覆盖情况']
    
    return operator_info


def generate_markdown_report(operator_info: Dict, stats: Dict) -> str:
    """生成Markdown格式的报告"""
    lines = []
    
    lines.append("# Kernel函数测试覆盖分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## 统计摘要")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 算子总数 | {stats['total_operators']} |")
    lines.append(f"| Kernel函数总数 | {stats['total_kernels']} |")
    lines.append(f"| 有测试覆盖的Kernel | {stats['covered_kernels']} |")
    lines.append(f"| 无测试覆盖的Kernel | {stats['uncovered_kernels']} |")
    lines.append(f"| Kernel测试覆盖率 | {stats['coverage_rate']:.1f}% |")
    lines.append(f"| 测试用例总数 | {stats['total_tests']} |")
    lines.append("")
    
    lines.append("## 无测试用例的算子")
    lines.append("")
    if stats['uncovered_operators']:
        for op in sorted(stats['uncovered_operators']):
            lines.append(f"- {op}")
    else:
        lines.append("所有算子都有测试覆盖")
    lines.append("")
    
    lines.append("## 详细表格")
    lines.append("")
    lines.append("| 算子名称 | Kernel函数 | Kernel定义路径 | 所属模块 | 功能描述 | 测试用例路径 | 测试覆盖情况 | 备注 |")
    lines.append("|----------|------------|----------------|----------|----------|--------------|--------------|------|")
    
    sorted_operators = sorted(operator_info.keys())
    
    for op_name in sorted_operators:
        info = operator_info[op_name]
        module = info['module']
        desc = info['description']
        test_paths = '|'.join(sorted(info['test_paths'])) if info['test_paths'] else '无测试用例'
        coverage = info['coverage_status']
        
        for i, kernel in enumerate(info['kernels']):
            kernel_name = kernel['name']
            kernel_path = f"{kernel['path']}:{kernel['line']}"
            
            if i == 0:
                lines.append(f"| {op_name} | {kernel_name} | {kernel_path} | {module} | {desc} | {test_paths} | {coverage} | |")
            else:
                lines.append(f"| | {kernel_name} | {kernel_path} | | | | | |")
    
    lines.append("")
    lines.append("## 特殊测试场景分析")
    lines.append("")
    lines.append("### 多kernel合并测试")
    lines.append("")
    lines.append("单个算子的多个kernel共享同一测试用例，测试该算子的完整功能。")
    lines.append("")
    lines.append("### 多算子多kernel综合测试")
    lines.append("")
    lines.append("测试用例覆盖多个算子的多个kernel，验证相关功能的正确性。主要原因：")
    lines.append("1. 共享kernel机制：Common算子提供的kernel被多个算子共享使用")
    lines.append("2. 相似架构：GatedDeltaRule、KDA、GLA等算子具有相似的架构设计")
    lines.append("3. 代码复用：通过共享kernel减少代码重复，提高维护效率")
    lines.append("")
    
    lines.append("## 关键发现")
    lines.append("")
    lines.append("1. **算子级测试为主**：大多数测试用例针对整个算子的功能进行测试，而非单独测试每个kernel")
    lines.append("2. **共享kernel间接测试**：共享kernel通过使用它们的算子的测试用例间接验证")
    lines.append("3. **变体测试丰富**：同一算子通常有多个测试变体（varlen、gqa、transpose_state等）")
    lines.append("4. **部分算子缺少测试**：ABC、RWKV4、Rebased、bitlinear、norm_gate等算子暂未实现测试用例")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*本报告由自动化脚本生成*")
    
    return '\n'.join(lines)


def main():
    print("加载输入文件...")
    mapping_rows = load_csv('tmp/operators_kernels_mapping.csv')
    test_rows = load_csv('tmp/test_cases_inventory.csv')
    coverage_rows = load_csv('tmp/kernel_test_coverage.csv')
    special_rows = load_csv('tmp/special_test_scenarios.csv')
    
    print(f"  - 算子kernel映射: {len(mapping_rows)}条")
    print(f"  - 测试用例清单: {len(test_rows)}条")
    print(f"  - Kernel测试覆盖: {len(coverage_rows)}条")
    print(f"  - 特殊测试场景: {len(special_rows)}条")
    
    operator_info = build_operator_info(mapping_rows, coverage_rows)
    
    total_kernels = len(mapping_rows)
    covered_kernels = sum(1 for r in coverage_rows if r['测试用例路径'])
    unique_covered = len(set(r['Kernel函数名称'] for r in coverage_rows if r['测试用例路径']))
    
    stats = {
        'total_operators': len(operator_info),
        'total_kernels': total_kernels,
        'covered_kernels': unique_covered,
        'uncovered_kernels': total_kernels - unique_covered,
        'coverage_rate': unique_covered / total_kernels * 100 if total_kernels > 0 else 0,
        'total_tests': len(test_rows),
        'uncovered_operators': [op for op, info in operator_info.items() 
                                 if not info['test_paths']]
    }
    
    print("\n统计信息:")
    print(f"  - 算子总数: {stats['total_operators']}")
    print(f"  - Kernel总数: {stats['total_kernels']}")
    print(f"  - 有测试覆盖的Kernel: {stats['covered_kernels']}")
    print(f"  - 覆盖率: {stats['coverage_rate']:.1f}%")
    
    report = generate_markdown_report(operator_info, stats)
    
    output_file = 'kernel_analysis_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n输出文件: {output_file}")


if __name__ == '__main__':
    main()
