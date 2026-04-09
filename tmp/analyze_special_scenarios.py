#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子任务4：分析特殊测试场景
输入：
  - tmp/kernel_test_coverage.csv（子任务3输出）
输出：
  - tmp/special_test_scenarios.csv（特殊测试场景分析）
"""

import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_csv(filepath: str) -> List[Dict[str, str]]:
    """加载CSV文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_special_scenarios(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """分析特殊测试场景"""
    
    test_to_kernels: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    test_to_operators: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    test_to_coverage_type: Dict[Tuple[str, str], str] = {}
    
    for row in rows:
        test_path = row['测试用例路径']
        test_func = row['测试函数名称']
        kernel = row['Kernel函数名称']
        operator = row['所属算子']
        coverage_type = row['测试覆盖情况']
        
        if test_path and test_func:
            key = (test_path, test_func)
            test_to_kernels[key].add(kernel)
            test_to_operators[key].add(operator)
            test_to_coverage_type[key] = coverage_type
    
    results = []
    
    for (test_path, test_func), operators in test_to_operators.items():
        kernels = test_to_kernels[(test_path, test_func)]
        coverage_type = test_to_coverage_type[(test_path, test_func)]
        
        if len(kernels) > 1 or len(operators) > 1:
            if len(operators) == 1 and len(kernels) > 1:
                scenario_type = "多kernel合并测试"
                reason = f"单个算子({list(operators)[0]})的多个kernel共享同一测试用例，测试该算子的完整功能"
            elif len(operators) > 1 and len(kernels) == 1:
                scenario_type = "多算子共享kernel测试"
                reason = f"多个算子共享同一个kernel函数({list(kernels)[0]})，通过同一测试验证"
            elif len(operators) > 1 and len(kernels) > 1:
                scenario_type = "多算子多kernel综合测试"
                if "共享kernel" in coverage_type:
                    reason = "共享kernel被多个算子使用，测试用例验证多个算子的功能"
                else:
                    reason = "测试用例覆盖多个算子的多个kernel，验证相关功能的正确性"
            else:
                scenario_type = "其他特殊场景"
                reason = "需要进一步分析"
            
            result = {
                '测试用例路径': test_path,
                '测试函数名称': test_func,
                '特殊场景类型': scenario_type,
                '覆盖的算子列表': '|'.join(sorted(operators)),
                '覆盖的Kernel列表': '|'.join(sorted(kernels)),
                '共享原因': reason,
                '备注': f"算子数: {len(operators)}, Kernel数: {len(kernels)}"
            }
            results.append(result)
    
    return results


def categorize_scenarios(results: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """按场景类型分类"""
    categories = defaultdict(list)
    for r in results:
        categories[r['特殊场景类型']].append(r)
    return categories


def main():
    rows = load_csv('tmp/kernel_test_coverage.csv')
    
    print(f"加载 {len(rows)} 条kernel测试覆盖记录")
    
    special_scenarios = analyze_special_scenarios(rows)
    
    print(f"发现 {len(special_scenarios)} 个特殊测试场景")
    
    categories = categorize_scenarios(special_scenarios)
    print("\n按场景类型统计:")
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  - {cat}: {len(items)}个")
    
    output_file = 'tmp/special_test_scenarios.csv'
    fieldnames = ['测试用例路径', '测试函数名称', '特殊场景类型', '覆盖的算子列表', '覆盖的Kernel列表', '共享原因', '备注']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(special_scenarios)
    
    print(f"\n输出文件: {output_file}")
    
    print("\n" + "="*60)
    print("特殊测试场景详细分析")
    print("="*60)
    
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n### {cat} ({len(items)}个)")
        print("-"*60)
        
        for item in items[:5]:
            print(f"\n测试: {item['测试函数名称']}")
            print(f"  文件: {item['测试用例路径']}")
            print(f"  算子: {item['覆盖的算子列表']}")
            print(f"  Kernel数: {item['备注']}")
            print(f"  原因: {item['共享原因']}")
        
        if len(items) > 5:
            print(f"\n  ... 还有 {len(items)-5} 个同类场景")


if __name__ == '__main__':
    main()
