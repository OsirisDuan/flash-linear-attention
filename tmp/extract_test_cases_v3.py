#!/usr/bin/env python3
"""
Extract test cases from test files and generate inventory CSV with improved accuracy.
"""
import ast
import os
import csv
import re
from pathlib import Path
from collections import defaultdict

def extract_imports(tree):
    """Extract imported modules and functions from AST."""
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    
    return imports

def extract_test_info(file_path):
    """Extract test function information from a Python test file."""
    tests = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = extract_imports(tree)
        
        # Determine the main operator/module being tested from imports
        main_test_object = determine_main_test_object(file_path, imports)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_name = node.name
                
                # Determine test type based on function name and content
                test_type = determine_test_type(test_name, node, content)
                
                # Determine specific test object
                test_object = determine_specific_test_object(test_name, main_test_object, node, imports)
                
                # Determine test object type
                test_object_type = determine_test_object_type(file_path, test_name)
                
                # Extract docstring if exists and clean it
                docstring = ast.get_docstring(node) or ""
                # Remove newlines and extra spaces from docstring
                docstring = ' '.join(docstring.split())
                docstring = docstring[:150] if docstring else ""
                
                tests.append({
                    'test_name': test_name,
                    'test_type': test_type,
                    'test_object': test_object,
                    'test_object_type': test_object_type,
                    'docstring': docstring
                })
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return tests

def determine_main_test_object(file_path, imports):
    """Determine the main test object from file path and imports."""
    file_name = Path(file_path).stem  # e.g., test_gla -> gla
    
    # Remove 'test_' prefix
    if file_name.startswith('test_'):
        operator_name = file_name[5:]
    else:
        operator_name = file_name
    
    # Map file names to operator names
    operator_mapping = {
        'attn': 'parallel_attention',
        'based': 'based',
        'comba': 'comba',
        'delta': 'delta_rule',
        'delta_product': 'delta_product',
        'deltaformer': 'deltaformer',
        'dplr_delta': 'dplr_delta',
        'forgetting_attn': 'forgetting_attention',
        'gated_delta': 'gated_delta_rule',
        'gated_delta_product': 'gated_delta_product',
        'gla': 'gla',
        'gsa': 'gsa',
        'hgrn': 'hgrn',
        'index': 'index',
        'intracard_cache': 'intracard_cache',
        'iplr_delta': 'iplr_delta',
        'kda': 'kda',
        'linear_attn': 'linear_attention',
        'log_linear_attn': 'log_linear_attention',
        'mesa': 'mesa',
        'nsa': 'nsa',
        'oja': 'oja',
        'path_attn': 'path_attention',
        'retention': 'retention',
        'rwkv6': 'rwkv6',
        'rwkv7': 'rwkv7',
        'simple_gla': 'simple_gla',
        'solve_tril': 'solve_tril',
        'titans': 'titans',
        'ttt': 'ttt',
        'utils': 'utils',
        'activation': 'activation',
        'activations_noncontiguous': 'activation',
        'conv': 'convolution',
        'cross_entropy': 'cross_entropy',
        'grpo': 'grpo',
        'kl_div': 'kl_div',
        'l2norm': 'l2norm',
        'l2warp': 'l2_warp',
        'layernorm': 'layer_norm',
        'layernorm_gated': 'layer_norm_gated',
        'rotary': 'rotary',
        'token_shift': 'token_shift',
        'cp_bwd_gk_offset': 'context_parallel',
        'cp_conv': 'context_parallel_convolution',
        'cp_gdn': 'context_parallel_gdn',
        'cp_kda': 'context_parallel_kda',
        'layer_cache_layer_idx': 'layer_cache',
    }
    
    return operator_mapping.get(operator_name, operator_name)

def determine_test_type(test_name, node, content):
    """Determine test type based on function name and content."""
    test_lower = test_name.lower()
    
    # Performance test indicators
    if any(keyword in test_lower for keyword in ['perf', 'benchmark', 'speed', 'throughput']):
        return "性能测试"
    
    # Integration test indicators
    if any(keyword in test_lower for keyword in ['integration', 'e2e', 'end_to_end', 'full']):
        return "集成测试"
    
    # Check if test involves multiple components
    func_source = ast.get_source_segment(content, node) or ""
    if 'distributed' in func_source or 'world_size' in func_source:
        return "集成测试"
    
    return "单元测试"

def determine_specific_test_object(test_name, main_object, node, imports):
    """Determine specific test object from test name and context."""
    test_lower = test_name.lower()
    
    # Common test patterns
    patterns = {
        'chunk': 'chunk',
        'fused_recurrent': 'fused_recurrent',
        'parallel': 'parallel',
        'varlen': 'varlen',
        'inference': 'inference',
        'decoding': 'decoding',
        'prefill': 'prefill',
        'gate': 'gate',
        'naive': 'naive',
        'backward': 'backward',
        'forward': 'forward',
        'cache': 'cache',
        'transpose_state': 'transpose_state',
        'gqa': 'gqa',
        'intracard': 'intracard',
    }
    
    # Check for specific patterns
    for pattern, obj_type in patterns.items():
        if pattern in test_lower:
            return f"{main_object}_{obj_type}"
    
    return main_object

def determine_test_object_type(file_path, test_name):
    """Determine test object type based on file path and test name."""
    path_str = str(file_path).replace('\\', '/')
    
    if '/modules/' in path_str:
        return "模块"
    elif '/layers/' in path_str:
        return "层"
    elif '/context_parallel/' in path_str:
        return "算子"
    elif '/ops/' in path_str:
        return "算子"
    
    return "算子"

def scan_test_directory(base_path, test_dirs):
    """Scan all test directories and collect test information."""
    all_tests = []
    
    for test_dir in test_dirs:
        dir_path = base_path / test_dir
        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            continue
        
        for py_file in dir_path.glob('test_*.py'):
            tests = extract_test_info(py_file)
            for test in tests:
                relative_path = str(py_file.relative_to(base_path))
                all_tests.append({
                    'file_path': relative_path,
                    **test
                })
    
    return all_tests

def generate_summary(all_tests):
    """Generate summary statistics."""
    summary = {
        'total_tests': len(all_tests),
        'by_type': defaultdict(int),
        'by_object_type': defaultdict(int),
        'by_file': defaultdict(int),
    }
    
    for test in all_tests:
        summary['by_type'][test['test_type']] += 1
        summary['by_object_type'][test['test_object_type']] += 1
        summary['by_file'][test['file_path']] += 1
    
    return summary

def main():
    base_path = Path(r'd:\code\flash-linear-attention')
    
    # Test directories to scan (excluding tests/models/)
    test_dirs = [
        'tests/ops',
        'tests/modules', 
        'tests/context_parallel',
        'tests/layers'
    ]
    
    print("Scanning test directories...")
    all_tests = scan_test_directory(base_path, test_dirs)
    
    # Generate summary
    summary = generate_summary(all_tests)
    
    print(f"\n=== Test Inventory Summary ===")
    print(f"Total tests: {summary['total_tests']}")
    print(f"\nBy test type:")
    for test_type, count in sorted(summary['by_type'].items()):
        print(f"  {test_type}: {count}")
    print(f"\nBy test object type:")
    for obj_type, count in sorted(summary['by_object_type'].items()):
        print(f"  {obj_type}: {count}")
    print(f"\nTest files: {len(summary['by_file'])}")
    
    # Write to CSV
    output_path = base_path / 'tmp' / 'test_cases_inventory.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '测试文件路径',
            '测试函数名称', 
            '测试类型',
            '测试对象',
            '测试对象类型',
            '备注'
        ])
        
        for test in all_tests:
            writer.writerow([
                test['file_path'],
                test['test_name'],
                test['test_type'],
                test['test_object'],
                test['test_object_type'],
                test['docstring']
            ])
    
    print(f"\nOutput written to: {output_path}")

if __name__ == '__main__':
    main()
