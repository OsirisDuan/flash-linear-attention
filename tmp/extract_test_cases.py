#!/usr/bin/env python3
"""
Extract test cases from test files and generate inventory CSV.
"""
import ast
import os
import csv
from pathlib import Path

def extract_test_info(file_path):
    """Extract test function information from a Python test file."""
    tests = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_name = node.name
                
                # Determine test type based on function name and content
                test_type = "单元测试"
                if "perf" in test_name.lower() or "benchmark" in test_name.lower():
                    test_type = "性能测试"
                elif "integration" in test_name.lower() or "e2e" in test_name.lower():
                    test_type = "集成测试"
                
                # Try to determine test object from function name
                test_object = extract_test_object(test_name, node, content)
                
                # Determine test object type
                test_object_type = "算子"
                if "module" in test_name.lower() or "layer" in test_name.lower():
                    test_object_type = "模块"
                elif "kernel" in test_name.lower():
                    test_object_type = "Kernel函数"
                
                # Extract docstring if exists
                docstring = ast.get_docstring(node) or ""
                
                tests.append({
                    'test_name': test_name,
                    'test_type': test_type,
                    'test_object': test_object,
                    'test_object_type': test_object_type,
                    'docstring': docstring[:100] if docstring else ""
                })
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return tests

def extract_test_object(test_name, node, content):
    """Try to extract what the test is testing from function name and imports."""
    # Common patterns in test names
    # test_gla -> GLA
    # test_chunk -> chunk
    # test_fused_recurrent -> fused_recurrent
    
    # Remove 'test_' prefix
    name_without_prefix = test_name[5:]
    
    # Split by underscore and try to identify the operator
    parts = name_without_prefix.split('_')
    
    # Look for imports in the file to match operators
    operators = []
    
    # Common operator names mapping
    operator_keywords = [
        'gla', 'delta', 'based', 'comba', 'deltaformer', 'dplr', 'iplr',
        'forgetting', 'gated', 'gsa', 'hgrn', 'kda', 'linear_attn',
        'log_linear', 'mesa', 'nsa', 'oja', 'path_attn', 'retention',
        'rwkv', 'simple_gla', 'solve_tril', 'titans', 'ttt', 'attn',
        'layernorm', 'rmsnorm', 'groupnorm', 'conv', 'rotary', 'activation',
        'cross_entropy', 'kl_div', 'l2norm', 'token_shift', 'index',
        'intracard_cache', 'utils', 'grpo', 'l2warp', 'cp'
    ]
    
    # Find which operator this test is for
    for keyword in operator_keywords:
        if keyword in test_name.lower():
            return keyword
    
    # If no keyword found, return the first meaningful part
    if parts:
        return parts[0]
    
    return "unknown"

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
    
    print(f"Total tests found: {len(all_tests)}")
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    main()
