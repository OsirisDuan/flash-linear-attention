#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5: 综合评分计算
整合各维度评分数据，采用双轨制计算每个 kernel 的加权总分
输出:
  - t5_kernel_tech_complexity.csv (技术复杂度支撑数据)
  - t5_kernel_test_gap_scores.csv (测试缺失度支撑数据)
  - t5_kernel_priority_scores.csv (全局排名)
  - t5_module_kernel_priority_scores.csv (共享模块内部排名)
"""

import csv
import os
import re
from collections import Counter, defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")

T2_CSV = os.path.join(OUTPUT_DIR, "t2_kernel_operator_model_mapping.csv")
T3_CSV = os.path.join(OUTPUT_DIR, "t3_kernel_reuse_stats.csv")
T4_CSV = os.path.join(OUTPUT_DIR, "t4_kernel_criticality.csv")
TEST_COVERAGE_CSV = os.path.join(OUTPUT_DIR, "kernel_test_coverage.csv")

OUTPUT_TECH_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_tech_complexity.csv")
OUTPUT_TEST_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_test_gap_scores.csv")
OUTPUT_GLOBAL_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_priority_scores.csv")
OUTPUT_MODULE_CSV = os.path.join(OUTPUT_DIR, "t5_module_kernel_priority_scores.csv")

WEIGHTS = {
    "model_heat": 0.35,
    "reuse": 0.30,
    "criticality": 0.20,
    "test_gap": 0.05,
    "tech_complexity": 0.10,
}


def find_func_body(content, start_line):
    lines = content.split("\n")
    start_idx = start_line - 1
    
    if start_idx >= len(lines):
        return [], 0
    
    if "@triton.jit" in lines[start_idx]:
        def_idx = start_idx + 1
        while def_idx < len(lines) and "def " not in lines[def_idx]:
            def_idx += 1
        if def_idx >= len(lines):
            return [], 0
        start_idx = def_idx
    
    first_line = lines[start_idx]
    if "def " not in first_line:
        return [], 0
    
    paren_depth = first_line.count("(") - first_line.count(")")
    body_start = start_idx
    
    while paren_depth > 0 and body_start < len(lines) - 1:
        body_start += 1
        paren_depth += lines[body_start].count("(") - lines[body_start].count(")")
    
    body_start += 1
    
    body_lines = []
    base_indent = None
    for i in range(body_start, len(lines)):
        line = lines[i]
        if line.strip() == "":
            body_lines.append(line)
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        if base_indent is None:
            base_indent = current_indent
        
        if current_indent < base_indent and line.strip():
            break
        
        body_lines.append(line)
    
    return body_lines, body_start + 1


def count_tl_dot(body_lines):
    count = 0
    for line in body_lines:
        count += len(re.findall(r"tl\.dot\s*\(", line))
    return count


def calc_tech_complexity(kernel_type, line_count, dot_count):
    """
    技术复杂度分级:
    - cv/cube + >50行 + 多个tl.dot() → 极高(5分)
    - cv/cube + >30行 → 高(4分)
    - cv/cube + ≤30行 或 vv + >30行 → 中(3分)
    - vv + 15-30行 → 低(2分)
    - vv + <15行 → 极低(1分)
    """
    if kernel_type == "cv/cube":
        if line_count > 50 and dot_count >= 2:
            return "极高", 5, f"cv/cube+{line_count}行+{dot_count}个tl.dot"
        elif line_count > 30:
            return "高", 4, f"cv/cube+{line_count}行"
        else:
            return "中", 3, f"cv/cube+{line_count}行"
    else:
        if line_count > 30:
            return "中", 3, f"vv+{line_count}行"
        elif line_count >= 15:
            return "低", 2, f"vv+{line_count}行"
        else:
            return "极低", 1, f"vv+{line_count}行"


def calc_test_gap_score(coverage_status, test_func_count, test_path_count):
    """
    测试缺失度分级:
    - 无测试用例 → 严重缺失(5分)
    - 有测试但单一场景(1个测试函数) → 明显不足(4分)
    - 基本功能测试(2-4个测试函数或1个测试文件) → 部分覆盖(3分)
    - 功能+梯度测试(5+个测试函数且2+个测试文件) → 基本覆盖(2分)
    - 全面覆盖(10+个测试函数且3+个测试文件) → 充分覆盖(1分)
    - 未统计 → 0分
    """
    if coverage_status == "无测试用例":
        return "严重缺失", 5
    
    if coverage_status == "":
        return "未统计", 0
    
    if test_func_count <= 1:
        return "明显不足", 4
    elif test_func_count <= 4 or test_path_count <= 1:
        return "部分覆盖", 3
    elif test_func_count < 10 or test_path_count < 3:
        return "基本覆盖", 2
    else:
        return "充分覆盖", 1


def get_priority_level(score):
    if score >= 4.0:
        return "P0"
    elif score >= 3.0:
        return "P1"
    elif score >= 2.0:
        return "P2"
    else:
        return "P3"


def main():
    print("=" * 60)
    print("T5: 综合评分计算")
    print("=" * 60)
    
    print("\n步骤 5.1: 读取各维度数据")
    
    t2_data = {}
    with open(T2_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["Kernel函数名称"], row["Kernel源码路径"], row["Kernel行号"])
            t2_data[key] = row
    print(f"  T2 数据: {len(t2_data)} 条")
    
    t3_data = {}
    with open(T3_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["Kernel函数名称"], row["Kernel源码路径"], row["Kernel行号"])
            t3_data[key] = row
    print(f"  T3 数据: {len(t3_data)} 条")
    
    t4_data = {}
    with open(T4_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["Kernel函数名称"], row["Kernel源码路径"], row["Kernel行号"])
            t4_data[key] = row
    print(f"  T4 数据: {len(t4_data)} 条")
    
    test_coverage = defaultdict(list)
    with open(TEST_COVERAGE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kernel_name = row["Kernel函数名称"]
            test_coverage[kernel_name].append(row)
    print(f"  测试覆盖数据: {len(test_coverage)} 个 kernel")
    
    all_kernels = set(t2_data.keys()) | set(t3_data.keys()) | set(t4_data.keys())
    print(f"  总 kernel 数: {len(all_kernels)}")
    
    print("\n步骤 5.2: 计算技术复杂度得分")
    
    source_cache = {}
    tech_results = []
    
    for key in all_kernels:
        kernel_name, source_path, line_no = key
        
        row = t2_data.get(key) or t3_data.get(key) or t4_data.get(key)
        kernel_type = row.get("kernel类型", "vv") if row else "vv"
        kernel_source = row.get("kernel来源", "核心算子") if row else "核心算子"
        
        full_path = os.path.join(BASE_DIR, source_path)
        line_count = 0
        dot_count = 0
        
        if os.path.exists(full_path):
            if full_path not in source_cache:
                with open(full_path, "r", encoding="utf-8") as f:
                    source_cache[full_path] = f.read()
            
            content = source_cache[full_path]
            body_lines, _ = find_func_body(content, int(line_no))
            line_count = len([l for l in body_lines if l.strip()])
            dot_count = count_tl_dot(body_lines)
        
        level, score, reason = calc_tech_complexity(kernel_type, line_count, dot_count)
        
        tech_results.append({
            "Kernel函数名称": kernel_name,
            "kernel类型": kernel_type,
            "kernel来源": kernel_source,
            "函数行数": line_count,
            "tl.dot调用次数": dot_count,
            "技术复杂度级别": level,
            "技术复杂度得分": score,
            "判定依据": reason,
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })
    
    with open(OUTPUT_TECH_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Kernel函数名称", "kernel类型", "kernel来源", "函数行数", 
                      "tl.dot调用次数", "技术复杂度级别", "技术复杂度得分", "判定依据",
                      "Kernel源码路径", "Kernel行号"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tech_results)
    print(f"  输出: {OUTPUT_TECH_CSV}")
    
    tech_data = {}
    for r in tech_results:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        tech_data[key] = r
    
    print("\n步骤 5.3: 计算测试缺失度得分")
    
    test_results = []
    for key in all_kernels:
        kernel_name, source_path, line_no = key
        
        row = t2_data.get(key) or t3_data.get(key) or t4_data.get(key)
        kernel_source = row.get("kernel来源", "核心算子") if row else "核心算子"
        
        coverage_rows = test_coverage.get(kernel_name, [])
        
        if coverage_rows:
            coverage_statuses = set(r["测试覆盖情况"] for r in coverage_rows)
            if "无测试用例" in coverage_statuses:
                coverage_status = "无测试用例"
            else:
                coverage_status = list(coverage_statuses)[0]
            
            test_paths = set(r.get("测试用例路径", "") for r in coverage_rows if r.get("测试用例路径"))
            test_funcs = set(r.get("测试函数名称", "") for r in coverage_rows if r.get("测试函数名称"))
            test_func_count = len(test_funcs)
            test_path_count = len(test_paths)
        else:
            coverage_status = ""
            test_paths = set()
            test_funcs = set()
            test_func_count = 0
            test_path_count = 0
        
        level, score = calc_test_gap_score(coverage_status, test_func_count, test_path_count)
        
        test_results.append({
            "Kernel函数名称": kernel_name,
            "kernel来源": kernel_source,
            "测试覆盖情况": coverage_status,
            "测试用例列表": ",".join(sorted(test_funcs)) if test_funcs else "",
            "测试缺失度级别": level,
            "测试缺失度得分": score,
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })
    
    with open(OUTPUT_TEST_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Kernel函数名称", "kernel来源", "测试覆盖情况", "测试用例列表",
                      "测试缺失度级别", "测试缺失度得分", "Kernel源码路径", "Kernel行号"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_results)
    print(f"  输出: {OUTPUT_TEST_CSV}")
    
    test_data = {}
    for r in test_results:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        test_data[key] = r
    
    print("\n步骤 5.4: 计算加权总分 — 全局排名")
    
    global_results = []
    for key in all_kernels:
        kernel_name, source_path, line_no = key
        
        t2_row = t2_data.get(key, {})
        t3_row = t3_data.get(key, {})
        t4_row = t4_data.get(key, {})
        tech_row = tech_data.get(key, {})
        test_row = test_data.get(key, {})
        
        kernel_type = t2_row.get("kernel类型", tech_row.get("kernel类型", "vv"))
        kernel_source = t2_row.get("kernel来源", t3_row.get("kernel来源", "核心算子"))
        
        model_heat_score = int(t2_row.get("模型热度得分", 1)) if t2_row.get("模型热度得分") else 1
        
        reuse_score = int(t3_row.get("全局排名复用度得分", 1)) if t3_row.get("全局排名复用度得分") else 1
        
        criticality_score = int(t4_row.get("核心程度得分", 3)) if t4_row.get("核心程度得分") else 3
        
        tech_score = int(tech_row.get("技术复杂度得分", 3)) if tech_row.get("技术复杂度得分") else 3
        
        test_score = int(test_row.get("测试缺失度得分", 0)) if test_row.get("测试缺失度得分") else 0
        
        weighted_score = (
            model_heat_score * WEIGHTS["model_heat"]
            + reuse_score * WEIGHTS["reuse"]
            + criticality_score * WEIGHTS["criticality"]
            + test_score * WEIGHTS["test_gap"]
            + tech_score * WEIGHTS["tech_complexity"]
        )
        
        adjustment = 0.0
        
        final_score = weighted_score + adjustment
        final_score = min(final_score, 5.5)
        
        priority = get_priority_level(final_score)
        
        model_list = t2_row.get("使用的模型列表", "")
        op_list = t3_row.get("共享算子/模块列表", "")
        
        global_results.append({
            "Kernel函数名称": kernel_name,
            "kernel类型": kernel_type,
            "kernel来源": kernel_source,
            "模型热度得分": model_heat_score,
            "复用度得分": reuse_score,
            "核心程度得分": criticality_score,
            "测试缺失度得分": test_score,
            "技术复杂度得分": tech_score,
            "加权总分": round(weighted_score, 2),
            "调整加成": adjustment,
            "最终总分": round(final_score, 2),
            "优先级等级": priority,
            "使用的模型列表": model_list,
            "共享算子/模块列表": op_list,
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })
    
    global_results.sort(key=lambda x: -x["最终总分"])
    
    with open(OUTPUT_GLOBAL_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Kernel函数名称", "kernel类型", "kernel来源", "模型热度得分",
                      "复用度得分", "核心程度得分", "测试缺失度得分", "技术复杂度得分",
                      "加权总分", "调整加成", "最终总分", "优先级等级",
                      "使用的模型列表", "共享算子/模块列表", "Kernel源码路径", "Kernel行号"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(global_results)
    print(f"  输出: {OUTPUT_GLOBAL_CSV}")
    
    print("\n步骤 5.4b: 计算加权总分 — 共享模块内部排名")
    
    module_results = []
    for key in all_kernels:
        kernel_name, source_path, line_no = key
        
        t2_row = t2_data.get(key, {})
        t3_row = t3_data.get(key, {})
        t4_row = t4_data.get(key, {})
        tech_row = tech_data.get(key, {})
        test_row = test_data.get(key, {})
        
        kernel_source = t2_row.get("kernel来源", t3_row.get("kernel来源", ""))
        
        if kernel_source != "共享模块":
            continue
        
        kernel_type = t2_row.get("kernel类型", tech_row.get("kernel类型", "vv"))
        
        if "modules/" in source_path:
            module_name = source_path.split("modules/")[1].split("/")[0].replace(".py", "")
        else:
            module_name = "common"
        
        model_heat_score = int(t2_row.get("模型热度得分", 5)) if t2_row.get("模型热度得分") else 5
        
        reuse_score = int(t3_row.get("模块内排名复用度得分", 1)) if t3_row.get("模块内排名复用度得分") else 1
        
        criticality_score = int(t4_row.get("核心程度得分", 3)) if t4_row.get("核心程度得分") else 3
        
        tech_score = int(tech_row.get("技术复杂度得分", 3)) if tech_row.get("技术复杂度得分") else 3
        
        test_score = int(test_row.get("测试缺失度得分", 0)) if test_row.get("测试缺失度得分") else 0
        
        weighted_score = (
            model_heat_score * WEIGHTS["model_heat"]
            + reuse_score * WEIGHTS["reuse"]
            + criticality_score * WEIGHTS["criticality"]
            + test_score * WEIGHTS["test_gap"]
            + tech_score * WEIGHTS["tech_complexity"]
        )
        
        module_results.append({
            "Kernel函数名称": kernel_name,
            "kernel类型": kernel_type,
            "所属模块": module_name,
            "模型热度得分": model_heat_score,
            "复用度得分": reuse_score,
            "核心程度得分": criticality_score,
            "测试缺失度得分": test_score,
            "技术复杂度得分": tech_score,
            "加权总分": round(weighted_score, 2),
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })
    
    module_results.sort(key=lambda x: -x["加权总分"])
    
    for i, r in enumerate(module_results):
        r["模块内排名"] = i + 1
        r["模块内优先级等级"] = get_priority_level(r["加权总分"])
    
    with open(OUTPUT_MODULE_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Kernel函数名称", "kernel类型", "所属模块", "模型热度得分",
                      "复用度得分", "核心程度得分", "测试缺失度得分", "技术复杂度得分",
                      "加权总分", "模块内排名", "模块内优先级等级", "Kernel源码路径", "Kernel行号"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(module_results)
    print(f"  输出: {OUTPUT_MODULE_CSV}")
    
    print("\n" + "=" * 60)
    print("T5 执行结果汇总")
    print("=" * 60)
    
    print(f"\n总体统计:")
    print(f"  总 kernel 数: {len(all_kernels)}")
    print(f"  核心算子 kernel: {sum(1 for r in global_results if r['kernel来源'] == '核心算子')}")
    print(f"  共享模块 kernel: {sum(1 for r in global_results if r['kernel来源'] == '共享模块')}")
    
    print(f"\n全局排名优先级分布:")
    for p in ["P0", "P1", "P2", "P3"]:
        count = sum(1 for r in global_results if r["优先级等级"] == p)
        print(f"  {p}: {count}")
    
    print(f"\n核心算子 kernel 优先级分布:")
    core_results = [r for r in global_results if r["kernel来源"] == "核心算子"]
    for p in ["P0", "P1", "P2", "P3"]:
        count = sum(1 for r in core_results if r["优先级等级"] == p)
        print(f"  {p}: {count}")
    
    print(f"\n共享模块 kernel 优先级分布:")
    shared_results = [r for r in global_results if r["kernel来源"] == "共享模块"]
    for p in ["P0", "P1", "P2", "P3"]:
        count = sum(1 for r in shared_results if r["优先级等级"] == p)
        print(f"  {p}: {count}")
    
    print(f"\n技术复杂度分布:")
    for level in ["极高", "高", "中", "低", "极低"]:
        count = sum(1 for r in tech_results if r["技术复杂度级别"] == level)
        print(f"  {level}: {count}")
    
    print(f"\n测试缺失度分布:")
    for level in ["严重缺失", "明显不足", "部分覆盖", "基本覆盖", "充分覆盖", "未统计"]:
        count = sum(1 for r in test_results if r["测试缺失度级别"] == level)
        print(f"  {level}: {count}")
    
    print(f"\nTop 10 P0 级 Kernel:")
    p0_kernels = [r for r in global_results if r["优先级等级"] == "P0"][:10]
    for r in p0_kernels:
        print(f"  {r['Kernel函数名称']:50s} {r['最终总分']:.2f} {r['kernel来源']}")
    
    print("\n" + "=" * 60)
    print("质量验证")
    print("=" * 60)
    
    print("\n验证1: 共享模块 kernel 复用度得分降权")
    shared_with_high_reuse = [r for r in shared_results if r["复用度得分"] > 2]
    if shared_with_high_reuse:
        print(f"  警告: {len(shared_with_high_reuse)} 个共享模块 kernel 复用度得分 > 2")
    else:
        print(f"  OK: 所有共享模块 kernel 复用度得分 <= 2")
    
    print("\n验证2: 测试缺失度得分范围")
    invalid_test_scores = [r for r in global_results if r["测试缺失度得分"] not in [0, 1, 2, 3, 4, 5]]
    if invalid_test_scores:
        print(f"  警告: {len(invalid_test_scores)} 个 kernel 测试缺失度得分不在 0-5 范围")
    else:
        print(f"  OK: 所有测试缺失度得分在 0-5 范围")
    
    print("\n验证3: 最终总分范围")
    invalid_final_scores = [r for r in global_results if r["最终总分"] < 1 or r["最终总分"] > 5.5]
    if invalid_final_scores:
        print(f"  警告: {len(invalid_final_scores)} 个 kernel 最终总分不在 1-5.5 范围")
    else:
        print(f"  OK: 所有最终总分在 1-5.5 范围")
    
    print("\n验证4: 模块内排名连续性")
    ranks = [r["模块内排名"] for r in module_results]
    if ranks and (min(ranks) != 1 or max(ranks) != len(ranks)):
        print(f"  警告: 模块内排名不连续 (min={min(ranks)}, max={max(ranks)}, count={len(ranks)})")
    else:
        print(f"  OK: 模块内排名连续 (1-{len(ranks)})")
    
    print("\n" + "=" * 60)
    print("T5 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
