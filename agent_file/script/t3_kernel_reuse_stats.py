#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3: Kernel复用度统计
统计每个 kernel 被多少个算子/模块共享使用
输出: t3_kernel_reuse_stats.csv
"""

import csv
import os
import re
from collections import defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")
FLA_OPS_DIR = os.path.join(BASE_DIR, "fla", "ops")
FLA_MODULES_DIR = os.path.join(BASE_DIR, "fla", "modules")

T2_CSV = os.path.join(OUTPUT_DIR, "t2_kernel_operator_model_mapping.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "t3_kernel_reuse_stats.csv")

REUSE_SCORE_MAP = {
    1: (1, "极低"),
    2: (2, "低"),
    3: (3, "中"),
    4: (4, "高"),
    5: (5, "极高"),
}


def get_n_models_from_t2(t2_row):
    model_list = t2_row.get("使用的模型列表", "")
    if not model_list:
        return 0
    return len(model_list.split(","))


def scan_common_and_utils_imports():
    """扫描 fla/ops/ 下所有算子对 common 和 utils 的 import 引用"""
    common_imports = defaultdict(set)
    utils_imports = defaultdict(set)

    for root, dirs, files in os.walk(FLA_OPS_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, FLA_OPS_DIR).replace("\\", "/")

            parts = rel_path.split("/")
            if len(parts) < 2:
                continue

            op_name = parts[0]
            if op_name in ("common", "utils", "__pycache__"):
                continue

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            for line in content.splitlines():
                line = line.strip()
                m = re.match(r"from\s+fla\.ops\.common\.(\w+)\s+import\s+(.+)", line)
                if m:
                    module = m.group(1)
                    common_imports[module].add(op_name)

                m = re.match(r"from\s+fla\.ops\.utils\.(\w+)\s+import\s+(.+)", line)
                if m:
                    module = m.group(1)
                    utils_imports[module].add(op_name)

    return common_imports, utils_imports


def scan_cross_operator_imports():
    """扫描 fla/ops/ 下算子之间的跨算子 import 引用"""
    cross_imports = defaultdict(set)

    for root, dirs, files in os.walk(FLA_OPS_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, FLA_OPS_DIR).replace("\\", "/")

            parts = rel_path.split("/")
            if len(parts) < 2:
                continue

            consumer_op = parts[0]
            if consumer_op in ("common", "utils", "__pycache__"):
                continue

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            for line in content.splitlines():
                line = line.strip()
                m = re.match(r"from\s+fla\.ops\.(\w+)\.", line)
                if m:
                    imported_op = m.group(1)
                    if imported_op not in ("common", "utils") and imported_op != consumer_op:
                        cross_imports[imported_op].add(consumer_op)

    return cross_imports


def scan_modules_imports():
    """扫描 fla/modules/ 下模块被哪些文件引用"""
    module_imports = defaultdict(set)

    for root, dirs, files in os.walk(FLA_MODULES_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, FLA_MODULES_DIR).replace("\\", "/")
            module_file = rel_path.replace(".py", "")
            if "/" in module_file:
                module_file = module_file.split("/")[0]

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            for line in content.splitlines():
                line = line.strip()
                m = re.match(r"from\s+fla\.modules\s+import\s+(.+)", line)
                if m:
                    imported_classes = [c.strip() for c in m.group(1).split(",")]
                    for ic in imported_classes:
                        module_imports[ic].add(module_file)

                m = re.match(r"from\s+fla\.modules\.(\w+)\s+import\s+(.+)", line)
                if m:
                    mod_name = m.group(1)
                    module_imports[mod_name].add(module_file)

    return module_imports


def scan_fla_modules_usage():
    """扫描 fla/models 和 fla/layers 对 fla.modules 的引用
    返回: 模块名 -> 使用它的模型名称集合
    """
    models_dir = os.path.join(BASE_DIR, "fla", "models")
    layers_dir = os.path.join(BASE_DIR, "fla", "layers")

    module_to_models = defaultdict(set)

    for model_subdir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_subdir)
        if not os.path.isdir(model_path):
            continue
        model_name = model_subdir

        for fname in os.listdir(model_path):
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(model_path, fname)

            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            for line in content.splitlines():
                line = line.strip()
                m = re.match(r"from\s+fla\.modules\s+import\s+(.+)", line)
                if m:
                    imported_classes = [c.strip() for c in m.group(1).split(",")]
                    for ic in imported_classes:
                        module_to_models[ic].add(model_name)

                m = re.match(r"from\s+fla\.modules\.(\w+)\s+import\s+(.+)", line)
                if m:
                    mod_name = m.group(1)
                    module_to_models[mod_name].add(model_name)

    layer_to_models = defaultdict(set)
    for layer_fname in os.listdir(layers_dir):
        if not layer_fname.endswith(".py"):
            continue
        layer_name = layer_fname.replace(".py", "")
        layer_path = os.path.join(layers_dir, layer_fname)

        with open(layer_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        for line in content.splitlines():
            line = line.strip()
            m = re.match(r"from\s+fla\.modules\s+import\s+(.+)", line)
            if m:
                imported_classes = [c.strip() for c in m.group(1).split(",")]
                for ic in imported_classes:
                    layer_to_models[ic].add(layer_name)

            m = re.match(r"from\s+fla\.modules\.(\w+)\s+import\s+(.+)", line)
            if m:
                mod_name = m.group(1)
                layer_to_models[mod_name].add(layer_name)

    return module_to_models, layer_to_models


def classify_kernel_source(source_path):
    if source_path.startswith("fla/modules/"):
        return "共享模块"
    if source_path.startswith("fla/ops/utils/"):
        return "共享模块"
    if source_path.startswith("fla/ops/common/"):
        return "共享模块"
    return "核心算子"


def get_module_name_for_kernel(source_path):
    if source_path.startswith("fla/modules/"):
        parts = source_path.replace("fla/modules/", "").split("/")
        if parts[0].endswith(".py"):
            return parts[0].replace(".py", "")
        return parts[0]
    if source_path.startswith("fla/ops/common/"):
        parts = source_path.replace("fla/ops/common/", "").split("/")
        return parts[0].replace(".py", "")
    if source_path.startswith("fla/ops/utils/"):
        parts = source_path.replace("fla/ops/utils/", "").split("/")
        return parts[0].replace(".py", "")
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("T3: Kernel复用度统计")
    print("=" * 60)

    print("\n步骤 3.1: 读取 T2 数据")
    t2_rows = []
    with open(T2_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t2_rows.append(row)
    print(f"  读取 {len(t2_rows)} 条 kernel 记录")

    print("\n步骤 3.2: 扫描 import 链")
    common_imports, utils_imports = scan_common_and_utils_imports()
    print(f"  common 模块: {len(common_imports)} 个被引用")
    for mod in sorted(common_imports.keys()):
        ops = sorted(common_imports[mod])
        print(f"    {mod}: {ops}")

    print(f"  utils 模块: {len(utils_imports)} 个被引用")
    for mod in sorted(utils_imports.keys()):
        ops = sorted(utils_imports[mod])
        print(f"    {mod}: {ops}")

    cross_imports = scan_cross_operator_imports()
    print(f"  跨算子引用: {len(cross_imports)} 个被引用算子")
    for imported_op in sorted(cross_imports.keys()):
        consumers = sorted(cross_imports[imported_op])
        print(f"    {imported_op} <- {consumers}")

    module_usage, layer_module_usage = scan_fla_modules_usage()
    print(f"  fla.modules 使用: {len(module_usage)} 个模块被模型引用")
    for mod_name in sorted(module_usage.keys()):
        models = sorted(module_usage[mod_name])
        print(f"    {mod_name}: {len(models)} 个模型 ({', '.join(models[:5])}{'...' if len(models) > 5 else ''})")
    print(f"  fla.modules 使用: {len(layer_module_usage)} 个模块被 Layer 引用")
    for mod_name in sorted(layer_module_usage.keys()):
        layers = sorted(layer_module_usage[mod_name])
        print(f"    {mod_name}: {', '.join(layers)}")

    print("\n步骤 3.3: 计算每个 kernel 的复用度")

    results = []
    for row in t2_rows:
        kernel_name = row["Kernel函数名称"]
        source_path = row["Kernel源码路径"]
        line_no = row["Kernel行号"]
        op_or_module = row["所属算子/模块"]
        kernel_source = row["kernel来源"]

        unique_key = f"{source_path}:{line_no}"

        consumer_ops = set()

        if kernel_source == "核心算子":
            op_name = op_or_module
            consumer_ops.add(op_name)

            op_lower = op_name.lower()
            if op_lower in cross_imports:
                consumer_ops.update(cross_imports[op_lower])
            else:
                for key in cross_imports:
                    if key.lower() == op_lower:
                        consumer_ops.update(cross_imports[key])
                        break

        elif source_path.startswith("fla/ops/common/"):
            module_file = get_module_name_for_kernel(source_path)
            if module_file and module_file in common_imports:
                consumer_ops.update(common_imports[module_file])

        elif source_path.startswith("fla/ops/utils/"):
            module_file = get_module_name_for_kernel(source_path)
            if module_file and module_file in utils_imports:
                consumer_ops.update(utils_imports[module_file])

        elif source_path.startswith("fla/modules/"):
            module_file = get_module_name_for_kernel(source_path)
            if module_file:
                consumer_ops.add(module_file)
                if module_file in module_usage:
                    consumer_ops.update(module_usage[module_file])
                if module_file in layer_module_usage:
                    for layer_name in layer_module_usage[module_file]:
                        consumer_ops.add(f"layer:{layer_name}")

        reuse_count = len(consumer_ops)
        consumer_list = sorted(consumer_ops)

        n_models = get_n_models_from_t2(row)

        if reuse_count >= 4:
            raw_score = 5
            reuse_level = "极高"
        elif reuse_count == 3:
            raw_score = 4
            reuse_level = "高"
        elif reuse_count == 2:
            raw_score = 3
            reuse_level = "中"
        elif n_models >= 2:
            raw_score = 2
            reuse_level = "低"
        else:
            raw_score = 1
            reuse_level = "极低"

        if kernel_source == "共享模块":
            global_score = min(raw_score, 2)
        else:
            global_score = raw_score

        module_internal_score = raw_score

        is_common_kernel = source_path.startswith("fla/ops/common/")

        results.append({
            "Kernel函数名称": kernel_name,
            "kernel来源": kernel_source,
            "共享算子/模块数量": reuse_count,
            "共享算子/模块列表": ",".join(consumer_list),
            "使用模型数量": n_models,
            "复用度级别": reuse_level,
            "原始复用度得分": raw_score,
            "全局排名复用度得分": global_score,
            "模块内排名复用度得分": module_internal_score,
            "是否为common目录kernel": "是" if is_common_kernel else "否",
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })

    print(f"  处理 {len(results)} 条记录")

    print("\n步骤 3.4: 输出结果")
    fieldnames = [
        "Kernel函数名称", "kernel来源", "共享算子/模块数量", "共享算子/模块列表",
        "使用模型数量", "复用度级别", "原始复用度得分", "全局排名复用度得分",
        "模块内排名复用度得分", "是否为common目录kernel",
        "Kernel源码路径", "Kernel行号"
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  输出文件: {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("T3 执行结果汇总")
    print("=" * 60)

    total = len(results)
    core_kernels = [r for r in results if r["kernel来源"] == "核心算子"]
    shared_kernels = [r for r in results if r["kernel来源"] == "共享模块"]

    print(f"\n总体统计:")
    print(f"  总 kernel 数: {total}")
    print(f"  核心算子 kernel: {len(core_kernels)}")
    print(f"  共享模块 kernel: {len(shared_kernels)}")

    print(f"\n复用度分布:")
    for level in ["极高", "高", "中", "低", "极低"]:
        count = sum(1 for r in results if r["复用度级别"] == level)
        print(f"  {level}: {count}")

    print(f"\n核心算子 kernel 复用度分布:")
    for level in ["极高", "高", "中", "低", "极低"]:
        count = sum(1 for r in core_kernels if r["复用度级别"] == level)
        print(f"  {level}: {count}")

    print(f"\n共享模块 kernel 复用度分布:")
    for level in ["极高", "高", "中", "低", "极低"]:
        count = sum(1 for r in shared_kernels if r["复用度级别"] == level)
        print(f"  {level}: {count}")

    print(f"\n高复用度 kernel (>=3个算子/模块):")
    high_reuse = [r for r in results if r["共享算子/模块数量"] >= 3]
    for r in sorted(high_reuse, key=lambda x: -x["共享算子/模块数量"])[:20]:
        print(f"  {r['Kernel函数名称']} ({r['kernel来源']}): {r['共享算子/模块数量']} - {r['共享算子/模块列表']}")

    print(f"\n双轨制验证:")
    shared_with_high_global = [r for r in shared_kernels if r["全局排名复用度得分"] > 2]
    if shared_with_high_global:
        print(f"  警告: {len(shared_with_high_global)} 个共享模块 kernel 全局得分 > 2")
    else:
        print(f"  OK: 所有共享模块 kernel 全局得分 <= 2")

    shared_raw_gt_2 = [r for r in shared_kernels if r["原始复用度得分"] > 2]
    shared_internal_gt_2 = [r for r in shared_kernels if r["模块内排名复用度得分"] > 2]
    print(f"  共享模块 kernel 原始得分 > 2: {len(shared_raw_gt_2)}")
    print(f"  共享模块 kernel 模块内得分 > 2: {len(shared_internal_gt_2)}")

    print("\n" + "=" * 60)
    print("T3 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
