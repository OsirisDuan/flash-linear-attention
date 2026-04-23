#!/usr/bin/env python3
"""
排查 t2_kernel_operator_model_mapping.csv 中 kernel 归属算子不完整的问题。
核心问题：operators_kernels_mapping.csv 中每个 kernel 只关联了一个算子，
但 fla/ops/common/ 和 fla/ops/utils/ 下的 kernel 被多个算子共享使用。
"""

import csv
import os
import re
from collections import defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
FLA_OPS_DIR = os.path.join(BASE_DIR, "fla", "ops")
T2_CSV = os.path.join(BASE_DIR, "agent_file", "tmp_file", "t2_kernel_operator_model_mapping.csv")
OPS_CSV = os.path.join(BASE_DIR, "agent_file", "tmp_file", "operators_kernels_mapping.csv")


def scan_ops_imports():
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
                    names = [n.strip().split(" as ")[0].strip() for n in m.group(2).split(",")]
                    for name in names:
                        common_imports[f"common/{module}"].add(op_name)

                m = re.match(r"from\s+fla\.ops\.utils\.(\w+)\s+import\s+(.+)", line)
                if m:
                    module = m.group(1)
                    names = [n.strip().split(" as ")[0].strip() for n in m.group(2).split(",")]
                    for name in names:
                        utils_imports[f"utils/{module}"].add(op_name)

    return common_imports, utils_imports


def load_ops_kernels():
    """加载 operators_kernels_mapping.csv"""
    rows = []
    with open(OPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_actual_ops_for_kernel(kernel_name, source_path, common_imports, utils_imports):
    """获取一个 kernel 实际被哪些算子使用"""
    if source_path.startswith("fla/ops/common/"):
        parts = source_path.replace("fla/ops/common/", "").split("/")
        module_key = f"common/{parts[0].replace('.py', '')}"
        if module_key in common_imports:
            return common_imports[module_key]
    elif source_path.startswith("fla/ops/utils/"):
        parts = source_path.replace("fla/ops/utils/", "").split("/")
        module_key = f"utils/{parts[0].replace('.py', '')}"
        if module_key in utils_imports:
            return utils_imports[module_key]
    return set()


def main():
    common_imports, utils_imports = scan_ops_imports()
    ops_kernels = load_ops_kernels()

    print("=" * 80)
    print("1. fla/ops/common/ 被引用情况")
    print("=" * 80)
    for module_key in sorted(common_imports.keys()):
        ops = sorted(common_imports[module_key])
        print(f"  {module_key}: {len(ops)} 个算子 -> {ops}")

    print()
    print("=" * 80)
    print("2. fla/ops/utils/ 被引用情况")
    print("=" * 80)
    for module_key in sorted(utils_imports.keys()):
        ops = sorted(utils_imports[module_key])
        print(f"  {module_key}: {len(ops)} 个算子 -> {ops}")

    print()
    print("=" * 80)
    print("3. CSV 中归属不完整的 kernel（实际被多个算子使用但只关联了一个）")
    print("=" * 80)

    issues = []
    for row in ops_kernels:
        kernel_name = row["Kernel函数名称"]
        source_path = row["Kernel源码路径"]
        csv_op = row["算子名称"]

        actual_ops = get_actual_ops_for_kernel(kernel_name, source_path, common_imports, utils_imports)
        if not actual_ops:
            continue

        if csv_op not in actual_ops and csv_op.lower() not in [o.lower() for o in actual_ops]:
            issues.append({
                "kernel": kernel_name,
                "csv_op": csv_op,
                "actual_ops": sorted(actual_ops),
                "source": source_path,
                "type": "WRONG_OP",
            })
        elif len(actual_ops) > 1:
            csv_ops_set = {csv_op}
            if actual_ops != csv_ops_set and actual_ops - csv_ops_set:
                issues.append({
                    "kernel": kernel_name,
                    "csv_op": csv_op,
                    "actual_ops": sorted(actual_ops),
                    "source": source_path,
                    "type": "MISSING_OPS",
                })

    wrong_ops = [i for i in issues if i["type"] == "WRONG_OP"]
    missing_ops = [i for i in issues if i["type"] == "MISSING_OPS"]

    print(f"\n  归属错误（WRONG_OP）: {len(wrong_ops)} 条")
    for i in wrong_ops[:20]:
        print(f"    {i['kernel']}: CSV={i['csv_op']}, 实际={i['actual_ops']}")

    print(f"\n  归属不完整（MISSING_OPS）: {len(missing_ops)} 条")
    for i in missing_ops[:30]:
        missing = sorted(set(i['actual_ops']) - {i['csv_op']})
        print(f"    {i['kernel']}: CSV={i['csv_op']}, 缺少={missing}")

    print()
    print("=" * 80)
    print("4. 按源码文件汇总缺失的算子关联")
    print("=" * 80)

    source_missing = defaultdict(lambda: defaultdict(set))
    for i in missing_ops:
        for op in i["actual_ops"]:
            if op != i["csv_op"]:
                source_missing[i["source"]][op].add(i["kernel"])

    for source in sorted(source_missing.keys()):
        print(f"\n  {source}:")
        for op in sorted(source_missing[source].keys()):
            kernels = sorted(source_missing[source][op])
            print(f"    {op}: {len(kernels)} kernels -> {kernels[:5]}{'...' if len(kernels) > 5 else ''}")

    print()
    print("=" * 80)
    print("5. 影响统计：这些缺失关联导致 T2 中模型映射缺失")
    print("=" * 80)

    from t2_kernel_operator_model_mapping import OP_NAME_ALIASES, HEAT_SCORE

    T1_OP_CSV = os.path.join(BASE_DIR, "agent_file", "tmp_file", "t1_model_operator_mapping.csv")
    op_to_models = {}
    with open(T1_OP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row["算子名称"]
            model = row["模型名称"]
            heat = row["模型热度级别"]
            if not op_name:
                continue
            key = op_name.lower()
            alias_key = OP_NAME_ALIASES.get(key, key)
            if alias_key not in op_to_models:
                op_to_models[alias_key] = {}
            op_to_models[alias_key][model] = heat

    affected_kernels = set()
    affected_models = set()
    for i in missing_ops:
        for op in i["actual_ops"]:
            if op != i["csv_op"]:
                op_key = OP_NAME_ALIASES.get(op.lower(), op.lower())
                if op_key in op_to_models:
                    affected_kernels.add(i["kernel"])
                    for m in op_to_models[op_key]:
                        affected_models.add(m)

    print(f"  受影响的 kernel 数: {len(affected_kernels)}")
    print(f"  受影响可补充的模型数: {len(affected_models)}")

    print()
    print("=" * 80)
    print("6. 具体影响：每个缺失算子关联的模型")
    print("=" * 80)
    for op in sorted(set(op for i in missing_ops for op in i["actual_ops"] if op != i["csv_op"])):
        op_key = OP_NAME_ALIASES.get(op.lower(), op.lower())
        if op_key in op_to_models:
            models = sorted(op_to_models[op_key].keys())
            print(f"  {op} (key={op_key}): {len(models)} 个模型 -> {models}")


if __name__ == "__main__":
    main()
