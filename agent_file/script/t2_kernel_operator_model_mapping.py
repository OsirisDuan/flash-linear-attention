#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2: 算子/模块-Kernel映射整合
整合已有数据，建立"Kernel → 算子/模块 → 模型"的完整映射链
输出: t2_kernel_operator_model_mapping.csv
"""

import csv
import os
import re
from collections import defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")
FLA_OPS_DIR = os.path.join(BASE_DIR, "fla", "ops")

OPS_KERNELS_CSV = os.path.join(OUTPUT_DIR, "operators_kernels_mapping.csv")
T1_OP_CSV = os.path.join(OUTPUT_DIR, "t1_model_operator_mapping.csv")
T1_MOD_CSV = os.path.join(OUTPUT_DIR, "t1_model_module_mapping.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "t2_kernel_operator_model_mapping.csv")

HEAT_SCORE = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}

OP_NAME_ALIASES = {
    "deltarule": "delta_rule",
    "gateddeltarule": "gated_delta_rule",
    "gateddeltaproduct": "gated_delta_product",
    "gatedojarule": "gated_oja_rule",
    "loglinearattn": "log_linear_attn",
    "mesanet": "mesa_net",
    "pathattn": "path_attn",
    "simplegla": "simple_gla",
    "dplr": "dplr",
    "iplr": "iplr",
    "cp": "cp",
    "ttt": "ttt",
    "rwkv4": "rwkv4",
    "based": "based",
    "rebased": "rebased",
    "common": "common",
}

MODULE_OP_TO_CLASS = {
    "cross_entropy": "FusedCrossEntropyLoss",
    "linear_cross_entropy": "FusedLinearCrossEntropyLoss",
    "kl_div": "FusedKLDivLoss",
    "layernorm": "RMSNorm",
    "layernorm_gated": "LayerNormGated",
    "l2norm": "L2Norm",
    "norm_gate": "FusedRMSNormSwishGate",
    "bitlinear": "FusedBitLinear",
    "rotary": "RotaryEmbedding",
    "token_shift": "TokenShift",
    "causal_conv1d": "ShortConvolution",
    "grpo": "GRPO",
    "swiglu": "GatedMLP",
    "swish": "GatedMLP",
    "sigmoid": "GatedMLP",
    "logsigmoid": "GatedMLP",
}

LAYER_TO_MODULE_IMPORTS = {
    "abc": ["activations"],
    "gla": ["activations"],
    "hgrn": ["activations"],
    "hgrn2": ["activations", "layernorm"],
    "lightnet": ["fused_norm_gate"],
    "log_linear_mamba2": ["layernorm_gated", "convolution"],
    "mamba": ["activations", "convolution"],
    "mamba2": ["activations", "layernorm_gated", "convolution"],
    "mesa_net": ["l2norm"],
    "multiscale_retention": ["rotary"],
    "path_attn": ["l2norm"],
    "rodimus": ["layernorm_gated"],
    "rwkv6": ["activations", "token_shift"],
    "rwkv7": ["l2norm", "token_shift"],
    "simple_gla": ["activations"],
    "bitattn": ["fused_bitlinear"],
    "gsa": ["layernorm"],
}

MODULE_FILE_TO_MODULE_NAME = {
    "activations": "activations",
    "convolution": "conv",
    "fused_bitlinear": "fused_bitlinear",
    "fused_norm_gate": "fused_norm_gate",
    "l2norm": "l2norm",
    "layernorm": "layernorm",
    "layernorm_gated": "layernorm_gated",
    "rotary": "rotary",
    "token_shift": "token_shift",
}


def classify_kernel_source(source_path):
    if source_path.startswith("fla/modules/"):
        return "共享模块"
    if source_path.startswith("fla/ops/utils/"):
        return "共享模块"
    if source_path.startswith("fla/ops/common/"):
        return "共享模块"
    return "核心算子"


def get_module_name_for_kernel(op_name, source_path):
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
    return op_name


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
    """扫描 fla/ops/ 下算子之间的跨算子 import 引用（非 common/utils）
    返回: imported_op -> set(consumer_ops) 的映射
    即被引用算子 -> 使用它的算子集合
    """
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


def load_ops_kernels():
    rows = []
    with open(OPS_KERNELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_t1_operator_mapping():
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
            if key not in op_to_models:
                op_to_models[key] = {}
            op_to_models[key][model] = heat
    return op_to_models


def load_t1_module_mapping():
    mod_to_models = {}
    with open(T1_MOD_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["模块名称"]
            model = row["模型名称"]
            heat = row["模型热度级别"]
            if class_name not in mod_to_models:
                mod_to_models[class_name] = {}
            mod_to_models[class_name][model] = heat
    return mod_to_models


def build_layer_model_mapping():
    layer_to_models = {}
    with open(T1_OP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_module = row["Layer模块"]
            model = row["模型名称"]
            heat = row["模型热度级别"]
            if layer_module not in layer_to_models:
                layer_to_models[layer_module] = {}
            layer_to_models[layer_module][model] = heat
    return layer_to_models


def build_indirect_module_models(layer_to_models, mod_to_models):
    indirect = {}
    for layer_module, module_files in LAYER_TO_MODULE_IMPORTS.items():
        if layer_module not in layer_to_models:
            continue
        layer_models = layer_to_models[layer_module]
        for module_file in module_files:
            mod_name = MODULE_FILE_TO_MODULE_NAME.get(module_file, module_file)
            if mod_name not in indirect:
                indirect[mod_name] = {}
            for model, heat in layer_models.items():
                if model not in indirect[mod_name] or HEAT_SCORE.get(heat, 0) > HEAT_SCORE.get(indirect[mod_name].get(model, "D"), 0):
                    indirect[mod_name][model] = heat
    return indirect


def resolve_op_key(op_name, op_to_models):
    key = op_name.lower()
    alias_key = OP_NAME_ALIASES.get(key, key)
    if key in op_to_models:
        return key
    if alias_key in op_to_models:
        return alias_key
    return None


def resolve_models_for_kernel(op_name, source_path, op_to_models, mod_to_models,
                               indirect_mod_models, common_imports, utils_imports,
                               cross_imports):
    kernel_source = classify_kernel_source(source_path)
    module_file = get_module_name_for_kernel(op_name, source_path)

    if kernel_source == "核心算子":
        op_key = resolve_op_key(op_name, op_to_models)
        all_models = {}
        if op_key:
            all_models = dict(op_to_models[op_key])

        consumer_ops = cross_imports.get(op_name, set())
        if not consumer_ops:
            consumer_ops = cross_imports.get(op_name.lower(), set())
        if not consumer_ops:
            alias_key = OP_NAME_ALIASES.get(op_name.lower(), op_name.lower())
            consumer_ops = cross_imports.get(alias_key, set())
        for cop in consumer_ops:
            op_key_c = resolve_op_key(cop, op_to_models)
            if op_key_c:
                for m, h in op_to_models[op_key_c].items():
                    if m not in all_models or HEAT_SCORE.get(h, 0) > HEAT_SCORE.get(all_models.get(m, "D"), 0):
                        all_models[m] = h

        return all_models, kernel_source, op_name

    if source_path.startswith("fla/ops/common/"):
        consumer_ops = common_imports.get(module_file, set())
        all_models = {}
        for cop in consumer_ops:
            op_key = resolve_op_key(cop, op_to_models)
            if op_key:
                for m, h in op_to_models[op_key].items():
                    if m not in all_models or HEAT_SCORE.get(h, 0) > HEAT_SCORE.get(all_models.get(m, "D"), 0):
                        all_models[m] = h
        return all_models, kernel_source, f"common/{module_file}"

    if source_path.startswith("fla/ops/utils/"):
        consumer_ops = utils_imports.get(module_file, set())
        all_models = {}
        for cop in consumer_ops:
            op_key = resolve_op_key(cop, op_to_models)
            if op_key:
                for m, h in op_to_models[op_key].items():
                    if m not in all_models or HEAT_SCORE.get(h, 0) > HEAT_SCORE.get(all_models.get(m, "D"), 0):
                        all_models[m] = h
        if module_file in indirect_mod_models:
            for m, h in indirect_mod_models[module_file].items():
                if m not in all_models or HEAT_SCORE.get(h, 0) > HEAT_SCORE.get(all_models.get(m, "D"), 0):
                    all_models[m] = h
        return all_models, kernel_source, f"utils/{module_file}"

    if source_path.startswith("fla/modules/"):
        class_name = MODULE_OP_TO_CLASS.get(op_name)
        all_models = {}
        if class_name and class_name in mod_to_models:
            all_models = dict(mod_to_models[class_name])
        elif op_name in mod_to_models:
            all_models = dict(mod_to_models[op_name])
        elif module_file in mod_to_models:
            all_models = dict(mod_to_models[module_file])
        else:
            for cname, models in mod_to_models.items():
                if module_file.lower() in cname.lower() or cname.lower() in module_file.lower():
                    all_models = dict(models)
                    break

        if module_file in indirect_mod_models:
            for m, h in indirect_mod_models[module_file].items():
                if m not in all_models or HEAT_SCORE.get(h, 0) > HEAT_SCORE.get(all_models.get(m, "D"), 0):
                    all_models[m] = h
        return all_models, kernel_source, module_file

    return {}, kernel_source, module_file


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("步骤 2.1: 读取已有核心算子映射数据")
    print("=" * 60)
    ops_kernels = load_ops_kernels()
    print(f"  读取 {len(ops_kernels)} 条 kernel 记录")

    print()
    print("=" * 60)
    print("步骤 2.2: 扫描 common/utils import 链 + 跨算子引用")
    print("=" * 60)
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

    print()
    print("=" * 60)
    print("步骤 2.3-2.4: 整合模型信息")
    print("=" * 60)
    op_to_models = load_t1_operator_mapping()
    mod_to_models = load_t1_module_mapping()
    layer_to_models = build_layer_model_mapping()
    indirect_mod_models = build_indirect_module_models(layer_to_models, mod_to_models)
    print(f"  核心算子映射: {len(op_to_models)} 个算子")
    print(f"  共享模块映射(直接): {len(mod_to_models)} 个模块类")
    print(f"  Layer→模块间接映射: {len(indirect_mod_models)} 个模块")

    print()
    print("  间接映射详情:")
    for mod_name in sorted(indirect_mod_models.keys()):
        models = sorted(indirect_mod_models[mod_name].keys())
        print(f"    {mod_name}: {len(models)} 个模型")

    print()
    print("=" * 60)
    print("步骤 2.5: 合并输出")
    print("=" * 60)

    results = []
    unmatched_ops = set()

    for row in ops_kernels:
        kernel_name = row["Kernel函数名称"]
        kernel_type = row.get("kernel类型", row.get("kernel 类型", "")).strip()
        source_path = row["Kernel源码路径"]
        line_no = row["Kernel行号"]
        op_name = row["算子名称"]

        models_heat, kernel_source, module_name = resolve_models_for_kernel(
            op_name, source_path, op_to_models, mod_to_models,
            indirect_mod_models, common_imports, utils_imports, cross_imports
        )

        if not models_heat:
            unmatched_ops.add(f"{op_name} ({source_path})")

        model_list = sorted(models_heat.keys())
        heat_list = [models_heat[m] for m in model_list]
        max_heat = max(heat_list) if heat_list else ""
        heat_score = HEAT_SCORE.get(max_heat, 0) if max_heat else 0

        results.append({
            "Kernel函数名称": kernel_name,
            "kernel类型": kernel_type,
            "kernel来源": kernel_source,
            "所属算子/模块": module_name,
            "使用的模型列表": ",".join(model_list),
            "各模型热度级别": ",".join(heat_list),
            "最高模型热度级别": max_heat,
            "模型热度得分": heat_score,
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })

    fieldnames = [
        "Kernel函数名称", "kernel类型", "kernel来源", "所属算子/模块",
        "使用的模型列表", "各模型热度级别", "最高模型热度级别",
        "模型热度得分", "Kernel源码路径", "Kernel行号"
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  输出 {len(results)} 条记录")

    print()
    print("=" * 60)
    print("T2 执行结果汇总")
    print("=" * 60)

    total = len(results)
    core_kernels = [r for r in results if r["kernel来源"] == "核心算子"]
    shared_kernels = [r for r in results if r["kernel来源"] == "共享模块"]
    with_models = [r for r in results if r["使用的模型列表"]]
    without_models = [r for r in results if not r["使用的模型列表"]]

    print(f"  总 kernel 数: {total}")
    print(f"  核心算子 kernel: {len(core_kernels)}")
    print(f"  共享模块 kernel: {len(shared_kernels)}")
    print(f"  有模型映射: {len(with_models)}")
    print(f"  无模型映射: {len(without_models)}")

    unique_core = len(set(r["Kernel函数名称"] for r in core_kernels))
    unique_shared = len(set(r["Kernel函数名称"] for r in shared_kernels))
    print(f"  去重核心算子 kernel: {unique_core}")
    print(f"  去重共享模块 kernel: {unique_shared}")

    print(f"\n按来源统计:")
    for source in ["核心算子", "共享模块"]:
        recs = [r for r in results if r["kernel来源"] == source]
        with_m = [r for r in recs if r["使用的模型列表"]]
        print(f"  {source}: {len(recs)} 条, 有模型映射 {len(with_m)} 条")

    print(f"\n按最高热度级别统计:")
    for level in ["S", "A", "B", "C", "D"]:
        recs = [r for r in results if r["最高模型热度级别"] == level]
        if recs:
            print(f"  {level}级: {len(recs)} 条 kernel")

    no_heat = [r for r in results if not r["最高模型热度级别"]]
    if no_heat:
        print(f"  无热度: {len(no_heat)} 条 kernel")

    print(f"\n无模型映射的算子/模块:")
    for op in sorted(unmatched_ops):
        print(f"  {op}")

    print(f"\n共享模块 kernel 按模块统计:")
    shared_by_module = {}
    for r in shared_kernels:
        mod = r["所属算子/模块"]
        if mod not in shared_by_module:
            shared_by_module[mod] = {"count": 0, "models": set()}
        shared_by_module[mod]["count"] += 1
        if r["使用的模型列表"]:
            for m in r["使用的模型列表"].split(","):
                shared_by_module[mod]["models"].add(m)
    for mod in sorted(shared_by_module.keys()):
        info = shared_by_module[mod]
        print(f"  {mod}: {info['count']} kernels, {len(info['models'])} 个模型")

    print(f"\n输出文件: {OUTPUT_CSV}")

    print()
    print("=" * 60)
    print("质量验证")
    print("=" * 60)

    print(f"  [覆盖率] 总 kernel 数: {total} (核心 {len(core_kernels)} + 共享 {len(shared_kernels)})")
    print(f"  [覆盖率] 有模型映射比例: {len(with_models)}/{total} = {len(with_models)/total*100:.1f}%")

    s_level_kernels = [r for r in results if r["最高模型热度级别"] == "S"]
    if s_level_kernels:
        print(f"  [热度验证] S级 kernel 数: {len(s_level_kernels)}")
        for r in s_level_kernels[:5]:
            print(f"    {r['Kernel函数名称']}: {r['使用的模型列表']}")

    shared_in_modules = [r for r in results if r["kernel来源"] == "共享模块" and "fla/modules/" in r["Kernel源码路径"]]
    print(f"  [来源验证] fla/modules/ 下的 kernel: {len(shared_in_modules)} 条 (预期约 43)")

    shared_in_common = [r for r in results if "fla/ops/common/" in r["Kernel源码路径"]]
    shared_in_common_with = [r for r in shared_in_common if r["使用的模型列表"]]
    print(f"  [来源验证] fla/ops/common/ 下的 kernel: {len(shared_in_common)} 条, 有模型映射 {len(shared_in_common_with)} 条")

    shared_in_utils = [r for r in results if "fla/ops/utils/" in r["Kernel源码路径"]]
    shared_in_utils_with = [r for r in shared_in_utils if r["使用的模型列表"]]
    print(f"  [来源验证] fla/ops/utils/ 下的 kernel: {len(shared_in_utils)} 条, 有模型映射 {len(shared_in_utils_with)} 条")

    core_no_model = [r for r in core_kernels if not r["使用的模型列表"]]
    print(f"  [核心算子] 无模型映射: {len(core_no_model)} 条")
    core_ops_no_model = sorted(set(r["所属算子/模块"] for r in core_no_model))
    if core_ops_no_model:
        print(f"    涉及算子: {', '.join(core_ops_no_model)}")

    print()
    print("=" * 60)
    print("关键验证：chunk_fwd_kernel_o 归属")
    print("=" * 60)
    for r in results:
        if r["Kernel函数名称"] == "chunk_fwd_kernel_o":
            print(f"  kernel: {r['Kernel函数名称']}")
            print(f"  来源: {r['kernel来源']}")
            print(f"  所属算子/模块: {r['所属算子/模块']}")
            print(f"  使用的模型列表: {r['使用的模型列表']}")
            print(f"  最高模型热度级别: {r['最高模型热度级别']}")
            print(f"  路径: {r['Kernel源码路径']}:{r['Kernel行号']}")
            print()

    print()
    print("=" * 60)
    print("跨算子引用验证")
    print("=" * 60)
    cross_ref_kernels = set()
    for imported_op, consumers in cross_imports.items():
        for r in results:
            if r["所属算子/模块"].lower() == imported_op.lower() and r["kernel来源"] == "核心算子":
                csv_models = set(r["使用的模型列表"].split(",")) if r["使用的模型列表"] else set()
                for cop in consumers:
                    cop_key = resolve_op_key(cop, op_to_models)
                    if cop_key:
                        cop_models = set(op_to_models[cop_key].keys())
                        missing = cop_models - csv_models
                        if missing:
                            cross_ref_kernels.add(r["Kernel函数名称"])
    if cross_ref_kernels:
        print(f"  受跨算子引用影响的 kernel: {len(cross_ref_kernels)} 个")
        for kn in sorted(cross_ref_kernels):
            print(f"    {kn}")
    else:
        print(f"  无跨算子引用遗漏")


if __name__ == "__main__":
    main()
