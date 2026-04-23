#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1: 模型-算子/模块映射分析（v4）
建立 模型 → Layer → 核心算子 + 模型 → 共享模块 的完整映射关系
输出:
  - t1_model_operator_mapping.csv: 核心算子映射
  - t1_model_module_mapping.csv: 共享模块映射
"""

import ast
import csv
import os
import re

BASE_DIR = r"d:\code\flash-linear-attention"
MODELS_DIR = os.path.join(BASE_DIR, "fla", "models")
LAYERS_DIR = os.path.join(BASE_DIR, "fla", "layers")
MODULES_DIR = os.path.join(BASE_DIR, "fla", "modules")
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")
SCRIPT_DIR = os.path.join(BASE_DIR, "agent_file", "script")

OP_OUTPUT = os.path.join(OUTPUT_DIR, "t1_model_operator_mapping.csv")
MOD_OUTPUT = os.path.join(OUTPUT_DIR, "t1_model_module_mapping.csv")

MODEL_HEAT_MAP = {
    "GatedDeltaNet": "S",
    "KDA": "S",
    "MLA": "S",
    "NSA": "A",
    "Mamba2": "A",
    "GLA": "A",
    "DeltaNet": "A",
    "RetNet": "A",
    "HGRN": "A",
    "Samba": "A",
    "ForgettingTransformer": "A",
    "GSA": "B",
    "RWKV6": "B",
    "HGRN2": "B",
    "Based": "B",
    "Rebased": "B",
    "BitNet": "B",
    "LightNet": "B",
    "LinearAttention": "B",
    "RWKV7": "C",
    "MesaNet": "C",
    "Comba": "C",
    "PaTHAttention": "C",
    "MoM": "C",
    "Rodimus": "C",
    "DeltaFormer": "C",
    "GatedDeltaProduct": "C",
    "LogLinearMamba2": "C",
    "Mamba": "C",
    "SimpleGLA": "C",
    "ABC": "D",
    "Transformer": "D",
}

LAYER_CLASS_TO_MODULE = {
    "ABCAttention": "abc",
    "Attention": "attn",
    "BasedLinearAttention": "based",
    "BitAttention": "bitattn",
    "Comba": "comba",
    "DeltaNet": "delta_net",
    "DeltaFormerAttention": "deltaformer",
    "ForgettingAttention": "forgetting_attn",
    "GatedDeltaNet": "gated_deltanet",
    "GatedDeltaProduct": "gated_deltaproduct",
    "GatedLinearAttention": "gla",
    "GatedSlotAttention": "gsa",
    "HGRNAttention": "hgrn",
    "HGRN2Attention": "hgrn2",
    "KimiDeltaAttention": "kda",
    "LightNetAttention": "lightnet",
    "LinearAttention": "linear_attn",
    "LogLinearMamba2": "log_linear_mamba2",
    "Mamba": "mamba",
    "Mamba2": "mamba2",
    "MesaNet": "mesa_net",
    "MomAttention": "mom",
    "MultiScaleRetention": "multiscale_retention",
    "MultiheadLatentAttention": "mla",
    "NativeSparseAttention": "nsa",
    "PaTHAttention": "path_attn",
    "RWKV6Attention": "rwkv6",
    "RWKV7Attention": "rwkv7",
    "ReBasedLinearAttention": "rebased",
    "RodimusAttention": "rodimus",
    "SlidingWindowSharedKeyAttention": "rodimus",
}

AUXILIARY_LAYER_CLASSES = {"LerpLinear", "align_multiple"}
AUXILIARY_OPS = {"utils"}

EXTERNAL_LIB_LAYERS = {
    "attn": "flash_attn",
    "bitattn": "flash_attn",
    "mla": "flash_attn",
    "mamba": "mamba_ssm",
    "mamba2": "mamba_ssm",
}

MODULE_CLASS_TO_SOURCE = {
    "FusedCrossEntropyLoss": "fla/modules/fused_cross_entropy.py",
    "FusedLinearCrossEntropyLoss": "fla/modules/fused_linear_cross_entropy.py",
    "FusedKLDivLoss": "fla/modules/fused_kl_div.py",
    "RMSNorm": "fla/modules/layernorm.py",
    "RMSNormLinear": "fla/modules/layernorm.py",
    "LayerNorm": "fla/modules/layernorm.py",
    "LayerNormLinear": "fla/modules/layernorm.py",
    "GroupNorm": "fla/modules/layernorm.py",
    "GroupNormLinear": "fla/modules/layernorm.py",
    "GatedMLP": "fla/modules/mlp.py",
    "BitLinear": "fla/modules/fused_bitlinear.py",
    "FusedBitLinear": "fla/modules/fused_bitlinear.py",
    "RotaryEmbedding": "fla/modules/rotary.py",
    "TokenShift": "fla/modules/token_shift.py",
    "L2Norm": "fla/modules/l2norm.py",
    "ImplicitLongConvolution": "fla/modules/convolution.py",
    "LongConvolution": "fla/modules/convolution.py",
    "ShortConvolution": "fla/modules/convolution.py",
    "FusedRMSNormSwishGate": "fla/modules/fused_norm_gate.py",
    "FusedRMSNormSwishGateNorm": "fla/modules/fused_norm_gate.py",
    "FusedLayerNormSwishGate": "fla/modules/fused_norm_gate.py",
    "FusedLayerNormSwishGateNorm": "fla/modules/fused_norm_gate.py",
    "LayerNormGated": "fla/modules/layernorm_gated.py",
}


def parse_imports_from_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append((module, names))
    except SyntaxError:
        for line in content.splitlines():
            line = line.strip()
            m = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", line)
            if m:
                module = m.group(1)
                names_str = m.group(2)
                names = [n.strip().split(" as ")[0].strip() for n in names_str.split(",")]
                imports.append((module, names))
    return imports


def count_triton_kernels(filepath):
    if not os.path.isfile(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return len(re.findall(r"@triton\.jit", content))


def get_module_source_path(class_name, module_subpath=None):
    if module_subpath:
        full_path = os.path.join(BASE_DIR, "fla", "modules", module_subpath)
        if os.path.isfile(full_path):
            return "fla/modules/" + module_subpath
    return MODULE_CLASS_TO_SOURCE.get(class_name, "fla/modules")


def resolve_module_file(class_name, module_str):
    if module_str == "fla.modules":
        return get_module_source_path(class_name)
    elif module_str.startswith("fla.modules."):
        sub = module_str.replace("fla.modules.", "")
        parts = sub.split(".")
        if len(parts) >= 2:
            candidate = sub + ".py"
            full = os.path.join(MODULES_DIR, candidate)
            if os.path.isfile(full):
                return "fla/modules/" + candidate
        candidate = parts[0] + ".py"
        full = os.path.join(MODULES_DIR, candidate)
        if os.path.isfile(full):
            return "fla/modules/" + candidate
        sub_dir = os.path.join(MODULES_DIR, parts[0])
        if os.path.isdir(sub_dir):
            for sub_file in os.listdir(sub_dir):
                if sub_file.endswith(".py") and not sub_file.startswith("__"):
                    fp = os.path.join(sub_dir, sub_file)
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        if re.search(rf"\b{re.escape(class_name)}\b", f.read()):
                            return f"fla/modules/{parts[0]}/{sub_file}"
            return f"fla/modules/{parts[0]}"
    return "fla/modules"


def get_model_name(dir_name):
    mapping = {
        "abc": "ABC",
        "bitnet": "BitNet",
        "comba": "Comba",
        "delta_net": "DeltaNet",
        "deltaformer": "DeltaFormer",
        "forgetting_transformer": "ForgettingTransformer",
        "gated_deltanet": "GatedDeltaNet",
        "gated_deltaproduct": "GatedDeltaProduct",
        "gla": "GLA",
        "gsa": "GSA",
        "hgrn": "HGRN",
        "hgrn2": "HGRN2",
        "kda": "KDA",
        "lightnet": "LightNet",
        "linear_attn": "LinearAttention",
        "log_linear_mamba2": "LogLinearMamba2",
        "mamba": "Mamba",
        "mamba2": "Mamba2",
        "mesa_net": "MesaNet",
        "mla": "MLA",
        "mom": "MoM",
        "nsa": "NSA",
        "path_attn": "PaTHAttention",
        "retnet": "RetNet",
        "rodimus": "Rodimus",
        "rwkv6": "RWKV6",
        "rwkv7": "RWKV7",
        "samba": "Samba",
        "transformer": "Transformer",
    }
    return mapping.get(dir_name, dir_name)


def get_layer_modules_from_model(model_dir):
    layer_modules = []
    for fname in os.listdir(model_dir):
        if fname.startswith("modeling_") and fname.endswith(".py"):
            filepath = os.path.join(model_dir, fname)
            imports = parse_imports_from_file(filepath)
            for module, names in imports:
                if module == "fla.layers":
                    for name in names:
                        if name in AUXILIARY_LAYER_CLASSES:
                            continue
                        layer_module = LAYER_CLASS_TO_MODULE.get(name, name.lower())
                        layer_modules.append((layer_module, name))
                elif module.startswith("fla.layers."):
                    layer_module = module.replace("fla.layers.", "").split(".")[0]
                    for name in names:
                        if name in AUXILIARY_LAYER_CLASSES:
                            continue
                        layer_modules.append((layer_module, name))
    return layer_modules


def get_modules_from_model(model_dir):
    modules = []
    for fname in os.listdir(model_dir):
        if fname.startswith("modeling_") and fname.endswith(".py"):
            filepath = os.path.join(model_dir, fname)
            imports = parse_imports_from_file(filepath)
            for module, names in imports:
                if module == "fla.modules":
                    for name in names:
                        modules.append((name, module))
                elif module.startswith("fla.modules."):
                    for name in names:
                        modules.append((name, module))
    return modules


def get_ops_from_layer(layer_module):
    layer_filepath = os.path.join(LAYERS_DIR, layer_module + ".py")
    if not os.path.isfile(layer_filepath):
        return []
    imports = parse_imports_from_file(layer_filepath)
    ops = []
    for module, names in imports:
        if not module.startswith("fla.ops."):
            continue
        ops_path = module.replace("fla.ops.", "")
        ops_module = ops_path.split(".")[0]
        if ops_module in AUXILIARY_OPS:
            continue
        ops.append({
            "算子名称": ops_module,
            "算子源码路径": "fla/ops/" + ops_module,
            "算子函数": ", ".join(names),
        })
    return ops


def get_layer_type(layer_module, layer_class):
    if layer_class == "Attention":
        return "标准注意力"
    if layer_module in EXTERNAL_LIB_LAYERS:
        return f"外部库({EXTERNAL_LIB_LAYERS[layer_module]})"
    return "fla核心算子"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    op_results = []
    mod_results = []

    model_dirs = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d)) and not d.startswith("__")
    ])

    print(f"发现 {len(model_dirs)} 个模型目录\n")
    print("=" * 60)
    print("步骤 1.1-1.3: 解析核心算子链")
    print("=" * 60)

    for dir_name in model_dirs:
        model_name = get_model_name(dir_name)
        model_dir = os.path.join(MODELS_DIR, dir_name)
        heat_level = MODEL_HEAT_MAP.get(model_name, "C")

        layer_modules = get_layer_modules_from_model(model_dir)
        if not layer_modules:
            print(f"  {model_name} ({heat_level}): 未找到 layer import")

        for layer_module, layer_class in layer_modules:
            layer_type = get_layer_type(layer_module, layer_class)
            ops_list = get_ops_from_layer(layer_module)

            if ops_list:
                for op in ops_list:
                    op_results.append({
                        "模型名称": model_name,
                        "模型热度级别": heat_level,
                        "Layer名称": layer_class,
                        "Layer模块": layer_module,
                        "Layer类型": layer_type,
                        "算子名称": op["算子名称"],
                        "算子源码路径": op["算子源码路径"],
                        "算子函数": op["算子函数"],
                    })
            else:
                op_results.append({
                    "模型名称": model_name,
                    "模型热度级别": heat_level,
                    "Layer名称": layer_class,
                    "Layer模块": layer_module,
                    "Layer类型": layer_type,
                    "算子名称": "",
                    "算子源码路径": "",
                    "算子函数": "",
                })

    seen = set()
    unique_op_results = []
    for r in op_results:
        key = (r["模型名称"], r["Layer名称"], r["算子名称"])
        if key not in seen:
            seen.add(key)
            unique_op_results.append(r)

    print(f"  核心算子映射记录: {len(unique_op_results)} 条")
    core_records = [r for r in unique_op_results if r["算子名称"]]
    op_count = len(set(r["算子名称"] for r in core_records))
    print(f"  涉及核心算子数: {op_count}")

    print()
    print("=" * 60)
    print("步骤 1.4-1.6: 解析共享模块链")
    print("=" * 60)

    for dir_name in model_dirs:
        model_name = get_model_name(dir_name)
        model_dir = os.path.join(MODELS_DIR, dir_name)
        heat_level = MODEL_HEAT_MAP.get(model_name, "C")

        modules = get_modules_from_model(model_dir)

        for class_name, module_str in modules:
            source_path = resolve_module_file(class_name, module_str)
            abs_source = os.path.join(BASE_DIR, source_path)

            has_triton = False
            kernel_count = 0

            if os.path.isfile(abs_source):
                kernel_count = count_triton_kernels(abs_source)
                has_triton = kernel_count > 0
            elif os.path.isdir(abs_source):
                for root, dirs, files in os.walk(abs_source):
                    for f in files:
                        if f.endswith(".py"):
                            fp = os.path.join(root, f)
                            kc = count_triton_kernels(fp)
                            if kc > 0:
                                has_triton = True
                                kernel_count += kc

            mod_results.append({
                "模型名称": model_name,
                "模型热度级别": heat_level,
                "模块名称": class_name,
                "模块源码路径": source_path,
                "含TritonKernel": has_triton,
                "Kernel数量": kernel_count,
            })

    seen_mod = set()
    unique_mod_results = []
    for r in mod_results:
        key = (r["模型名称"], r["模块名称"])
        if key not in seen_mod:
            seen_mod.add(key)
            unique_mod_results.append(r)

    print(f"  共享模块映射记录: {len(unique_mod_results)} 条")
    triton_mods = set(r["模块名称"] for r in unique_mod_results if r["含TritonKernel"])
    print(f"  含 Triton Kernel 的模块: {len(triton_mods)} 个")
    total_kernels = sum(r["Kernel数量"] for r in unique_mod_results if r["Kernel数量"] > 0)
    print(f"  共享模块中 Triton Kernel 总数: {total_kernels}")

    op_fieldnames = [
        "模型名称", "模型热度级别", "Layer名称", "Layer模块",
        "Layer类型", "算子名称", "算子源码路径", "算子函数"
    ]
    with open(OP_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=op_fieldnames)
        writer.writeheader()
        writer.writerows(unique_op_results)

    mod_fieldnames = [
        "模型名称", "模型热度级别", "模块名称",
        "模块源码路径", "含TritonKernel", "Kernel数量"
    ]
    with open(MOD_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mod_fieldnames)
        writer.writeheader()
        writer.writerows(unique_mod_results)

    print()
    print("=" * 60)
    print("T1 执行结果汇总")
    print("=" * 60)

    model_count = len(set(r["模型名称"] for r in unique_op_results))
    print(f"模型目录数: {len(model_dirs)}")
    print(f"涉及模型数: {model_count}")
    print(f"核心算子映射记录: {len(unique_op_results)}")
    print(f"共享模块映射记录: {len(unique_mod_results)}")

    print(f"\n按热度级别统计（核心算子映射）:")
    for level in ["S", "A", "B", "C", "D"]:
        records = [r for r in core_records if r["模型热度级别"] == level]
        models_at_level = sorted(set(r["模型名称"] for r in records))
        if models_at_level:
            print(f"  {level}级: {len(records)}条映射, 涉及模型: {', '.join(models_at_level)}")

    print(f"\n各模型使用的核心算子:")
    model_ops = {}
    for r in unique_op_results:
        if r["算子名称"]:
            model_ops.setdefault(r["模型名称"], set()).add(r["算子名称"])
    for model in sorted(model_ops.keys()):
        ops = sorted(model_ops[model])
        print(f"  {model}: {', '.join(ops)}")

    no_ops_models = sorted(set(
        r["模型名称"] for r in unique_op_results
        if not any(rr["算子名称"] for rr in unique_op_results if rr["模型名称"] == r["模型名称"])
    ))
    if no_ops_models:
        print(f"\n完全无核心算子的模型（仅使用外部库或标准注意力）:")
        for m in no_ops_models:
            layers = sorted(set(
                f"{r['Layer名称']}({r['Layer类型']})"
                for r in unique_op_results if r["模型名称"] == m
            ))
            print(f"  {m}: {', '.join(layers)}")

    print(f"\n算子被模型使用统计:")
    op_models = {}
    for r in unique_op_results:
        if r["算子名称"]:
            op_models.setdefault(r["算子名称"], set()).add(r["模型名称"])
    for op in sorted(op_models.keys(), key=lambda x: -len(op_models[x])):
        models = sorted(op_models[op])
        print(f"  {op}: {len(models)}个模型 -> {', '.join(models)}")

    print(f"\n各模型使用的共享模块:")
    model_mods = {}
    for r in unique_mod_results:
        model_mods.setdefault(r["模型名称"], []).append(r)
    for model in sorted(model_mods.keys()):
        mods = model_mods[model]
        triton_mods_list = [m for m in mods if m["含TritonKernel"]]
        non_triton = [m for m in mods if not m["含TritonKernel"]]
        parts = []
        for m in triton_mods_list:
            parts.append(f"{m['模块名称']}(T:{m['Kernel数量']})")
        for m in non_triton:
            parts.append(f"{m['模块名称']}")
        print(f"  {model}: {', '.join(parts)}")

    print(f"\n含 Triton Kernel 的共享模块详情:")
    triton_mod_detail = {}
    for r in unique_mod_results:
        if r["含TritonKernel"]:
            triton_mod_detail.setdefault(r["模块名称"], {
                "source": r["模块源码路径"],
                "kernel_count": r["Kernel数量"],
                "models": set()
            })
            triton_mod_detail[r["模块名称"]]["models"].add(r["模型名称"])
    for mod_name in sorted(triton_mod_detail.keys()):
        info = triton_mod_detail[mod_name]
        models = sorted(info["models"])
        print(f"  {mod_name}: {info['kernel_count']} kernels, {len(models)}个模型 -> {', '.join(models)}")

    print(f"\n输出文件:")
    print(f"  核心算子映射: {OP_OUTPUT}")
    print(f"  共享模块映射: {MOD_OUTPUT}")

    print()
    print("=" * 60)
    print("质量验证")
    print("=" * 60)

    print(f"  [完整性] 模型目录数: {len(model_dirs)} (期望 >= 29)")
    all_models = set(r["模型名称"] for r in unique_op_results) | set(r["模型名称"] for r in unique_mod_results)
    print(f"  [完整性] 映射到的模型数: {len(all_models)}")

    expected_common = {"RMSNorm", "FusedCrossEntropyLoss", "GatedMLP"}
    for model in sorted(all_models):
        model_mod_set = set(r["模块名称"] for r in unique_mod_results if r["模型名称"] == model)
        missing = expected_common - model_mod_set
        if missing:
            print(f"  [模块映射] {model}: 缺少常见模块 {missing}")

    triton_mod_names = set(r["模块名称"] for r in unique_mod_results if r["含TritonKernel"])
    print(f"  [共享模块覆盖率] 含@triton.jit的模块: {len(triton_mod_names)} 个")
    print(f"    {', '.join(sorted(triton_mod_names))}")


if __name__ == "__main__":
    main()
