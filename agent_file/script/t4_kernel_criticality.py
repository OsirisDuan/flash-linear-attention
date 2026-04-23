#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4: Kernel核心程度判定
判定每个 kernel 在模型执行路径中的位置和重要性
输出: t4_kernel_criticality.csv
"""

import csv
import os
import re
from collections import Counter, defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")

T2_CSV = os.path.join(OUTPUT_DIR, "t2_kernel_operator_model_mapping.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "t4_kernel_criticality.csv")


def classify_core_kernel(name, kernel_type):
    """
    核心算子 kernel 分类规则
    
    规则优先级（从高到低）：
    1. naive → 边缘(1分)
    2. fused + fwd(无intra/inter/preprocess) → 核心(5分)
    3. fused + bwd → 重要(4分)
    4. fused → 优化(2分)
    5. cumsum/preprocess/mask → 辅助(3分)
    6. bwd → 重要(4分)
    7. fwd(无intra/inter/preprocess) → 核心(5分)
    8. fwd(有intra/inter/preprocess) → 辅助(3分)
    9. 未匹配 + cv/cube → 核心(5分)
    10. 未匹配 + vv → 辅助(3分)
    """
    name_lower = name.lower()
    
    # 规则1: naive → 边缘
    if 'naive' in name_lower:
        return '边缘', 1, '含naive'
    
    # 规则2-4: fused 处理
    if 'fused' in name_lower:
        if 'fwd' in name_lower and 'intra' not in name_lower and 'inter' not in name_lower and 'preprocess' not in name_lower:
            return '核心', 5, 'fused+fwd(无intra/inter/preprocess)'
        elif 'bwd' in name_lower:
            return '重要', 4, 'fused+bwd'
        else:
            return '优化', 2, '含fused'
    
    # 规则5: cumsum/preprocess/mask → 辅助
    if 'cumsum' in name_lower or 'preprocess' in name_lower or 'mask' in name_lower:
        return '辅助', 3, '含cumsum/preprocess/mask'
    
    # 规则6: bwd → 重要
    if 'bwd' in name_lower:
        return '重要', 4, '含bwd'
    
    # 规则7-8: fwd 处理
    if 'fwd' in name_lower:
        if 'intra' in name_lower or 'inter' in name_lower or 'preprocess' in name_lower:
            return '重要', 4, 'fwd+intra/inter/preprocess(前向分步)'
        else:
            return '核心', 5, 'fwd(无intra/inter/preprocess)'
    
    # 规则9-10: 未匹配，根据 kernel 类型判断
    # 补充规则：排序工具
    if any(kw in name_lower for kw in ['sort', 'merge', 'swap', 'argsort', 'bitonic']):
        return '辅助', 3, '排序工具'
    
    # 补充规则：数据搬运/预处理
    if any(kw in name_lower for kw in ['copy', 'save', 'prepare', 'gather']):
        return '辅助', 3, '数据搬运/预处理'
    
    # 补充规则：topk 操作
    if 'topk' in name_lower:
        return '辅助', 3, 'topk操作'
    
    # 补充规则：解码步骤
    if 'decoding' in name_lower:
        return '核心', 5, '解码步骤'
    
    # 补充规则：mixing 操作
    if 'mix' in name_lower:
        return '核心', 5, 'mixing操作'
    
    # 补充规则：update 操作
    if 'update' in name_lower:
        if kernel_type == 'cv/cube':
            return '核心', 5, 'update+cv/cube'
        else:
            return '辅助', 3, 'update+vv'
    
    # 补充规则：gate 操作
    if 'gate' in name_lower:
        if kernel_type == 'cv/cube':
            return '核心', 5, 'gate+cv/cube'
        else:
            return '辅助', 3, 'gate+vv'
    
    # 默认：根据 kernel 类型
    if kernel_type == 'cv/cube':
        return '核心', 5, 'cv/cube类型默认核心'
    else:
        return '辅助', 3, 'vv类型默认辅助'


def classify_shared_kernel(name, kernel_type):
    """
    共享模块 kernel 分类规则
    
    所有共享模块 kernel 默认为辅助(3分)，但可根据函数名进一步细分
    """
    name_lower = name.lower()
    
    # 规则：norm 相关
    if any(kw in name_lower for kw in ['norm', 'rms', 'l2norm']):
        return '辅助', 3, '归一化组件'
    
    # 规则：损失函数
    if any(kw in name_lower for kw in ['cross_entropy', 'kl_div', 'linear_cross_entropy', 'grpo']):
        return '辅助', 3, '损失函数'
    
    # 规则：卷积组件
    if any(kw in name_lower for kw in ['conv', 'causal']):
        return '辅助', 3, '卷积组件'
    
    # 规则：激活函数
    if any(kw in name_lower for kw in ['activation', 'swiglu', 'swish', 'sigmoid', 'logsigmoid', 'softplus', 'relu', 'tanh', 'exp', 'log']):
        return '辅助', 3, '激活函数'
    
    # 规则：位置编码/辅助操作
    if any(kw in name_lower for kw in ['rotary', 'token_shift', 'l2', 'position']):
        return '辅助', 3, '位置编码/辅助操作'
    
    # 规则：门控组件
    if any(kw in name_lower for kw in ['gate', 'norm_gated']):
        return '辅助', 3, '门控组件'
    
    # 规则：cumsum
    if 'cumsum' in name_lower:
        return '辅助', 3, 'cumsum操作'
    
    # 规则：chunk 相关（可能是核心计算）
    if 'chunk' in name_lower:
        if 'fwd' in name_lower and 'intra' not in name_lower and 'inter' not in name_lower:
            if kernel_type == 'cv/cube':
                return '辅助', 3, '共享模块chunk+fwd+cv/cube(降级为辅助)'
            else:
                return '辅助', 3, '共享模块chunk+fwd+vv'
        elif 'bwd' in name_lower:
            return '辅助', 3, '共享模块chunk+bwd'
        else:
            return '辅助', 3, '共享模块chunk操作'
    
    # 规则：fused 相关
    if 'fused' in name_lower:
        return '辅助', 3, '共享模块fused操作'
    
    # 默认：辅助
    return '辅助', 3, '共享模块默认辅助'


def main():
    print("=" * 60)
    print("T4: Kernel核心程度判定")
    print("=" * 60)
    
    print("\n步骤 4.1: 读取 T2 数据")
    t2_rows = []
    with open(T2_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t2_rows.append(row)
    print(f"  读取 {len(t2_rows)} 条 kernel 记录")
    
    print("\n步骤 4.2: 判定每个 kernel 的核心程度")
    results = []
    for row in t2_rows:
        kernel_name = row["Kernel函数名称"]
        source_path = row["Kernel源码路径"]
        line_no = row["Kernel行号"]
        kernel_source = row["kernel来源"]
        kernel_type = row["kernel类型"]
        
        if kernel_source == "核心算子":
            level, score, reason = classify_core_kernel(kernel_name, kernel_type)
        else:
            level, score, reason = classify_shared_kernel(kernel_name, kernel_type)
        
        results.append({
            "Kernel函数名称": kernel_name,
            "kernel来源": kernel_source,
            "函数名特征": reason,
            "kernel类型": kernel_type,
            "核心程度级别": level,
            "核心程度得分": score,
            "判定依据": reason,
            "Kernel源码路径": source_path,
            "Kernel行号": line_no,
        })
    
    print(f"  处理 {len(results)} 条记录")
    
    print("\n步骤 4.3: 输出结果")
    fieldnames = [
        "Kernel函数名称", "kernel来源", "函数名特征", "kernel类型",
        "核心程度级别", "核心程度得分", "判定依据",
        "Kernel源码路径", "Kernel行号"
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  输出文件: {OUTPUT_CSV}")
    
    print("\n" + "=" * 60)
    print("T4 执行结果汇总")
    print("=" * 60)
    
    total = len(results)
    core_kernels = [r for r in results if r["kernel来源"] == "核心算子"]
    shared_kernels = [r for r in results if r["kernel来源"] == "共享模块"]
    
    print(f"\n总体统计:")
    print(f"  总 kernel 数: {total}")
    print(f"  核心算子 kernel: {len(core_kernels)}")
    print(f"  共享模块 kernel: {len(shared_kernels)}")
    
    print(f"\n核心程度分布:")
    for level in ["核心", "重要", "辅助", "优化", "边缘"]:
        count = sum(1 for r in results if r["核心程度级别"] == level)
        print(f"  {level}: {count}")
    
    print(f"\n核心算子 kernel 核心程度分布:")
    for level in ["核心", "重要", "辅助", "优化", "边缘"]:
        count = sum(1 for r in core_kernels if r["核心程度级别"] == level)
        print(f"  {level}: {count}")
    
    print(f"\n共享模块 kernel 核心程度分布:")
    for level in ["核心", "重要", "辅助", "优化", "边缘"]:
        count = sum(1 for r in shared_kernels if r["核心程度级别"] == level)
        print(f"  {level}: {count}")
    
    print(f"\n核心程度得分分布:")
    for score in range(1, 6):
        count = sum(1 for r in results if r["核心程度得分"] == score)
        print(f"  {score}分: {count}")
    
    print(f"\n判定依据统计:")
    reason_counter = Counter(r["判定依据"] for r in results)
    for reason, count in reason_counter.most_common(15):
        print(f"  {reason}: {count}")
    
    print("\n" + "=" * 60)
    print("质量验证")
    print("=" * 60)
    
    # 验证1: 同一算子的 fwd kernel 应比 bwd kernel 核心程度更高或相等
    print("\n验证1: fwd vs bwd 核心程度一致性")
    op_kernels = defaultdict(list)
    for r in core_kernels:
        op = r["Kernel源码路径"].split("/")[2] if "/" in r["Kernel源码路径"] else "unknown"
        op_kernels[op].append(r)
    
    inconsistent = []
    for op, kernels in op_kernels.items():
        fwd_kernels = [k for k in kernels if 'fwd' in k["Kernel函数名称"].lower() and 'bwd' not in k["Kernel函数名称"].lower()]
        bwd_kernels = [k for k in kernels if 'bwd' in k["Kernel函数名称"].lower()]
        if fwd_kernels and bwd_kernels:
            fwd_max = max(k["核心程度得分"] for k in fwd_kernels)
            bwd_max = max(k["核心程度得分"] for k in bwd_kernels)
            if fwd_max < bwd_max:
                inconsistent.append((op, fwd_max, bwd_max))
    
    if inconsistent:
        print(f"  警告: {len(inconsistent)} 个算子的 fwd 核心程度低于 bwd")
        for op, fwd, bwd in inconsistent[:5]:
            print(f"    {op}: fwd={fwd}, bwd={bwd}")
    else:
        print(f"  OK: 所有算子的 fwd 核心程度 >= bwd")
    
    # 验证2: 共享模块 kernel 应全部为辅助
    print("\n验证2: 共享模块 kernel 核心程度")
    non_aux_shared = [r for r in shared_kernels if r["核心程度级别"] != "辅助"]
    if non_aux_shared:
        print(f"  警告: {len(non_aux_shared)} 个共享模块 kernel 不是辅助级别")
        for r in non_aux_shared[:5]:
            print(f"    {r['Kernel函数名称']}: {r['核心程度级别']}")
    else:
        print(f"  OK: 所有共享模块 kernel 都是辅助级别")
    
    # 验证3: cv/cube 类型 kernel 应更可能是核心
    print("\n验证3: cv/cube 类型 kernel 核心程度")
    cv_core = [r for r in results if r["kernel类型"] == "cv/cube" and r["核心程度级别"] == "核心"]
    cv_non_core = [r for r in results if r["kernel类型"] == "cv/cube" and r["核心程度级别"] != "核心"]
    print(f"  cv/cube 类型 kernel: {len([r for r in results if r['kernel类型'] == 'cv/cube'])} 个")
    print(f"  其中核心级别: {len(cv_core)} 个")
    print(f"  其中非核心级别: {len(cv_non_core)} 个")
    
    print("\n" + "=" * 60)
    print("T4 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
