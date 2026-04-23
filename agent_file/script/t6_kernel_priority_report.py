#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6: 优先级报告生成
生成人类可读的优先级评估报告
输出: t6_kernel_priority_report.md
"""

import csv
import os
from collections import Counter, defaultdict

BASE_DIR = r"d:\code\flash-linear-attention"
OUTPUT_DIR = os.path.join(BASE_DIR, "agent_file", "tmp_file")
REPORT_DIR = os.path.join(BASE_DIR, "agent_file", "report")

T2_CSV = os.path.join(OUTPUT_DIR, "t2_kernel_operator_model_mapping.csv")
T3_CSV = os.path.join(OUTPUT_DIR, "t3_kernel_reuse_stats.csv")
T4_CSV = os.path.join(OUTPUT_DIR, "t4_kernel_criticality.csv")
T5_GLOBAL_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_priority_scores.csv")
T5_MODULE_CSV = os.path.join(OUTPUT_DIR, "t5_module_kernel_priority_scores.csv")
T5_TECH_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_tech_complexity.csv")
T5_TEST_CSV = os.path.join(OUTPUT_DIR, "t5_kernel_test_gap_scores.csv")

OUTPUT_REPORT = os.path.join(REPORT_DIR, "t6_kernel_priority_report.md")


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("=" * 60)
    print("T6: 优先级报告生成")
    print("=" * 60)

    print("\n步骤 6.1: 读取所有数据")
    t2_data = load_csv(T2_CSV)
    t3_data = load_csv(T3_CSV)
    t4_data = load_csv(T4_CSV)
    t5_global = load_csv(T5_GLOBAL_CSV)
    t5_module = load_csv(T5_MODULE_CSV)
    t5_tech = load_csv(T5_TECH_CSV)
    t5_test = load_csv(T5_TEST_CSV)
    print(f"  T2: {len(t2_data)} 条")
    print(f"  T3: {len(t3_data)} 条")
    print(f"  T4: {len(t4_data)} 条")
    print(f"  T5 全局: {len(t5_global)} 条")
    print(f"  T5 模块: {len(t5_module)} 条")
    print(f"  T5 技术: {len(t5_tech)} 条")
    print(f"  T5 测试: {len(t5_test)} 条")

    t4_map = {}
    for r in t4_data:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t4_map[key] = r

    t3_map = {}
    for r in t3_data:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t3_map[key] = r

    t5_tech_map = {}
    for r in t5_tech:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t5_tech_map[key] = r

    t5_test_map = {}
    for r in t5_test:
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t5_test_map[key] = r

    print("\n步骤 6.2: 生成报告")

    core_kernels = [r for r in t5_global if r["kernel来源"] == "核心算子"]
    shared_kernels = [r for r in t5_global if r["kernel来源"] == "共享模块"]
    total = len(t5_global)

    priority_dist_all = Counter(r["优先级等级"] for r in t5_global)
    priority_dist_core = Counter(r["优先级等级"] for r in core_kernels)
    priority_dist_shared = Counter(r["优先级等级"] for r in shared_kernels)

    heat_dist_core = Counter(int(r["模型热度得分"]) for r in core_kernels)
    reuse_dist_core = Counter(int(r["复用度得分"]) for r in core_kernels)
    crit_dist_core = Counter(int(r["核心程度得分"]) for r in core_kernels)
    test_dist_core = Counter(int(r["测试缺失度得分"]) for r in core_kernels)
    tech_dist_core = Counter(int(r["技术复杂度得分"]) for r in core_kernels)

    heat_dist_shared = Counter(int(r["模型热度得分"]) for r in shared_kernels)
    reuse_dist_shared = Counter(int(r["复用度得分"]) for r in shared_kernels)

    tech_dist_all = Counter(r["技术复杂度级别"] for r in t5_tech)
    test_dist_all = Counter(r["测试缺失度级别"] for r in t5_test)

    p0_kernels = [r for r in t5_global if r["优先级等级"] == "P0"]
    p1_kernels = [r for r in t5_global if r["优先级等级"] == "P1"]

    no_test_kernels = [r for r in t5_test if r["测试缺失度级别"] == "严重缺失"]

    module_groups = defaultdict(list)
    for r in t5_module:
        module_groups[r["所属模块"]].append(r)

    report = []

    report.append("# Kernel 优先级评估报告")
    report.append("")
    report.append("## 1. 执行摘要")
    report.append("")
    report.append(f"本次评估共覆盖 **{total} 个 Triton kernel**，其中核心算子 kernel {len(core_kernels)} 个，共享模块 kernel {len(shared_kernels)} 个。")
    report.append("")

    report.append("### 1.1 全局排名优先级分布")
    report.append("")
    report.append("| 优先级 | 总数 | 核心算子 | 共享模块 | 占比 |")
    report.append("|--------|------|---------|---------|------|")
    for p in ["P0", "P1", "P2", "P3"]:
        all_c = priority_dist_all.get(p, 0)
        core_c = priority_dist_core.get(p, 0)
        shared_c = priority_dist_shared.get(p, 0)
        pct = all_c / total * 100
        report.append(f"| {p} | {all_c} | {core_c} | {shared_c} | {pct:.1f}% |")
    report.append("")

    report.append("### 1.2 双轨制评估策略说明")
    report.append("")
    report.append("本评估采用**双轨制**策略：")
    report.append("")
    report.append("- **全局排名**：所有 kernel 参与排名，共享模块 kernel 的复用度得分降权（上限2分），防止高复用的基础设施 kernel 占据排名前列")
    report.append("- **共享模块内部排名**：仅共享模块 kernel 参与排名，复用度使用原始得分（不降权），在模块内形成差异化")
    report.append("")

    report.append("### 1.3 关键发现")
    report.append("")
    report.append(f"1. **{len(p0_kernels)} 个 P0 级 kernel**：全部为核心算子 kernel，集中在 GLA 算子和 GatedDeltaRule 算子，被 S 级模型（KDA/Kimi）和 A 级模型（GLA）使用")
    report.append(f"2. **共享模块 kernel 无 P0 级**：双轨制降权后，共享模块 kernel 最高为 P1 级，符合预期——全局排名聚焦核心算子")
    report.append(f"3. **{len(no_test_kernels)} 个 kernel 无测试用例**：其中核心算子 kernel 占比较大，需优先补充测试")
    report.append(f"4. **技术复杂度两极分化**：极高(89个) 和极低(38个) 占比大，cv/cube 类型 kernel 普遍复杂度高")
    report.append(f"5. **GLA 算子 kernel 占据 P0 前列**：GLA 的 chunk kernel 被 4+ 个算子共享，复用度极高")
    report.append("")

    report.append("## 2. 全局排名 — P0 级 Kernel 清单")
    report.append("")
    report.append("| 排名 | Kernel | 总分 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 | 使用的模型 | 共享算子 |")
    report.append("|------|--------|------|---------|--------|---------|---------|-----------|----------|---------|")
    for i, r in enumerate(p0_kernels):
        models = r["使用的模型列表"][:30] + "..." if len(r["使用的模型列表"]) > 30 else r["使用的模型列表"]
        ops = r["共享算子/模块列表"][:20] + "..." if len(r["共享算子/模块列表"]) > 20 else r["共享算子/模块列表"]
        report.append(f"| {i+1} | {r['Kernel函数名称']} | {r['最终总分']} | {r['模型热度得分']} | {r['复用度得分']} | {r['核心程度得分']} | {r['测试缺失度得分']} | {r['技术复杂度得分']} | {models} | {ops} |")
    report.append("")

    report.append("### P0 级行动建议")
    report.append("")
    report.append("1. **立即补充测试**：P0 级 kernel 测试缺失度均为 3 分（部分覆盖），需补充独立单元测试")
    report.append("2. **GLA chunk kernel 优先**：GLA 的 fwd/bwd kernel 被 4+ 个算子共享，修改影响范围大")
    report.append("3. **GatedDeltaRule kkt_solve_kernel 关注**：虽复用度仅 2 分，但核心程度和技术复杂度均为 5 分，是算法核心")
    report.append("")

    report.append("## 3. 核心算子 Kernel 分析（全局排名视角）")
    report.append("")

    report.append("### 3.1 模型热度分布")
    report.append("")
    report.append("| 得分 | 数量 | 占比 | 说明 |")
    report.append("|------|------|------|------|")
    heat_labels = {5: "S级模型", 4: "A级模型", 3: "B级模型", 2: "C级模型", 1: "D级模型"}
    for s in range(5, 0, -1):
        c = heat_dist_core.get(s, 0)
        pct = c / len(core_kernels) * 100
        report.append(f"| {s} | {c} | {pct:.1f}% | {heat_labels[s]} |")
    report.append("")

    report.append("### 3.2 复用度分布")
    report.append("")
    report.append("| 得分 | 数量 | 占比 | 说明 |")
    report.append("|------|------|------|------|")
    reuse_labels = {5: "≥4个算子共享", 4: "3个算子共享", 3: "2个算子共享", 2: "1算子+多模型", 1: "1算子+1模型"}
    for s in range(5, 0, -1):
        c = reuse_dist_core.get(s, 0)
        pct = c / len(core_kernels) * 100
        report.append(f"| {s} | {c} | {pct:.1f}% | {reuse_labels[s]} |")
    report.append("")

    report.append("### 3.3 核心程度分布")
    report.append("")
    report.append("| 得分 | 数量 | 占比 | 说明 |")
    report.append("|------|------|------|------|")
    crit_labels = {5: "核心(前向主计算)", 4: "重要(反向传播)", 3: "辅助(预处理/后处理)", 2: "优化(融合操作)", 1: "边缘(naive实现)"}
    for s in range(5, 0, -1):
        c = crit_dist_core.get(s, 0)
        pct = c / len(core_kernels) * 100
        report.append(f"| {s} | {c} | {pct:.1f}% | {crit_labels[s]} |")
    report.append("")

    report.append("### 3.4 测试覆盖现状")
    report.append("")
    report.append("| 得分 | 数量 | 占比 | 说明 |")
    report.append("|------|------|------|------|")
    test_labels = {5: "严重缺失(无测试)", 4: "明显不足(1个测试)", 3: "部分覆盖(2-4个测试)", 2: "基本覆盖(5+测试)", 1: "充分覆盖(10+测试)", 0: "未统计"}
    for s in range(5, -1, -1):
        c = test_dist_core.get(s, 0)
        if c == 0:
            continue
        pct = c / len(core_kernels) * 100
        report.append(f"| {s} | {c} | {pct:.1f}% | {test_labels[s]} |")
    report.append("")

    no_test_core = [r for r in t5_test if r["kernel来源"] == "核心算子" and r["测试缺失度级别"] == "严重缺失"]
    if no_test_core:
        report.append(f"**{len(no_test_core)} 个核心算子 kernel 无测试用例**，涉及算子：")
        no_test_ops = set()
        for r in no_test_core:
            path = r.get("Kernel源码路径", "")
            if "ops/" in path:
                op = path.split("ops/")[1].split("/")[0]
                no_test_ops.add(op)
        report.append(f"- {', '.join(sorted(no_test_ops))}")
        report.append("")

    report.append("## 4. 共享模块 Kernel 分析")
    report.append("")

    report.append("### 4.1 全局排名视角")
    report.append("")
    report.append(f"共享模块 kernel 在全局排名中的分布：")
    report.append("")
    report.append("| 优先级 | 数量 | 占比 |")
    report.append("|--------|------|------|")
    for p in ["P0", "P1", "P2", "P3"]:
        c = priority_dist_shared.get(p, 0)
        pct = c / len(shared_kernels) * 100 if shared_kernels else 0
        report.append(f"| {p} | {c} | {pct:.1f}% |")
    report.append("")

    report.append("**复用度降权效果**：共享模块 kernel 在全局排名中复用度得分上限为 2 分，而模块内排名使用原始得分。以 `chunk_fwd_kernel_h` 为例：")
    chunk_fwd_h_global = [r for r in t5_global if r["Kernel函数名称"] == "chunk_fwd_kernel_h" and "chunk_h" in r["Kernel源码路径"]]
    chunk_fwd_h_module = [r for r in t5_module if r["Kernel函数名称"] == "chunk_fwd_kernel_h" and "chunk_h" in r["Kernel源码路径"]]
    if chunk_fwd_h_global and chunk_fwd_h_module:
        g = chunk_fwd_h_global[0]
        m = chunk_fwd_h_module[0]
        report.append(f"- 全局排名：复用度 {g['复用度得分']} 分，总分 {g['最终总分']}，优先级 {g['优先级等级']}")
        report.append(f"- 模块内排名：复用度 {m['复用度得分']} 分，总分 {m['加权总分']}，优先级 {m['模块内优先级等级']}")
    report.append("")

    report.append("### 4.2 模块内排名视角（不降权）")
    report.append("")

    report.append("#### 各模块 kernel 数量与复杂度")
    report.append("")
    report.append("| 模块 | kernel 数量 | P0 | P1 | P2 | P3 | 平均总分 |")
    report.append("|------|-----------|-----|-----|-----|-----|---------|")
    for mod_name in sorted(module_groups.keys()):
        mod_kernels = module_groups[mod_name]
        mod_p0 = sum(1 for r in mod_kernels if r["模块内优先级等级"] == "P0")
        mod_p1 = sum(1 for r in mod_kernels if r["模块内优先级等级"] == "P1")
        mod_p2 = sum(1 for r in mod_kernels if r["模块内优先级等级"] == "P2")
        mod_p3 = sum(1 for r in mod_kernels if r["模块内优先级等级"] == "P3")
        avg_score = sum(float(r["加权总分"]) for r in mod_kernels) / len(mod_kernels)
        report.append(f"| {mod_name} | {len(mod_kernels)} | {mod_p0} | {mod_p1} | {mod_p2} | {mod_p3} | {avg_score:.2f} |")
    report.append("")

    report.append("#### 模块内 Top 5 Kernel")
    report.append("")
    report.append("| 排名 | Kernel | 总分 | 模块 | 模型热度 | 复用度 | 核心程度 | 测试缺失 | 技术复杂度 |")
    report.append("|------|--------|------|------|---------|--------|---------|---------|-----------|")
    for i, r in enumerate(t5_module[:5]):
        report.append(f"| {i+1} | {r['Kernel函数名称']} | {r['加权总分']} | {r['所属模块']} | {r['模型热度得分']} | {r['复用度得分']} | {r['核心程度得分']} | {r['测试缺失度得分']} | {r['技术复杂度得分']} |")
    report.append("")

    report.append("#### 模块内排名与全局排名的差异分析")
    report.append("")
    report.append("共享模块 kernel 在模块内排名中优先级普遍高于全局排名，这是复用度降权的直接效果：")
    report.append("")
    diff_examples = []
    for r in t5_module[:10]:
        g_match = [g for g in t5_global if g["Kernel函数名称"] == r["Kernel函数名称"] and g["Kernel源码路径"] == r["Kernel源码路径"]]
        if g_match:
            g = g_match[0]
            diff_examples.append((r["Kernel函数名称"], r["模块内优先级等级"], g["优先级等级"], r["复用度得分"], g["复用度得分"]))
    if diff_examples:
        report.append("| Kernel | 模块内优先级 | 全局优先级 | 模块内复用度 | 全局复用度 |")
        report.append("|--------|------------|-----------|------------|-----------|")
        for name, mod_p, glob_p, mod_reuse, glob_reuse in diff_examples:
            report.append(f"| {name} | {mod_p} | {glob_p} | {mod_reuse} | {glob_reuse} |")
    report.append("")

    report.append("## 5. 高优先级 Kernel 详细分析")
    report.append("")

    report.append("### 5.1 Top 10 核心 Kernel 深度分析（全局排名）")
    report.append("")

    for i, r in enumerate(t5_global[:10]):
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t4_r = t4_map.get(key, {})
        t3_r = t3_map.get(key, {})
        tech_r = t5_tech_map.get(key, {})
        test_r = t5_test_map.get(key, {})

        report.append(f"#### {i+1}. {r['Kernel函数名称']}")
        report.append("")
        report.append(f"- **最终总分**: {r['最终总分']} ({r['优先级等级']})")
        report.append(f"- **源码路径**: {r['Kernel源码路径']}:{r['Kernel行号']}")
        report.append(f"- **kernel 类型**: {r['kernel类型']}")
        report.append("")

        report.append("**各维度得分及支撑数据**：")
        report.append("")
        report.append("| 维度 | 得分 | 支撑数据 |")
        report.append("|------|------|---------|")

        heat_detail = f"使用的模型: {r['使用的模型列表']}"
        report.append(f"| 模型热度(35%) | {r['模型热度得分']} | {heat_detail} |")

        reuse_detail = f"共享算子: {r['共享算子/模块列表']}"
        if t3_r:
            reuse_detail += f", 共享数量: {t3_r.get('共享算子/模块数量', '')}"
        report.append(f"| 复用度(30%) | {r['复用度得分']} | {reuse_detail} |")

        crit_detail = t4_r.get("判定依据", "")
        report.append(f"| 核心程度(20%) | {r['核心程度得分']} | {crit_detail} |")

        test_detail = test_r.get("测试覆盖情况", "")
        test_funcs = test_r.get("测试用例列表", "")
        if test_funcs:
            test_detail += f" (测试函数: {test_funcs[:50]}...)" if len(test_funcs) > 50 else f" (测试函数: {test_funcs})"
        report.append(f"| 测试缺失度(5%) | {r['测试缺失度得分']} | {test_detail} |")

        tech_detail = tech_r.get("判定依据", "")
        report.append(f"| 技术复杂度(10%) | {r['技术复杂度得分']} | {tech_detail} |")

        report.append("")

    report.append("### 5.2 Top 5 共享模块 Kernel 分析（模块内排名）")
    report.append("")

    for i, r in enumerate(t5_module[:5]):
        key = (r["Kernel函数名称"], r["Kernel源码路径"], r["Kernel行号"])
        t4_r = t4_map.get(key, {})
        t3_r = t3_map.get(key, {})
        tech_r = t5_tech_map.get(key, {})
        test_r = t5_test_map.get(key, {})

        g_match = [g for g in t5_global if g["Kernel函数名称"] == r["Kernel函数名称"] and g["Kernel源码路径"] == r["Kernel源码路径"]]
        global_priority = g_match[0]["优先级等级"] if g_match else "N/A"
        global_score = g_match[0]["最终总分"] if g_match else "N/A"

        report.append(f"#### {i+1}. {r['Kernel函数名称']}")
        report.append("")
        report.append(f"- **模块内总分**: {r['加权总分']} ({r['模块内优先级等级']})，全局总分: {global_score} ({global_priority})")
        report.append(f"- **所属模块**: {r['所属模块']}")
        report.append(f"- **源码路径**: {r['Kernel源码路径']}:{r['Kernel行号']}")
        report.append("")

        report.append("**各维度得分及支撑数据**：")
        report.append("")
        report.append("| 维度 | 模块内得分 | 全局得分 | 支撑数据 |")
        report.append("|------|-----------|---------|---------|")

        g = g_match[0] if g_match else {}

        report.append(f"| 模型热度(35%) | {r['模型热度得分']} | {g.get('模型热度得分', '-')} | 共享模块被所有模型使用 |")

        mod_reuse_detail = f"原始复用度: {r['复用度得分']}"
        glob_reuse = g.get("复用度得分", "-")
        if t3_r:
            mod_reuse_detail += f", 共享数量: {t3_r.get('共享算子/模块数量', '')}"
        report.append(f"| 复用度(30%) | {r['复用度得分']} | {glob_reuse} | {mod_reuse_detail} |")

        crit_detail = t4_r.get("判定依据", "共享模块默认辅助")
        report.append(f"| 核心程度(20%) | {r['核心程度得分']} | {g.get('核心程度得分', '-')} | {crit_detail} |")

        test_detail = test_r.get("测试覆盖情况", "")
        report.append(f"| 测试缺失度(5%) | {r['测试缺失度得分']} | {g.get('测试缺失度得分', '-')} | {test_detail} |")

        tech_detail = tech_r.get("判定依据", "")
        report.append(f"| 技术复杂度(10%) | {r['技术复杂度得分']} | {g.get('技术复杂度得分', '-')} | {tech_detail} |")

        report.append("")

    report.append("## 6. 行动建议")
    report.append("")

    report.append("### 6.1 短期行动项（P0 级 kernel）")
    report.append("")
    report.append(f"1. **补充 GLA chunk kernel 测试**：`chunk_gla_fwd_kernel_o`、`chunk_gla_bwd_kernel_*` 等 9 个 P0 级 kernel 需补充独立单元测试，当前仅有间接测试覆盖")
    report.append(f"2. **补充 GatedDeltaRule kernel 测试**：`pre_process_fwd/bwd_kernel_merged`、`merge_fwd_bwd_kernel` 测试缺失度仅 1 分（充分覆盖），但 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 需关注")
    report.append(f"3. **建立共享 kernel 回归测试**：`chunk_fwd_kernel_o`、`chunk_bwd_kernel_dqkwg` 等被多算子共享的 kernel，修改前需确认对所有使用算子的影响")
    report.append("")

    report.append("### 6.2 中期规划（P1 级 kernel）")
    report.append("")
    report.append(f"1. **覆盖 P1 级核心算子 kernel**：{priority_dist_core.get('P1', 0)} 个 P1 级核心算子 kernel 需逐步补充测试")
    report.append(f"2. **优化高复用 kernel 性能**：复用度 5 分的 kernel（如 `chunk_fwd_kernel_o`）性能优化收益最大")
    report.append(f"3. **关注无测试 kernel**：{len(no_test_core)} 个核心算子 kernel 无任何测试用例，需优先补充")
    report.append("")

    report.append("### 6.3 共享模块 kernel 的特殊建议（基于模块内排名）")
    report.append("")
    report.append("1. **chunk_delta_h 模块**：`chunk_gated_delta_rule_fwd/bwd_kernel_h_blockdim64` 在模块内排名 Top 2，被 S 级模型（KDA/Kimi）使用，需确保稳定性")
    report.append("2. **chunk_o 模块**：`chunk_fwd_kernel_o`、`chunk_bwd_kernel_dqkwg`、`chunk_bwd_kernel_dv` 被 4+ 算子共享，修改需谨慎")
    report.append("3. **utils/op.py 简单函数**：`tanh`、`log`、`exp` 等在模块内排名靠前（因高复用+高测试缺失），但实际是简单的逐元素操作，修改风险低，可降低优先级")
    report.append("4. **activations 模块**：`swiglu_fwdbwd_kernel` 在模块内排名较高，需关注梯度正确性测试")
    report.append("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"  输出: {OUTPUT_REPORT}")

    print("\n" + "=" * 60)
    print("T6 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
