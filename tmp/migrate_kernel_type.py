#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将operators_kernels_mapping.csv中的kernel类型列迁移到kernel_test_coverage.csv
"""

import csv

mapping_path = "tmp/operators_kernels_mapping.csv"
coverage_path = "tmp/kernel_test_coverage.csv"

with open(mapping_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    kernel_type_map = {}
    for row in reader:
        name = row["Kernel函数名称"]
        ktype = row["kernel类型"]
        if name not in kernel_type_map:
            kernel_type_map[name] = ktype

with open(coverage_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

kernel_idx = fieldnames.index("Kernel函数名称")
new_fieldnames = fieldnames[:kernel_idx + 1] + ["kernel类型"] + fieldnames[kernel_idx + 1:]

not_found = []
for row in rows:
    name = row["Kernel函数名称"]
    if name in kernel_type_map:
        row["kernel类型"] = kernel_type_map[name]
    else:
        row["kernel类型"] = "unknown"
        not_found.append(name)

with open(coverage_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"处理完成!")
print(f"  总行数: {len(rows)}")
print(f"  成功匹配: {len(rows) - len(not_found)}")
print(f"  未匹配: {len(not_found)}")

if not_found:
    unique_not_found = sorted(set(not_found))
    print(f"\n未匹配的kernel函数 ({len(unique_not_found)}个):")
    for name in unique_not_found:
        print(f"  - {name}")

type_counts = {}
for row in rows:
    t = row["kernel类型"]
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"\nkernel类型统计:")
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")
