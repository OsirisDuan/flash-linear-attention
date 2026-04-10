#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为operators_kernels_mapping.csv添加kernel类型列
- 包含tl.dot() -> cv/cube
- 不包含tl.dot() -> vv
"""

import csv
import os
import re

BASE_DIR = r"d:\code\flash-linear-attention"


def read_file_lines(filepath):
    abs_path = os.path.join(BASE_DIR, filepath.replace("/", os.sep))
    if not os.path.isfile(abs_path):
        return None
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def find_def_line(lines, kernel_name, declared_line):
    if lines is None:
        return None

    target_idx = declared_line - 1
    if 0 <= target_idx < len(lines):
        if re.search(r'\bdef\s+' + re.escape(kernel_name) + r'\b', lines[target_idx]):
            return target_idx

    for offset in range(-10, 11):
        idx = target_idx + offset
        if 0 <= idx < len(lines):
            if re.search(r'\bdef\s+' + re.escape(kernel_name) + r'\b', lines[idx]):
                return idx

    for i, line in enumerate(lines):
        if re.search(r'\bdef\s+' + re.escape(kernel_name) + r'\b', line):
            return i

    return None


def find_func_body_start(lines, def_line_idx):
    paren_depth = 0
    found_open = False
    for i in range(def_line_idx, len(lines)):
        for ch in lines[i]:
            if ch == '(':
                paren_depth += 1
                found_open = True
            elif ch == ')':
                paren_depth -= 1
            if found_open and paren_depth == 0:
                colon_pos = lines[i].find(':', lines[i].rfind(')'))
                if colon_pos != -1:
                    return i + 1
                else:
                    for j in range(i + 1, len(lines)):
                        if ':' in lines[j]:
                            return j + 1
                    return i + 1
    return def_line_idx + 1


def extract_function_body(lines, def_line_idx):
    func_start = def_line_idx
    for i in range(def_line_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("@"):
            func_start = i
        else:
            break

    body_start = find_func_body_start(lines, def_line_idx)

    body_indent = None
    for i in range(body_start, len(lines)):
        stripped = lines[i].rstrip()
        if stripped and not stripped.startswith("#"):
            body_indent = len(lines[i]) - len(lines[i].lstrip())
            break

    if body_indent is None:
        end_idx = len(lines)
    else:
        end_idx = body_start
        while end_idx < len(lines):
            line = lines[end_idx]
            stripped = line.strip()
            if not stripped:
                end_idx += 1
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent < body_indent:
                break
            end_idx += 1

    func_lines = lines[func_start:end_idx]
    return "".join(func_lines), func_start + 1


def search_kernel_in_repo(kernel_name):
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "build", "dist", "egg-info"}]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            if re.search(r'\bdef\s+' + re.escape(kernel_name) + r'\b', content):
                rel_path = os.path.relpath(fpath, BASE_DIR).replace(os.sep, "/")
                for i, line in enumerate(content.splitlines()):
                    if re.search(r'\bdef\s+' + re.escape(kernel_name) + r'\b', line):
                        return rel_path, i + 1
    return None, None


def check_tl_dot(func_code):
    if func_code is None:
        return "unknown"
    if "tl.dot(" in func_code:
        return "cv/cube"
    return "vv"


def main():
    csv_path = os.path.join(BASE_DIR, "tmp", "operators_kernels_mapping.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    kernel_idx = fieldnames.index("Kernel函数名称")
    new_fieldnames = fieldnames[:kernel_idx + 1] + ["kernel类型"] + fieldnames[kernel_idx + 1:]

    file_cache = {}
    corrected_count = 0
    not_found_count = 0

    for row in rows:
        kernel_name = row["Kernel函数名称"]
        src_path = row["Kernel源码路径"]
        line_no = int(row["Kernel行号"])

        if src_path not in file_cache:
            file_cache[src_path] = read_file_lines(src_path)
        lines = file_cache[src_path]

        def_idx = find_def_line(lines, kernel_name, line_no)

        if def_idx is not None:
            func_code, actual_line = extract_function_body(lines, def_idx)
            if actual_line != line_no:
                print(f"修正行号: {kernel_name} {src_path}:{line_no} -> :{actual_line}")
                row["Kernel行号"] = str(actual_line)
                corrected_count += 1
        else:
            new_path, new_line = search_kernel_in_repo(kernel_name)
            if new_path:
                print(f"修正路径: {kernel_name} {src_path}:{line_no} -> {new_path}:{new_line}")
                row["Kernel源码路径"] = new_path
                row["Kernel行号"] = str(new_line)
                corrected_count += 1
                if new_path not in file_cache:
                    file_cache[new_path] = read_file_lines(new_path)
                lines = file_cache[new_path]
                def_idx = find_def_line(lines, kernel_name, new_line)
                if def_idx is not None:
                    func_code, actual_line = extract_function_body(lines, def_idx)
                else:
                    func_code = None
            else:
                print(f"未找到: {kernel_name} (原路径: {src_path}:{line_no})")
                not_found_count += 1
                func_code = None

        row["kernel类型"] = check_tl_dot(func_code)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n处理完成!")
    print(f"  修正路径/行号: {corrected_count}条")
    print(f"  未找到kernel: {not_found_count}条")

    type_counts = {}
    for row in rows:
        t = row["kernel类型"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nkernel类型统计:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"\ncv/cube类型的kernel:")
    for row in rows:
        if row["kernel类型"] == "cv/cube":
            print(f"  {row['Kernel函数名称']} ({row['Kernel源码路径']}:{row['Kernel行号']})")


if __name__ == "__main__":
    main()
