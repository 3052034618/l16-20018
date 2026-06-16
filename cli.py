"""
命令行入口 — 批量跑场景、对比启发函数、性能测试。

用法:
  python cli.py run <map_file.json> [--heuristic name]
  python cli.py batch <scenario_dir>
  python cli.py compare <grid_map.json> <navmesh_map.json>
  python cli.py demo
"""

import argparse
import os
import sys
import time

from pathfinding.loader import load_map
from pathfinding.benchmark import (
    run_grid_benchmark,
    run_navmesh_benchmark,
    format_result_table,
    compare_grid_vs_navmesh,
)
from pathfinding.heuristics import (
    manhattan_distance,
    euclidean_distance,
    diagonal_distance,
    zero_heuristic,
)


HEURISTIC_MAP = {
    "zero": zero_heuristic,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "diagonal": diagonal_distance,
    "octile": diagonal_distance,
}


def cmd_run(args):
    """跑单个地图文件"""
    filepath = args.map_file
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在: {filepath}")
        return 1

    map_info = load_map(filepath)
    name = map_info["name"]
    map_type = map_info["map_type"]
    start = map_info["start"]
    goal = map_info["goal"]

    print(f"地图: {name} ({map_type})")
    print(f"起点: {start} → 终点: {goal}")
    print()

    if map_type == "grid":
        grid_map = map_info["map"]

        if args.heuristic:
            h_name = args.heuristic
            h_func = HEURISTIC_MAP.get(h_name)
            if not h_func:
                print(f"错误: 未知启发函数 '{h_name}'")
                print(f"  可选: {', '.join(HEURISTIC_MAP.keys())}")
                return 1
            heuristics = {h_name: h_func}
        else:
            heuristics = None

        results = run_grid_benchmark(grid_map, start, goal, heuristics=heuristics)
        print(format_result_table(results, title="网格地图寻路结果"))

        if args.show_path and results:
            best = results[0]
            if best.found:
                print()
                print("路径可视化:")
                print(grid_map.display_path(best.path))

    elif map_type == "navmesh":
        navmesh = map_info["map"]
        result = run_navmesh_benchmark(navmesh, start, goal)
        print(format_result_table([result], title="导航网格寻路结果"))

        if result.found and args.show_path:
            print()
            print(f"平滑路径 ({len(result.smooth_path)} 个拐点):")
            for i, pt in enumerate(result.smooth_path):
                print(f"  {i}: ({pt[0]:.2f}, {pt[1]:.2f})")

    print()
    return 0


def cmd_batch(args):
    """批量跑目录下所有场景"""
    scenario_dir = args.dir
    if not os.path.isdir(scenario_dir):
        print(f"错误: 目录不存在: {scenario_dir}")
        return 1

    json_files = sorted([
        f for f in os.listdir(scenario_dir)
        if f.endswith('.json')
    ])

    if not json_files:
        print(f"错误: 目录中没有 JSON 文件: {scenario_dir}")
        return 1

    print(f"批量运行 {len(json_files)} 个场景...")
    print("=" * 90)

    all_results = []
    grid_count = 0
    navmesh_count = 0

    for fname in json_files:
        filepath = os.path.join(scenario_dir, fname)
        try:
            map_info = load_map(filepath)
        except Exception as e:
            print(f"[跳过] {fname}: 加载失败 - {e}")
            continue

        map_type = map_info["map_type"]
        name = map_info["name"]
        start = map_info["start"]
        goal = map_info["goal"]

        if map_type == "grid":
            grid_count += 1
            results = run_grid_benchmark(map_info["map"], start, goal,
                                         heuristics={"diagonal": diagonal_distance})
            r = results[0]
            status = "✓" if r.found else "✗"
            extra = ""
            if not r.start_valid:
                extra = "(起点无效)"
            elif not r.goal_valid:
                extra = "(终点无效)"
            print(f"  [Grid] {name:<30} {status}  代价={r.total_cost:>7.2f}  "
                  f"展开={r.nodes_expanded:>6}  耗时={r.time_ms:>7.3f}ms  {extra}")
            all_results.append((name, "grid", r))

        elif map_type == "navmesh":
            navmesh_count += 1
            r = run_navmesh_benchmark(map_info["map"], start, goal)
            status = "✓" if r.found else "✗"
            extra = ""
            if not r.start_valid:
                extra = "(起点无效)"
            elif not r.goal_valid:
                extra = "(终点无效)"
            print(f"  [NavM] {name:<30} {status}  代价={r.total_cost:>7.2f}  "
                  f"展开={r.nodes_expanded:>6}  耗时={r.time_ms:>7.3f}ms  {extra}")
            all_results.append((name, "navmesh", r))

    print()
    print(f"完成: 共 {len(json_files)} 个场景, "
          f"网格 {grid_count} 个, 导航网格 {navmesh_count} 个")

    found_count = sum(1 for _, _, r in all_results if r.found)
    print(f"      成功 {found_count} 个, 失败 {len(all_results) - found_count} 个")

    return 0


def cmd_compare(args):
    """对比网格与导航网格的性能"""
    if not os.path.exists(args.grid_map):
        print(f"错误: 网格地图文件不存在: {args.grid_map}")
        return 1
    if not os.path.exists(args.navmesh_map):
        print(f"错误: 导航网格文件不存在: {args.navmesh_map}")
        return 1

    grid_info = load_map(args.grid_map)
    nm_info = load_map(args.navmesh_map)

    if grid_info["map_type"] != "grid":
        print(f"错误: {args.grid_map} 不是网格地图")
        return 1
    if nm_info["map_type"] != "navmesh":
        print(f"错误: {args.navmesh_map} 不是导航网格")
        return 1

    print(compare_grid_vs_navmesh(
        grid_info["map"], grid_info["start"], grid_info["goal"],
        nm_info["map"], nm_info["start"], nm_info["goal"],
    ))

    return 0


def cmd_demo(args):
    """运行演示 (等价于 demo.py)"""
    import demo
    return 0


def main():
    parser = argparse.ArgumentParser(description="寻路引擎命令行工具")
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_run = sub.add_parser("run", help="跑单个地图文件")
    p_run.add_argument("map_file", help="JSON 地图文件路径")
    p_run.add_argument("--heuristic", "-H", default=None,
                       help="启发函数: zero/manhattan/euclidean/diagonal (默认全部)")
    p_run.add_argument("--show-path", "-p", action="store_true",
                       help="显示路径可视化")
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="批量跑目录下所有场景")
    p_batch.add_argument("dir", help="场景 JSON 文件目录")
    p_batch.set_defaults(func=cmd_batch)

    p_cmp = sub.add_parser("compare", help="对比网格与导航网格性能")
    p_cmp.add_argument("grid_map", help="网格地图 JSON 文件")
    p_cmp.add_argument("navmesh_map", help="导航网格 JSON 文件")
    p_cmp.set_defaults(func=cmd_compare)

    p_demo = sub.add_parser("demo", help="运行完整演示")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
