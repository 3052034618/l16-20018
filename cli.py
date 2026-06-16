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
from pathfinding.reporting import (
    generate_report,
    format_report_summary,
    export_csv,
    export_markdown,
)
from pathfinding.regression import (
    save_baseline,
    load_baseline,
    compare_with_baseline,
    format_regression_report,
    export_regression_markdown,
)
from pathfinding.stress_test import (
    run_stress_test,
    format_stress_report,
    run_multiple_stress_tests,
    format_multi_stress_report,
    build_matrix_configs,
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

    filter_type = args.type
    only_failures = args.only_failures
    verbose_path = args.verbose
    export_maps_dir = args.export_maps

    heuristics = None
    if args.heuristics:
        heuristics = {}
        for h_name in args.heuristics:
            h_func = HEURISTIC_MAP.get(h_name)
            if not h_func:
                print(f"警告: 未知启发函数 '{h_name}', 已跳过")
                continue
            heuristics[h_name] = h_func
        if not heuristics:
            print("错误: 没有有效的启发函数")
            return 1

    if export_maps_dir:
        os.makedirs(export_maps_dir, exist_ok=True)

    all_results = []
    grid_count = 0
    navmesh_count = 0
    shown_count = 0

    print(f"批量运行 {len(json_files)} 个场景..." +
          (f" (筛选: {filter_type})" if filter_type else "") +
          (" (仅失败)" if only_failures else ""))
    print("=" * 110)

    for fname in json_files:
        filepath = os.path.join(scenario_dir, fname)
        try:
            map_info = load_map(filepath)
        except Exception as e:
            print(f"  [跳过] {fname}: 加载失败 - {e}")
            continue

        map_type = map_info["map_type"]
        if filter_type and map_type != filter_type:
            continue

        name = map_info["name"]
        start = map_info["start"]
        goal = map_info["goal"]

        if map_type == "grid":
            grid_count += 1
            grid_map = map_info["map"]
            results = run_grid_benchmark(grid_map, start, goal,
                                         heuristics=heuristics)

            any_failure = any(not r.found for r in results)
            if only_failures and not any_failure:
                all_results.append((name, "grid", results, map_info))
                continue

            shown_count += 1
            print(f"\n  ▶ [Grid] {name}")
            print(format_result_table(results, title=""))
            for r in results:
                if r.found:
                    print(f"    {r.heuristic_name}: {r.path_summary(verbose=verbose_path)}")

            if export_maps_dir:
                for r in results:
                    if r.found and r.path:
                        safe_h = r.heuristic_name.replace(" ", "_").replace("(", "").replace(")", "")
                        map_fname = f"{name}_{safe_h}.txt"
                        map_path = os.path.join(export_maps_dir, map_fname)
                        with open(map_path, 'w', encoding='utf-8') as f:
                            f.write(f"场景: {name}  启发: {r.heuristic_name}\n")
                            f.write(f"代价={r.total_cost:.2f} 展开={r.nodes_expanded}\n\n")
                            f.write(grid_map.display_path(r.path))
                        print(f"    导出: {map_path}")

            all_results.append((name, "grid", results, map_info))

        elif map_type == "navmesh":
            navmesh_count += 1
            r = run_navmesh_benchmark(map_info["map"], start, goal)

            if only_failures and r.found:
                all_results.append((name, "navmesh", [r], map_info))
                continue

            shown_count += 1
            print(f"\n  ▶ [NavM] {name}")
            print(format_result_table([r], title=""))
            if r.found:
                print(f"    路径: {r.path_summary(verbose=verbose_path)}")

            all_results.append((name, "navmesh", [r], map_info))

    total_scenarios = len(all_results)
    print()
    print("=" * 110)
    print(f"完成: 共 {total_scenarios} 个场景 (显示 {shown_count} 个), "
          f"网格 {grid_count} 个, 导航网格 {navmesh_count} 个")

    success_count = 0
    for _, _, results, _ in all_results:
        if isinstance(results, list) and results and results[0].found:
            success_count += 1
        elif hasattr(results, 'found') and results.found:
            success_count += 1
    print(f"      成功 {success_count} 个, 失败 {total_scenarios - success_count} 个")

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


def cmd_report(args):
    """跑场景验收报告, 可导出 CSV/Markdown"""
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

    filter_type = args.type
    verbose_path = args.verbose
    export_maps_dir = args.export_maps

    if export_maps_dir:
        os.makedirs(export_maps_dir, exist_ok=True)

    all_data = []
    for fname in json_files:
        filepath = os.path.join(scenario_dir, fname)
        try:
            map_info = load_map(filepath)
        except Exception as e:
            print(f"[跳过] {fname}: 加载失败 - {e}")
            continue

        map_type = map_info["map_type"]
        if filter_type and map_type != filter_type:
            continue

        name = map_info["name"]
        start = map_info["start"]
        goal = map_info["goal"]
        expectations = map_info.get("expectations", {})

        if map_type == "grid":
            results = run_grid_benchmark(map_info["map"], start, goal)
            all_data.append((name, "grid", results, expectations, map_info))
        elif map_type == "navmesh":
            r = run_navmesh_benchmark(map_info["map"], start, goal)
            all_data.append((name, "navmesh", [r], expectations, map_info))

    report_data = [(n, mt, rs, exp) for n, mt, rs, exp, _ in all_data]
    reports = generate_report(report_data)

    print(format_report_summary(reports, verbose=verbose_path))

    if export_maps_dir:
        for name, map_type, results, expectations, map_info in all_data:
            if map_type == "grid":
                grid_map = map_info["map"]
                for r in results:
                    if r.found and r.path:
                        safe_h = r.heuristic_name.replace(" ", "_").replace("(", "").replace(")", "")
                        map_fname = f"{name}_{safe_h}.txt"
                        map_path = os.path.join(export_maps_dir, map_fname)
                        with open(map_path, 'w', encoding='utf-8') as f:
                            f.write(f"场景: {name}  启发: {r.heuristic_name}\n")
                            f.write(f"代价={r.total_cost:.2f} 展开={r.nodes_expanded}\n\n")
                            f.write(grid_map.display_path(r.path))

    if args.csv:
        export_csv(reports, args.csv)
        print(f"\nCSV 报告已导出: {args.csv}")

    if args.markdown:
        export_markdown(reports, args.markdown)
        print(f"Markdown 报告已导出: {args.markdown}")

    if args.save_baseline:
        save_baseline(report_data, args.save_baseline)
        print(f"基线已保存: {args.save_baseline}")

    return 0


def cmd_stress(args):
    """随机压力测试"""
    if args.matrix:
        sizes = [(int(s.split('x')[0]), int(s.split('x')[1])) for s in args.sizes]
        densities = [float(d) for d in args.densities]
        seeds = args.seed if args.seed else [42, 123]
        merge_sizes = [int(m) for m in args.merge_sizes]
        terrain_ratios = None
        if args.terrain:
            terrain_ratios = {}
            for t in args.terrain:
                name, ratio = t.split('=')
                terrain_ratios[name] = float(ratio)

        configs = build_matrix_configs(
            sizes=sizes,
            densities=densities,
            seeds=seeds,
            merge_sizes=merge_sizes,
            terrain_ratios=terrain_ratios,
        )
        print(f"矩阵式压力测试: {len(configs)} 组配置...")
        results = run_multiple_stress_tests(configs)
        output = format_multi_stress_report(results)
        print(output)

        if args.export:
            with open(args.export, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"\n报告已导出: {args.export}")

    elif args.multi:
        configs = []
        seeds = args.seed if args.seed else [42, 123, 456, 789, 1024]
        for seed in seeds:
            cfg = {
                "width": args.width,
                "height": args.height,
                "obstacle_density": args.obstacle_density,
                "seed": seed,
                "run_navmesh": not args.no_navmesh,
                "navmesh_merge_size": args.merge_size,
            }
            if args.terrain:
                terrain_ratios = {}
                for t in args.terrain:
                    name, ratio = t.split('=')
                    terrain_ratios[name] = float(ratio)
                cfg["terrain_ratios"] = terrain_ratios
            configs.append(cfg)

        results = run_multiple_stress_tests(configs)
        output = format_multi_stress_report(results)
        print(output)

        if args.export:
            with open(args.export, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"\n报告已导出: {args.export}")
    else:
        seed = args.seed[0] if args.seed else 42
        terrain_ratios = None
        if args.terrain:
            terrain_ratios = {}
            for t in args.terrain:
                name, ratio = t.split('=')
                terrain_ratios[name] = float(ratio)

        result = run_stress_test(
            width=args.width,
            height=args.height,
            obstacle_density=args.obstacle_density,
            terrain_ratios=terrain_ratios,
            seed=seed,
            run_navmesh=not args.no_navmesh,
            navmesh_merge_size=args.merge_size,
        )
        print(format_stress_report(result))

    return 0


def cmd_regress(args):
    """回归基线对比"""
    scenario_dir = args.dir
    if not os.path.isdir(scenario_dir):
        print(f"错误: 目录不存在: {scenario_dir}")
        return 1

    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        print(f"错误: 基线文件不存在: {baseline_path}")
        print("  提示: 先用 report --save-baseline <file> 生成基线")
        return 1

    json_files = sorted([
        f for f in os.listdir(scenario_dir)
        if f.endswith('.json')
    ])

    if not json_files:
        print(f"错误: 目录中没有 JSON 文件: {scenario_dir}")
        return 1

    filter_type = args.type

    all_data = []
    for fname in json_files:
        filepath = os.path.join(scenario_dir, fname)
        try:
            map_info = load_map(filepath)
        except Exception as e:
            print(f"[跳过] {fname}: 加载失败 - {e}")
            continue

        map_type = map_info["map_type"]
        if filter_type and map_type != filter_type:
            continue

        name = map_info["name"]
        start = map_info["start"]
        goal = map_info["goal"]
        expectations = map_info.get("expectations", {})

        if map_type == "grid":
            results = run_grid_benchmark(map_info["map"], start, goal)
            all_data.append((name, "grid", results, expectations))
        elif map_type == "navmesh":
            r = run_navmesh_benchmark(map_info["map"], start, goal)
            all_data.append((name, "navmesh", [r], expectations))

    baseline = load_baseline(baseline_path)
    items = compare_with_baseline(all_data, baseline)

    print(format_regression_report(items))

    if args.save_baseline:
        save_baseline(all_data, args.save_baseline)
        print(f"\n新基线已保存: {args.save_baseline}")

    if args.markdown:
        export_regression_markdown(items, args.markdown)
        print(f"回归报告已导出: {args.markdown}")

    has_regression = any(i.has_regression for i in items)
    return 1 if has_regression else 0


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
    p_batch.add_argument("--type", "-t", default=None, choices=["grid", "navmesh"],
                         help="只跑某类地图: grid 或 navmesh")
    p_batch.add_argument("--only-failures", "-f", action="store_true",
                         help="只显示失败的用例")
    p_batch.add_argument("--heuristics", "-H", nargs="+", default=None,
                         help="指定启发函数列表 (仅网格), 如: -H manhattan euclidean")
    p_batch.add_argument("--verbose", "-v", action="store_true",
                         help="显示完整路径坐标")
    p_batch.add_argument("--export-maps", default=None, metavar="DIR",
                         help="导出网格路径文本图到指定目录")
    p_batch.set_defaults(func=cmd_batch)

    p_cmp = sub.add_parser("compare", help="对比网格与导航网格性能")
    p_cmp.add_argument("grid_map", help="网格地图 JSON 文件")
    p_cmp.add_argument("navmesh_map", help="导航网格 JSON 文件")
    p_cmp.set_defaults(func=cmd_compare)

    p_report = sub.add_parser("report", help="场景验收报告 (支持 CSV/Markdown 导出)")
    p_report.add_argument("dir", help="场景 JSON 文件目录")
    p_report.add_argument("--type", "-t", default=None, choices=["grid", "navmesh"],
                          help="只跑某类地图: grid 或 navmesh")
    p_report.add_argument("--verbose", "-v", action="store_true",
                          help="显示完整路径坐标")
    p_report.add_argument("--export-maps", default=None, metavar="DIR",
                          help="导出网格路径文本图到指定目录")
    p_report.add_argument("--csv", default=None,
                          help="导出 CSV 报告到指定文件")
    p_report.add_argument("--markdown", "--md", default=None,
                          help="导出 Markdown 报告到指定文件")
    p_report.add_argument("--save-baseline", default=None, metavar="FILE",
                          help="保存结果为基线 JSON 文件")
    p_report.set_defaults(func=cmd_report)

    p_stress = sub.add_parser("stress", help="随机压力测试 (对比启发函数和 navmesh 性能)")
    p_stress.add_argument("--width", "-w", type=int, default=50,
                          help="地图宽度 (默认 50)")
    p_stress.add_argument("--height", type=int, default=50,
                          help="地图高度 (默认 50)")
    p_stress.add_argument("--obstacle-density", "-d", type=float, default=0.2,
                          help="障碍密度 0.0~1.0 (默认 0.2)")
    p_stress.add_argument("--seed", "-s", nargs="+", type=int, default=None,
                          help="随机种子 (可多个用于多组测试)")
    p_stress.add_argument("--terrain", "-T", nargs="+", default=None,
                          help="地形比例, 如 forest=0.1 swamp=0.05")
    p_stress.add_argument("--no-navmesh", action="store_true",
                          help="不跑导航网格对比")
    p_stress.add_argument("--multi", "-m", action="store_true",
                          help="多组测试汇总模式")
    p_stress.add_argument("--merge-size", type=int, default=1,
                          help="NavMesh 合并粒度 (1=每格, 2=2x2, 4=4x4, 默认 1)")
    p_stress.add_argument("--matrix", action="store_true",
                          help="矩阵式测试: 多尺寸×多密度×多粒度")
    p_stress.add_argument("--sizes", nargs="+", default=["30x30", "50x50"],
                          help="矩阵模式地图尺寸 (如 30x30 50x50 100x100)")
    p_stress.add_argument("--densities", nargs="+", default=["0.1", "0.2", "0.3"],
                          help="矩阵模式障碍密度列表 (如 0.1 0.2 0.3)")
    p_stress.add_argument("--merge-sizes", nargs="+", default=["1", "2"],
                          help="矩阵模式 NavMesh 粒度列表 (如 1 2 4)")
    p_stress.add_argument("--export", default=None, metavar="FILE",
                          help="导出报告到文件")
    p_stress.set_defaults(func=cmd_stress)

    p_regress = sub.add_parser("regress", help="回归基线对比")
    p_regress.add_argument("dir", help="场景 JSON 文件目录")
    p_regress.add_argument("--baseline", "-b", required=True,
                           help="基线 JSON 文件路径")
    p_regress.add_argument("--type", "-t", default=None, choices=["grid", "navmesh"],
                           help="只跑某类地图: grid 或 navmesh")
    p_regress.add_argument("--save-baseline", default=None, metavar="FILE",
                           help="同时保存新基线 (更新基线)")
    p_regress.add_argument("--markdown", "--md", default=None,
                           help="导出回归对比 Markdown 报告")
    p_regress.set_defaults(func=cmd_regress)

    p_demo = sub.add_parser("demo", help="运行完整演示")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
