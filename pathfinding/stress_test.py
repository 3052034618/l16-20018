"""
随机压力测试模块 — 生成随机地图并对比不同启发函数和导航网格的性能。

支持:
  - 随机网格地图 (可配置尺寸、障碍密度、地形比例、随机种子)
  - 随机导航网格 (网格多边形化, 可配置粒度 — 多个格子合并为一个多边形)
  - 多种启发函数对比
  - 网格 vs 导航网格性能对比
  - 多档规模 + 多密度 + 多 navmesh 粒度的矩阵式测试
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .grid import GridMap, TerrainType, TERRAIN_COSTS
from .navmesh import NavMesh
from .benchmark import (
    PathfindingResult,
    run_grid_benchmark,
    run_navmesh_benchmark,
    format_result_table,
    compare_grid_vs_navmesh,
    _path_length,
)
from .heuristics import (
    zero_heuristic,
    manhattan_distance,
    euclidean_distance,
    diagonal_distance,
    chebyshev_distance,
)


DEFAULT_HEURISTICS = {
    "zero (Dijkstra)": zero_heuristic,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "diagonal (octile)": diagonal_distance,
    "chebyshev": chebyshev_distance,
}


def generate_random_grid(
    width: int,
    height: int,
    obstacle_density: float = 0.2,
    terrain_ratios: Optional[Dict[TerrainType, float]] = None,
    seed: Optional[int] = None,
    allow_diagonal: bool = True,
) -> Tuple[GridMap, Tuple[int, int], Tuple[int, int]]:
    """
    生成随机网格地图。

    Args:
        width: 地图宽度
        height: 地图高度
        obstacle_density: 障碍密度 (0.0 ~ 1.0)
        terrain_ratios: 地形类型比例, 如 {TerrainType.FOREST: 0.1, TerrainType.SWAMP: 0.05}
        seed: 随机种子
        allow_diagonal: 是否允许对角线移动

    Returns:
        (grid_map, start, goal)
    """
    if seed is not None:
        random.seed(seed)

    grid = GridMap(width=width, height=height, allow_diagonal=allow_diagonal)

    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_density:
                grid.set_terrain(x, y, TerrainType.WALL)

    if terrain_ratios:
        for terrain_type, ratio in terrain_ratios.items():
            if terrain_type == TerrainType.WALL:
                continue
            count = int(width * height * ratio)
            placed = 0
            attempts = 0
            while placed < count and attempts < count * 10:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if grid.get_terrain(x, y) == TerrainType.PLAIN:
                    grid.set_terrain(x, y, terrain_type)
                    placed += 1
                attempts += 1

    start = _find_free_cell(grid, width, height, top_left=True)
    goal = _find_free_cell(grid, width, height, top_left=False)

    return grid, start, goal


def _find_free_cell(
    grid: GridMap,
    width: int,
    height: int,
    top_left: bool = True,
) -> Tuple[int, int]:
    """找到一个无障碍格子作为起点/终点。"""
    if top_left:
        for y in range(height):
            for x in range(width):
                if grid.get_terrain(x, y) != TerrainType.WALL:
                    return (x, y)
    else:
        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                if grid.get_terrain(x, y) != TerrainType.WALL:
                    return (x, y)
    return (0, 0)


def grid_to_navmesh(
    grid: GridMap,
    cell_size: float = 1.0,
    merge_size: int = 1,
) -> Tuple[NavMesh, Tuple[float, float], Tuple[float, float]]:
    """
    将网格地图转换为导航网格。

    merge_size 控制粒度: merge_size=N 时, 每 N×N 个格子合并为一个多边形。
    merge_size=1 时每个非障碍格子对应一个多边形 (最细粒度)。
    merge_size=2 时每 2×2 格子合并 (4 倍粒度)。
    合并块内含障碍时跳过, 仅含可通行格子时取平均代价。

    Args:
        grid: 网格地图
        cell_size: 每个格子对应的世界坐标大小
        merge_size: 合并粒度 (1=每格一个多边形, 2=2x2, 3=3x3, ...)

    Returns:
        (navmesh, nm_start, nm_goal)
    """
    nm = NavMesh()
    ms = max(1, merge_size)

    for block_y in range(0, grid.height, ms):
        for block_x in range(0, grid.width, ms):
            passable_cells = []
            total_cost = 0.0
            any_wall = False

            for dy in range(ms):
                for dx in range(ms):
                    cx, cy = block_x + dx, block_y + dy
                    if cx >= grid.width or cy >= grid.height:
                        any_wall = True
                        break
                    terrain = grid.get_terrain(cx, cy)
                    if terrain == TerrainType.WALL:
                        any_wall = True
                        break
                    total_cost += TERRAIN_COSTS.get(terrain, 1.0)
                    passable_cells.append((cx, cy))
                if any_wall:
                    break

            if any_wall or not passable_cells:
                continue

            avg_cost = total_cost / len(passable_cells)
            x0 = block_x * cell_size
            y0 = block_y * cell_size
            x1 = min(block_x + ms, grid.width) * cell_size
            y1 = min(block_y + ms, grid.height) * cell_size

            nm.add_polygon(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                cost=avg_cost,
            )

    return nm


def run_stress_test(
    width: int = 50,
    height: int = 50,
    obstacle_density: float = 0.2,
    terrain_ratios: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    run_navmesh: bool = True,
    navmesh_cell_size: float = 1.0,
    navmesh_merge_size: int = 1,
    heuristics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    运行一次随机压力测试。

    Args:
        width: 地图宽度
        height: 地图高度
        obstacle_density: 障碍密度
        terrain_ratios: 地形比例 (用名字) {"forest": 0.1, "swamp": 0.05}
        seed: 随机种子
        run_navmesh: 是否也跑导航网格
        navmesh_cell_size: 导航网格单元大小
        navmesh_merge_size: navmesh 合并粒度 (1=每格一多边形, 2=2x2...)
        heuristics: 启发函数字典

    Returns:
        包含网格结果、navmesh结果、对比数据的字典
    """
    terrain_map = {
        "plain": TerrainType.PLAIN,
        "forest": TerrainType.FOREST,
        "swamp": TerrainType.SWAMP,
        "hill": TerrainType.HILL,
    }

    ratios = None
    if terrain_ratios:
        ratios = {}
        for name, r in terrain_ratios.items():
            tt = terrain_map.get(name)
            if tt:
                ratios[tt] = r

    grid, start, goal = generate_random_grid(
        width, height, obstacle_density, ratios, seed, allow_diagonal=True,
    )

    if heuristics is None:
        heuristics = DEFAULT_HEURISTICS

    grid_results = run_grid_benchmark(grid, start, goal, heuristics=heuristics)

    result = {
        "width": width,
        "height": height,
        "obstacle_density": obstacle_density,
        "seed": seed,
        "grid_results": grid_results,
        "grid_start": start,
        "grid_goal": goal,
        "navmesh_merge_size": navmesh_merge_size,
    }

    if run_navmesh:
        nm = grid_to_navmesh(grid, cell_size=navmesh_cell_size, merge_size=navmesh_merge_size)
        nm_start = (start[0] + 0.5, start[1] + 0.5)
        nm_goal = (goal[0] + 0.5, goal[1] + 0.5)
        nm_result = run_navmesh_benchmark(nm, nm_start, nm_goal)
        result["navmesh_result"] = nm_result
        result["navmesh_start"] = nm_start
        result["navmesh_goal"] = nm_goal
        result["navmesh_poly_count"] = len(nm.polygons)

    return result


def format_stress_report(result: Dict[str, Any]) -> str:
    """格式化压力测试报告为可读文本。"""
    lines = []

    lines.append("随机压力测试报告")
    lines.append("=" * 90)
    lines.append(f"地图大小: {result['width']} × {result['height']}")
    lines.append(f"障碍密度: {result['obstacle_density']:.1%}")
    lines.append(f"随机种子: {result['seed']}")
    lines.append(f"NavMesh粒度: merge={result.get('navmesh_merge_size', 1)}")
    lines.append(f"起点: {result['grid_start']}  终点: {result['grid_goal']}")
    lines.append("")

    lines.append("【网格地图 - 各启发函数对比】")
    lines.append(format_result_table(result["grid_results"], title=""))
    lines.append("")

    if "navmesh_result" in result:
        nm_r = result["navmesh_result"]
        lines.append(f"【导航网格 - {result['navmesh_poly_count']} 个多边形 (merge={result.get('navmesh_merge_size', 1)})】")
        lines.append(format_result_table([nm_r], title=""))
        lines.append("")

        grid_best = None
        for r in result["grid_results"]:
            if r.found and (grid_best is None or r.nodes_expanded < grid_best.nodes_expanded):
                grid_best = r

        if grid_best and nm_r.found:
            if nm_r.nodes_expanded > 0:
                node_ratio = grid_best.nodes_expanded / nm_r.nodes_expanded
                lines.append(f"节点展开比 (Grid/NavMesh): {node_ratio:.1f}x")
            if nm_r.time_ms > 0:
                time_ratio = grid_best.time_ms / nm_r.time_ms
                lines.append(f"耗时比 (Grid/NavMesh): {time_ratio:.1f}x")

    return '\n'.join(lines)


def run_multiple_stress_tests(
    configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """运行多组压力测试配置。"""
    results = []
    for cfg in configs:
        r = run_stress_test(**cfg)
        results.append(r)
    return results


def format_multi_stress_report(results: List[Dict[str, Any]]) -> str:
    """格式化多组压力测试汇总报告。"""
    lines = []
    lines.append("多组压力测试汇总")
    lines.append("=" * 110)

    header = (
        f"{'尺寸':<12} {'密度':<6} {'粒度':<4} {'种子':<6} "
        f"{'Grid展开':>8} {'Grid耗时':>10} {'Grid代价':>8} "
        f"{'NM展开':>8} {'NM耗时':>10} {'NM代价':>8} {'NM多边形':>8} "
        f"{'节点比':>6} {'时间比':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        grid_best = None
        for gr in r["grid_results"]:
            if gr.found and (grid_best is None or gr.nodes_expanded < grid_best.nodes_expanded):
                grid_best = gr

        nm_r = r.get("navmesh_result")
        size_str = f"{r['width']}x{r['height']}"
        density_str = f"{r['obstacle_density']:.0%}"
        merge_str = str(r.get("navmesh_merge_size", 1))
        seed_str = str(r.get("seed", "—"))

        grid_exp = f"{grid_best.nodes_expanded}" if grid_best else "—"
        grid_time = f"{grid_best.time_ms:.1f}ms" if grid_best else "—"
        grid_cost = f"{grid_best.total_cost:.1f}" if grid_best else "—"

        nm_exp = f"{nm_r.nodes_expanded}" if (nm_r and nm_r.found) else "—"
        nm_time = f"{nm_r.time_ms:.1f}ms" if (nm_r and nm_r.found) else "—"
        nm_cost = f"{nm_r.total_cost:.1f}" if (nm_r and nm_r.found) else "—"
        nm_polys = f"{r.get('navmesh_poly_count', '—')}" if nm_r else "—"

        node_ratio = "—"
        time_ratio = "—"

        if grid_best and nm_r and grid_best.found and nm_r.found:
            if nm_r.nodes_expanded > 0:
                node_ratio = f"{grid_best.nodes_expanded / nm_r.nodes_expanded:.1f}x"
            if nm_r.time_ms > 0:
                time_ratio = f"{grid_best.time_ms / nm_r.time_ms:.1f}x"

        lines.append(
            f"{size_str:<12} {density_str:<6} {merge_str:<4} {seed_str:<6} "
            f"{grid_exp:>8} {grid_time:>10} {grid_cost:>8} "
            f"{nm_exp:>8} {nm_time:>10} {nm_cost:>8} {nm_polys:>8} "
            f"{node_ratio:>6} {time_ratio:>6}"
        )

    return '\n'.join(lines)


def build_matrix_configs(
    sizes: Optional[List[Tuple[int, int]]] = None,
    densities: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    merge_sizes: Optional[List[int]] = None,
    terrain_ratios: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    构建矩阵式测试配置 — 对所有 (尺寸×密度×种子×粒度) 组合生成配置。

    Args:
        sizes: [(width, height), ...] 如 [(30,30), (50,50), (100,100)]
        densities: [0.1, 0.2, 0.3, ...]
        seeds: [42, 123, 456, ...]
        merge_sizes: [1, 2, 4, ...] navmesh 粒度
        terrain_ratios: 地形比例

    Returns:
        配置字典列表
    """
    if sizes is None:
        sizes = [(30, 30), (50, 50), (100, 100)]
    if densities is None:
        densities = [0.1, 0.2, 0.3]
    if seeds is None:
        seeds = [42, 123]
    if merge_sizes is None:
        merge_sizes = [1, 2, 4]

    configs = []
    for w, h in sizes:
        for d in densities:
            for s in seeds:
                for ms in merge_sizes:
                    cfg = {
                        "width": w,
                        "height": h,
                        "obstacle_density": d,
                        "seed": s,
                        "navmesh_merge_size": ms,
                    }
                    if terrain_ratios:
                        cfg["terrain_ratios"] = terrain_ratios
                    configs.append(cfg)
    return configs
