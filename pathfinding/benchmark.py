"""
基准测试与批处理模块 — 支持批量跑场景、对比启发函数、输出详细统计。

提供功能:
  - run_benchmark: 单张地图 + 多个启发函数 → 对比表
  - run_batch: 批量 JSON 地图 → 汇总报告
  - format_result_table: 格式化输出结果表格
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .grid import GridMap
from .navmesh import NavMesh
from .smoothing import smooth_grid_path
from .heuristics import (
    diagonal_distance,
    euclidean_distance,
    manhattan_distance,
    zero_heuristic,
)


DEFAULT_GRID_HEURISTICS = {
    "zero (Dijkstra)": zero_heuristic,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "diagonal (octile)": diagonal_distance,
}


class PathfindingResult:
    """单次寻路的结果封装。"""

    def __init__(self):
        self.map_name: str = ""
        self.map_type: str = ""
        self.heuristic_name: str = ""
        self.found: bool = False
        self.start_valid: bool = True
        self.goal_valid: bool = True
        self.path_nodes: int = 0
        self.total_cost: float = 0.0
        self.nodes_expanded: int = 0
        self.nodes_generated: int = 0
        self.max_open_size: int = 0
        self.time_ms: float = 0.0
        self.raw_path_length: float = 0.0
        self.smooth_path_length: float = 0.0
        self.corridor_length: int = 0
        self.path: List = []
        self.smooth_path: List = []
        self.corridor: List = []

    def summary(self) -> str:
        if not self.found:
            reason = ""
            if not self.start_valid:
                reason = " (起点无效)"
            elif not self.goal_valid:
                reason = " (终点无效)"
            return f"未找到路径{reason}"
        return (
            f"代价={self.total_cost:.2f}, "
            f"展开={self.nodes_expanded}, "
            f"生成={self.nodes_generated}, "
            f"耗时={self.time_ms:.2f}ms"
        )

    def path_summary(self, verbose: bool = False) -> str:
        """格式化路径摘要。verbose=True 显示完整坐标, 否则只显示首尾。"""
        if not self.found:
            return "无路径"
        if self.map_type == "grid":
            if not self.path:
                return "无路径"
            n = len(self.path)
            if verbose:
                pts = [f"({p[0]},{p[1]})" for p in self.path]
                return f"[{n}步] " + "→".join(pts)
            if n <= 4:
                pts = [f"({p[0]},{p[1]})" for p in self.path]
                return f"[{n}步] " + "→".join(pts)
            return (f"[{n}步] ({self.path[0][0]},{self.path[0][1]})"
                    f"→...({self.path[n//2][0]},{self.path[n//2][1]})..."
                    f"→({self.path[-1][0]},{self.path[-1][1]})")
        else:
            if not self.smooth_path:
                return "无路径"
            n = len(self.smooth_path)
            if verbose:
                pts = [f"({p[0]:.1f},{p[1]:.1f})" for p in self.smooth_path]
                return f"[{n}点] " + "→".join(pts)
            if n <= 4:
                pts = [f"({p[0]:.1f},{p[1]:.1f})" for p in self.smooth_path]
                return f"[{n}点] " + "→".join(pts)
            return (f"[{n}点] ({self.smooth_path[0][0]:.1f},{self.smooth_path[0][1]:.1f})"
                    f"→...→({self.smooth_path[-1][0]:.1f},{self.smooth_path[-1][1]:.1f})")


def run_grid_benchmark(
    grid_map: GridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristics: Optional[Dict[str, Callable]] = None,
    do_smoothing: bool = True,
) -> List[PathfindingResult]:
    """
    在网格地图上用多种启发函数跑寻路, 返回结果列表。

    Args:
        grid_map: 网格地图
        start: 起点格子坐标
        goal: 终点格子坐标
        heuristics: {name: heuristic_func} 字典, None 则用默认集
        do_smoothing: 是否执行路径平滑

    Returns:
        每个启发函数对应一个 PathfindingResult
    """
    if heuristics is None:
        heuristics = DEFAULT_GRID_HEURISTICS

    results = []
    obstacle_polys = grid_map.get_obstacle_polygons() if do_smoothing else []

    for name, h_func in heuristics.items():
        result = PathfindingResult()
        result.map_type = "grid"
        result.heuristic_name = name

        t0 = time.perf_counter()
        path, cost, stats = grid_map.find_path_detail(start, goal, heuristic=h_func)
        t1 = time.perf_counter()

        result.found = stats["found"]
        result.start_valid = stats.get("start_valid", True)
        result.goal_valid = stats.get("goal_valid", True)
        result.path_nodes = len(path)
        result.total_cost = cost
        result.nodes_expanded = stats["nodes_expanded"]
        result.nodes_generated = stats["nodes_generated"]
        result.max_open_size = stats["max_open_size"]
        result.time_ms = (t1 - t0) * 1000.0
        result.path = path

        if path:
            world_pts = grid_map.path_to_world_coords(path)
            result.raw_path_length = _path_length(world_pts)

            if do_smoothing:
                smoothed = smooth_grid_path(path, obstacle_polys, world_coords=True)
                result.smooth_path = smoothed
                result.smooth_path_length = _path_length(smoothed)

        results.append(result)

    return results


def run_navmesh_benchmark(
    navmesh: NavMesh,
    start: Tuple[float, float],
    goal: Tuple[float, float],
) -> PathfindingResult:
    """
    在导航网格上跑寻路, 返回结果。

    NavMesh 只支持一种(质心欧几里得)启发函数, 因此只返回单个结果。
    """
    result = PathfindingResult()
    result.map_type = "navmesh"
    result.heuristic_name = "centroid euclidean"

    t0 = time.perf_counter()
    smooth_path, corridor, cost, stats = navmesh.find_path_detail(start, goal)
    t1 = time.perf_counter()

    result.found = stats["found"]
    result.start_valid = stats.get("start_valid", True)
    result.goal_valid = stats.get("goal_valid", True)
    result.path_nodes = len(smooth_path)
    result.total_cost = cost
    result.nodes_expanded = stats["nodes_expanded"]
    result.nodes_generated = stats["nodes_generated"]
    result.max_open_size = stats["max_open_size"]
    result.time_ms = (t1 - t0) * 1000.0
    result.smooth_path = smooth_path
    result.corridor = corridor
    result.corridor_length = len(corridor)
    result.smooth_path_length = _path_length(smooth_path)

    return result


def format_result_table(
    results: List[PathfindingResult],
    title: str = "",
) -> str:
    """
    将一组寻路结果格式化为可读的表格字符串。
    """
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * 80)

    if not results:
        lines.append("(无结果)")
        return '\n'.join(lines)

    map_type = results[0].map_type

    if map_type == "grid":
        header = f"{'启发函数':<20} {'找到':<4} {'代价':>8} {'展开节点':>8} {'生成节点':>8} {'Open峰值':>8} {'耗时ms':>8} {'原始长':>8} {'平滑长':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in results:
            found = "✓" if r.found else "✗"
            cost = f"{r.total_cost:.2f}" if r.found else "—"
            raw_len = f"{r.raw_path_length:.2f}" if r.found else "—"
            smooth_len = f"{r.smooth_path_length:.2f}" if r.found and r.smooth_path_length > 0 else "—"
            lines.append(
                f"{r.heuristic_name:<20} {found:<4} {cost:>8} "
                f"{r.nodes_expanded:>8} {r.nodes_generated:>8} "
                f"{r.max_open_size:>8} {r.time_ms:>8.3f} "
                f"{raw_len:>8} {smooth_len:>8}"
            )

    elif map_type == "navmesh":
        header = f"{'方法':<22} {'找到':<4} {'代价':>8} {'多边形数':>8} {'展开节点':>8} {'生成节点':>8} {'Open峰值':>8} {'耗时ms':>8} {'路径长':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in results:
            found = "✓" if r.found else "✗"
            cost = f"{r.total_cost:.2f}" if r.found else "—"
            plen = f"{r.smooth_path_length:.2f}" if r.found else "—"
            lines.append(
                f"{r.heuristic_name:<22} {found:<4} {cost:>8} "
                f"{r.corridor_length:>8} {r.nodes_expanded:>8} "
                f"{r.nodes_generated:>8} {r.max_open_size:>8} "
                f"{r.time_ms:>8.3f} {plen:>8}"
            )

    return '\n'.join(lines)


def compare_grid_vs_navmesh(
    grid_map: GridMap,
    grid_start: Tuple[int, int],
    grid_goal: Tuple[int, int],
    navmesh: NavMesh,
    nm_start: Tuple[float, float],
    nm_goal: Tuple[float, float],
) -> str:
    """
    对比网格和导航网格在同一场景下的性能, 返回对比表格字符串。
    """
    grid_results = run_grid_benchmark(grid_map, grid_start, grid_goal,
                                      heuristics={"diagonal (octile)": diagonal_distance})
    nm_result = run_navmesh_benchmark(navmesh, nm_start, nm_goal)

    lines = []
    lines.append("网格 vs 导航网格 性能对比")
    lines.append("=" * 90)

    header = (
        f"{'方式':<22} {'找到':<4} {'代价':>8} "
        f"{'展开节点':>10} {'生成节点':>10} {'Open峰值':>10} "
        f"{'耗时ms':>10} {'路径长':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in grid_results:
        found = "✓" if r.found else "✗"
        cost = f"{r.total_cost:.2f}" if r.found else "—"
        plen = f"{r.smooth_path_length:.2f}" if r.found else "—"
        lines.append(
            f"{'Grid (A*+octile)':<22} {found:<4} {cost:>8} "
            f"{r.nodes_expanded:>10} {r.nodes_generated:>10} {r.max_open_size:>10} "
            f"{r.time_ms:>10.3f} {plen:>10}"
        )

    r = nm_result
    found = "✓" if r.found else "✗"
    cost = f"{r.total_cost:.2f}" if r.found else "—"
    plen = f"{r.smooth_path_length:.2f}" if r.found else "—"
    lines.append(
        f"{'NavMesh (A*+funnel)':<22} {found:<4} {cost:>8} "
        f"{r.nodes_expanded:>10} {r.nodes_generated:>10} {r.max_open_size:>10} "
        f"{r.time_ms:>10.3f} {plen:>10}"
    )

    if grid_results and grid_results[0].found and nm_result.found:
        grid_expanded = grid_results[0].nodes_expanded
        nm_expanded = nm_result.nodes_expanded
        if nm_expanded > 0 and grid_expanded > 0:
            ratio = grid_expanded / nm_expanded
            lines.append("")
            lines.append(
                f"节点展开比 (Grid/NavMesh): {ratio:.1f}x"
            )
            if grid_results[0].time_ms > 0 and nm_result.time_ms > 0:
                time_ratio = grid_results[0].time_ms / nm_result.time_ms
                lines.append(
                    f"耗时比 (Grid/NavMesh): {time_ratio:.1f}x"
                )

    return '\n'.join(lines)


def _path_length(points: List) -> float:
    """计算路径的几何总长度。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total
