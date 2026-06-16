"""
导航网格(NavMesh)模块 — 基于凸多边形的地图表示与路径查找。

导航网格相比格子地图在大地图上的优势
=====================================

1. 内存效率:
   格子地图: O(W × H) 存储, 1000×1000 地图需要 100 万个格子
   导航网格: O(P) 存储, P 为多边形数量, 通常几百到几千个
   大地图上, navmesh 内存占用可低几个数量级。

2. 搜索效率:
   格子地图 A* 探索节点数与地图面积成正比
   NavMesh A* 只在多边形之间跳转, 探索节点数与多边形数成正比
   大地图上, navmesh 搜索速度显著更快。

3. 路径质量:
   格子路径是锯齿状的离散点序列, 需额外平滑
   NavMesh 路径经漏斗算法后是连续直线段, 更自然流畅

4. 精度:
   格子地图的分辨率受格子大小限制
   NavMesh 可表示任意形状的可通行区域, 精度不受限

5. 动态更新:
   添加/移除障碍物时, 只需修改受影响的多边形
   格子地图可能需要更新大量格子

多边形相邻关系如何表示
=======================

NavMesh 的图结构: 每个凸多边形是一个节点, 共享边的多边形互为邻居。

相邻关系的表示方法:
  1. 边邻接表(Edge Adjacency):
     对每条边, 记录其两侧的多边形 ID。
     shared_edge_map: {(v1, v2): [poly_a_id, poly_b_id]}
     查询: 给定多边形, 遍历其所有边, 在 map 中查找邻居

  2. 多边形邻接表(Polygon Adjacency):
     对每个多边形, 直接存储邻居 ID 列表。
     adjacency: {poly_id: [(neighbor_id, shared_edge), ...]}

本实现采用方法 2, 同时保存共享边信息, 用于:
  - A* 计算移动代价(通过共享边中点的距离)
  - 漏斗算法提取走廊
"""

import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .core import AStar
from .geometry import (
    Point,
    line_segment_intersection,
    point_in_polygon,
    point_in_polygon_or_on_edge,
    point_on_polygon_edge,
    point_to_point_distance,
    point_to_segment_distance,
    polygon_centroid,
    shared_edge,
)
from .heuristics import euclidean_distance


class NavMeshPolygon:
    """导航网格中的一个凸多边形。"""

    def __init__(self, poly_id: int, vertices: List[Point], cost: float = 1.0):
        self.poly_id = poly_id
        self.vertices = vertices
        self.cost = cost
        self.centroid = polygon_centroid(vertices)
        self.neighbors: List[Tuple[int, Tuple[Point, Point]]] = []

    def add_neighbor(self, neighbor_id: int, shared: Tuple[Point, Point]):
        self.neighbors.append((neighbor_id, shared))

    def __repr__(self):
        return f"Poly({self.poly_id}, verts={len(self.vertices)}, neighbors={len(self.neighbors)})"


class NavMesh:
    """
    导航网格: 由凸多边形组成的可通行区域图。

    多边形之间的相邻关系通过共享边建立。
    A* 在多边形粒度上搜索, 从起点所在多边形到终点所在多边形。
    """

    def __init__(self):
        self.polygons: Dict[int, NavMeshPolygon] = {}
        self._next_id = 0

    def add_polygon(self, vertices: List[Point], cost: float = 1.0) -> int:
        """
        添加一个凸多边形到导航网格, 返回其 ID。
        自动检测与已有多边形的共享边并建立邻接关系。
        """
        poly_id = self._next_id
        self._next_id += 1

        polygon = NavMeshPolygon(poly_id, vertices, cost)

        for existing_id, existing_poly in self.polygons.items():
            edge = shared_edge(polygon.vertices, existing_poly.vertices)
            if edge:
                polygon.add_neighbor(existing_id, edge)
                existing_poly.add_neighbor(poly_id, edge)

        self.polygons[poly_id] = polygon
        return poly_id

    def find_polygon(self, point: Point) -> Optional[int]:
        """查找包含给定点的多边形 ID。点在边上时返回第一个包含它的多边形。"""
        for poly_id, polygon in self.polygons.items():
            if point_in_polygon_or_on_edge(point, polygon.vertices):
                return poly_id
        return None

    def find_all_polygons(self, point: Point) -> List[int]:
        """查找所有包含给定点的多边形 ID (用于处理共享边上的点)。"""
        result = []
        for poly_id, polygon in self.polygons.items():
            if point_in_polygon_or_on_edge(point, polygon.vertices):
                result.append(poly_id)
        return result

    def point_on_shared_edge(self, point: Point) -> Optional[Tuple[int, int]]:
        """检查点是否在两个多边形的共享边上, 返回 (poly_a_id, poly_b_id) 或 None。"""
        polys = self.find_all_polygons(point)
        if len(polys) >= 2:
            return (polys[0], polys[1])
        return None

    def get_neighbors_with_cost(
        self,
        poly_id: int,
        goal_poly_id: int,
        goal_point: Optional[Point] = None,
    ) -> List[Tuple[int, float]]:
        """
        获取多边形的邻居及移动代价。

        代价计算: 从当前多边形质心到共享边中点的距离 × 目标多边形地形代价。
        这是启发式的近似 — 精确代价取决于最终路径上的穿越点。

        不同地形代价的并入方式:
          move_cost = distance × target_polygon.cost
          即穿越代价更高的多边形需要付出更多。
        """
        if poly_id not in self.polygons:
            return []

        current = self.polygons[poly_id]
        result = []

        for neighbor_id, shared in current.neighbors:
            neighbor = self.polygons[neighbor_id]

            mid_shared = (
                (shared[0][0] + shared[1][0]) / 2,
                (shared[0][1] + shared[1][1]) / 2,
            )

            dist = point_to_point_distance(current.centroid, mid_shared)
            move_cost = dist * neighbor.cost

            result.append((neighbor_id, move_cost))

        return result

    def create_astar(
        self,
        goal_poly_id: int,
        goal_point: Optional[Point] = None,
        heuristic: Optional[Callable] = None,
    ) -> AStar:
        """
        创建适用于本导航网格的 A* 实例。

        启发函数: 默认使用多边形质心间的欧几里得距离。
        由于质心距离 ≤ 实际最短路径距离(凸多边形性质), 满足可容许性。
        """
        if heuristic is None:
            def default_heuristic(n, goal):
                if n in self.polygons and goal in self.polygons:
                    return euclidean_distance(
                        self.polygons[n].centroid,
                        self.polygons[goal].centroid,
                    )
                return 0.0
            heuristic = default_heuristic

        def get_neighbors(poly_id):
            return self.get_neighbors_with_cost(poly_id, goal_poly_id, goal_point)

        return AStar(
            get_neighbors=get_neighbors,
            heuristic=heuristic,
        )

    def find_path(
        self,
        start: Point,
        goal: Point,
        heuristic: Optional[Callable] = None,
    ) -> Tuple[List[Point], List[int], float]:
        """
        在导航网格上从 start 到 goal 寻路。

        Returns:
            smooth_path: 漏斗算法平滑后的路径点列表
            corridor: 经过的多边形 ID 序列
            cost: 路径总代价 (按实际折线路径与多边形代价计算)
        """
        smooth_path, corridor, cost, _ = self.find_path_detail(start, goal, heuristic)
        return smooth_path, corridor, cost

    def find_path_detail(
        self,
        start: Point,
        goal: Point,
        heuristic: Optional[Callable] = None,
    ) -> Tuple[List[Point], List[int], float, dict]:
        """
        在导航网格上寻路, 返回详细统计信息。

        Returns:
            smooth_path: 漏斗算法平滑后的路径点列表
            corridor: 经过的多边形 ID 序列
            cost: 路径总代价 (按实际折线路径与多边形代价计算)
            stats: 统计信息 (nodes_expanded, nodes_generated, etc.)
        """
        start_polys = self.find_all_polygons(start)
        goal_polys = self.find_all_polygons(goal)

        if not start_polys or not goal_polys:
            stats = {"nodes_expanded": 0, "nodes_generated": 0, "max_open_size": 0, "found": False,
                     "start_valid": bool(start_polys), "goal_valid": bool(goal_polys)}
            return [], [], float('inf'), stats

        if start_polys[0] == goal_polys[0] or set(start_polys) & set(goal_polys):
            same_poly = (set(start_polys) & set(goal_polys)).pop()
            cost = point_to_point_distance(start, goal) * self.polygons[same_poly].cost
            stats = {"nodes_expanded": 0, "nodes_generated": 0, "max_open_size": 0, "found": True,
                     "start_valid": True, "goal_valid": True}
            return [start, goal], [same_poly], cost, stats

        best_result = None
        best_cost = float('inf')

        for sp in start_polys:
            for gp in goal_polys:
                astar = self.create_astar(gp, goal, heuristic)
                poly_path, _, astar_stats = astar.find_path_detail(sp, gp)

                if not poly_path:
                    continue

                from .smoothing import funnel_smooth

                corridor_polygons = [self.polygons[pid].vertices for pid in poly_path]
                smooth_path = funnel_smooth(corridor_polygons, start, goal)

                actual_cost = self.compute_path_cost(smooth_path, poly_path)

                if actual_cost < best_cost:
                    best_cost = actual_cost
                    best_result = (smooth_path, poly_path, actual_cost, astar_stats)

        if best_result is None:
            stats = {"nodes_expanded": 0, "nodes_generated": 0, "max_open_size": 0, "found": False,
                     "start_valid": True, "goal_valid": True}
            return [], [], float('inf'), stats

        smooth_path, corridor, cost, astar_stats = best_result
        stats = {
            **astar_stats,
            "start_valid": True,
            "goal_valid": True,
        }
        return smooth_path, corridor, cost, stats

    def compute_path_cost(
        self,
        smooth_path: List[Point],
        corridor: List[int],
    ) -> float:
        """
        根据实际折线路径和多边形地形代价计算总代价。

        对于路径上每个线段, 计算其在每个多边形内的长度,
        乘以该多边形的 cost, 累加得到总代价。

        Args:
            smooth_path: 平滑后的路径点列表
            corridor: 经过的多边形 ID 序列

        Returns:
            总代价 (几何长度 × 地形代价)
        """
        if len(smooth_path) < 2:
            return 0.0

        total_cost = 0.0

        shared_edges = []
        for i in range(len(corridor) - 1):
            edge = shared_edge(
                self.polygons[corridor[i]].vertices,
                self.polygons[corridor[i + 1]].vertices,
            )
            shared_edges.append(edge)

        path_t = [0.0]
        current_poly_idx = 0
        seg_start_idx = 0
        seg_start_t = 0.0

        cumulative_len = 0.0
        seg_lengths = []
        for i in range(len(smooth_path) - 1):
            length = point_to_point_distance(smooth_path[i], smooth_path[i + 1])
            seg_lengths.append(length)
            cumulative_len += length

        total_length = cumulative_len
        if total_length < 1e-10:
            start_poly = corridor[0]
            return 0.0

        dist_so_far = 0.0
        for seg_i in range(len(smooth_path) - 1):
            p1 = smooth_path[seg_i]
            p2 = smooth_path[seg_i + 1]
            seg_len = seg_lengths[seg_i]

            crossings = []

            for edge_i, edge in enumerate(shared_edges[current_poly_idx:], start=current_poly_idx):
                if edge is None:
                    continue
                result = line_segment_intersection(p1, p2, edge[0], edge[1])
                if result is not None:
                    t_in_seg, pt = result
                    if t_in_seg > 1e-10 and t_in_seg < 1 - 1e-10:
                        crossings.append((t_in_seg, edge_i))

            crossings.sort(key=lambda x: x[0])

            prev_t = 0.0
            for t_in_seg, edge_idx in crossings:
                seg_dist = (t_in_seg - prev_t) * seg_len
                total_cost += seg_dist * self.polygons[corridor[current_poly_idx]].cost
                prev_t = t_in_seg
                current_poly_idx = max(current_poly_idx, edge_idx + 1)
                if current_poly_idx >= len(corridor):
                    current_poly_idx = len(corridor) - 1

            remaining_dist = (1.0 - prev_t) * seg_len
            if remaining_dist > 1e-10:
                total_cost += remaining_dist * self.polygons[corridor[current_poly_idx]].cost

        return total_cost

    def __repr__(self):
        lines = [f"NavMesh(polygons={len(self.polygons)})"]
        for pid, poly in self.polygons.items():
            neighbor_ids = [n for n, _ in poly.neighbors]
            lines.append(f"  {pid}: neighbors={neighbor_ids}, cost={poly.cost}")
        return '\n'.join(lines)
