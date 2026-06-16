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
    point_in_polygon,
    point_to_point_distance,
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
        """查找包含给定点的多边形 ID。"""
        for poly_id, polygon in self.polygons.items():
            if point_in_polygon(point, polygon.vertices):
                return poly_id
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
            cost: 路径总代价
        """
        start_poly = self.find_polygon(start)
        goal_poly = self.find_polygon(goal)

        if start_poly is None or goal_poly is None:
            return [], [], float('inf')

        if start_poly == goal_poly:
            return [start, goal], [start_poly], point_to_point_distance(start, goal)

        astar = self.create_astar(goal_poly, goal, heuristic)
        poly_path, cost = astar.find_path(start_poly, goal_poly)

        if not poly_path:
            return [], [], float('inf')

        from .smoothing import funnel_smooth

        corridor_polygons = [self.polygons[pid].vertices for pid in poly_path]
        smooth_path = funnel_smooth(corridor_polygons, start, goal)

        return smooth_path, poly_path, cost

    def __repr__(self):
        lines = [f"NavMesh(polygons={len(self.polygons)})"]
        for pid, poly in self.polygons.items():
            neighbor_ids = [n for n, _ in poly.neighbors]
            lines.append(f"  {pid}: neighbors={neighbor_ids}, cost={poly.cost}")
        return '\n'.join(lines)
