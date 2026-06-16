"""
几何工具模块 — 为导航网格和路径平滑提供几何计算。

包括:
  - 点在多边形内判断 (射线法)
  - 线段相交检测
  - 线段与多边形边的视线检测
  - 向量运算
"""

import math
from typing import List, Tuple


Point = Tuple[float, float]


def cross_2d(o: Point, a: Point, b: Point) -> float:
    """二维叉积 (OA × OB), 用于判断三点转向方向。"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """
    射线法判断点是否在多边形内部。
    从点向右发射水平射线, 计算与多边形边的交叉次数:
    奇数次 → 内部, 偶数次 → 外部。
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    判断线段 p1-p2 与 p3-p4 是否相交(不含端点共线退化)。
    使用叉积符号判断: 若两线段相互"跨越", 则相交。
    """
    d1 = cross_2d(p3, p4, p1)
    d2 = cross_2d(p3, p4, p2)
    d3 = cross_2d(p1, p2, p3)
    d4 = cross_2d(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if abs(d1) < 1e-10 and _on_segment(p3, p4, p1):
        return True
    if abs(d2) < 1e-10 and _on_segment(p3, p4, p2):
        return True
    if abs(d3) < 1e-10 and _on_segment(p1, p2, p3):
        return True
    if abs(d4) < 1e-10 and _on_segment(p1, p2, p4):
        return True

    return False


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """判断点 r 是否在线段 pq 上(已知 r 在 pq 所在直线上)。"""
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def line_of_sight(p1: Point, p2: Point, obstacle_polygons: List[List[Point]]) -> bool:
    """
    视线检测: 判断从 p1 到 p2 的直线是否被任何障碍多边形遮挡。
    用于网格路径平滑 — 若两点间有视线, 则可直接通过, 无需绕行。
    """
    for polygon in obstacle_polygons:
        n = len(polygon)
        for i in range(n):
            if segments_intersect(p1, p2, polygon[i], polygon[(i + 1) % n]):
                return False
    return True


def point_to_point_distance(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def polygon_centroid(polygon: List[Point]) -> Point:
    """计算多边形质心。"""
    n = len(polygon)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


def line_segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    """
    计算两线段的交点(若相交)。
    返回 (t, point) 其中 t 是 p1→p2 上的参数, 或 None。
    """
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]
    d2y = p4[1] - p3[1]

    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return None

    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = p1[0] + t * d1x
        iy = p1[1] + t * d1y
        return (t, (ix, iy))
    return None


def shared_edge(poly_a: List[Point], poly_b: List[Point]):
    """
    找到两个多边形的共享边。
    返回共享边的两个端点, 或 None。
    由于浮点精度, 使用距离容差判断顶点重合。
    """
    eps = 1e-6
    shared = []
    for pa in poly_a:
        for pb in poly_b:
            if point_to_point_distance(pa, pb) < eps:
                shared.append(pa)
                break
    if len(shared) >= 2:
        return (shared[0], shared[1])
    return None
