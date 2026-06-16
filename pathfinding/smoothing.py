"""
路径平滑模块 — 网格路径的视线平滑 与 导航网格的漏斗(Funnel)算法。

找到的网格路径呈锯齿状如何通过视线检测做平滑
=============================================
网格 A* 的路径由离散格子组成, 移动方向在格子间频繁切换,
呈现"锯齿状"折线, 不适合角色直接跟随。

视线平滑 (Line-of-Sight Smoothing) 的核心思想:
  若路径上两个非相邻节点之间存在无障碍直线(视线),
  则中间的节点可以跳过, 用直线段替代折线。

算法(贪心视线检测):
  1. 从路径起点 start 开始
  2. 从路径末尾向前查找, 找到与 start 有视线的最远节点 farthest
  3. 将 farthest 加入平滑路径, 以 farthest 为新起点重复步骤 2
  4. 直到到达终点

这等价于一种简化: 逐步"拉直"路径, 用长直线段替代短折线段。
结果路径更短、更平滑, 适合角色移动。

导航网格上的漏斗(Funnel)算法
==============================
在 navmesh 中, A* 返回的是多边形序列(走廊),
需要将走廊"压缩"为穿过共享边的最短路径。

漏斗算法:
  将走廊的共享边视为"漏斗口", 维护左右两条边界线,
  逐步收紧漏斗直到顶点处"夹紧", 此时产生一个路径拐点。
  漏斗算法在凸走廊中保证找到最短路径。
"""

from typing import List, Optional, Tuple

from .geometry import (
    Point,
    cross_2d,
    line_of_sight,
    point_to_point_distance,
)


def smooth_grid_path(
    path: List[Tuple[int, int]],
    obstacle_polygons: List[List[Point]],
    world_coords: bool = True,
) -> List[Point]:
    """
    使用视线检测对网格路径做平滑。

    将格子坐标转为世界坐标(格子中心), 然后用贪心视线检测
    逐步拉直路径: 从当前点出发, 找到视线可达的最远点。

    Args:
        path: 网格 A* 返回的格子坐标路径
        obstacle_polygons: 障碍物的多边形列表
        world_coords: 是否将格子坐标转为世界坐标(格子中心)

    Returns:
        平滑后的世界坐标路径
    """
    if len(path) <= 2:
        if world_coords:
            return [(x + 0.5, y + 0.5) for x, y in path]
        return list(path)

    if world_coords:
        waypoints = [(x + 0.5, y + 0.5) for x, y in path]
    else:
        waypoints = list(path)

    smoothed = [waypoints[0]]
    current = 0

    while current < len(waypoints) - 1:
        farthest = current + 1
        for i in range(len(waypoints) - 1, current + 1, -1):
            if line_of_sight(waypoints[current], waypoints[i], obstacle_polygons):
                farthest = i
                break
        smoothed.append(waypoints[farthest])
        current = farthest

    return smoothed


def funnel_smooth(
    corridor: List[List[Point]],
    start: Point,
    goal: Point,
) -> List[Point]:
    """
    漏斗(Funnel)算法: 在导航网格走廊中找到最短路径。

    corridor 是 A* 返回的多边形序列形成的"走廊"。
    每对相邻多边形之间有一条共享边(portal)。
    漏斗算法维护一个"漏斗"区域, 由左右边界界定,
    逐步收紧直到漏斗口夹紧, 产生路径拐点。

    算法步骤:
      1. 初始化漏斗: 顶点 = start, 左右边界 = start
      2. 对每条共享边(portal), 更新左右边界
      3. 若新左边界在右边界右侧(或反之), 漏斗"夹紧",
         将当前顶点加入路径, 以夹紧侧的边界点为新的漏斗顶点
      4. 最后将 goal 加入路径

    Args:
        corridor: 多边形列表(按路径顺序), 每个多边形是顶点列表
        start: 起始点坐标
        goal: 目标点坐标

    Returns:
        平滑后的路径点列表
    """
    if not corridor:
        return [start, goal]

    portals = _extract_portals(corridor)
    return _funnel_algorithm(portals, start, goal)


def _extract_portals(corridor: List[List[Point]]) -> List[Tuple[Point, Point]]:
    """
    从走廊(多边形序列)中提取共享边(portal)列表。

    每条 portal 是两个相邻多边形的共享边, 用 (left, right) 端点表示。
    漏斗算法要求左右端点相对于"行走方向"保持一致:
      当从当前多边形走向下一个多边形时, left 在左侧, right 在右侧。

    通过计算行走方向与共享边端点的叉积来确定正确的左右分配。
    """
    from .geometry import shared_edge, polygon_centroid

    portals = []
    for i in range(len(corridor) - 1):
        edge = shared_edge(corridor[i], corridor[i + 1])
        if edge:
            centroid_from = polygon_centroid(corridor[i])
            centroid_to = polygon_centroid(corridor[i + 1])
            dx = centroid_to[0] - centroid_from[0]
            dy = centroid_to[1] - centroid_from[1]

            ax, ay = edge[0]
            bx, by = edge[1]
            cross_a = dx * (ay - centroid_from[1]) - dy * (ax - centroid_from[0])
            cross_b = dx * (by - centroid_from[1]) - dy * (bx - centroid_from[0])

            if cross_a > cross_b:
                portals.append((edge[0], edge[1]))
            else:
                portals.append((edge[1], edge[0]))
        else:
            ci = corridor[i]
            mid = ((ci[0][0] + ci[1][0]) / 2, (ci[0][1] + ci[1][1]) / 2)
            portals.append((mid, mid))
    return portals


def _funnel_algorithm(
    portals: List[Tuple[Point, Point]],
    start: Point,
    goal: Point,
) -> List[Point]:
    """
    漏斗算法核心实现。

    维护三个关键点:
      - apex: 漏斗顶点(当前路径点)
      - left_bound: 左边界点(在行走方向左侧)
      - right_bound: 右边界点(在行走方向右侧)

    叉积约定: cross(O, A, B) > 0 表示 B 在 OA 的左侧。

    对每条新的 portal (new_left, new_right):
      1. 右侧收紧: 若 new_right 在当前右边界左侧(漏斗收紧)
         - 若 new_right 未越过左边界: 更新右边界
         - 若 new_right 越过左边界: 漏斗夹紧, 记录拐点
      2. 左侧收紧: 若 new_left 在当前左边界右侧(漏斗收紧)
         - 若 new_left 未越过右边界: 更新左边界
         - 若 new_left 越过右边界: 漏斗夹紧, 记录拐点
    """
    if not portals:
        return [start, goal]

    path = [start]

    apex = start
    left_bound = start
    right_bound = start

    left_idx = -1
    right_idx = -1

    for i, (new_left, new_right) in enumerate(portals):

        if right_bound == apex or cross_2d(apex, right_bound, new_right) > 0:
            if left_bound != apex and cross_2d(apex, left_bound, new_right) > 0:
                path.append(left_bound)
                apex = left_bound
                left_bound = apex
                right_bound = apex
                left_idx = right_idx = i
                continue
            right_bound = new_right
            right_idx = i

        if left_bound == apex or cross_2d(apex, left_bound, new_left) < 0:
            if right_bound != apex and cross_2d(apex, right_bound, new_left) < 0:
                path.append(right_bound)
                apex = right_bound
                left_bound = apex
                right_bound = apex
                left_idx = right_idx = i
                continue
            left_bound = new_left
            left_idx = i

    path.append(goal)
    return path
