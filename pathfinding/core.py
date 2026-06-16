"""
A* 核心算法模块 — 通用 A* 实现, 支持可配置启发函数。

A* 算法核心思想
==============
A* 在每次迭代中选择 f(n) = g(n) + h(n) 最小的节点 n 展开:
  - open_set:  待展开的节点集合(优先队列)
  - closed_set: 已展开的节点集合
  - g(n): 起点到 n 的已知最短代价
  - h(n): n 到终点的启发式估计
  - f(n) = g(n) + h(n): n 的优先级

算法流程:
  1. 将起点加入 open_set, g(start) = 0
  2. 从 open_set 取出 f 值最小的节点 current
  3. 若 current 是终点, 回溯重建路径
  4. 否则展开 current: 对每个邻居 n':
     - 计算 tentative_g = g(current) + cost(current, n')
     - 若 tentative_g < g(n'), 更新 g(n') 和 parent
     - 将 n' 加入 open_set
  5. 重复直到 open_set 为空(无路径)或找到终点

不同地形的移动代价如何并入路径总代价
=====================================
在 A* 中, 移动代价通过 g(n) 的更新并入路径:
  g(neighbor) = g(current) + move_cost(current, neighbor)

move_cost 由地图决定, 可以反映:
  - 平地: cost = 1
  - 沼泽: cost = 3 (移动更慢)
  - 山地: cost = 5 (移动最慢)
  - 障碍: 不可通行, 不生成邻居

这样, A* 在计算路径时会自动选择总代价最小的路径:
  虽然"穿过沼泽"的几何距离短, 但代价高;
  "绕行平地"虽然距离长, 但总代价可能更低。

地形代价只影响 g(n), 不影响 h(n) — h(n) 仍应低估真实代价,
通常用几何距离(欧几里得/曼哈顿)除以最低单位代价来保证可容许性。
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from .heuristics import euclidean_distance


class AStar:
    """
    通用 A* 寻路器。

    通过注入 get_neighbors 和 heuristic 函数, 可适配不同的地图表示:
      - 网格地图: get_neighbors 返回相邻格子和移动代价
      - 导航网格: get_neighbors 返回相邻多边形和穿越代价
    """

    def __init__(
        self,
        get_neighbors: Callable[[Any], List[Tuple[Any, float]]],
        heuristic: Callable[[Any, Any], float] = euclidean_distance,
        goal_test: Optional[Callable[[Any, Any], bool]] = None,
    ):
        """
        Args:
            get_neighbors: 给定节点, 返回 [(邻居, 移动代价), ...]
            heuristic: 启发函数 h(n, goal) → float
            goal_test: 可选的自定义目标测试, 默认 n == goal
        """
        self.get_neighbors = get_neighbors
        self.heuristic = heuristic
        self.goal_test = goal_test or (lambda n, goal: n == goal)

    def find_path(
        self, start: Any, goal: Any
    ) -> Tuple[List[Any], float]:
        """
        执行 A* 搜索, 返回 (路径, 总代价)。
        若无可达路径, 返回 ([], float('inf'))。

        Returns:
            path: 从 start 到 goal 的节点列表
            cost: 路径总代价
        """
        open_set: List[Tuple[float, int, Any]] = []
        counter = 0

        g_score: Dict[Any, float] = {start: 0.0}
        came_from: Dict[Any, Any] = {}
        closed_set = set()

        f_start = self.heuristic(start, goal)
        heapq.heappush(open_set, (f_start, counter, start))
        counter += 1

        open_set_check = {start}

        while open_set:
            f_val, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            open_set_check.discard(current)

            if self.goal_test(current, goal):
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current]

            closed_set.add(current)

            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_check:
                        heapq.heappush(open_set, (f_neighbor, counter, neighbor))
                        counter += 1
                        open_set_check.add(neighbor)
                    else:
                        heapq.heappush(open_set, (f_neighbor, counter, neighbor))
                        counter += 1

        return [], float('inf')

    @staticmethod
    def _reconstruct_path(came_from: Dict[Any, Any], current: Any) -> List[Any]:
        """从 came_from 字典回溯重建路径。"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
