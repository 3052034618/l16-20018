"""
启发函数模块 — 为 A* 提供可配置的启发式估计。

A* 如何用启发函数引导搜索方向以减少探索节点
=================================================
A* 的节点优先级由 f(n) = g(n) + h(n) 决定:
  - g(n): 从起点到 n 的实际代价（已知）
  - h(n): 从 n 到终点的启发式估计（预测）

h(n) 的作用相当于"指南针": 它为每个节点估计到目标的剩余距离,
使 A* 优先朝目标方向展开搜索, 而非像 Dijkstra 那样均匀地向所有方向扩散。

当 h(n) = 0 时, A* 退化为 Dijkstra, 会探索所有 g 值小于最优路径长度的节点。
当 h(n) 接近真实代价时, A* 几乎只沿最优路径探索, 极大减少扩展节点数。

启发函数低估实际代价为何是保证最优路径的前提
=============================================
A* 最优性定理: 若 h(n) 对所有节点 n 满足 h(n) ≤ h*(n)
(其中 h*(n) 是 n 到目标的真实最短代价), 则 A* 一定找到最优路径。

直观理解:
  - 若 h(n) 高估了真实代价, A* 可能"过早地"认为某条路径代价过高而跳过它,
    导致最终返回的路径不是最优的。
  - 若 h(n) 低估, A* 会保留更多候选节点, 保证不会遗漏更优路径。
    低估越多, 保留的候选越多, 搜索越慢但保证最优。
    低估越少(越接近真实值), 搜索越快且仍保证最优。

可容许性(admissibility): h(n) ≤ h*(n) 对所有 n 成立
一致性(consistency): h(n) ≤ c(n, n') + h(n') 对所有边 (n, n') 成立
  一致性 ⇒ 可容许性, 且保证已展开节点的 g 值即为最优, 无需重新展开。
"""

import math
from typing import Tuple


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    曼哈顿距离: |dx| + |dy|
    适用于只允许四方向移动(上下左右)的网格。
    在四方向网格上是可容许的(不会高估)。
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    欧几里得距离: sqrt(dx² + dy²)
    适用于允许任意方向移动或八方向移动的网格。
    在四方向网格上也是可容许的(直线距离 ≤ 折线距离)。
    在八方向网格上同样可容许。
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    切比雪夫距离: max(|dx|, |dy|)
    适用于八方向移动且对角线代价与直线相同的网格。
    在此设定下是可容许的。
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def diagonal_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    对角线距离 (Octile distance):
      当对角线移动代价为 √2, 直线移动代价为 1 时,
      D * (|dx| + |dy|) + (D2 - 2*D) * min(|dx|, |dy|)
      其中 D=1, D2=√2

    这是对角线移动代价为 √2 的八方向网格上的最优启发函数,
    精确估计了无障碍时的最短路径长度, 可容许且一致。
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1.0
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def zero_heuristic(a, b) -> float:
    """
    零启发函数: h(n) = 0
    使 A* 退化为 Dijkstra 算法, 保证最优但无方向引导, 探索节点最多。
    作为对照基线使用。
    """
    return 0.0
