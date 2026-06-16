"""
网格地图模块 — 基于格子的地图表示, 支持障碍物和多种地形。

不同地形的移动代价如何并入路径总代价
=====================================
每个格子有一个 terrain_cost 值, 表示"进入"该格子的代价:
  - 平地 (PLAIN): 1.0   — 正常速度
  - 森林 (FOREST): 2.0  — 穿越较慢
  - 沼泽 (SWAMP): 4.0   — 非常缓慢
  - 山地 (HILL): 3.0    — 翻越较慢
  - 障碍 (WALL): ∞     — 不可通行

从格子 A 移动到格子 B 的代价为:
  move_cost = distance(A, B) × terrain_cost(B)

其中 distance(A, B) 是几何距离:
  - 四方向邻居: 1.0
  - 八方向对角邻居: √2 ≈ 1.414

这样, 总路径代价 = Σ distance(i, i+1) × terrain_cost(i+1),
既反映了几何距离, 又反映了地形难度。
"""

import math
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from .core import AStar
from .heuristics import diagonal_distance, euclidean_distance


class TerrainType(Enum):
    PLAIN = "plain"
    FOREST = "forest"
    SWAMP = "swamp"
    HILL = "hill"
    WALL = "wall"


TERRAIN_COSTS = {
    TerrainType.PLAIN: 1.0,
    TerrainType.FOREST: 2.0,
    TerrainType.SWAMP: 4.0,
    TerrainType.HILL: 3.0,
    TerrainType.WALL: float('inf'),
}


class GridMap:
    """
    网格地图: 用二维数组存储每个格子的地形类型。

    支持:
      - 四方向(上下左右)或八方向(含对角线)移动
      - 不同地形类型和对应移动代价
      - 障碍物(不可通行)
      - 对角线移动需检查"角落切割"避免穿墙
    """

    def __init__(
        self,
        width: int,
        height: int,
        allow_diagonal: bool = True,
        terrain_costs: Optional[Dict[TerrainType, float]] = None,
    ):
        self.width = width
        self.height = height
        self.allow_diagonal = allow_diagonal
        self.terrain_costs = terrain_costs or dict(TERRAIN_COSTS)
        self.grid: List[List[TerrainType]] = [
            [TerrainType.PLAIN for _ in range(width)]
            for _ in range(height)
        ]

    def set_terrain(self, x: int, y: int, terrain: TerrainType):
        """设置 (x, y) 处的地形类型。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = terrain

    def get_terrain(self, x: int, y: int) -> TerrainType:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return TerrainType.WALL

    def get_cost(self, x: int, y: int) -> float:
        return self.terrain_costs.get(self.get_terrain(x, y), float('inf'))

    def is_passable(self, x: int, y: int) -> bool:
        return self.get_terrain(x, y) != TerrainType.WALL

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        获取 pos 的所有可通行邻居及移动代价。

        对角线移动需检查角落切割:
          从 (x,y) 到 (x+1,y+1) 需要 (x+1,y) 和 (x,y+1) 都可通行,
          否则会"穿过"两个对角格子的墙角。
        """
        x, y = pos
        neighbors = []

        directions = [
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (1, 0, 1.0),
        ]
        if self.allow_diagonal:
            directions += [
                (-1, -1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (1, 1, math.sqrt(2)),
            ]

        for dx, dy, dist in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if not self.is_passable(nx, ny):
                continue

            if dx != 0 and dy != 0:
                if not self.is_passable(x + dx, y) or not self.is_passable(x, y + dy):
                    continue

            move_cost = dist * self.get_cost(nx, ny)
            neighbors.append(((nx, ny), move_cost))

        return neighbors

    def create_astar(
        self,
        heuristic: Optional[Callable] = None,
    ) -> AStar:
        """
        创建适用于本网格地图的 A* 实例。

        默认使用 diagonal_distance 启发函数(对八方向最优)。
        若 allow_diagonal=False, 默认使用曼哈顿距离。
        """
        if heuristic is None:
            from .heuristics import manhattan_distance
            heuristic = diagonal_distance if self.allow_diagonal else manhattan_distance
        return AStar(
            get_neighbors=self.get_neighbors,
            heuristic=heuristic,
        )

    def get_obstacle_polygons(self) -> List[List[Tuple[float, float]]]:
        """
        将障碍格子转为正方形多边形列表, 用于视线检测。
        每个障碍格 (x, y) 对应正方形 [(x,y), (x+1,y), (x+1,y+1), (x, y+1)]。
        """
        polygons = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_passable(x, y):
                    polygons.append([
                        (x, y),
                        (x + 1, y),
                        (x + 1, y + 1),
                        (x, y + 1),
                    ])
        return polygons

    def path_to_world_coords(
        self, path: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """将格子坐标转换为世界坐标(格子中心)。"""
        return [(x + 0.5, y + 0.5) for x, y in path]

    def __repr__(self) -> str:
        symbols = {
            TerrainType.PLAIN: '.',
            TerrainType.FOREST: 'F',
            TerrainType.SWAMP: 'S',
            TerrainType.HILL: 'H',
            TerrainType.WALL: '#',
        }
        lines = []
        for y in range(self.height):
            line = ''.join(symbols.get(self.grid[y][x], '?') for x in range(self.width))
            lines.append(line)
        return '\n'.join(lines)

    def display_path(self, path: List[Tuple[int, int]]) -> str:
        """可视化路径, 用 * 标记路径上的格子。"""
        path_set = set(path)
        symbols = {
            TerrainType.PLAIN: '.',
            TerrainType.FOREST: 'F',
            TerrainType.SWAMP: 'S',
            TerrainType.HILL: 'H',
            TerrainType.WALL: '#',
        }
        lines = []
        for y in range(self.height):
            line = ''
            for x in range(self.width):
                if (x, y) in path_set:
                    line += '*'
                else:
                    line += symbols.get(self.grid[y][x], '?')
            lines.append(line)
        return '\n'.join(lines)
