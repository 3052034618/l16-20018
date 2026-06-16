"""生成测试场景 JSON 文件到 scenarios/ 目录。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathfinding.grid import GridMap, TerrainType
from pathfinding.loader import save_grid_map, save_navmesh
from pathfinding.navmesh import NavMesh


SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")


def save_json(filepath: str, data: dict):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  生成: {os.path.basename(filepath)}")


def gen_grid_simple():
    """10x10 简单网格, 少量障碍"""
    grid = GridMap(width=10, height=10, allow_diagonal=True)
    grid.set_terrain(4, 3, TerrainType.WALL)
    grid.set_terrain(4, 4, TerrainType.WALL)
    grid.set_terrain(4, 5, TerrainType.WALL)
    start = (0, 0)
    goal = (9, 9)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_simple.json"), grid, start, goal)


def gen_grid_obstacle_wall():
    """一道贯穿全图的墙, 留一个缺口 — 路径必须绕过去"""
    grid = GridMap(width=15, height=10, allow_diagonal=True)
    for y in range(10):
        if y != 5:
            grid.set_terrain(7, y, TerrainType.WALL)
    start = (0, 4)
    goal = (14, 6)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_wall_gap.json"), grid, start, goal)


def gen_grid_unreachable():
    """墙完全隔开左右两半 — 不可达"""
    grid = GridMap(width=12, height=10, allow_diagonal=True)
    for y in range(10):
        grid.set_terrain(6, y, TerrainType.WALL)
    start = (0, 5)
    goal = (11, 5)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_unreachable.json"), grid, start, goal)


def gen_grid_start_in_wall():
    """起点在障碍里 — 无效起点"""
    grid = GridMap(width=10, height=10, allow_diagonal=True)
    grid.set_terrain(2, 2, TerrainType.WALL)
    grid.set_terrain(2, 3, TerrainType.WALL)
    grid.set_terrain(3, 2, TerrainType.WALL)
    start = (2, 2)
    goal = (9, 9)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_start_in_wall.json"), grid, start, goal)


def gen_grid_goal_in_wall():
    """终点在障碍里 — 无效终点"""
    grid = GridMap(width=10, height=10, allow_diagonal=True)
    grid.set_terrain(7, 7, TerrainType.WALL)
    grid.set_terrain(7, 8, TerrainType.WALL)
    grid.set_terrain(8, 7, TerrainType.WALL)
    grid.set_terrain(8, 8, TerrainType.WALL)
    start = (0, 0)
    goal = (7, 7)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_goal_in_wall.json"), grid, start, goal)


def gen_grid_multi_terrain():
    """多地形: 沼泽、森林、山地 — 测试代价绕行"""
    grid = GridMap(width=20, height=12, allow_diagonal=True)

    for x in range(6, 14):
        for y in range(3, 9):
            grid.set_terrain(x, y, TerrainType.SWAMP)

    for x in range(2, 5):
        for y in range(2, 6):
            grid.set_terrain(x, y, TerrainType.FOREST)

    for x in range(15, 19):
        for y in range(5, 9):
            grid.set_terrain(x, y, TerrainType.HILL)

    for y in range(1, 8):
        grid.set_terrain(5, y, TerrainType.WALL)
    grid.set_terrain(5, 8, TerrainType.PLAIN)

    start = (0, 0)
    goal = (19, 11)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_multi_terrain.json"), grid, start, goal)


def gen_grid_maze():
    """迷宫式地图 — 多障碍多拐点"""
    grid = GridMap(width=21, height=15, allow_diagonal=False)

    for x in range(21):
        grid.set_terrain(x, 0, TerrainType.WALL)
        grid.set_terrain(x, 14, TerrainType.WALL)
    for y in range(15):
        grid.set_terrain(0, y, TerrainType.WALL)
        grid.set_terrain(20, y, TerrainType.WALL)

    for x in range(2, 18):
        grid.set_terrain(x, 3, TerrainType.WALL)
    grid.set_terrain(5, 3, TerrainType.PLAIN)

    for x in range(4, 20):
        grid.set_terrain(x, 7, TerrainType.WALL)
    grid.set_terrain(10, 7, TerrainType.PLAIN)

    for y in range(3, 12):
        grid.set_terrain(10, y, TerrainType.WALL)
    grid.set_terrain(10, 3, TerrainType.PLAIN)
    grid.set_terrain(10, 10, TerrainType.PLAIN)

    for x in range(2, 10):
        grid.set_terrain(x, 11, TerrainType.WALL)

    start = (1, 1)
    goal = (19, 13)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_maze.json"), grid, start, goal)


def gen_grid_large():
    """大地图 (100x100) — 用于性能测试"""
    grid = GridMap(width=100, height=100, allow_diagonal=True)

    for x in range(100):
        grid.set_terrain(x, 0, TerrainType.WALL)
        grid.set_terrain(x, 99, TerrainType.WALL)
    for y in range(100):
        grid.set_terrain(0, y, TerrainType.WALL)
        grid.set_terrain(99, y, TerrainType.WALL)

    for x in range(10, 80):
        grid.set_terrain(x, 20, TerrainType.WALL)
    for x in range(30, 90):
        grid.set_terrain(x, 40, TerrainType.WALL)
    for x in range(10, 70):
        grid.set_terrain(x, 60, TerrainType.WALL)
    for x in range(20, 90):
        grid.set_terrain(x, 80, TerrainType.WALL)

    for y in range(0, 20):
        grid.set_terrain(15, y, TerrainType.WALL)
    for y in range(20, 60):
        grid.set_terrain(50, y, TerrainType.WALL)
    for y in range(40, 80):
        grid.set_terrain(75, y, TerrainType.WALL)

    grid.set_terrain(15, 10, TerrainType.PLAIN)
    grid.set_terrain(50, 30, TerrainType.PLAIN)
    grid.set_terrain(75, 55, TerrainType.PLAIN)

    start = (1, 1)
    goal = (98, 98)
    save_grid_map(os.path.join(SCENARIO_DIR, "grid_large.json"), grid, start, goal)


def gen_navmesh_simple():
    """简单的 2x2 多边形网格"""
    nm = NavMesh()
    nm.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], cost=1.0)
    nm.add_polygon([(5, 0), (10, 0), (10, 5), (5, 5)], cost=1.0)
    nm.add_polygon([(0, 5), (5, 5), (5, 10), (0, 10)], cost=1.0)
    nm.add_polygon([(5, 5), (10, 5), (10, 10), (5, 10)], cost=1.0)
    start = (1.0, 1.0)
    goal = (9.0, 9.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_simple.json"), nm, start, goal)


def gen_navmesh_terrain_costs():
    """不同多边形有不同地形代价 — 测试代价计算"""
    nm = NavMesh()
    nm.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)], cost=1.0)
    nm.add_polygon([(4, 0), (8, 0), (8, 4), (4, 4)], cost=3.0)
    nm.add_polygon([(0, 4), (4, 4), (4, 8), (0, 8)], cost=1.0)
    nm.add_polygon([(4, 4), (8, 4), (8, 8), (4, 8)], cost=5.0)
    nm.add_polygon([(8, 0), (12, 0), (12, 4), (8, 4)], cost=1.0)
    nm.add_polygon([(8, 4), (12, 4), (12, 8), (8, 8)], cost=1.0)
    nm.add_polygon([(0, 8), (4, 8), (4, 12), (0, 12)], cost=1.0)
    nm.add_polygon([(4, 8), (8, 8), (8, 12), (4, 12)], cost=2.0)
    nm.add_polygon([(8, 8), (12, 8), (12, 12), (8, 12)], cost=1.0)
    start = (1.0, 1.0)
    goal = (11.0, 11.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_terrain_costs.json"), nm, start, goal)


def gen_navmesh_l_corridor():
    """L形走廊 — 漏斗算法产生拐点"""
    nm = NavMesh()
    nm.add_polygon([(0, 0), (8, 0), (8, 2), (0, 2)], cost=1.0)
    nm.add_polygon([(8, 0), (10, 0), (10, 2), (8, 2)], cost=1.0)
    nm.add_polygon([(8, 2), (10, 2), (10, 6), (8, 6)], cost=1.0)
    nm.add_polygon([(8, 6), (10, 6), (10, 10), (8, 10)], cost=1.0)
    start = (1.0, 1.0)
    goal = (9.0, 9.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_l_corridor.json"), nm, start, goal)


def gen_navmesh_shared_edge_start():
    """起点落在共享边上"""
    nm = NavMesh()
    nm.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], cost=1.0)
    nm.add_polygon([(5, 0), (10, 0), (10, 5), (5, 5)], cost=2.0)
    nm.add_polygon([(0, 5), (5, 5), (5, 10), (0, 10)], cost=3.0)
    nm.add_polygon([(5, 5), (10, 5), (10, 10), (5, 10)], cost=1.0)
    start = (5.0, 2.5)
    goal = (8.0, 8.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_shared_edge_start.json"), nm, start, goal)


def gen_navmesh_unreachable():
    """两组不连通的多边形 — 不可达"""
    nm = NavMesh()
    nm.add_polygon([(0, 0), (3, 0), (3, 3), (0, 3)], cost=1.0)
    nm.add_polygon([(3, 0), (6, 0), (6, 3), (3, 3)], cost=1.0)
    nm.add_polygon([(10, 0), (13, 0), (13, 3), (10, 3)], cost=1.0)
    nm.add_polygon([(13, 0), (16, 0), (16, 3), (13, 3)], cost=1.0)
    start = (1.0, 1.0)
    goal = (14.0, 1.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_unreachable.json"), nm, start, goal)


def gen_navmesh_start_outside():
    """起点在所有多边形之外"""
    nm = NavMesh()
    nm.add_polygon([(2, 2), (6, 2), (6, 6), (2, 6)], cost=1.0)
    nm.add_polygon([(6, 2), (10, 2), (10, 6), (6, 6)], cost=1.0)
    start = (0.0, 0.0)
    goal = (8.0, 4.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_start_outside.json"), nm, start, goal)


def gen_navmesh_large():
    """较大的导航网格 (20x20 格子对应 ~100 个多边形) — 用于性能对比"""
    nm = NavMesh()

    cols = 10
    rows = 10
    for r in range(rows):
        for c in range(cols):
            x0 = c * 5.0
            y0 = r * 5.0
            x1 = (c + 1) * 5.0
            y1 = (r + 1) * 5.0
            cost = 1.0
            if (r + c) % 5 == 0:
                cost = 2.0
            elif (r * c) % 7 == 0:
                cost = 3.0
            nm.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], cost=cost)

    start = (1.0, 1.0)
    goal = (49.0, 49.0)
    save_navmesh(os.path.join(SCENARIO_DIR, "navmesh_large.json"), nm, start, goal)


def main():
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    print("生成测试场景到 scenarios/ 目录...\n")

    print("【网格地图场景】")
    gen_grid_simple()
    gen_grid_obstacle_wall()
    gen_grid_unreachable()
    gen_grid_start_in_wall()
    gen_grid_goal_in_wall()
    gen_grid_multi_terrain()
    gen_grid_maze()
    gen_grid_large()

    print("\n【导航网格场景】")
    gen_navmesh_simple()
    gen_navmesh_terrain_costs()
    gen_navmesh_l_corridor()
    gen_navmesh_shared_edge_start()
    gen_navmesh_unreachable()
    gen_navmesh_start_outside()
    gen_navmesh_large()

    print(f"\n完成! 共生成 {len(os.listdir(SCENARIO_DIR))} 个场景文件。")


if __name__ == "__main__":
    main()
