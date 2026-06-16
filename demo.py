"""
寻路引擎演示 — 网格地图与导航网格两种场景的完整示例。

演示内容:
  1. 网格地图 A* 寻路 (含多种地形和障碍)
  2. 不同启发函数的对比
  3. 网格路径的视线平滑
  4. 导航网格 A* 寻路
  5. 漏斗算法路径平滑
"""

import math

from pathfinding.heuristics import (
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    diagonal_distance,
    zero_heuristic,
)
from pathfinding.grid import GridMap, TerrainType
from pathfinding.navmesh import NavMesh
from pathfinding.smoothing import smooth_grid_path, funnel_smooth


def separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_grid_astar():
    separator("场景一: 网格地图 A* 寻路 (含多种地形与障碍)")

    grid = GridMap(width=20, height=15, allow_diagonal=True)

    for x in range(5, 15):
        grid.set_terrain(x, 4, TerrainType.WALL)
    grid.set_terrain(14, 4, TerrainType.PLAIN)

    for x in range(3, 10):
        for y in range(8, 11):
            grid.set_terrain(x, y, TerrainType.SWAMP)

    for x in range(12, 18):
        for y in range(9, 13):
            grid.set_terrain(x, y, TerrainType.FOREST)

    grid.set_terrain(2, 2, TerrainType.HILL)
    grid.set_terrain(3, 2, TerrainType.HILL)
    grid.set_terrain(2, 3, TerrainType.HILL)

    for x in range(6, 12):
        grid.set_terrain(x, 11, TerrainType.WALL)

    print("地图图例: .平地 F森林 S沼泽 H山地 #障碍 *路径\n")
    print(grid)
    print()

    start = (1, 1)
    goal = (18, 13)

    heuristics = {
        "零启发(Dijkstra)": zero_heuristic,
        "曼哈顿距离": manhattan_distance,
        "欧几里得距离": euclidean_distance,
        "对角线距离(Octile)": diagonal_distance,
    }

    results = {}

    for name, h_func in heuristics.items():
        astar = grid.create_astar(heuristic=h_func)
        path, cost = astar.find_path(start, goal)
        results[name] = (path, cost)
        if path:
            print(f"[{name}]")
            print(f"  路径长度: {len(path)} 步, 总代价: {cost:.2f}")
            print(f"  路径: {path[:5]}...{path[-3:]}")
        else:
            print(f"[{name}] 未找到路径!")
        print()

    best_path = results["对角线距离(Octile)"][0]
    if best_path:
        print("路径可视化 (使用对角线距离启发):")
        print(grid.display_path(best_path))
        print()

    print("--- 启发函数效果分析 ---")
    print("零启发(Dijkstra): 无方向引导, 探索所有 g < optimal 的节点, 最慢")
    print("曼哈顿距离:       只考虑水平+垂直距离, 低估对角线移动, 较慢")
    print("欧几里得距离:     考虑直线距离, 比曼哈顿更接近真实, 较快")
    print("对角线距离:       最匹配八方向移动的启发, 低估最少, 最快且最优")


def demo_grid_smoothing():
    separator("场景二: 网格路径的视线平滑")

    grid = GridMap(width=25, height=12, allow_diagonal=True)

    for y in range(2, 10):
        grid.set_terrain(8, y, TerrainType.WALL)
    grid.set_terrain(8, 9, TerrainType.PLAIN)

    for y in range(2, 8):
        grid.set_terrain(16, y, TerrainType.WALL)
    grid.set_terrain(16, 2, TerrainType.PLAIN)

    print("地图 (两道墙各留一个缺口):\n")
    print(grid)
    print()

    start = (1, 1)
    goal = (23, 10)

    astar = grid.create_astar()
    path, cost = astar.find_path(start, goal)

    if path:
        print(f"A* 原始路径: {len(path)} 个节点, 代价 {cost:.2f}")
        print("原始路径(锯齿状):")
        print(grid.display_path(path))
        print()

        obstacle_polys = grid.get_obstacle_polygons()
        smoothed = smooth_grid_path(path, obstacle_polys, world_coords=True)

        print(f"平滑后路径: {len(smoothed)} 个关键点")
        print("平滑路径点:")
        for i, pt in enumerate(smoothed):
            print(f"  {i}: ({pt[0]:.1f}, {pt[1]:.1f})")

        print()
        print("--- 路径平滑原理 ---")
        print("原始网格路径由逐格移动产生, 呈锯齿状折线。")
        print("视线平滑: 从当前点出发, 找到视线可达的最远点, 用直线替代折线。")
        print("若两点间无障碍遮挡, 则可直接通过, 跳过中间所有折线节点。")
        print("结果: 路径更短、更平滑, 角色移动更自然。")

        original_length = 0
        world_pts = grid.path_to_world_coords(path)
        for i in range(len(world_pts) - 1):
            original_length += math.sqrt(
                (world_pts[i + 1][0] - world_pts[i][0]) ** 2 +
                (world_pts[i + 1][1] - world_pts[i][1]) ** 2
            )
        smooth_length = 0
        for i in range(len(smoothed) - 1):
            smooth_length += math.sqrt(
                (smoothed[i + 1][0] - smoothed[i][0]) ** 2 +
                (smoothed[i + 1][1] - smoothed[i][1]) ** 2
            )
        print(f"\n原始路径几何长度: {original_length:.2f}")
        print(f"平滑路径几何长度: {smooth_length:.2f}")
        print(f"缩短比例: {(1 - smooth_length / original_length) * 100:.1f}%")


def demo_navmesh():
    separator("场景三: 导航网格(NavMesh)寻路")

    navmesh = NavMesh()

    p0 = navmesh.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)], cost=1.0)
    p1 = navmesh.add_polygon([(4, 0), (8, 0), (8, 4), (4, 4)], cost=1.0)
    p2 = navmesh.add_polygon([(8, 0), (12, 0), (12, 4), (8, 4)], cost=2.0)
    p3 = navmesh.add_polygon([(0, 4), (4, 4), (4, 8), (0, 8)], cost=1.0)
    p4 = navmesh.add_polygon([(4, 4), (8, 4), (8, 8), (4, 8)], cost=1.5)
    p5 = navmesh.add_polygon([(8, 4), (12, 4), (12, 8), (8, 8)], cost=1.0)
    p6 = navmesh.add_polygon([(0, 8), (4, 8), (4, 12), (0, 12)], cost=1.0)
    p7 = navmesh.add_polygon([(4, 8), (8, 8), (8, 12), (4, 12)], cost=1.0)
    p8 = navmesh.add_polygon([(8, 8), (12, 8), (12, 12), (8, 12)], cost=2.5)

    print("导航网格结构:")
    print(navmesh)
    print()

    print("多边形邻接关系:")
    for pid, poly in navmesh.polygons.items():
        for nid, shared in poly.neighbors:
            print(f"  多边形 {pid} ↔ 多边形 {nid}, 共享边: {shared[0]}-{shared[1]}")
    print()

    start = (1.0, 1.0)
    goal = (11.0, 11.0)

    print(f"寻路: {start} → {goal}\n")

    smooth_path, poly_path, cost = navmesh.find_path(start, goal)

    if smooth_path:
        print(f"多边形路径: {poly_path}")
        print(f"路径总代价: {cost:.2f}")
        print(f"漏斗算法平滑路径 ({len(smooth_path)} 个关键点):")
        for i, pt in enumerate(smooth_path):
            print(f"  {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
    else:
        print("未找到路径!")

    print()
    print("--- 导航网格 vs 格子地图 ---")
    print("内存: 此场景用 9 个多边形代替 12×12=144 个格子")
    print("搜索: A* 仅在 9 个多边形间跳转, 而非 144 个格子")
    print("路径: 漏斗算法直接生成连续直线段, 无需额外平滑")
    print("地形: 不同多边形可有不同 cost, 代价并入方式与格子相同")
    print("可扩展: 大地图上多边形数远小于格子数, 优势更明显")


def demo_navmesh_with_obstacle():
    separator("场景四: 导航网格绕障碍寻路 (含漏斗平滑)")

    navmesh = NavMesh()

    navmesh.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], cost=1.0)
    navmesh.add_polygon([(5, 0), (10, 0), (10, 3), (5, 3)], cost=1.0)
    navmesh.add_polygon([(5, 3), (10, 3), (10, 5), (5, 5)], cost=1.0)
    navmesh.add_polygon([(0, 5), (5, 5), (5, 10), (0, 10)], cost=1.0)
    navmesh.add_polygon([(5, 5), (10, 5), (10, 10), (5, 10)], cost=1.0)

    print("导航网格 (5个多边形, 2×2+3布局):")
    print(navmesh)
    print()

    start = (1.0, 1.0)
    goal = (9.0, 9.0)

    print(f"寻路: {start} → {goal}\n")

    smooth_path, poly_path, cost = navmesh.find_path(start, goal)

    if smooth_path:
        print(f"多边形路径: {poly_path}")
        print(f"路径总代价: {cost:.2f}")
        print(f"漏斗平滑路径:")
        for i, pt in enumerate(smooth_path):
            print(f"  {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
    else:
        print("未找到路径!")

    print()

    navmesh2 = NavMesh()
    navmesh2.add_polygon([(0, 0), (8, 0), (8, 2), (0, 2)], cost=1.0)
    navmesh2.add_polygon([(8, 0), (10, 0), (10, 2), (8, 2)], cost=1.0)
    navmesh2.add_polygon([(8, 2), (10, 2), (10, 5), (8, 5)], cost=1.0)
    navmesh2.add_polygon([(8, 5), (10, 5), (10, 8), (8, 8)], cost=1.0)

    print("--- L形走廊: 漏斗算法产生拐点 ---")
    print("  Poly0: (0,0)-(8,0)-(8,2)-(0,2)   底部水平走廊")
    print("  Poly1: (8,0)-(10,0)-(10,2)-(8,2)  右下角")
    print("  Poly2: (8,2)-(10,2)-(10,5)-(8,5)  右侧垂直走廊(下)")
    print("  Poly3: (8,5)-(10,5)-(10,8)-(8,8)  右侧垂直走廊(上)")
    print()

    start2 = (1.0, 1.0)
    goal2 = (9.0, 7.0)

    smooth2, poly2, cost2 = navmesh2.find_path(start2, goal2)
    if smooth2:
        print(f"寻路: {start2} → {goal2}")
        print(f"多边形路径: {poly2}")
        print(f"路径代价: {cost2:.2f}")
        print(f"漏斗平滑路径 ({len(smooth2)} 个关键点):")
        for i, pt in enumerate(smooth2):
            print(f"  {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
        print()
        print("漏斗算法在L形拐角处产生拐点 (8.00, 2.00):")
        print("  从 (1,1) 沿水平走廊走到拐角 (8,2)")
        print("  然后沿垂直走廊走到 (9,7)")
        print("  直线 (1,1)→(9,7) 会穿过走廊外的空间, 故漏斗收紧并产生拐点")

    print()
    print("--- 漏斗算法原理 ---")
    print("1. A* 返回多边形序列(走廊), 如 [Poly0, Poly3, Poly4]")
    print("2. 提取相邻多边形的共享边(portal), 确定左右端点")
    print("3. 维护'漏斗': 以当前顶点为尖端, 左右边界沿 portal 端点延伸")
    print("4. 每条新 portal 更新边界: 若漏斗收紧则更新, 若夹紧则产生拐点")
    print("5. 最终路径穿过共享边的最优点, 形成走廊内的最短折线")


def demo_terrain_cost_analysis():
    separator("场景五: 地形代价对路径选择的影响")

    grid = GridMap(width=15, height=5, allow_diagonal=True)

    for x in range(3, 12):
        for y in range(1, 4):
            grid.set_terrain(x, y, TerrainType.SWAMP)

    print("地图: 中间是沼泽(S), 上下是平地(.)\n")
    print(grid)
    print()

    start = (0, 2)
    goal = (14, 2)

    astar = grid.create_astar()
    path, cost = astar.find_path(start, goal)

    if path:
        print(f"路径 (代价 {cost:.2f}):")
        print(grid.display_path(path))
        print()
        print("分析: A* 选择绕行平地而非穿越沼泽, 因为:")
        print("  直线穿越沼泽: 14格 × cost=4 = 56")
        print("  绕行平地:     虽然距离更长, 但 cost=1, 总代价更低")

    grid2 = GridMap(width=15, height=5, allow_diagonal=True)
    for x in range(3, 12):
        for y in range(1, 4):
            grid2.set_terrain(x, y, TerrainType.FOREST)

    print("\n--- 对比: 将沼泽改为森林 (cost=2) ---\n")
    print(grid2)

    astar2 = grid2.create_astar()
    path2, cost2 = astar2.find_path(start, goal)

    if path2:
        print(f"路径 (代价 {cost2:.2f}):")
        print(grid2.display_path(path2))
        print()
        print("森林代价较低, A* 可能选择直穿而非绕行, 取决于具体代价对比。")


def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           寻路引擎演示 — A* on Grid & NavMesh                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    demo_grid_astar()
    demo_grid_smoothing()
    demo_terrain_cost_analysis()
    demo_navmesh()
    demo_navmesh_with_obstacle()

    separator("总结: 核心概念回顾")

    print("""
1. A* 如何用启发函数引导搜索方向以减少探索节点:
   f(n) = g(n) + h(n) 为节点优先级, h(n) 估计到目标的剩余距离,
   使 A* 优先朝目标方向展开, 减少无效探索。
   h 越接近真实值, 探索越少; h=0 则退化为 Dijkstra。

2. 启发函数低估实际代价为何是保证最优路径的前提:
   低估(h ≤ h*)保证 A* 不会过早排除更优路径。
   若 h 高估, A* 可能认为某路径代价过高而跳过, 导致非最优结果。
   低估越多越慢但保证最优; 低估越少越快且仍最优。

3. 不同地形的移动代价如何并入路径总代价:
   move_cost = distance × terrain_cost
   A* 通过 g(n) 更新累加: g(neighbor) = g(current) + move_cost
   地形代价使 A* 自动权衡"短距离高代价"vs"长距离低代价"。

4. 网格路径锯齿状如何通过视线检测做漏斗平滑:
   视线平滑: 贪心地从当前点找视线可达的最远点, 用直线替代折线。
   漏斗算法: 在 navmesh 走廊中维护左右边界, 收紧至夹紧时产生拐点。
   两者都将锯齿路径压缩为关键拐点的最短折线。

5. 导航网格相比格子地图的优势及多边形相邻关系:
   优势: 内存少(多边形数 << 格子数), 搜索快, 路径质量高, 精度不限。
   相邻关系: 共享边的多边形互为邻居, 用邻接表存储 (poly_id → [(neighbor_id, shared_edge)])
   """)


if __name__ == "__main__":
    main()
