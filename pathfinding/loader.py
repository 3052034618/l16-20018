"""
JSON 地图加载器 — 支持从 JSON 文件加载网格地图和导航网格。

网格地图 JSON 格式:
{
  "type": "grid",
  "width": 20,
  "height": 15,
  "allow_diagonal": true,
  "terrain_costs": {
    "plain": 1.0,
    "forest": 2.0,
    "swamp": 4.0,
    "hill": 3.0,
    "wall": 999
  },
  "terrain": [
    [0, 0, 1, 1, ...],
    ...
  ],
  "terrain_symbols": {
    ".": "plain",
    "F": "forest",
    "S": "swamp",
    "H": "hill",
    "#": "wall"
  },
  "terrain_string": [
    "....................",
    "..HH................",
    ...
  ],
  "start": [1, 1],
  "goal": [18, 13]
}

导航网格 JSON 格式:
{
  "type": "navmesh",
  "polygons": [
    {"vertices": [[0,0],[4,0],[4,4],[0,4]], "cost": 1.0},
    ...
  ],
  "start": [1.0, 1.0],
  "goal": [11.0, 11.0]
}
"""

import json
import os
from typing import Any, Dict, List, Tuple

from .grid import GridMap, TerrainType
from .navmesh import NavMesh


TERRAIN_TYPE_MAP = {
    "plain": TerrainType.PLAIN,
    "forest": TerrainType.FOREST,
    "swamp": TerrainType.SWAMP,
    "hill": TerrainType.HILL,
    "wall": TerrainType.WALL,
}

SYMBOL_TO_TERRAIN = {
    ".": TerrainType.PLAIN,
    "F": TerrainType.FOREST,
    "S": TerrainType.SWAMP,
    "H": TerrainType.HILL,
    "#": TerrainType.WALL,
}


def load_map(filepath: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载地图。

    Returns:
        dict with keys:
          - map_type: "grid" or "navmesh"
          - map: GridMap or NavMesh instance
          - start: start point (tuple)
          - goal: goal point (tuple)
          - name: map name (from filename)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    map_type = data.get("type", "grid")
    name = os.path.splitext(os.path.basename(filepath))[0]

    if map_type == "grid":
        grid_map, start, goal = _load_grid_map(data)
        return {
            "map_type": "grid",
            "map": grid_map,
            "start": start,
            "goal": goal,
            "name": name,
        }
    elif map_type == "navmesh":
        navmesh, start, goal = _load_navmesh(data)
        return {
            "map_type": "navmesh",
            "map": navmesh,
            "start": start,
            "goal": goal,
            "name": name,
        }
    else:
        raise ValueError(f"Unknown map type: {map_type}")


def _load_grid_map(data: Dict) -> Tuple[GridMap, Tuple[int, int], Tuple[int, int]]:
    width = data["width"]
    height = data["height"]
    allow_diagonal = data.get("allow_diagonal", True)

    custom_costs = data.get("terrain_costs")
    terrain_costs = None
    if custom_costs:
        from .grid import TERRAIN_COSTS
        terrain_costs = dict(TERRAIN_COSTS)
        for key, cost in custom_costs.items():
            tt = TERRAIN_TYPE_MAP.get(key)
            if tt:
                terrain_costs[tt] = float(cost)

    grid_map = GridMap(width=width, height=height, allow_diagonal=allow_diagonal,
                       terrain_costs=terrain_costs)

    if "terrain_string" in data:
        symbol_map = dict(SYMBOL_TO_TERRAIN)
        if "terrain_symbols" in data:
            for sym, terr_name in data["terrain_symbols"].items():
                tt = TERRAIN_TYPE_MAP.get(terr_name)
                if tt:
                    symbol_map[sym] = tt

        rows = data["terrain_string"]
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in symbol_map:
                    grid_map.set_terrain(x, y, symbol_map[ch])

    elif "terrain" in data:
        terrain_grid = data["terrain"]
        for y, row in enumerate(terrain_grid):
            for x, terrain_id in enumerate(row):
                terrain_name = _terrain_id_to_name(terrain_id)
                tt = TERRAIN_TYPE_MAP.get(terrain_name, TerrainType.PLAIN)
                grid_map.set_terrain(x, y, tt)

    start = tuple(data.get("start", [0, 0]))
    goal = tuple(data.get("goal", [width - 1, height - 1]))

    return grid_map, start, goal


def _terrain_id_to_name(terrain_id: int) -> str:
    mapping = {
        0: "plain",
        1: "forest",
        2: "swamp",
        3: "hill",
        4: "wall",
    }
    return mapping.get(terrain_id, "plain")


def _load_navmesh(data: Dict) -> Tuple[NavMesh, Tuple[float, float], Tuple[float, float]]:
    navmesh = NavMesh()

    for poly_data in data.get("polygons", []):
        vertices = [tuple(v) for v in poly_data["vertices"]]
        cost = float(poly_data.get("cost", 1.0))
        navmesh.add_polygon(vertices, cost)

    start = tuple(data.get("start", [0.0, 0.0]))
    goal = tuple(data.get("goal", [1.0, 1.0]))

    return navmesh, start, goal


def save_grid_map(filepath: str, grid_map: GridMap,
                  start: Tuple[int, int], goal: Tuple[int, int]):
    """保存网格地图为 JSON 文件。"""
    terrain_symbols = {v: k for k, v in SYMBOL_TO_TERRAIN.items()}

    rows = []
    for y in range(grid_map.height):
        row = []
        for x in range(grid_map.width):
            tt = grid_map.get_terrain(x, y)
            row.append(terrain_symbols.get(tt, '?'))
        rows.append(''.join(row))

    data = {
        "type": "grid",
        "width": grid_map.width,
        "height": grid_map.height,
        "allow_diagonal": grid_map.allow_diagonal,
        "terrain_string": rows,
        "start": list(start),
        "goal": list(goal),
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_navmesh(filepath: str, navmesh: NavMesh,
                 start: Tuple[float, float], goal: Tuple[float, float]):
    """保存导航网格为 JSON 文件。"""
    polygons = []
    for pid in sorted(navmesh.polygons.keys()):
        poly = navmesh.polygons[pid]
        polygons.append({
            "vertices": [list(v) for v in poly.vertices],
            "cost": poly.cost,
        })

    data = {
        "type": "navmesh",
        "polygons": polygons,
        "start": list(start),
        "goal": list(goal),
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
