"""
回归基线模块 — 保存/加载历史测试结果, 对比差异, 标出回归和改善。

功能:
  - 将场景测试结果序列化为 JSON 基线文件
  - 加载历史基线并与当前结果对比
  - 标出: 代价变差、展开节点暴涨、可达性变化
  - 按场景类型和启发函数聚合排行榜
  - 导出回归对比报告为 Markdown
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import PathfindingResult


class RegressionItem:
    """单个场景的回归对比结果。"""

    def __init__(
        self,
        scenario_name: str,
        map_type: str,
        heuristic_name: str,
    ):
        self.scenario_name = scenario_name
        self.map_type = map_type
        self.heuristic_name = heuristic_name
        self.current: Optional[Dict[str, Any]] = None
        self.baseline: Optional[Dict[str, Any]] = None
        self.regressions: List[str] = []
        self.improvements: List[str] = []
        self.is_new = False
        self.is_removed = False
        self.cost_delta_pct: Optional[float] = None
        self.node_delta_pct: Optional[float] = None

    @property
    def has_regression(self) -> bool:
        return len(self.regressions) > 0

    @property
    def has_change(self) -> bool:
        return bool(self.regressions or self.improvements or self.is_new or self.is_removed)


def result_to_dict(result: PathfindingResult) -> Dict[str, Any]:
    """将 PathfindingResult 序列化为可 JSON 化的字典。"""
    d = {
        "heuristic_name": result.heuristic_name,
        "found": result.found,
        "total_cost": round(result.total_cost, 6),
        "nodes_expanded": result.nodes_expanded,
        "nodes_generated": result.nodes_generated,
        "max_open_size": result.max_open_size,
        "time_ms": round(result.time_ms, 4),
        "raw_path_length": round(result.raw_path_length, 6),
        "smooth_path_length": round(result.smooth_path_length, 6),
    }
    return d


def save_baseline(
    all_data: List[Tuple[str, str, List[PathfindingResult], Dict[str, Any]]],
    filepath: str,
) -> None:
    """
    将测试结果保存为基线 JSON 文件。

    Args:
        all_data: [(scenario_name, map_type, [results], expectations), ...]
        filepath: 保存路径
    """
    baseline = {}
    for name, map_type, results, expectations in all_data:
        key = f"{map_type}:{name}"
        baseline[key] = {
            "scenario_name": name,
            "map_type": map_type,
            "results": [result_to_dict(r) for r in results],
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)


def load_baseline(filepath: str) -> Dict[str, Any]:
    """加载基线 JSON 文件。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_with_baseline(
    all_data: List[Tuple[str, str, List[PathfindingResult], Dict[str, Any]]],
    baseline: Dict[str, Any],
    cost_threshold: float = 0.05,
    node_threshold: float = 0.5,
) -> List[RegressionItem]:
    """
    将当前测试结果与基线对比。

    Args:
        all_data: 当前测试结果
        baseline: 加载的基线数据
        cost_threshold: 代价涨幅超过此比例视为回归 (默认 5%)
        node_threshold: 展开节点涨幅超过此比例视为暴涨 (默认 50%)

    Returns:
        RegressionItem 列表
    """
    items = []
    cost_thresh = max(cost_threshold, 0.001)
    node_thresh = max(node_threshold, 0.1)

    for name, map_type, results, expectations in all_data:
        key = f"{map_type}:{name}"

        if key not in baseline:
            for r in results:
                item = RegressionItem(name, map_type, r.heuristic_name)
                item.current = result_to_dict(r)
                item.is_new = True
                items.append(item)
            continue

        baseline_entry = baseline[key]
        baseline_results = {r["heuristic_name"]: r for r in baseline_entry["results"]}

        for r in results:
            item = RegressionItem(name, map_type, r.heuristic_name)
            item.current = result_to_dict(r)

            h_name = r.heuristic_name
            if h_name not in baseline_results:
                item.is_new = True
                items.append(item)
                continue

            b = baseline_results[h_name]
            item.baseline = b

            if not b["found"] and r.found:
                item.improvements.append("路径从不可达变为可达")
            elif b["found"] and not r.found:
                item.regressions.append("路径从可达变为不可达")
            elif b["found"] and r.found:
                b_cost = b["total_cost"]
                c_cost = r.total_cost
                if b_cost > 0:
                    item.cost_delta_pct = ((c_cost - b_cost) / b_cost) * 100
                if c_cost > b_cost * (1 + cost_thresh):
                    pct = item.cost_delta_pct
                    item.regressions.append(
                        f"代价上涨 {pct:.1f}% ({b_cost:.2f} → {c_cost:.2f})"
                    )
                elif c_cost < b_cost * (1 - cost_thresh):
                    pct = abs(item.cost_delta_pct) if item.cost_delta_pct else 0
                    item.improvements.append(
                        f"代价下降 {pct:.1f}% ({b_cost:.2f} → {c_cost:.2f})"
                    )

                b_nodes = b["nodes_expanded"]
                c_nodes = r.nodes_expanded
                if b_nodes > 0:
                    item.node_delta_pct = ((c_nodes - b_nodes) / b_nodes) * 100
                if b_nodes > 0 and c_nodes > b_nodes * (1 + node_thresh):
                    pct = item.node_delta_pct
                    item.regressions.append(
                        f"展开节点暴涨 {pct:.0f}% ({b_nodes} → {c_nodes})"
                    )
                elif b_nodes > 0 and c_nodes < b_nodes * (1 - node_thresh):
                    pct = abs(item.node_delta_pct) if item.node_delta_pct else 0
                    item.improvements.append(
                        f"展开节点减少 {pct:.0f}% ({b_nodes} → {c_nodes})"
                    )

            items.append(item)

    current_keys = {f"{mt}:{n}" for n, mt, _, _ in all_data}
    for key, entry in baseline.items():
        if key not in current_keys:
            for br in entry["results"]:
                item = RegressionItem(entry["scenario_name"], entry["map_type"], br["heuristic_name"])
                item.baseline = br
                item.is_removed = True
                items.append(item)

    return items


def _build_rankings(items: List[RegressionItem], top_n: int = 10):
    """
    构建代价上涨和节点暴涨排行榜, 按场景类型和启发函数聚合。

    Returns:
        (cost_rank, node_rank, cost_by_type, node_by_type, cost_by_heuristic, node_by_heuristic)
    """
    RANK_EPS = 0.01
    cost_items = [
        i for i in items
        if i.cost_delta_pct is not None and i.cost_delta_pct > RANK_EPS
        and i.current and i.baseline and i.current.get("found") and i.baseline.get("found")
    ]
    cost_rank = sorted(cost_items, key=lambda x: x.cost_delta_pct, reverse=True)[:top_n]

    node_items = [
        i for i in items
        if i.node_delta_pct is not None and i.node_delta_pct > RANK_EPS
        and i.current and i.baseline and i.current.get("found") and i.baseline.get("found")
    ]
    node_rank = sorted(node_items, key=lambda x: x.node_delta_pct, reverse=True)[:top_n]

    type_cost = {}
    for i in cost_items:
        mt = i.map_type
        if mt not in type_cost:
            type_cost[mt] = {"count": 0, "total_pct": 0.0, "worst": i}
        type_cost[mt]["count"] += 1
        type_cost[mt]["total_pct"] += i.cost_delta_pct
        if i.cost_delta_pct > type_cost[mt]["worst"].cost_delta_pct:
            type_cost[mt]["worst"] = i

    type_node = {}
    for i in node_items:
        mt = i.map_type
        if mt not in type_node:
            type_node[mt] = {"count": 0, "total_pct": 0.0, "worst": i}
        type_node[mt]["count"] += 1
        type_node[mt]["total_pct"] += i.node_delta_pct
        if i.node_delta_pct > type_node[mt]["worst"].node_delta_pct:
            type_node[mt]["worst"] = i

    h_cost = {}
    for i in cost_items:
        h = i.heuristic_name
        if h not in h_cost:
            h_cost[h] = {"count": 0, "total_pct": 0.0, "worst": i}
        h_cost[h]["count"] += 1
        h_cost[h]["total_pct"] += i.cost_delta_pct
        if i.cost_delta_pct > h_cost[h]["worst"].cost_delta_pct:
            h_cost[h]["worst"] = i

    h_node = {}
    for i in node_items:
        h = i.heuristic_name
        if h not in h_node:
            h_node[h] = {"count": 0, "total_pct": 0.0, "worst": i}
        h_node[h]["count"] += 1
        h_node[h]["total_pct"] += i.node_delta_pct
        if i.node_delta_pct > h_node[h]["worst"].node_delta_pct:
            h_node[h]["worst"] = i

    return cost_rank, node_rank, type_cost, type_node, h_cost, h_node


def format_regression_report(items: List[RegressionItem]) -> str:
    """格式化回归对比报告为可读文本。"""
    lines = []
    lines.append("回归基线对比报告")
    lines.append("=" * 90)

    regressed = [i for i in items if i.has_regression]
    improved = [i for i in items if i.improvements and not i.has_regression]
    new_items = [i for i in items if i.is_new]
    removed_items = [i for i in items if i.is_removed]
    unchanged = [i for i in items if not i.has_change]

    lines.append(f"总计: {len(items)} 项对比")
    lines.append(f"  回归: {len(regressed)} 项")
    lines.append(f"  改善: {len(improved)} 项")
    lines.append(f"  新增: {len(new_items)} 项")
    lines.append(f"  移除: {len(removed_items)} 项")
    lines.append(f"  无变化: {len(unchanged)} 项")
    lines.append("")

    cost_rank, node_rank, type_cost, type_node, h_cost, h_node = _build_rankings(items)

    if cost_rank:
        lines.append("【代价上涨 TOP 榜】")
        lines.append("-" * 60)
        for rank, item in enumerate(cost_rank, 1):
            b_cost = item.baseline["total_cost"] if item.baseline else 0
            c_cost = item.current["total_cost"] if item.current else 0
            lines.append(
                f"  {rank:>2}. [{item.map_type}] {item.scenario_name} / {item.heuristic_name}  "
                f"+{item.cost_delta_pct:.1f}% ({b_cost:.2f} → {c_cost:.2f})"
            )
        lines.append("")

    if node_rank:
        lines.append("【展开节点暴涨 TOP 榜】")
        lines.append("-" * 60)
        for rank, item in enumerate(node_rank, 1):
            b_nodes = item.baseline["nodes_expanded"] if item.baseline else 0
            c_nodes = item.current["nodes_expanded"] if item.current else 0
            lines.append(
                f"  {rank:>2}. [{item.map_type}] {item.scenario_name} / {item.heuristic_name}  "
                f"+{item.node_delta_pct:.0f}% ({b_nodes} → {c_nodes})"
            )
        lines.append("")

    if type_cost:
        lines.append("【按场景类型聚合 — 代价上涨】")
        lines.append("-" * 60)
        for mt, info in sorted(type_cost.items()):
            avg = info["total_pct"] / info["count"]
            w = info["worst"]
            lines.append(
                f"  [{mt}] 受影响 {info['count']} 项, 平均上涨 {avg:.1f}%, "
                f"最差: {w.scenario_name} (+{w.cost_delta_pct:.1f}%)"
            )
        lines.append("")

    if type_node:
        lines.append("【按场景类型聚合 — 节点暴涨】")
        lines.append("-" * 60)
        for mt, info in sorted(type_node.items()):
            avg = info["total_pct"] / info["count"]
            w = info["worst"]
            lines.append(
                f"  [{mt}] 受影响 {info['count']} 项, 平均上涨 {avg:.0f}%, "
                f"最差: {w.scenario_name} (+{w.node_delta_pct:.0f}%)"
            )
        lines.append("")

    if h_cost:
        lines.append("【按启发函数聚合 — 代价上涨】")
        lines.append("-" * 60)
        for h, info in sorted(h_cost.items(), key=lambda x: x[1]["total_pct"], reverse=True):
            avg = info["total_pct"] / info["count"]
            w = info["worst"]
            lines.append(
                f"  {h}: 受影响 {info['count']} 项, 平均上涨 {avg:.1f}%, "
                f"最差: {w.scenario_name} (+{w.cost_delta_pct:.1f}%)"
            )
        lines.append("")

    if h_node:
        lines.append("【按启发函数聚合 — 节点暴涨】")
        lines.append("-" * 60)
        for h, info in sorted(h_node.items(), key=lambda x: x[1]["total_pct"], reverse=True):
            avg = info["total_pct"] / info["count"]
            w = info["worst"]
            lines.append(
                f"  {h}: 受影响 {info['count']} 项, 平均上涨 {avg:.0f}%, "
                f"最差: {w.scenario_name} (+{w.node_delta_pct:.0f}%)"
            )
        lines.append("")

    if regressed:
        lines.append("【回归详情 — 需要关注】")
        lines.append("-" * 60)
        for item in regressed:
            lines.append(f"  ✗ [{item.map_type}] {item.scenario_name} / {item.heuristic_name}")
            for msg in item.regressions:
                lines.append(f"    ✗ {msg}")
            lines.append("")

    if improved:
        lines.append("【改善详情】")
        lines.append("-" * 60)
        for item in improved:
            lines.append(f"  ✓ [{item.map_type}] {item.scenario_name} / {item.heuristic_name}")
            for msg in item.improvements:
                lines.append(f"    ✓ {msg}")
            lines.append("")

    if new_items:
        lines.append("【新增场景】")
        lines.append("-" * 60)
        for item in new_items:
            found = "可达" if item.current and item.current.get("found") else "不可达"
            lines.append(f"  + [{item.map_type}] {item.scenario_name} / {item.heuristic_name} ({found})")
        lines.append("")

    if removed_items:
        lines.append("【已移除场景】")
        lines.append("-" * 60)
        for item in removed_items:
            lines.append(f"  - [{item.map_type}] {item.scenario_name} / {item.heuristic_name}")
        lines.append("")

    return '\n'.join(lines)


def export_regression_markdown(
    items: List[RegressionItem],
    filepath: str,
) -> None:
    """导出回归对比报告为 Markdown 文件。"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 回归基线对比报告\n\n")

        regressed = [i for i in items if i.has_regression]
        improved = [i for i in items if i.improvements and not i.has_regression]
        new_items = [i for i in items if i.is_new]
        removed_items = [i for i in items if i.is_removed]
        unchanged = [i for i in items if not i.has_change]

        f.write(f"**总计: {len(items)} 项对比**\n\n")
        f.write(f"| 类别 | 数量 |\n|------|------|\n")
        f.write(f"| 回归 | {len(regressed)} |\n")
        f.write(f"| 改善 | {len(improved)} |\n")
        f.write(f"| 新增 | {len(new_items)} |\n")
        f.write(f"| 移除 | {len(removed_items)} |\n")
        f.write(f"| 无变化 | {len(unchanged)} |\n\n")

        cost_rank, node_rank, type_cost, type_node, h_cost, h_node = _build_rankings(items)

        if cost_rank:
            f.write("## 🔥 代价上涨 TOP 榜\n\n")
            f.write("| # | 场景 | 类型 | 启发函数 | 基线代价 | 当前代价 | 涨幅 |\n")
            f.write("|---|------|------|----------|----------|----------|------|\n")
            for rank, item in enumerate(cost_rank, 1):
                b_cost = item.baseline["total_cost"] if item.baseline else 0
                c_cost = item.current["total_cost"] if item.current else 0
                f.write(f"| {rank} | {item.scenario_name} | {item.map_type} | "
                        f"{item.heuristic_name} | {b_cost:.2f} | {c_cost:.2f} | "
                        f"+{item.cost_delta_pct:.1f}% |\n")
            f.write("\n")

        if node_rank:
            f.write("## 📈 展开节点暴涨 TOP 榜\n\n")
            f.write("| # | 场景 | 类型 | 启发函数 | 基线节点 | 当前节点 | 涨幅 |\n")
            f.write("|---|------|------|----------|----------|----------|------|\n")
            for rank, item in enumerate(node_rank, 1):
                b_nodes = item.baseline["nodes_expanded"] if item.baseline else 0
                c_nodes = item.current["nodes_expanded"] if item.current else 0
                f.write(f"| {rank} | {item.scenario_name} | {item.map_type} | "
                        f"{item.heuristic_name} | {b_nodes} | {c_nodes} | "
                        f"+{item.node_delta_pct:.0f}% |\n")
            f.write("\n")

        if type_cost:
            f.write("## 按场景类型聚合 — 代价上涨\n\n")
            f.write("| 类型 | 受影响项 | 平均涨幅 | 最差场景 | 最差涨幅 |\n")
            f.write("|------|----------|----------|----------|----------|\n")
            for mt, info in sorted(type_cost.items()):
                avg = info["total_pct"] / info["count"]
                w = info["worst"]
                f.write(f"| {mt} | {info['count']} | +{avg:.1f}% | "
                        f"{w.scenario_name} | +{w.cost_delta_pct:.1f}% |\n")
            f.write("\n")

        if type_node:
            f.write("## 按场景类型聚合 — 节点暴涨\n\n")
            f.write("| 类型 | 受影响项 | 平均涨幅 | 最差场景 | 最差涨幅 |\n")
            f.write("|------|----------|----------|----------|----------|\n")
            for mt, info in sorted(type_node.items()):
                avg = info["total_pct"] / info["count"]
                w = info["worst"]
                f.write(f"| {mt} | {info['count']} | +{avg:.0f}% | "
                        f"{w.scenario_name} | +{w.node_delta_pct:.0f}% |\n")
            f.write("\n")

        if h_cost:
            f.write("## 按启发函数聚合 — 代价上涨\n\n")
            f.write("| 启发函数 | 受影响项 | 平均涨幅 | 最差场景 | 最差涨幅 |\n")
            f.write("|----------|----------|----------|----------|----------|\n")
            for h, info in sorted(h_cost.items(), key=lambda x: x[1]["total_pct"], reverse=True):
                avg = info["total_pct"] / info["count"]
                w = info["worst"]
                f.write(f"| {h} | {info['count']} | +{avg:.1f}% | "
                        f"{w.scenario_name} | +{w.cost_delta_pct:.1f}% |\n")
            f.write("\n")

        if h_node:
            f.write("## 按启发函数聚合 — 节点暴涨\n\n")
            f.write("| 启发函数 | 受影响项 | 平均涨幅 | 最差场景 | 最差涨幅 |\n")
            f.write("|----------|----------|----------|----------|----------|\n")
            for h, info in sorted(h_node.items(), key=lambda x: x[1]["total_pct"], reverse=True):
                avg = info["total_pct"] / info["count"]
                w = info["worst"]
                f.write(f"| {h} | {info['count']} | +{avg:.0f}% | "
                        f"{w.scenario_name} | +{w.node_delta_pct:.0f}% |\n")
            f.write("\n")

        if regressed:
            f.write("## ❌ 回归 — 需要关注\n\n")
            f.write("| 场景 | 类型 | 启发函数 | 问题 |\n")
            f.write("|------|------|----------|------|\n")
            for item in regressed:
                problems = "; ".join(item.regressions)
                f.write(f"| {item.scenario_name} | {item.map_type} | "
                        f"{item.heuristic_name} | {problems} |\n")
            f.write("\n")

        if improved:
            f.write("## ✅ 改善\n\n")
            f.write("| 场景 | 类型 | 启发函数 | 改善 |\n")
            f.write("|------|------|----------|------|\n")
            for item in improved:
                imp = "; ".join(item.improvements)
                f.write(f"| {item.scenario_name} | {item.map_type} | "
                        f"{item.heuristic_name} | {imp} |\n")
            f.write("\n")

        if new_items:
            f.write("## 🆕 新增场景\n\n")
            for item in new_items:
                found = "可达" if item.current and item.current.get("found") else "不可达"
                f.write(f"- [{item.map_type}] {item.scenario_name} / {item.heuristic_name} ({found})\n")
            f.write("\n")

        if removed_items:
            f.write("## 🗑️ 已移除场景\n\n")
            for item in removed_items:
                f.write(f"- [{item.map_type}] {item.scenario_name} / {item.heuristic_name}\n")
            f.write("\n")

        if unchanged:
            f.write("## 不变\n\n")
            f.write("| 场景 | 类型 | 启发函数 | 代价 | 展开节点 |\n")
            f.write("|------|------|----------|------|----------|\n")
            for item in unchanged:
                c = item.current
                cost = f"{c['total_cost']:.2f}" if c and c.get("found") else "—"
                nodes = c.get("nodes_expanded", "—") if c else "—"
                f.write(f"| {item.scenario_name} | {item.map_type} | "
                        f"{item.heuristic_name} | {cost} | {nodes} |\n")
            f.write("\n")
