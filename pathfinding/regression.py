"""
回归基线模块 — 保存/加载历史测试结果, 对比差异, 标出回归和改善。

功能:
  - 将场景测试结果序列化为 JSON 基线文件
  - 加载历史基线并与当前结果对比
  - 标出: 代价变差、展开节点暴涨、可达性变化
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
                if c_cost > b_cost * (1 + cost_thresh):
                    pct = ((c_cost - b_cost) / b_cost) * 100
                    item.regressions.append(
                        f"代价上涨 {pct:.1f}% ({b_cost:.2f} → {c_cost:.2f})"
                    )
                elif c_cost < b_cost * (1 - cost_thresh):
                    pct = ((b_cost - c_cost) / b_cost) * 100
                    item.improvements.append(
                        f"代价下降 {pct:.1f}% ({b_cost:.2f} → {c_cost:.2f})"
                    )

                b_nodes = b["nodes_expanded"]
                c_nodes = r.nodes_expanded
                if b_nodes > 0 and c_nodes > b_nodes * (1 + node_thresh):
                    pct = ((c_nodes - b_nodes) / b_nodes) * 100
                    item.regressions.append(
                        f"展开节点暴涨 {pct:.0f}% ({b_nodes} → {c_nodes})"
                    )
                elif b_nodes > 0 and c_nodes < b_nodes * (1 - node_thresh):
                    pct = ((b_nodes - c_nodes) / b_nodes) * 100
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

    if regressed:
        lines.append("【回归 — 需要关注】")
        lines.append("-" * 60)
        for item in regressed:
            lines.append(f"  ✗ [{item.map_type}] {item.scenario_name} / {item.heuristic_name}")
            for msg in item.regressions:
                lines.append(f"    ✗ {msg}")
            lines.append("")

    if improved:
        lines.append("【改善】")
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
