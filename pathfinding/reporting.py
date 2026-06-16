"""
场景验收报告模块 — 校验寻路结果是否符合预期, 支持导出 CSV/Markdown。

JSON 场景文件中的 expectations 字段:
{
  "expectations": {
    "reachable": true,           // 是否可达
    "min_cost": 10.0,            // 代价下界 (可选)
    "max_cost": 20.0,            // 代价上界 (可选)
    "start_valid": true,         // 起点是否有效 (可选)
    "goal_valid": true,          // 终点是否有效 (可选)
    "min_path_length": 5,        // 最少路径点数 (可选)
    "max_path_length": 100,      // 最多路径点数 (可选)
    "max_nodes_expanded": 1000,  // 最多展开节点 (可选, 性能上限)
    "waypoints": [[1,1], [5,5]]  // 路径必须经过的关键点 (可选)
  }
}
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import PathfindingResult
from .geometry import point_to_point_distance


class ExpectationResult:
    """单个期望项的校验结果。"""

    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class ScenarioReport:
    """单个场景的验收报告。"""

    def __init__(self, scenario_name: str, map_type: str):
        self.scenario_name = scenario_name
        self.map_type = map_type
        self.expectations: Dict[str, Any] = {}
        self.results: List[PathfindingResult] = []
        self.checks: List[ExpectationResult] = []
        self.overall_passed = True

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) if self.checks else True


def validate_scenario(
    result: PathfindingResult,
    expectations: Dict[str, Any],
) -> List[ExpectationResult]:
    """
    校验单个寻路结果是否满足预期。

    Args:
        result: PathfindingResult 对象
        expectations: 预期字典

    Returns:
        校验结果列表
    """
    checks = []

    if "reachable" in expectations:
        expected_reachable = expectations["reachable"]
        actual_reachable = result.found
        passed = actual_reachable == expected_reachable
        checks.append(ExpectationResult(
            "reachable", passed,
            f"预期可达={expected_reachable}, 实际={'是' if actual_reachable else '否'}"
        ))

    if not result.found:
        if "start_valid" in expectations:
            expected = expectations["start_valid"]
            actual = result.start_valid
            passed = expected == actual
            checks.append(ExpectationResult(
                "start_valid", passed,
                f"预期起点有效={expected}, 实际={'是' if actual else '否'}"
            ))
        if "goal_valid" in expectations:
            expected = expectations["goal_valid"]
            actual = result.goal_valid
            passed = expected == actual
            checks.append(ExpectationResult(
                "goal_valid", passed,
                f"预期终点有效={expected}, 实际={'是' if actual else '否'}"
            ))
        return checks

    if "min_cost" in expectations:
        min_cost = float(expectations["min_cost"])
        passed = result.total_cost >= min_cost - 1e-9
        checks.append(ExpectationResult(
            "min_cost", passed,
            f"代价 >= {min_cost}, 实际={result.total_cost:.4f}"
        ))

    if "max_cost" in expectations:
        max_cost = float(expectations["max_cost"])
        passed = result.total_cost <= max_cost + 1e-9
        checks.append(ExpectationResult(
            "max_cost", passed,
            f"代价 <= {max_cost}, 实际={result.total_cost:.4f}"
        ))

    if "min_path_length" in expectations:
        min_len = int(expectations["min_path_length"])
        actual_len = (len(result.smooth_path) if result.smooth_path
                      else len(result.path))
        passed = actual_len >= min_len
        checks.append(ExpectationResult(
            "min_path_length", passed,
            f"路径点数 >= {min_len}, 实际={actual_len}"
        ))

    if "max_path_length" in expectations:
        max_len = int(expectations["max_path_length"])
        actual_len = (len(result.smooth_path) if result.smooth_path
                      else len(result.path))
        passed = actual_len <= max_len
        checks.append(ExpectationResult(
            "max_path_length", passed,
            f"路径点数 <= {max_len}, 实际={actual_len}"
        ))

    if "max_nodes_expanded" in expectations:
        max_exp = int(expectations["max_nodes_expanded"])
        passed = result.nodes_expanded <= max_exp
        checks.append(ExpectationResult(
            "max_nodes_expanded", passed,
            f"展开节点 <= {max_exp}, 实际={result.nodes_expanded}"
        ))

    if "waypoints" in expectations and result.smooth_path:
        waypoints = [tuple(wp) for wp in expectations["waypoints"]]
        all_passed = True
        failed_wps = []
        for wp in waypoints:
            found = False
            for pt in result.smooth_path:
                if point_to_point_distance(tuple(pt), wp) < 0.5:
                    found = True
                    break
            if not found:
                all_passed = False
                failed_wps.append(wp)
        checks.append(ExpectationResult(
            "waypoints", all_passed,
            f"路径经过关键点, 未通过: {failed_wps}" if failed_wps
            else "所有关键点均经过"
        ))

    return checks


def generate_report(
    all_results: List[Tuple[str, str, List[PathfindingResult], Dict[str, Any]]],
) -> List[ScenarioReport]:
    """
    生成一批场景的验收报告。

    Args:
        all_results: [(scenario_name, map_type, [results...], expectations), ...]

    Returns:
        ScenarioReport 列表
    """
    reports = []
    for name, map_type, results, expectations in all_results:
        report = ScenarioReport(name, map_type)
        report.expectations = expectations
        report.results = results

        primary = results[0] if results else None
        if primary and expectations:
            report.checks = validate_scenario(primary, expectations)
            report.overall_passed = report.passed

        reports.append(report)
    return reports


def format_report_summary(reports: List[ScenarioReport], verbose: bool = False) -> str:
    """格式化报告汇总为可读文本。verbose=True 显示完整路径坐标。"""
    lines = []
    lines.append("场景验收报告")
    lines.append("=" * 90)

    passed = sum(1 for r in reports if r.overall_passed)
    failed = len(reports) - passed
    lines.append(f"总计: {len(reports)} 个场景, 通过 {passed} 个, 失败 {failed} 个")
    lines.append("")

    for report in reports:
        status = "✓ 通过" if report.overall_passed else "✗ 失败"
        lines.append(f"  [{report.map_type:>4}] {report.scenario_name:<30} {status}")

        for check in report.checks:
            sym = "  ✓" if check.passed else "  ✗"
            lines.append(f"    {sym} {check.name:<20} {check.message}")

        if report.results and report.results[0].found:
            r = report.results[0]
            lines.append(f"       代价={r.total_cost:.4f}, 展开={r.nodes_expanded}, "
                         f"耗时={r.time_ms:.3f}ms")
            lines.append(f"       路径: {r.path_summary(verbose=verbose)}")
        lines.append("")

    return '\n'.join(lines)


def export_csv(reports: List[ScenarioReport], filepath: str) -> None:
    """导出报告为 CSV 文件。"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "场景名称", "地图类型", "整体结果", "启发函数",
            "是否找到", "总代价", "展开节点", "生成节点", "Open峰值",
            "耗时(ms)", "原始路径长", "平滑路径长",
        ])

        for report in reports:
            for r in report.results:
                writer.writerow([
                    report.scenario_name,
                    report.map_type,
                    "通过" if report.overall_passed else "失败",
                    r.heuristic_name,
                    "是" if r.found else "否",
                    f"{r.total_cost:.4f}" if r.found else "",
                    r.nodes_expanded,
                    r.nodes_generated,
                    r.max_open_size,
                    f"{r.time_ms:.3f}",
                    f"{r.raw_path_length:.4f}" if r.raw_path_length else "",
                    f"{r.smooth_path_length:.4f}" if r.smooth_path_length else "",
                ])


def export_markdown(reports: List[ScenarioReport], filepath: str) -> None:
    """导出报告为 Markdown 文件。"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 寻路引擎验收报告\n\n")

        passed = sum(1 for r in reports if r.overall_passed)
        failed = len(reports) - passed
        f.write(f"**总计: {len(reports)} 个场景, 通过 {passed} 个, 失败 {failed} 个**\n\n")

        f.write("## 汇总表\n\n")
        f.write("| 场景 | 类型 | 结果 | 代价 | 展开节点 | 耗时(ms) |\n")
        f.write("|------|------|------|------|----------|----------|\n")

        for report in reports:
            r = report.results[0] if report.results else None
            status = "✅ 通过" if report.overall_passed else "❌ 失败"
            cost = f"{r.total_cost:.2f}" if (r and r.found) else "—"
            expanded = r.nodes_expanded if r else "—"
            time_ms = f"{r.time_ms:.2f}" if r else "—"
            f.write(f"| {report.scenario_name} | {report.map_type} | "
                    f"{status} | {cost} | {expanded} | {time_ms} |\n")

        f.write("\n## 详细结果\n\n")

        for report in reports:
            status = "✅ 通过" if report.overall_passed else "❌ 失败"
            f.write(f"### {report.scenario_name} ({report.map_type}) — {status}\n\n")

            if report.checks:
                f.write("**验收项:**\n\n")
                for check in report.checks:
                    sym = "✅" if check.passed else "❌"
                    f.write(f"- {sym} **{check.name}**: {check.message}\n")
                f.write("\n")

            if report.results:
                f.write("**寻路结果:**\n\n")
                f.write("| 启发函数 | 找到 | 代价 | 展开 | 生成 | Open峰值 | 耗时(ms) | 原始长 | 平滑长 |\n")
                f.write("|----------|------|------|------|------|----------|----------|--------|--------|\n")
                for r in report.results:
                    found = "是" if r.found else "否"
                    cost = f"{r.total_cost:.2f}" if r.found else "—"
                    raw_len = f"{r.raw_path_length:.2f}" if r.raw_path_length else "—"
                    smooth_len = f"{r.smooth_path_length:.2f}" if r.smooth_path_length else "—"
                    f.write(f"| {r.heuristic_name} | {found} | {cost} | "
                            f"{r.nodes_expanded} | {r.nodes_generated} | "
                            f"{r.max_open_size} | {r.time_ms:.2f} | "
                            f"{raw_len} | {smooth_len} |\n")
            f.write("\n")
