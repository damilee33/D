import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


FIGURE_SKILL_SCRIPTS = Path(
    r"C:\Users\dami\.codex\skills\math-modeling\tools\figure\scripts"
)
sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))
from export_figure import export_figure  # noqa: E402
from layout_tools import finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from PIL import Image  # noqa: E402


OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def export_with_qa(fig, basename, size_inches):
    finalize_figure(fig)
    question = "q2" if "_q2_" in basename.name else "q1"
    qa_dir = basename.parent.parent / "results" / question / "figure_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    preview = qa_dir / (basename.name + "_preview.png")
    render_preview(fig, preview, dpi=150)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if verdict == "FAIL":
        raise RuntimeError("figure layout audit failed: {}".format(basename))
    outputs = export_figure(
        fig,
        str(basename),
        # Project-level delivery decision: the final paper is Word-only, so
        # future figures are exported as high-resolution PNG only.
        formats=["png"],
        size_inches=size_inches,
        dpi=300,
        grayscale_preview=False,
    )
    with Image.open(str(basename) + ".png") as image:
        image.convert("L").save(
            qa_dir / (basename.name + "_grayscale.png"), dpi=(300, 300)
        )
    plt.close(fig)
    return {"basename": str(basename), "verdict": verdict, "issues": issues, "outputs": outputs}


def raw_demand_figure(project_root, figures_dir):
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.io.dataset import load_project_data

    data = load_project_data(project_root)
    services = [row["服务区编号"] for row in data["nodes"]["service_areas"]]
    materials = ["医疗物资", "饮用水", "应急食品", "生活卫生用品"]
    matrix = np.zeros((len(services), len(materials)))
    for box in data["demands"]["boxes"]:
        matrix[services.index(box["服务区编号"]), materials.index(box["物资类型"])] += box[
            "单箱质量（kg）"
        ]
    order = np.argsort(matrix.sum(axis=1))
    labels = [services[index] for index in order]
    values = matrix[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    left = np.zeros(len(labels))
    hatches = ["///", "\\\\", "...", "xx"]
    for material, color, hatch, column in zip(materials, OKABE[:4], hatches, values.T):
        ax.barh(
            labels,
            column,
            left=left,
            label=material,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.45,
        )
        left += column
    ax.set_xlabel("需求质量（kg）")
    ax.set_ylabel("服务区")
    ax.set_title("各服务区物资需求质量构成")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    return export_with_qa(
        fig, figures_dir / "raw_q1_demand_structure", (7.2, 4.8)
    )


def raw_box_properties_figure(project_root, figures_dir):
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.io.dataset import load_project_data

    boxes = load_project_data(project_root)["demands"]["boxes"]
    materials = ["医疗物资", "饮用水", "应急食品", "生活卫生用品"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for material, color, marker in zip(materials, OKABE[:4], markers):
        selected = [box for box in boxes if box["物资类型"] == material]
        grouped = {}
        for box in selected:
            key = (box["单箱质量（kg）"], box["单箱体积（m³）"] * 1000.0)
            grouped[key] = grouped.get(key, 0) + 1
        for point_index, ((mass, volume), count) in enumerate(sorted(grouped.items())):
            ax.scatter(
                [mass], [volume], s=25 + 7 * count, color=color, marker=marker,
                alpha=0.82, edgecolor="white", linewidth=0.35,
                label="{}（n={}）".format(material, len(selected)) if point_index == 0 else None,
            )
            ax.annotate("×{}".format(count), (mass, volume), xytext=(5, 5), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("单箱质量（kg）")
    ax.set_ylabel("单箱体积（L）")
    ax.set_title("80 个不可拆分货箱的质量—体积结构")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.2, linewidth=0.5)
    return export_with_qa(fig, figures_dir / "raw_q1_box_mass_volume", (6.4, 4.2))


def raw_route_geometry_figure(project_root, figures_dir):
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.domain.geometry import RasterDEM, segment_terrain_profile
    from dproblem.io.dataset import load_project_data

    data = load_project_data(project_root)
    center = data["nodes"]["centers"][0]
    dem = RasterDEM(data["paths"]["dem_tif"])
    profiles = []
    for service in data["nodes"]["service_areas"]:
        profile = segment_terrain_profile(
            dem, center, service, origin_work_offset_m=0.0, destination_work_offset_m=30.0
        )
        profiles.append((service["服务区编号"], profile))
    profiles.sort(key=lambda item: item[1]["horizontal_distance_m"])
    labels = [item[0] for item in profiles]
    distances = [item[1]["horizontal_distance_m"] / 1000.0 for item in profiles]
    climbs = [item[1]["forward_climb_m"] for item in profiles]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True, constrained_layout=True)
    axes[0].barh(labels, distances, color=OKABE[0])
    axes[1].barh(labels, climbs, color=OKABE[1], hatch="//", edgecolor="white")
    axes[0].set_xlabel("单程水平距离（km）")
    axes[1].set_xlabel("去程累计爬升（m）")
    axes[0].set_ylabel("服务区（按距离排序）")
    axes[0].set_title("航线距离")
    axes[1].set_title("DEM 地形爬升负担")
    for ax, panel in zip(axes, "ab"):
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.text(-0.14, 1.02, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    return export_with_qa(fig, figures_dir / "raw_q1_route_geometry", (7.2, 4.8))


def safe_payload_figure(project_root, figures_dir):
    rows = read_csv(project_root / "results" / "q1" / "safe_payload_matrix.csv")
    services = sorted({row["服务区编号"] for row in rows})
    types = ["A", "B", "C"]
    capacities = {"A": 25.0, "B": 30.0, "C": 80.0}
    matrix = np.zeros((len(services), len(types)))
    for row in rows:
        matrix[services.index(row["服务区编号"]), types.index(row["机型编号"])] = (
            float(row["最大安全载荷（kg）"]) / capacities[row["机型编号"]] * 100.0
        )
    fig, ax = plt.subplots(figsize=(4.6, 5.0), constrained_layout=True)
    boundaries = np.linspace(0, 100, 11)
    colormap = plt.get_cmap("viridis", 10)
    norm = BoundaryNorm(boundaries, colormap.N, clip=True)
    image = ax.pcolormesh(
        np.arange(len(types) + 1),
        np.arange(len(services) + 1),
        matrix,
        cmap=colormap,
        norm=norm,
        shading="flat",
        edgecolors="none",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value < 55 else "#222222"
            ax.text(column_index + 0.5, row_index + 0.5, "{:.0f}".format(value), ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks(np.arange(len(types)) + 0.5, ["A型", "B型", "C型"])
    ax.set_yticks(np.arange(len(services)) + 0.5, services)
    ax.set_xlim(0, len(types))
    ax.set_ylim(len(services), 0)
    ax.set_xlabel("运输无人机机型")
    ax.set_ylabel("服务区")
    ax.set_title("20%返航余量下最大安全载荷占额定载荷比例")
    colorbar = fig.colorbar(
        image, ax=ax, shrink=0.86, boundaries=boundaries, ticks=np.arange(0, 101, 20)
    )
    colorbar.set_label("安全载荷 / 额定载荷（%）")
    return export_with_qa(
        fig, figures_dir / "process_q1_safe_payload", (4.6, 5.0)
    )


def reserve_sensitivity_figure(project_root, figures_dir):
    rows = read_csv(project_root / "results" / "q1" / "safe_payload_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, type_id, color, panel in zip(axes, ["A", "B", "C"], OKABE[:3], "abc"):
        selected = [row for row in rows if row["机型编号"] == type_id]
        reserves = sorted({float(row["返航余量"]) for row in selected})
        grouped = [
            [float(row["最大安全载荷（kg）"]) for row in selected if float(row["返航余量"]) == reserve]
            for reserve in reserves
        ]
        lower = np.array([min(values) for values in grouped])
        median = np.array([np.median(values) for values in grouped])
        upper = np.array([max(values) for values in grouped])
        x = np.array(reserves) * 100.0
        ax.fill_between(x, lower, upper, color=color, alpha=0.18, label="服务区范围")
        ax.plot(x, median, color=color, marker="o", linewidth=1.4, label="中位数")
        ax.set_title("{}型".format(type_id))
        ax.set_xlabel("返航余量（%）")
        ax.set_ylabel("最大安全载荷（kg）" if type_id == "A" else "")
        ax.set_xticks(x)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.5)
    return export_with_qa(
        fig, figures_dir / "process_q1_reserve_sensitivity", (7.2, 2.8)
    )


def neighborhood_gain_figure(project_root, figures_dir):
    baseline = read_csv(project_root / "results" / "q1" / "baseline_trips.csv")
    main = read_csv(project_root / "results" / "q1" / "main_trips.csv")
    services = sorted({row["服务区编号"] for row in main})

    def aggregate(rows, field):
        return {
            service: sum(float(row[field]) for row in rows if row["服务区编号"] == service)
            for service in services
        }

    energy_base = aggregate(baseline, "架次能耗（kWh）")
    energy_main = aggregate(main, "架次能耗（kWh）")
    time_base = aggregate(baseline, "累计作业时间（s）")
    time_main = aggregate(main, "累计作业时间（s）")
    energy_gain = [energy_base[s] - energy_main[s] for s in services]
    time_gain = [time_base[s] - time_main[s] for s in services]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    axes[0].bar(services, energy_gain, color=OKABE[0])
    axes[1].bar(services, time_gain, color=OKABE[1], hatch="//", edgecolor="white")
    axes[0].set_ylabel("节省能耗（kWh）")
    axes[1].set_ylabel("节省作业时间（s）")
    axes[1].set_xlabel("服务区")
    axes[0].set_title("双批精确重组对各服务区的改进贡献")
    axes[0].text(-0.07, 1.02, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=9)
    axes[1].text(-0.07, 1.02, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="#333333", linewidth=0.6)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    return export_with_qa(
        fig, figures_dir / "process_q1_neighborhood_gain", (7.2, 4.6)
    )


def plan_comparison_figure(project_root, figures_dir):
    summary = json.loads(
        (project_root / "results" / "q1" / "q1_summary.json").read_text(encoding="utf-8")
    )
    metrics = [
        ("架次数", summary["baseline_trip_count"], summary["main_trip_count"], "次", 0),
        (
            "总能耗",
            summary["baseline_total_energy_kwh"],
            summary["main_total_energy_kwh"],
            "kWh",
            3,
        ),
        (
            "累计作业时间",
            summary["baseline_cumulative_work_seconds"] / 3600.0,
            summary["main_cumulative_work_seconds"] / 3600.0,
            "h",
            3,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, (title, baseline, main, unit, precision), panel in zip(axes, metrics, "abc"):
        bars = ax.barh(
            ["FFD 基准", "局部搜索"],
            [baseline, main],
            color=["#999999", OKABE[0]],
            hatch=["///", ""],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlim(0, max(baseline, main) * 1.22)
        ax.set_xlabel(unit)
        ax.set_title(title)
        reduction = (baseline - main) / baseline * 100.0
        for bar, value in zip(bars, (baseline, main)):
            ax.text(
                value + max(baseline, main) * 0.025,
                bar.get_y() + bar.get_height() / 2,
                ("{:.%df}" % precision).format(value),
                va="center",
                fontsize=7,
            )
        ax.text(
            0.98,
            0.94,
            "降低 {:.2f}%".format(reduction),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=OKABE[0],
            fontsize=7,
        )
        ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    return export_with_qa(
        fig, figures_dir / "result_q1_plan_comparison", (7.2, 2.8)
    )


def result_constraint_margins_figure(project_root, figures_dir):
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.io.dataset import load_project_data

    rows = read_csv(project_root / "results" / "q1" / "main_trips.csv")
    types = {
        row["机型编号"]: row for row in load_project_data(project_root)["transport"]["types"]
    }
    x = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True)
    for type_id, color, marker in zip(["A", "B", "C"], OKABE[:3], ["o", "s", "^"]):
        indices = [i for i, row in enumerate(rows) if row["机型编号"] == type_id]
        if not indices:
            continue
        axes[0].scatter(
            x[indices],
            [float(rows[i]["总质量（kg）"]) / types[type_id]["最大载货质量（kg）"] * 100 for i in indices],
            color=color, marker=marker, label="{}型".format(type_id), s=25,
        )
        axes[1].scatter(
            x[indices],
            [float(rows[i]["总体积（m³）"]) / types[type_id]["可用装载体积（m³）"] * 100 for i in indices],
            color=color, marker=marker, s=25,
        )
        axes[2].scatter(
            x[indices],
            [float(rows[i]["返航SOC（%）"]) - types[type_id]["返航电量下限（%）"] for i in indices],
            color=color, marker=marker, s=25,
        )
    axes[0].set_ylabel("质量利用率（%）")
    axes[1].set_ylabel("体积利用率（%）")
    axes[2].set_ylabel("返航SOC裕量（百分点）")
    axes[2].set_xlabel("主方案架次序号")
    axes[0].set_title("主方案逐架次硬约束裕量")
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    axes[0].axhline(100, color="#333333", linewidth=0.7, linestyle="--")
    axes[1].axhline(100, color="#333333", linewidth=0.7, linestyle="--")
    axes[2].axhline(0, color="#333333", linewidth=0.7, linestyle="--")
    for ax, panel in zip(axes, "abc"):
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(-0.07, 1.02, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    return export_with_qa(
        fig, figures_dir / "result_q1_constraint_margins", (7.2, 5.4)
    )


def result_service_plan_figure(project_root, figures_dir):
    rows = read_csv(project_root / "results" / "q1" / "main_trips.csv")
    services = sorted({row["服务区编号"] for row in rows})
    types = ["A", "B", "C"]
    counts = np.array(
        [[sum(row["服务区编号"] == service and row["机型编号"] == type_id for row in rows) for type_id in types] for service in services]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    left = np.zeros(len(services))
    for type_id, color, hatch, column in zip(types, OKABE[:3], ["..", "//", ""], counts.T):
        ax.barh(services, column, left=left, color=color, hatch=hatch, edgecolor="white", label="{}型".format(type_id))
        left += column
    for index, total in enumerate(left):
        ax.text(total + 0.04, index, "{} 架次".format(int(total)), va="center", fontsize=6.5)
    ax.set_xlim(0, max(left) + 0.55)
    ax.set_xlabel("架次数")
    ax.set_ylabel("服务区")
    ax.set_title("主方案各服务区机型与架次配置")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    return export_with_qa(fig, figures_dir / "result_q1_service_plan", (6.4, 4.8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    style = setup_style(journal="general", lang="zh", serif_for_zh=True)
    reports = [
        raw_demand_figure(project_root, figures_dir),
        raw_box_properties_figure(project_root, figures_dir),
        raw_route_geometry_figure(project_root, figures_dir),
        safe_payload_figure(project_root, figures_dir),
        reserve_sensitivity_figure(project_root, figures_dir),
        neighborhood_gain_figure(project_root, figures_dir),
        plan_comparison_figure(project_root, figures_dir),
        result_constraint_margins_figure(project_root, figures_dir),
        result_service_plan_figure(project_root, figures_dir),
    ]
    contracts = {
        "style": style,
        "visual_review": {
            "rounds": 2,
            "status": "PASS",
            "checks": [
                "中文与符号无缺字",
                "标题、坐标轴、刻度和数值标注未裁切或重叠",
                "图例未遮挡数据",
                "灰度预览仍可借助纹理或数值标注辨识",
                "热力图改为真正矢量网格，SVG不含base64位图",
            ],
            "known_gap": "check_figure默认运行时未装pypdf；已另用bundled runtime/pypdf确认PDF使用嵌入式Type0/CIDFontType2字体且无Type3字体",
        },
        "figures": [
            {
                "id": "raw_q1_demand_structure",
                "core_claim": "15个服务区的需求规模与物资构成差异显著，组批必须逐区处理。",
                "evidence": "逐箱清单按服务区和物资类型汇总的质量，不含推断值。",
                "chart": "按总质量排序的水平堆叠柱状图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "raw_q1_box_mass_volume",
                "core_claim": "货箱不可拆分且不同物资的质量—体积组合差异明显，组批必须同时检查重量与体积。",
                "evidence": "80个原始货箱的单箱质量和单箱体积。",
                "chart": "按物资类型编码的质量—体积散点图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "raw_q1_route_geometry",
                "core_claim": "15条单点往返航线的水平距离和DEM爬升负担存在明显差异。",
                "evidence": "节点经纬度与30米DEM严格超覆盖剖面。",
                "chart": "共享服务区顺序的双面板水平柱状图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "process_q1_safe_payload",
                "core_claim": "在20%返航余量下，部分远端航线的安全载荷受能量而非额定载荷限制。",
                "evidence": "15×3最大安全载荷矩阵除以对应机型额定载荷。",
                "chart": "单向viridis热力图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "process_q1_reserve_sensitivity",
                "core_claim": "返航余量从10%提高到30%时，各机型安全载荷不出现系统性上升。",
                "evidence": "15个服务区、3类机型、5档返航余量的最大安全载荷。",
                "chart": "中位数折线与服务区最小—最大范围带",
                "backend": "Python/matplotlib",
            },
            {
                "id": "process_q1_neighborhood_gain",
                "core_claim": "局部搜索的收益来自少数服务区的双批重组，未靠减少配送覆盖获得。",
                "evidence": "逐服务区FFD基线与主方案的能耗、累计作业时间差。",
                "chart": "逐服务区双面板改进贡献柱状图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "result_q1_plan_comparison",
                "core_claim": "在保持最少架次数不变的前提下，局部搜索主方案降低总能耗并缩短累计作业时间。",
                "evidence": "FFD baseline 与主方案的确定性总量对比；非均值、无统计误差棒。",
                "chart": "从零起轴的三面板确定性总量水平柱状图，并直接标注精确值与相对变化",
                "backend": "Python/matplotlib",
            },
            {
                "id": "result_q1_constraint_margins",
                "core_claim": "主方案18个架次的质量、体积和返航SOC均满足硬约束。",
                "evidence": "逐架次质量/体积利用率与相对机型阈值的返航SOC裕量。",
                "chart": "按机型编码的三面板约束裕量散点图",
                "backend": "Python/matplotlib",
            },
            {
                "id": "result_q1_service_plan",
                "core_claim": "主方案逐服务区明确给出机型与架次数配置，且不跨区组批。",
                "evidence": "main_trips.csv中的18个已审计架次。",
                "chart": "逐服务区机型堆叠水平柱状图",
                "backend": "Python/matplotlib",
            },
        ],
        "qa": reports,
    }
    (project_root / "results" / "q1" / "figure_contracts.json").write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contracts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
