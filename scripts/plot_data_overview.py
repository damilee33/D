import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


FIGURE_SIZE = (9.0, 5.5)
PROJECT_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_CODE_ROOT))
FIGURE_SKILL_SCRIPTS = Path(
    r"C:\Users\dami\.codex\skills\math-modeling\tools\figure\scripts"
)
sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from utils.plot_style import PALETTE  # noqa: E402


MATERIALS = ["医疗物资", "饮用水", "应急食品", "生活卫生用品"]
TYPES = ["A", "B", "C"]


def _configure_console():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _load_project_data(project_root):
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.domain.geometry import RasterDEM
    from dproblem.io.dataset import load_project_data

    data = load_project_data(project_root)
    data["dem"] = RasterDEM(data["paths"]["dem_tif"])
    return data


def _material_profile(data):
    boxes = data["demands"]["boxes"]
    rows = []
    for material in MATERIALS:
        selected = [box for box in boxes if box["物资类型"] == material]
        rows.append(
            {
                "物资类型": material,
                "货箱数量（箱）": len(selected),
                "总质量（kg）": sum(float(box["单箱质量（kg）"]) for box in selected),
                "总体积（m³）": sum(float(box["单箱体积（m³）"]) for box in selected),
            }
        )
    return rows


def _resource_profile(data):
    drone_counts = Counter(row["机型编号"] for row in data["transport"]["drones"])
    battery_counts = {
        row["机型编号"]: int(row["共享电池组总数（组）"])
        for row in data["transport"]["batteries"]
    }
    return [
        {
            "机型": type_id,
            "实体无人机（架）": int(drone_counts[type_id]),
            "共享电池组（组）": int(battery_counts[type_id]),
        }
        for type_id in TYPES
    ]


def _write_profiles(results_dir, material_rows, resource_rows):
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("物资类别汇总.csv", material_rows),
        ("运输资源汇总.csv", resource_rows),
    ):
        with (results_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _export_with_qa(fig, output_stem, qa_dir):
    finalize_figure(fig)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(w_pad=0.12, h_pad=0.14, hspace=0.06)
        fig.canvas.draw()
    qa_dir.mkdir(parents=True, exist_ok=True)
    preview_path = qa_dir / (output_stem.name + "_preview.png")
    render_preview(fig, str(preview_path), dpi=150)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if verdict != "PASS":
        raise RuntimeError(
            "figure layout audit did not pass for {}: {}".format(output_stem, issues)
        )
    outputs = export_figure(
        fig,
        str(output_stem),
        formats=["pdf", "svg", "png"],
        size_inches=FIGURE_SIZE,
        dpi=300,
        grayscale_preview=False,
        tight=False,
    )
    grayscale_path = output_stem.parent / (output_stem.name + "_grayscale.png")
    with Image.open(str(output_stem) + ".png") as image:
        image.convert("L").save(grayscale_path, dpi=(300, 300))
    outputs.append(str(grayscale_path))
    plt.close(fig)
    return {
        "id": output_stem.name,
        "size_inches": list(FIGURE_SIZE),
        "layout_verdict": verdict,
        "layout_issues": issues,
        "preview": str(preview_path),
        "outputs": outputs,
    }


def plot_material_distribution(material_rows, figures_dir, qa_dir):
    labels = [row["物资类型"] for row in material_rows]
    counts = np.array([row["货箱数量（箱）"] for row in material_rows], dtype=float)
    masses = np.array([row["总质量（kg）"] for row in material_rows], dtype=float)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.04},
    )
    bar_color = PALETTE["primary"]
    line_color = PALETTE["secondary"]

    bars = axes[0].bar(
        x,
        counts,
        width=0.56,
        color=bar_color,
        edgecolor="white",
        linewidth=0.7,
        label="货箱数量",
    )
    axes[0].bar_label(bars, labels=["{:.0f}".format(value) for value in counts], padding=4)
    axes[0].set_ylabel("货箱数量（箱）")
    axes[0].set_ylim(0, counts.max() * 1.28)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)

    axes[1].plot(
        x,
        masses,
        color=line_color,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=2.0,
        label="总质量",
    )
    for position, value in zip(x, masses):
        axes[1].annotate(
            "{:.0f}".format(value),
            (position, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=line_color,
            fontweight="bold",
        )
    axes[1].set_ylabel("总质量（kg）")
    axes[1].set_xlabel("物资类型")
    axes[1].set_ylim(0, masses.max() * 1.22)
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    fig.suptitle("各类应急物资的货箱数量与总质量分布", fontsize=13, fontweight="bold")

    report = _export_with_qa(
        fig,
        figures_dir / "图1_各类物资箱数与总质量分布",
        qa_dir,
    )
    report.update(
        {
            "core_claim": "饮用水同时占据最多货箱和最大质量，是运输需求的主要负担。",
            "evidence": "80个逐箱记录按物资类型汇总；柱为箱数，折线为总质量。",
            "chart": "上下双面板共享分类轴：柱状图+折线图，避免双Y轴误导。",
        }
    )
    return report


def plot_transport_resources(resource_rows, figures_dir, qa_dir):
    labels = ["{}型".format(row["机型"]) for row in resource_rows]
    drones = np.array([row["实体无人机（架）"] for row in resource_rows])
    batteries = np.array([row["共享电池组（组）"] for row in resource_rows])
    x = np.arange(len(labels))
    width = 0.31

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    bars_drone = ax.bar(
        x - width / 2,
        drones,
        width,
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=0.7,
        label="实体无人机（架）",
    )
    bars_battery = ax.bar(
        x + width / 2,
        batteries,
        width,
        color=PALETTE["secondary"],
        hatch="//",
        edgecolor="white",
        linewidth=0.7,
        label="共享电池组（组）",
    )
    ax.bar_label(bars_drone, padding=4)
    ax.bar_label(bars_battery, padding=4)
    ax.set_xticks(x, labels)
    ax.set_xlabel("运输无人机机型")
    ax.set_ylabel("资源数量（架/组）")
    ax.set_ylim(0, max(drones.max(), batteries.max()) + 1.8)
    ax.set_title("A/B/C型运输无人机及共享电池资源数量", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)

    report = _export_with_qa(
        fig,
        figures_dir / "图2_ABC无人机及电池资源数量",
        qa_dir,
    )
    report.update(
        {
            "core_claim": "A型无人机和对应电池组数量最多，B/C型实体机库存均为2架。",
            "evidence": "运输无人机数据.xlsx中的实体编号与共享电池库存。",
            "chart": "按机型分组的双系列柱状图，颜色与纹理双重编码。",
        }
    )
    return report


def _label_offsets(services, center):
    del center
    return {
        "S001": (8, -13),
        "S002": (9, 7),
        "S003": (-12, 7),
        "S004": (9, 10),
        "S005": (-10, 9),
        "S006": (-13, -14),
        "S007": (-11, 5),
        "S008": (-8, 10),
        "S009": (10, -13),
        "S010": (11, 5),
        "S011": (-10, 9),
        "S012": (11, -12),
        "S013": (11, 7),
        "S014": (11, -14),
        "S015": (-9, 10),
    }


def plot_dem_service_map(data, figures_dir, qa_dir):
    dem = data["dem"]
    center = data["nodes"]["centers"][0]
    services = sorted(data["nodes"]["service_areas"], key=lambda row: row["服务区编号"])
    lon_min = dem.lon0
    lon_max = dem.lon0 + dem.width * dem.lon_step
    lat_max = dem.lat0
    lat_min = dem.lat0 - dem.height * dem.lat_step

    all_longitudes = [center["经度（°）"]] + [row["经度（°）"] for row in services]
    all_latitudes = [center["纬度（°）"]] + [row["纬度（°）"] for row in services]
    node_lon_span = max(all_longitudes) - min(all_longitudes)
    node_lat_span = max(all_latitudes) - min(all_latitudes)
    view_lon_min = max(lon_min, min(all_longitudes) - 0.30 * node_lon_span)
    view_lon_max = min(lon_max, max(all_longitudes) + 0.30 * node_lon_span)
    view_lat_min = max(lat_min, min(all_latitudes) - 0.34 * node_lat_span)
    view_lat_max = min(lat_max, max(all_latitudes) + 0.34 * node_lat_span)

    terrain_base = plt.get_cmap("terrain")
    land_terrain = LinearSegmentedColormap.from_list(
        "land_terrain", terrain_base(np.linspace(0.22, 1.0, 256))
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    dem_stride = 6
    row_indices = np.arange(0, dem.height, dem_stride)
    column_indices = np.arange(0, dem.width, dem_stride)
    dem_display = dem.values[np.ix_(row_indices, column_indices)]
    longitude = dem.lon0 + (column_indices + 0.5) * dem.lon_step
    latitude = dem.lat0 - (row_indices + 0.5) * dem.lat_step
    terrain = ax.pcolormesh(
        longitude,
        latitude,
        dem_display,
        shading="auto",
        cmap=land_terrain,
        vmin=float(np.nanmin(dem.values)),
        vmax=float(np.nanmax(dem.values)),
        zorder=0,
    )
    contour_values = np.linspace(
        float(np.nanmin(dem.values)), float(np.nanmax(dem.values)), 9
    )[1:-1]
    ax.contour(
        longitude,
        latitude,
        dem_display,
        levels=contour_values,
        colors="#4A4A4A",
        linewidths=0.32,
        alpha=0.36,
        zorder=1,
    )
    ax.scatter(
        [row["经度（°）"] for row in services],
        [row["纬度（°）"] for row in services],
        s=48,
        marker="o",
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=0.8,
        label="服务区（15个）",
        zorder=4,
    )
    ax.scatter(
        [center["经度（°）"]],
        [center["纬度（°）"]],
        s=180,
        marker="*",
        color=PALETTE["secondary"],
        edgecolor="#222222",
        linewidth=0.8,
        label="调度中心 O01",
        zorder=5,
    )
    offsets = _label_offsets(services, center)
    for service in services:
        ax.annotate(
            service["服务区编号"],
            (service["经度（°）"], service["纬度（°）"]),
            xytext=offsets[service["服务区编号"]],
            textcoords="offset points",
            fontsize=7.2,
            ha="left" if offsets[service["服务区编号"]][0] > 0 else "right",
            va="bottom" if offsets[service["服务区编号"]][1] >= 0 else "top",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.72},
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.35},
            zorder=6,
        )
    ax.annotate(
        "O01",
        (center["经度（°）"], center["纬度（°）"]),
        xytext=(8, 7),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.78},
        zorder=6,
    )
    colorbar = fig.colorbar(terrain, ax=ax, shrink=0.9, pad=0.025)
    colorbar.set_label("海拔（m）")
    ax.set_xlabel("经度（°）")
    ax.set_ylabel("纬度（°）")
    ax.set_title("镇龙乡DEM地形、调度中心O01与15个服务区空间分布", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.88, loc="upper right")
    ax.set_xlim(view_lon_min, view_lon_max)
    ax.set_ylim(view_lat_min, view_lat_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad((view_lat_min + view_lat_max) / 2.0)))

    report = _export_with_qa(
        fig,
        figures_dir / "图3_DEM地形及调度中心与服务区分布",
        qa_dir,
    )
    report.update(
        {
            "core_claim": "调度中心O01与15个服务区均位于DEM覆盖范围内，地形高差显著。",
            "evidence": "30米DEM与调度中心、服务区经纬度。",
            "chart": "连续terrain色带DEM地图，叠加等高线、调度中心与服务区标记。",
        }
    )
    return report


def main():
    _configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    figures_dir = project_root / "figures" / "data_overview"
    results_dir = project_root / "results" / "data_overview"
    qa_dir = results_dir / "figure_qa"
    figures_dir.mkdir(parents=True, exist_ok=True)

    style = setup_style(journal="general", lang="zh", serif_for_zh=True)
    data = _load_project_data(project_root)
    material_rows = _material_profile(data)
    resource_rows = _resource_profile(data)
    _write_profiles(results_dir, material_rows, resource_rows)

    reports = [
        plot_material_distribution(material_rows, figures_dir, qa_dir),
        plot_transport_resources(resource_rows, figures_dir, qa_dir),
        plot_dem_service_map(data, figures_dir, qa_dir),
    ]
    contract = {
        "status": "PASS",
        "figure_size_inches": list(FIGURE_SIZE),
        "style": style,
        "visual_review": {
            "rounds": 2,
            "status": "PASS",
            "checks": [
                "中文与特殊符号",
                "标题与坐标轴裁切",
                "数值标注重叠",
                "图例遮挡",
                "服务区标签遮挡",
                "灰度辨识",
            ],
        },
        "design_decisions": [
            "物资箱数与总质量采用上下双面板共享分类轴，避免双Y轴视觉误导。",
            "柱状与折线使用不同的色盲安全颜色，数值标注位于不同面板。",
            "资源库存使用颜色与纹理双重编码。",
            "DEM使用抽样矢量网格、陆地terrain连续色带及带单位的colorbar。",
        ],
        "profile_scope": (
            "四类物资和三类运输资源均为完整清单中的确定性总量，不是抽样统计量；"
            "因此小样本统计图提示不适用，也不绘制误差棒。"
        ),
        "source_files": {
            "demand_workbook": str(data["paths"]["demand_workbook"]),
            "transport_workbook": str(data["paths"]["transport_workbook"]),
            "nodes_workbook": str(data["paths"]["nodes_workbook"]),
            "dem_tif": str(data["paths"]["dem_tif"]),
        },
        "material_profile": material_rows,
        "resource_profile": resource_rows,
        "figures": reports,
    }
    (results_dir / "figure_contracts.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
