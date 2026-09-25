import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dproblem.common.hashing import sha256_file, stable_json_hash
from dproblem.config import load_model_config
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data


CORE_MANIFEST_KEYS = [
    "problem_document",
    "submission_template",
    "development_spec",
    "nodes_workbook",
    "demand_workbook",
    "transport_workbook",
    "relay_workbook",
    "communication_workbook",
    "dem_tif",
    "dem_mat",
    "geo_description",
    "geo_map",
]


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError("{}: expected {!r}, got {!r}".format(label, expected, actual))


def _assert_close(actual, expected, label, tolerance=1e-9):
    if abs(actual - expected) > tolerance:
        raise AssertionError("{}: expected {}, got {}".format(label, expected, actual))


def _assert_finite(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError("{} must be finite, got {!r}".format(label, value))


def _assert_positive(value, label, allow_zero=False):
    _assert_finite(value, label)
    if value < 0 or (value == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise AssertionError("{} must be {}, got {!r}".format(label, relation, value))


def _relative(path, project_root):
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def build_input_manifest(project_root, paths):
    files = []
    for key in CORE_MANIFEST_KEYS:
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "key": key,
                "path": _relative(path, project_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def inspect_dem(path):
    with Image.open(path) as image:
        values = np.asarray(image)
        tags = image.tag_v2
        nodata_raw = tags.get(42113)
        nodata = float(nodata_raw) if nodata_raw is not None else None
        valid = values if nodata is None else values[values != nodata]
        geo_keys = tuple(tags.get(34735, ()))
        epsg = 4326 if 4326 in geo_keys else None
        return {
            "width": image.width,
            "height": image.height,
            "dtype": str(values.dtype),
            "minimum_m": float(valid.min()),
            "maximum_m": float(valid.max()),
            "nodata_value": nodata,
            "nodata_pixel_count": int((values == nodata).sum()) if nodata is not None else 0,
            "pixel_scale": list(tags.get(33550, ())),
            "tiepoint": list(tags.get(33922, ())),
            "epsg": epsg,
        }


def _validate_demand_consistency(summary, boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[(box["服务区编号"], box["物资类型"])].append(box)
    _assert_equal(len(grouped), len(summary), "需求汇总组合数")
    for row in summary:
        key = (row["服务区编号"], row["物资类型"])
        rows = grouped[key]
        _assert_equal(len(rows), row["总需求箱数"], "{} 总箱数".format(key))
        _assert_equal(
            sum(box["是否首批保障"] == "是" for box in rows),
            row["首批必须送达箱数"],
            "{} 首批箱数".format(key),
        )
        for box in rows:
            _assert_close(box["单箱质量（kg）"], row["单箱质量（kg）"], "{} 单箱质量".format(key))
            _assert_close(box["单箱体积（m³）"], row["单箱体积（m³）"], "{} 单箱体积".format(key))
            _assert_equal(box["应急优先系数"], row["应急优先系数"], "{} 优先系数".format(key))
            _assert_equal(box["期望送达时间（s）"], row["期望送达时间（s）"], "{} 期望送达时间".format(key))
            if box["是否首批保障"] == "是":
                _assert_equal(box["首批截止时间（s）"], row["首批截止时间（s）"], "{} 首批截止时间".format(key))
            else:
                _assert_equal(box["首批截止时间（s）"], None, "{} 非首批截止时间应为空".format(key))


def _validate_physical_parameters(data, model_config):
    for row in data["nodes"]["centers"] + data["nodes"]["service_areas"]:
        for field in ("经度（°）", "纬度（°）", "海拔（m）"):
            _assert_finite(row[field], "{} {}".format(row.get("调度中心编号", row.get("服务区编号")), field))

    positive_transport_fields = (
        "含电池空载总质量（kg）",
        "最大载货质量（kg）",
        "可用装载体积（m³）",
        "计划巡航速度（m/s）",
        "空载标准航程（m）",
        "满载标准航程（m）",
        "电池可用能量（kWh）",
        "最大爬升速度（m/s）",
        "最大下降速度（m/s）",
        "爬升能耗效率",
    )
    nonnegative_transport_fields = (
        "工位固定准备时间（s）",
        "每箱装载时间（s）",
        "接收点基础交接时间（s）",
        "每箱增加交接时间（s）",
        "下降能耗效率",
    )
    for row in data["transport"]["types"]:
        type_id = row["机型编号"]
        for field in positive_transport_fields:
            _assert_positive(row[field], "运输机型 {} {}".format(type_id, field))
        for field in nonnegative_transport_fields:
            _assert_positive(row[field], "运输机型 {} {}".format(type_id, field), allow_zero=True)
        reserve = row["返航电量下限（%）"]
        _assert_finite(reserve, "运输机型 {} 返航电量下限".format(type_id))
        if not 0 <= reserve < 100:
            raise AssertionError("运输机型 {} 返航电量下限超界".format(type_id))
        if row["满载标准航程（m）"] > row["空载标准航程（m）"]:
            raise AssertionError("运输机型 {} 满载航程大于空载航程".format(type_id))
        if not 0 < row["爬升能耗效率"] <= 1:
            raise AssertionError("运输机型 {} 爬升效率超界".format(type_id))

    # The source workbook displays column E with one decimal place, but the
    # stored values are more precise.  These assertions prevent a future
    # manual transcription of the displayed 0.1/0.1/0.3 values from silently
    # replacing the authoritative 0.060/0.073/0.250 m^3 values.
    expected_usable_volumes = {"A": 0.060, "B": 0.073, "C": 0.250}
    actual_usable_volumes = {
        row["机型编号"]: row["可用装载体积（m³）"]
        for row in data["transport"]["types"]
    }
    _assert_equal(set(actual_usable_volumes), set(expected_usable_volumes), "运输机型体积参数集合")
    for type_id, expected in expected_usable_volumes.items():
        _assert_close(
            actual_usable_volumes[type_id],
            expected,
            "运输机型 {} 精确可用装载体积".format(type_id),
            tolerance=1e-12,
        )

    type_ids = {row["机型编号"] for row in data["transport"]["types"]}
    for row in data["transport"]["drones"]:
        if row["机型编号"] not in type_ids:
            raise AssertionError("运输无人机引用未知机型")
    _assert_equal({row["机型编号"] for row in data["transport"]["batteries"]}, type_ids, "电池机型集合")

    positive_relay_fields = (
        "含能源组件空载总质量（kg）",
        "中继通信模块质量（kg）",
        "计划起飞总质量（kg）",
        "计划巡航速度（m/s）",
        "巡航功率（kW）",
        "能源组件可用能量（kWh）",
        "最大爬升速度（m/s）",
        "最大下降速度（m/s）",
        "爬升能耗效率",
        "悬停功率（kW）",
        "通信附加功率（kW）",
        "最大悬停离地高度（m）",
    )
    for row in data["relay"]["types"]:
        for field in positive_relay_fields:
            _assert_positive(row[field], "中继机型 {}".format(field))
        if not 0 < row["爬升能耗效率"] <= 1:
            raise AssertionError("中继机型爬升效率超界")
        reserve = row["返航电量下限（%）"]
        _assert_finite(reserve, "中继机型返航电量下限")
        if not 0 <= reserve < 100:
            raise AssertionError("中继机型返航电量下限超界")

    for row in data["demands"]["boxes"]:
        _assert_positive(row["单箱质量（kg）"], "{} 质量".format(row["货箱编号"]))
        _assert_positive(row["单箱体积（m³）"], "{} 体积".format(row["货箱编号"]))
        _assert_positive(row["期望送达时间（s）"], "{} 期望送达时间".format(row["货箱编号"]))
        if row["首批截止时间（s）"] is not None:
            _assert_positive(row["首批截止时间（s）"], "{} 首批截止时间".format(row["货箱编号"]))

    for row in data["communication"]:
        _assert_finite(row["参数值"], "通信参数 {} {}".format(row["参数类别"], row["参数名称"]))
    frequency = next(row["参数值"] for row in data["communication"] if row["参数名称"] == "载波频率（MHz）")
    _assert_positive(frequency, "载波频率")

    energy_config = model_config["energy_model"]
    _assert_positive(energy_config["gravity_m_s2"], "配置重力加速度")
    _assert_positive(energy_config["equivalent_range_exponent"], "配置等效航程指数")


def build_data_audit(project_root):
    data = load_project_data(project_root)
    nodes = data["nodes"]
    demands = data["demands"]
    transport = data["transport"]
    relay = data["relay"]

    _assert_equal(len(nodes["centers"]), 1, "调度中心数量")
    _assert_equal(len(nodes["service_areas"]), 15, "服务区数量")
    _assert_equal(len(demands["boxes"]), 80, "货箱数量")
    _assert_equal(len({row["货箱编号"] for row in demands["boxes"]}), 80, "唯一货箱编号数量")
    _validate_demand_consistency(demands["summary"], demands["boxes"])

    service_ids = {row["服务区编号"] for row in nodes["service_areas"]}
    box_service_ids = {row["服务区编号"] for row in demands["boxes"]}
    _assert_equal(box_service_ids, service_ids, "货箱与服务区编号集合")

    total_mass = sum(row["单箱质量（kg）"] for row in demands["boxes"])
    total_volume = sum(row["单箱体积（m³）"] for row in demands["boxes"])
    first_batch = sum(row["是否首批保障"] == "是" for row in demands["boxes"])
    type_counts = Counter(row["物资类型"] for row in demands["boxes"])
    _assert_close(total_mass, 758.0, "总质量")
    _assert_close(total_volume, 2.011, "总体积", tolerance=1e-12)
    _assert_equal(first_batch, 30, "首批货箱数")
    _assert_equal(
        dict(type_counts),
        {"医疗物资": 16, "饮用水": 36, "应急食品": 19, "生活卫生用品": 9},
        "物资类型箱数",
    )

    drone_type_counts = Counter(row["机型编号"] for row in transport["drones"])
    battery_counts = {row["机型编号"]: row["共享电池组总数（组）"] for row in transport["batteries"]}
    _assert_equal(dict(drone_type_counts), {"A": 4, "B": 2, "C": 2}, "运输无人机数量")
    _assert_equal(battery_counts, {"A": 6, "B": 4, "C": 4}, "共享电池数量")
    _assert_equal(len(relay["drones"]), 2, "中继无人机数量")
    _assert_equal(relay["modules"][0]["共享能源组件总数（组）"], 6, "中继能源组件数量")
    _assert_equal(len(data["communication"]), 14, "通信参数数量")

    model_config = load_model_config(project_root)
    _validate_physical_parameters(data, model_config)

    dem = inspect_dem(data["paths"]["dem_tif"])
    _assert_equal((dem["width"], dem["height"]), (1486, 1309), "DEM 尺寸")
    _assert_equal(dem["epsg"], 4326, "DEM EPSG")
    _assert_close(dem["minimum_m"], 41.6891, "DEM 最小高程", tolerance=1e-4)
    _assert_close(dem["maximum_m"], 1132.8561, "DEM 最大高程", tolerance=1e-4)

    dem_grid = RasterDEM(data["paths"]["dem_tif"])
    for row in nodes["centers"] + nodes["service_areas"]:
        node_id = row.get("调度中心编号", row.get("服务区编号"))
        if not dem_grid.contains(row["经度（°）"], row["纬度（°）"]):
            raise AssertionError("节点 {} 位于 DEM 范围外".format(node_id))

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "counts": {
            "dispatch_centers": 1,
            "service_areas": len(nodes["service_areas"]),
            "demand_summary_rows": len(demands["summary"]),
            "cargo_boxes": len(demands["boxes"]),
            "first_batch_boxes": first_batch,
            "transport_drones": len(transport["drones"]),
            "relay_drones": len(relay["drones"]),
            "communication_parameters": len(data["communication"]),
        },
        "cargo": {
            "total_mass_kg": total_mass,
            "total_volume_m3": total_volume,
            "counts_by_material": dict(type_counts),
        },
        "transport_resources": {
            "drones_by_type": dict(drone_type_counts),
            "batteries_by_type": battery_counts,
            "usable_volume_m3_by_type": {
                row["机型编号"]: row["可用装载体积（m³）"]
                for row in transport["types"]
            },
        },
        "relay_resources": {
            "drones": len(relay["drones"]),
            "energy_modules": relay["modules"][0]["共享能源组件总数（组）"],
        },
        "dem": dem,
        "model_contract": {
            "energy_model_version": model_config["energy_model"]["version"],
            "energy_config_sha256": stable_json_hash(model_config["energy_model"]),
            "q1_work_time_version": model_config["q1_cumulative_work_time"]["version"],
            "q1_work_time_status": model_config["q1_cumulative_work_time"]["status"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def write_audit(project_root, output_dir):
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_project_data(project_root)
    manifest = build_input_manifest(project_root, data["paths"])
    audit = build_data_audit(project_root)
    manifest["manifest_sha256"] = stable_json_hash(manifest["files"])

    manifest_path = output_dir / "input_manifest.json"
    audit_path = output_dir / "data_audit.json"
    summary_path = output_dir / "数据审计摘要.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                "# P0 数据审计摘要",
                "",
                "- 状态：PASS",
                "- 服务区：{} 个".format(audit["counts"]["service_areas"]),
                "- 货箱：{} 箱，其中首批 {} 箱".format(audit["counts"]["cargo_boxes"], audit["counts"]["first_batch_boxes"]),
                "- 总质量：{:.3f} kg".format(audit["cargo"]["total_mass_kg"]),
                "- 总体积：{:.3f} m³".format(audit["cargo"]["total_volume_m3"]),
                "- 运输无人机：{} 架；中继无人机：{} 架".format(audit["counts"]["transport_drones"], audit["counts"]["relay_drones"]),
                "- 运输机型精确可用装载体积：A={:.3f}、B={:.3f}、C={:.3f} m³（读取单元格底层值）".format(
                    audit["transport_resources"]["usable_volume_m3_by_type"]["A"],
                    audit["transport_resources"]["usable_volume_m3_by_type"]["B"],
                    audit["transport_resources"]["usable_volume_m3_by_type"]["C"],
                ),
                "- DEM：{}×{}，EPSG:{}，高程 {:.4f}—{:.4f} m".format(audit["dem"]["width"], audit["dem"]["height"], audit["dem"]["epsg"], audit["dem"]["minimum_m"], audit["dem"]["maximum_m"]),
                "- 能耗口径：`{}`（答题者暂定）".format(audit["model_contract"]["energy_model_version"]),
                "- Q1 累计作业时间：`{}`（答题者定义；非并行 makespan）".format(audit["model_contract"]["q1_work_time_version"]),
                "",
                "> 原始输入仅被读取，未被改名、移动或覆盖。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, audit_path, summary_path
