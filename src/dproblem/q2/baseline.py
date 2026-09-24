"""Q2 EDF baseline with explicit drone and battery timelines."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dproblem.common.hashing import sha256_file, stable_json_hash
from dproblem.config import load_model_config
from dproblem.domain.charging import charge_time_to_full_seconds
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data
from dproblem.q2.route import RouteEvaluator


EPSILON = 1e-8


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hard_deadline(box):
    deadlines = []
    if box["物资类型"] == "医疗物资":
        deadlines.append(float(box["期望送达时间（s）"]))
    if box["是否首批保障"] == "是":
        deadlines.append(float(box["首批截止时间（s）"]))
    return min(deadlines) if deadlines else None


def _task_sort_key(task):
    hard = [deadline for deadline in (_hard_deadline(box) for box in task["boxes"]) if deadline is not None]
    return (
        min(hard) if hard else float("inf"),
        min(float(box["期望送达时间（s）"]) for box in task["boxes"]),
        -max(float(box["应急优先系数"]) for box in task["boxes"]),
        task["service_id"],
        tuple(box["货箱编号"] for box in task["boxes"]),
    )


def build_q1_tasks(project_root, boxes):
    q1_rows = _read_csv(Path(project_root) / "results" / "q1" / "main_trips.csv")
    boxes_by_id = {box["货箱编号"]: box for box in boxes}
    tasks = []
    for row in q1_rows:
        cargo_ids = row["货箱编号列表"].split(";")
        task_boxes = [boxes_by_id[cargo_id] for cargo_id in cargo_ids]
        tasks.append(
            {
                "parent_q1_trip_id": row["架次编号"],
                "service_id": row["服务区编号"],
                "type_id": row["机型编号"],
                "boxes": task_boxes,
            }
        )
    return sorted(tasks, key=_task_sort_key)


def schedule_q2_baseline(project_root):
    project_root = Path(project_root).resolve()
    data = load_project_data(project_root)
    config = load_model_config(project_root)
    evaluator = RouteEvaluator(
        RasterDEM(data["paths"]["dem_tif"]),
        data["nodes"]["centers"][0],
        data["nodes"]["service_areas"],
        config["energy_model"],
    )
    types = {row["机型编号"]: row for row in data["transport"]["types"]}
    drones = {
        row["无人机编号"]: {"type_id": row["机型编号"], "available": 0.0}
        for row in data["transport"]["drones"]
    }
    batteries = {}
    charge_full = {}
    for row in data["transport"]["batteries"]:
        type_id = row["机型编号"]
        charge_full[type_id] = float(row["等效完全充电时间（s）"])
        for index in range(1, int(row["共享电池组总数（组）"]) + 1):
            battery_id = "{}-BAT-{:02d}".format(type_id, index)
            batteries[battery_id] = {"type_id": type_id, "available": 0.0}

    tasks = build_q1_tasks(project_root, data["demands"]["boxes"])
    scheduled = []
    battery_rows = []
    timeline_rows = []
    for sequence, task in enumerate(tasks, start=1):
        type_id = task["type_id"]
        compatible_drones = sorted(
            drone_id for drone_id, state in drones.items() if state["type_id"] == type_id
        )
        compatible_batteries = sorted(
            battery_id for battery_id, state in batteries.items() if state["type_id"] == type_id
        )
        choices = []
        for drone_id in compatible_drones:
            for battery_id in compatible_batteries:
                choices.append(
                    (
                        max(drones[drone_id]["available"], batteries[battery_id]["available"]),
                        drone_id,
                        battery_id,
                    )
                )
        start, drone_id, battery_id = min(choices)
        route = evaluator.evaluate(
            task["boxes"], [task["service_id"]], types[type_id], start_seconds=start
        )
        if route is None:
            raise RuntimeError("Q1 batch became physically infeasible in Q2: {}".format(task))
        trip_id = "Q2B-{:03d}".format(sequence)
        route.update(
            {
                "架次编号": trip_id,
                "父Q1架次编号": task["parent_q1_trip_id"],
                "无人机编号": drone_id,
                "电池编号": battery_id,
                "货箱编号列表": [box["货箱编号"] for box in task["boxes"]],
            }
        )
        scheduled.append(route)
        drones[drone_id]["available"] = (
            route["返回O01时刻（s）"]
            + config["transport_timeline"]["transport_turnaround_seconds"]
        )
        soc_after = route["返航SOC（%）"] / 100.0
        charge_seconds = charge_time_to_full_seconds(soc_after, charge_full[type_id])
        charge_start = route["返回O01时刻（s）"]
        charge_end = charge_start + charge_seconds
        batteries[battery_id]["available"] = charge_end
        battery_rows.append(
            {
                "电池编号": battery_id,
                "机型编号": type_id,
                "架次编号": trip_id,
                "任务开始时刻（s）": start,
                "任务结束时刻（s）": charge_start,
                "任务后SOC（%）": route["返航SOC（%）"],
                "充电开始时刻（s）": charge_start,
                "充电结束时刻（s）": charge_end,
                "再次可用时刻（s）": charge_end,
            }
        )
        for event_index, event in enumerate(route["时间线"], start=1):
            timeline_rows.append(
                {
                    "架次编号": trip_id,
                    "无人机编号": drone_id,
                    "电池编号": battery_id,
                    "事件序号": event_index,
                    "阶段": event["阶段"],
                    "开始时刻（s）": event["开始时刻（s）"],
                    "结束时刻（s）": event["结束时刻（s）"],
                    "起点": event.get("起点", ""),
                    "终点": event.get("终点", ""),
                    "服务区编号": event.get("服务区编号", ""),
                }
            )

    box_rows = []
    hard_violations = []
    timeliness_loss = 0.0
    boxes_by_id = {box["货箱编号"]: box for box in data["demands"]["boxes"]}
    for trip in scheduled:
        for cargo_id, delivery in trip["逐箱交付时刻"].items():
            box = boxes_by_id[cargo_id]
            deadline = _hard_deadline(box)
            violation = deadline is not None and delivery > deadline + EPSILON
            if violation:
                hard_violations.append(
                    {"货箱编号": cargo_id, "交付时刻（s）": delivery, "硬截止（s）": deadline}
                )
            if deadline is None:
                timeliness_loss += float(box["应急优先系数"]) * max(
                    0.0, delivery - float(box["期望送达时间（s）"])
                )
            box_rows.append(
                {
                    "货箱编号": cargo_id,
                    "架次编号": trip["架次编号"],
                    "服务区编号": box["服务区编号"],
                    "物资类型": box["物资类型"],
                    "是否首批保障": box["是否首批保障"],
                    "交付完成时刻（s）": delivery,
                    "期望送达时间（s）": box["期望送达时间（s）"],
                    "硬截止时间（s）": "" if deadline is None else deadline,
                    "硬截止违例": violation,
                }
            )

    if len(box_rows) != len(boxes_by_id) or len({row["货箱编号"] for row in box_rows}) != len(boxes_by_id):
        raise AssertionError("Q2 baseline cargo coverage is not one-to-one")
    return {
        "data": data,
        "config": config,
        "trips": scheduled,
        "boxes": sorted(box_rows, key=lambda row: row["货箱编号"]),
        "battery_rows": battery_rows,
        "timeline_rows": timeline_rows,
        "summary": {
            "status": "PASS" if not hard_violations else "FAIL",
            "solution_version": "S2-baseline-v1.0.0-provisional_v0",
            "parent_version": "S1-v1.0.0-provisional_v0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "deterministic": True,
            "random_seed": None,
            "input_manifest_sha256": json.loads(
                (project_root / "results" / "00_audit" / "input_manifest.json").read_text(encoding="utf-8")
            )["manifest_sha256"],
            "model_config_sha256": stable_json_hash(config),
            "parent_q1_summary_sha256": sha256_file(project_root / "results" / "q1" / "q1_summary.json"),
            "solver": "q1_batches_single_service_edf_greedy_drone_battery_schedule",
            "trip_count": len(scheduled),
            "total_energy_kwh": sum(trip["架次能耗（kWh）"] for trip in scheduled),
            "transport_makespan_seconds": max(trip["返回O01时刻（s）"] for trip in scheduled),
            "timeliness_loss_weighted_seconds": timeliness_loss,
            "hard_deadline_violations": hard_violations,
            "cargo_boxes": len(box_rows),
            "transport_turnaround_seconds": config["transport_timeline"]["transport_turnaround_seconds"],
        },
    }


def write_q2_baseline_results(project_root, output_dir="results/q2_baseline"):
    project_root = Path(project_root).resolve()
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    solution = schedule_q2_baseline(project_root)
    trips = solution["trips"]
    trip_rows = [
        {
            "架次编号": trip["架次编号"],
            "无人机编号": trip["无人机编号"],
            "机型编号": trip["机型编号"],
            "电池编号": trip["电池编号"],
            "开始时刻（s）": trip["开始时刻（s）"],
            "访问服务区顺序": ";".join(trip["访问服务区顺序"]),
            "货箱编号列表": ";".join(trip["货箱编号列表"]),
            "返回O01时刻（s）": trip["返回O01时刻（s）"],
            "架次能耗（kWh）": trip["架次能耗（kWh）"],
            "返航SOC（%）": trip["返航SOC（%）"],
        }
        for trip in trips
    ]
    segment_rows = []
    for trip in trips:
        for index, segment in enumerate(trip["航段"], start=1):
            row = {"架次编号": trip["架次编号"], "航段序号": index}
            row.update(segment)
            segment_rows.append(row)
    _write_csv(output_dir / "baseline_trips.csv", trip_rows)
    _write_csv(output_dir / "baseline_box_deliveries.csv", solution["boxes"])
    _write_csv(output_dir / "baseline_segments.csv", segment_rows)
    _write_csv(output_dir / "baseline_timeline.csv", solution["timeline_rows"])
    _write_csv(output_dir / "baseline_battery_timeline.csv", solution["battery_rows"])
    (output_dir / "q2_baseline_summary.json").write_text(
        json.dumps(solution["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from dproblem.compat import import_openpyxl_load_workbook

    workbook = import_openpyxl_load_workbook()(solution["data"]["paths"]["submission_template"])
    sheet = workbook["Q2_运输架次"]
    headers = [cell.value for cell in sheet[1] if cell.value is not None]
    for row_index, row in enumerate(trip_rows, start=2):
        for column_index, header in enumerate(headers, start=1):
            sheet.cell(row=row_index, column=column_index, value=row[header])
    sheet = workbook["Q2_逐箱交付"]
    headers = [cell.value for cell in sheet[1] if cell.value is not None]
    for row_index, row in enumerate(solution["boxes"], start=2):
        for column_index, header in enumerate(headers, start=1):
            sheet.cell(row=row_index, column=column_index, value=row[header])
    workbook.save(output_dir / "结果提交_Q2_baseline.xlsx")
    workbook.close()
    return solution["summary"]

