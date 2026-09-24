"""Q2 multi-seed ALNS solver, audited output writer, and S2 metadata."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev

from dproblem.common.hashing import sha256_file, stable_json_hash
from dproblem.config import load_model_config
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data
from dproblem.q2.alns import (
    build_initial_routes,
    objective,
    pair_merge_descent,
    run_alns,
)
from dproblem.q2.baseline import _hard_deadline
from dproblem.q2.route import RouteEvaluator
from dproblem.q2.scheduler import schedule_routes


DEFAULT_SEEDS = (20260924, 20260925, 20260926, 20260927, 20260928)


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _constraint_audit(result, data):
    expected = {box["货箱编号"] for box in data["demands"]["boxes"]}
    cargo_ids = [cargo_id for trip in result["trips"] for cargo_id in trip["货箱编号列表"]]
    violations = []
    if set(cargo_ids) != expected or len(cargo_ids) != len(expected):
        violations.append("cargo_coverage")
    if result["hard_violations"]:
        violations.append("hard_deadline")
    for resource_field in ("无人机编号", "电池编号"):
        grouped = {}
        for trip in result["trips"]:
            grouped.setdefault(trip[resource_field], []).append(
                (trip["开始时刻（s）"], trip["返回O01时刻（s）"], trip["架次编号"])
            )
        for resource_id, intervals in grouped.items():
            intervals.sort()
            for left, right in zip(intervals, intervals[1:]):
                if left[1] > right[0] + 1e-8:
                    violations.append("{}_overlap_{}_{}_{}".format(resource_field, resource_id, left[2], right[2]))
    grouped_battery = {}
    for row in result["battery_rows"]:
        grouped_battery.setdefault(row["电池编号"], []).append(row)
    for battery_id, rows in grouped_battery.items():
        rows.sort(key=lambda row: row["任务开始时刻（s）"])
        for left, right in zip(rows, rows[1:]):
            if left["再次可用时刻（s）"] > right["任务开始时刻（s）"] + 1e-8:
                violations.append("battery_charge_overlap_{}".format(battery_id))
    if violations:
        raise AssertionError("Q2 audit failed: {}".format(";".join(violations)))
    return {
        "status": "PASS",
        "violations": 0,
        "cargo_boxes": len(cargo_ids),
        "trips": len(result["trips"]),
        "drone_resources_used": len({trip["无人机编号"] for trip in result["trips"]}),
        "battery_resources_used": len({trip["电池编号"] for trip in result["trips"]}),
    }


def _pareto_runs(runs):
    kept = []
    for candidate in runs:
        candidate_objective = objective(candidate["best"])
        dominated = False
        for other in runs:
            if other is candidate:
                continue
            other_objective = objective(other["best"])
            if all(left <= right for left, right in zip(other_objective, candidate_objective)) and any(
                left < right for left, right in zip(other_objective, candidate_objective)
            ):
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return sorted(kept, key=lambda run: objective(run["best"]))


def solve_q2(project_root, seeds=DEFAULT_SEEDS, iterations=500):
    project_root = Path(project_root).resolve()
    data = load_project_data(project_root)
    config = load_model_config(project_root)
    evaluator = RouteEvaluator(
        RasterDEM(data["paths"]["dem_tif"]),
        data["nodes"]["centers"][0],
        data["nodes"]["service_areas"],
        config["energy_model"],
    )
    initial_routes = build_initial_routes(project_root, data)
    initial = schedule_routes(initial_routes, data, config, evaluator, preserve_order=True)
    if initial["status"] != "PASS":
        raise RuntimeError("Q2 repaired initial schedule is infeasible: {}".format(initial))

    runs = []
    for seed in seeds:
        run = run_alns(initial_routes, data, config, evaluator, seed, iterations=iterations)
        merged_routes, merged_result, merge_log = pair_merge_descent(
            run["best_routes"], run["best"], data, config, evaluator
        )
        run["best_routes"] = merged_routes
        run["best"] = merged_result
        run["pair_merge_descent"] = merge_log
        runs.append(run)
    selected = min(runs, key=lambda run: objective(run["best"]))
    pareto_runs = _pareto_runs(runs)
    best = selected["best"]
    audit = _constraint_audit(best, data)

    baseline_summary = json.loads(
        (project_root / "results" / "q2_baseline" / "q2_baseline_summary.json").read_text(
            encoding="utf-8"
        )
    )
    input_manifest = json.loads(
        (project_root / "results" / "00_audit" / "input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metric_names = ("timeliness_loss", "makespan", "energy", "trip_count")
    stability = {}
    for index, name in enumerate(metric_names):
        values = [objective(run["best"])[index] for run in runs]
        stability[name] = {
            "best": min(values),
            "median": median(values),
            "mean": mean(values),
            "population_std": pstdev(values),
            "values": values,
        }
    summary = {
        "status": "PASS",
        "solution_version": "S2-v1.0.0-provisional_v0",
        "parent_version": "S1-v1.0.0-provisional_v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "model_config_sha256": stable_json_hash(config),
        "parent_s1_freeze_sha256": sha256_file(project_root / "results" / "q1" / "S1_FREEZE.json"),
        "energy_model_version": config["energy_model"]["version"],
        "transport_timeline_version": config["transport_timeline"]["version"],
        "solver": "urgency_repair_plus_multiseed_alns_plus_exact_pair_merge_descent",
        "seeds": list(seeds),
        "iterations_per_seed": iterations,
        "selected_seed": selected["seed"],
        "pareto_seed_count": len(pareto_runs),
        "pareto_seeds": [run["seed"] for run in pareto_runs],
        "initial_metrics": {
            "timeliness_loss_weighted_seconds": initial["timeliness_loss"],
            "transport_makespan_seconds": initial["makespan"],
            "total_energy_kwh": initial["energy"],
            "trip_count": initial["trip_count"],
        },
        "main_metrics": {
            "timeliness_loss_weighted_seconds": best["timeliness_loss"],
            "transport_makespan_seconds": best["makespan"],
            "total_energy_kwh": best["energy"],
            "trip_count": best["trip_count"],
            "multi_stop_trip_count": sum(
                len(trip["访问服务区顺序"]) > 1 for trip in best["trips"]
            ),
        },
        "baseline_status": baseline_summary["status"],
        "baseline_hard_deadline_violations": baseline_summary["hard_deadline_violations"],
        "constraint_audit": audit,
        "seed_stability": stability,
        "selected_operator_counts": selected["operator_counts"],
        "selected_operator_accepted": selected["operator_accepted"],
        "selected_operator_improvements": selected["operator_improvements"],
        "selected_pair_merge_descent": selected["pair_merge_descent"],
    }
    return {
        "data": data,
        "config": config,
        "initial": initial,
        "runs": runs,
        "selected": selected,
        "pareto_runs": pareto_runs,
        "best": best,
        "summary": summary,
    }


def write_q2_results(project_root, output_dir="results/q2", seeds=DEFAULT_SEEDS, iterations=500):
    project_root = Path(project_root).resolve()
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    solution = solve_q2(project_root, seeds=seeds, iterations=iterations)
    best = solution["best"]
    boxes_by_id = {
        box["货箱编号"]: box for box in solution["data"]["demands"]["boxes"]
    }
    trip_rows = []
    segment_rows = []
    timeline_rows = []
    box_rows = []
    for trip in best["trips"]:
        trip_rows.append(
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
        )
        for index, segment in enumerate(trip["航段"], start=1):
            row = {"架次编号": trip["架次编号"], "航段序号": index}
            row.update(segment)
            segment_rows.append(row)
        for index, event in enumerate(trip["时间线"], start=1):
            timeline_rows.append(
                {
                    "架次编号": trip["架次编号"],
                    "无人机编号": trip["无人机编号"],
                    "电池编号": trip["电池编号"],
                    "事件序号": index,
                    "阶段": event["阶段"],
                    "开始时刻（s）": event["开始时刻（s）"],
                    "结束时刻（s）": event["结束时刻（s）"],
                    "起点": event.get("起点", ""),
                    "终点": event.get("终点", ""),
                    "服务区编号": event.get("服务区编号", ""),
                }
            )
        for cargo_id in trip["货箱编号列表"]:
            box = boxes_by_id[cargo_id]
            deadline = _hard_deadline(box)
            box_rows.append(
                {
                    "货箱编号": cargo_id,
                    "架次编号": trip["架次编号"],
                    "服务区编号": box["服务区编号"],
                    "物资类型": box["物资类型"],
                    "是否首批保障": box["是否首批保障"],
                    "交付完成时刻（s）": trip["逐箱交付时刻"][cargo_id],
                    "期望送达时间（s）": box["期望送达时间（s）"],
                    "硬截止时间（s）": "" if deadline is None else deadline,
                    "硬截止违例": False,
                }
            )
    box_rows.sort(key=lambda row: row["货箱编号"])
    service_rows = []
    for service_id in sorted({row["服务区编号"] for row in box_rows}):
        selected = [row for row in box_rows if row["服务区编号"] == service_id]
        service_rows.append(
            {
                "服务区编号": service_id,
                "首箱交付时刻（s）": min(row["交付完成时刻（s）"] for row in selected),
                "末箱交付时刻（s）": max(row["交付完成时刻（s）"] for row in selected),
                "货箱数": len(selected),
                "硬截止违例数": sum(bool(row["硬截止违例"]) for row in selected),
            }
        )
    seed_rows = [
        {
            "随机种子": run["seed"],
            "迭代次数": run["iterations"],
            "及时性损失（加权s）": run["best"]["timeliness_loss"],
            "运输完成时间（s）": run["best"]["makespan"],
            "总能耗（kWh）": run["best"]["energy"],
            "架次数": run["best"]["trip_count"],
            "多点架次数": sum(len(trip["访问服务区顺序"]) > 1 for trip in run["best"]["trips"]),
            "是否选用": run is solution["selected"],
        }
        for run in solution["runs"]
    ]
    history_rows = [
        dict({"随机种子": run["seed"]}, **row)
        for run in solution["runs"]
        for row in run["history"]
    ]
    pareto_rows = [
        {
            "随机种子": run["seed"],
            "及时性损失（加权s）": run["best"]["timeliness_loss"],
            "运输完成时间（s）": run["best"]["makespan"],
            "总能耗（kWh）": run["best"]["energy"],
            "架次数": run["best"]["trip_count"],
            "多点架次数": sum(len(trip["访问服务区顺序"]) > 1 for trip in run["best"]["trips"]),
            "分层主方案": run is solution["selected"],
        }
        for run in solution["pareto_runs"]
    ]
    _write_csv(output_dir / "main_trips.csv", trip_rows)
    _write_csv(output_dir / "main_box_deliveries.csv", box_rows)
    _write_csv(output_dir / "main_segments.csv", segment_rows)
    _write_csv(output_dir / "main_timeline.csv", timeline_rows)
    _write_csv(output_dir / "main_battery_timeline.csv", best["battery_rows"])
    _write_csv(output_dir / "service_delivery_summary.csv", service_rows)
    _write_csv(output_dir / "seed_stability.csv", seed_rows)
    _write_csv(output_dir / "pareto_solutions.csv", pareto_rows)
    _write_csv(output_dir / "alns_convergence.csv", history_rows)
    (output_dir / "q2_summary.json").write_text(
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
    for row_index, row in enumerate(box_rows, start=2):
        for column_index, header in enumerate(headers, start=1):
            sheet.cell(row=row_index, column=column_index, value=row[header])
    workbook.save(output_dir / "结果提交_Q2.xlsx")
    workbook.close()
    return solution["summary"]
