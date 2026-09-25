"""Build and audit the communication-feasible Q3 joint schedule."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from dproblem.domain.charging import charge_time_to_full_seconds
from dproblem.domain.communication import CommunicationModel, audit_link
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data
from dproblem.q3.direct import verify_s2_freeze
from dproblem.q3.relay import relay_sortie_metrics, relay_travel_profile


EPSILON = 1e-7


# These shifts are relative to frozen S2.  They are the result of the Q3
# time-space feasibility search with official O01-return relay sorties.
TRIP_SHIFTS_SECONDS = {
    "Q2-005": 980.0,
    "Q2-006": 5474.388058583308,
    "Q2-008": 400.0,
    "Q2-009": 2202.629303887711,
    "Q2-011": 5900.0,
    "Q2-013": 800.0,
    "Q2-014": 980.0,
    "Q2-015": 700.0,
    "Q2-016": -3772.056700388106,
    "Q2-017": 2262.629303887711,
    "Q2-018": 980.0,
    "Q2-019": 3179.887279322106,
    "Q2-020": -1190.8995031364677,
    "Q2-021": -1190.8995031364677,
    "Q2-023": 700.0,
}

CARGO_SWAP = {
    "S002-WAT-01": "Q2-007",
    "S002-WAT-03": "Q2-006",
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _shift(trip_id):
    return float(TRIP_SHIFTS_SECONDS.get(trip_id, 0.0))


def _float_fields(row, fields):
    for field in fields:
        row[field] = float(row[field])
    return row


def _group_by(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return grouped


def _assign_transport_resources(trips, charge_full):
    """Reassign the shifted tasks using only the frozen S2 inventory."""

    inventory = defaultdict(lambda: {"drones": set(), "batteries": set()})
    for row in trips:
        inventory[row["机型编号"]]["drones"].add(row["无人机编号"])
        inventory[row["机型编号"]]["batteries"].add(row["电池编号"])

    violations = []
    for type_id, typed_trips in _group_by(trips, "机型编号").items():
        drone_available = {
            item: 0.0 for item in sorted(inventory[type_id]["drones"])
        }
        battery_available = {
            item: 0.0 for item in sorted(inventory[type_id]["batteries"])
        }
        for trip in sorted(
            typed_trips, key=lambda row: (row["开始时刻（s）"], row["架次编号"])
        ):
            start = trip["开始时刻（s）"]
            drone_id = next(
                (item for item, available in drone_available.items() if available <= start + EPSILON),
                None,
            )
            battery_id = next(
                (item for item, available in battery_available.items() if available <= start + EPSILON),
                None,
            )
            if drone_id is None:
                violations.append("drone_inventory_{}".format(trip["架次编号"]))
            else:
                trip["无人机编号"] = drone_id
                drone_available[drone_id] = trip["返回O01时刻（s）"]
            if battery_id is None:
                violations.append("battery_inventory_{}".format(trip["架次编号"]))
            else:
                trip["电池编号"] = battery_id
                battery_available[battery_id] = trip["返回O01时刻（s）"] + charge_time_to_full_seconds(
                    trip["返航SOC（%）"] / 100.0, charge_full[type_id]
                )
    return violations


def build_adjusted_transport(project_root):
    """Apply the audited local timing/resource edits to frozen S2."""

    project_root = Path(project_root).resolve()
    verify_s2_freeze(project_root)
    q2_dir = project_root / "results" / "q2"
    data = load_project_data(project_root)

    trips = read_csv(q2_dir / "main_trips.csv")
    for row in trips:
        trip_id = row["架次编号"]
        shift = _shift(trip_id)
        _float_fields(
            row,
            ("开始时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）", "返航SOC（%）"),
        )
        row["开始时刻（s）"] += shift
        row["返回O01时刻（s）"] += shift

    charge_full = {
        row["机型编号"]: float(row["等效完全充电时间（s）"])
        for row in data["transport"]["batteries"]
    }
    resource_violations = _assign_transport_resources(trips, charge_full)

    trip_by_id = {row["架次编号"]: row for row in trips}
    for cargo_id, target_trip in CARGO_SWAP.items():
        for row in trips:
            cargo = row["货箱编号列表"].split(";")
            if cargo_id in cargo:
                cargo.remove(cargo_id)
                row["货箱编号列表"] = ";".join(cargo)
                break
        target = trip_by_id[target_trip]
        cargo = target["货箱编号列表"].split(";")
        cargo.append(cargo_id)
        target["货箱编号列表"] = ";".join(cargo)

    segments = read_csv(q2_dir / "main_segments.csv")
    for row in segments:
        shift = _shift(row["架次编号"])
        _float_fields(
            row,
            (
                "载荷（kg）",
                "水平距离（m）",
                "爬升（m）",
                "下降（m）",
                "开始时刻（s）",
                "结束时刻（s）",
                "水平能耗（kWh）",
                "爬升能耗（kWh）",
                "总能耗（kWh）",
            ),
        )
        row["开始时刻（s）"] += shift
        row["结束时刻（s）"] += shift

    timeline = read_csv(q2_dir / "main_timeline.csv")
    for row in timeline:
        trip_id = row["架次编号"]
        shift = _shift(trip_id)
        _float_fields(row, ("开始时刻（s）", "结束时刻（s）"))
        row["开始时刻（s）"] += shift
        row["结束时刻（s）"] += shift
        row["无人机编号"] = trip_by_id[trip_id]["无人机编号"]
        row["电池编号"] = trip_by_id[trip_id]["电池编号"]

    deliveries = read_csv(q2_dir / "main_box_deliveries.csv")
    delivery_by_trip = {}
    for row in deliveries:
        trip_id = row["架次编号"]
        _float_fields(row, ("交付完成时刻（s）", "期望送达时间（s）"))
        row["交付完成时刻（s）"] += _shift(trip_id)
        row["硬截止时间（s）"] = (
            float(row["硬截止时间（s）"]) if row["硬截止时间（s）"] else None
        )
        delivery_by_trip.setdefault(trip_id, row["交付完成时刻（s）"])
    for row in deliveries:
        cargo_id = row["货箱编号"]
        if cargo_id in CARGO_SWAP:
            target_trip = CARGO_SWAP[cargo_id]
            row["架次编号"] = target_trip
            row["交付完成时刻（s）"] = delivery_by_trip[target_trip]
        row["硬截止违例"] = bool(
            row["硬截止时间（s）"] is not None
            and row["交付完成时刻（s）"] > row["硬截止时间（s）"] + EPSILON
        )

    transport_types = {row["机型编号"]: row for row in data["transport"]["types"]}
    battery_rows = []
    for trip in trips:
        type_id = trip["机型编号"]
        charge_start = trip["返回O01时刻（s）"]
        charge_end = charge_start + charge_time_to_full_seconds(
            trip["返航SOC（%）"] / 100.0,
            charge_full[type_id],
        )
        battery_rows.append(
            {
                "电池编号": trip["电池编号"],
                "机型编号": type_id,
                "架次编号": trip["架次编号"],
                "任务开始时刻（s）": trip["开始时刻（s）"],
                "任务结束时刻（s）": charge_start,
                "任务后SOC（%）": trip["返航SOC（%）"],
                "充电开始时刻（s）": charge_start,
                "充电结束时刻（s）": charge_end,
                "再次可用时刻（s）": charge_end,
            }
        )

    boxes = {row["货箱编号"]: row for row in data["demands"]["boxes"]}
    violations = list(resource_violations)
    for row in deliveries:
        if row["硬截止违例"]:
            violations.append("hard_deadline_{}".format(row["货箱编号"]))
    if len(deliveries) != 80 or len({row["货箱编号"] for row in deliveries}) != 80:
        violations.append("cargo_coverage")

    for resource_field in ("无人机编号", "电池编号"):
        grouped = defaultdict(list)
        for trip in trips:
            grouped[trip[resource_field]].append(
                (trip["开始时刻（s）"], trip["返回O01时刻（s）"], trip["架次编号"])
            )
        for resource_id, intervals in grouped.items():
            intervals.sort()
            for left, right in zip(intervals, intervals[1:]):
                if right[0] < left[1] - EPSILON:
                    violations.append(
                        "{}_overlap_{}_{}_{}".format(resource_field, resource_id, left[2], right[2])
                    )

    grouped_batteries = defaultdict(list)
    for row in battery_rows:
        grouped_batteries[row["电池编号"]].append(row)
    for battery_id, rows in grouped_batteries.items():
        rows.sort(key=lambda row: row["任务开始时刻（s）"])
        for left, right in zip(rows, rows[1:]):
            if right["任务开始时刻（s）"] < left["再次可用时刻（s）"] - EPSILON:
                violations.append(
                    "battery_charge_overlap_{}_{}_{}".format(
                        battery_id, left["架次编号"], right["架次编号"]
                    )
                )

    timeliness_loss = 0.0
    for row in deliveries:
        box = boxes[row["货箱编号"]]
        if box["首批截止时间（s）"] is None:
            timeliness_loss += float(box["应急优先系数"]) * max(
                0.0,
                row["交付完成时刻（s）"] - float(box["期望送达时间（s）"]),
            )

    minimum_hard_margin = min(
        row["硬截止时间（s）"] - row["交付完成时刻（s）"]
        for row in deliveries
        if row["硬截止时间（s）"] is not None
    )
    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "data": data,
        "trips": sorted(trips, key=lambda row: row["架次编号"]),
        "segments": sorted(segments, key=lambda row: (row["架次编号"], int(row["航段序号"]))),
        "timeline": sorted(timeline, key=lambda row: (row["架次编号"], int(row["事件序号"]))),
        "deliveries": sorted(deliveries, key=lambda row: row["货箱编号"]),
        "battery_rows": sorted(
            battery_rows, key=lambda row: (row["电池编号"], row["任务开始时刻（s）"])
        ),
        "timeliness_loss_weighted_seconds": timeliness_loss,
        "transport_makespan_seconds": max(row["返回O01时刻（s）"] for row in trips),
        "transport_energy_kwh": sum(row["架次能耗（kWh）"] for row in trips),
        "minimum_hard_deadline_margin_seconds": minimum_hard_margin,
        "transport_types": transport_types,
    }


def _candidate_rows(project_root):
    rows = {}
    for relative in (
        "results/q3_candidate_probe_5s/candidates.csv",
        "results/q3_grid_refinement/refined_candidates.csv",
        "results/q3_multicover_refinement/multicover_candidates.csv",
    ):
        for row in read_csv(Path(project_root) / relative):
            for field in ("lon", "lat", "alt_m", "height_agl_m", "ground_elevation_m"):
                row[field] = float(row[field])
            rows[row["candidate_id"]] = row
    return rows


def build_relay_sorties(project_root, transport):
    project_root = Path(project_root).resolve()
    plan = json.loads(
        (project_root / "results" / "q3" / "relay_path_plan.json").read_text(
            encoding="utf-8"
        )
    )
    if plan["status"] != "PASS":
        raise ValueError("robust relay path plan is not feasible")
    data = transport["data"]
    relay_type = data["relay"]["types"][0]
    candidates = _candidate_rows(project_root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    profiles = {
        candidate_id: relay_travel_profile(
            dem,
            data["nodes"]["centers"][0],
            candidates[candidate_id],
            relay_type,
            {"gravity_m_s2": 9.80665},
        )
        for candidate_id in set(plan["initial_pair"])
        | {
            candidate_id
            for action in plan["actions"]
            for candidate_id in (
                [action["destination"]]
                if action["action"] == "relocate"
                else action["destination_pair"]
            )
        }
    }

    drone_at = {
        plan["initial_pair"][0]: "R01",
        plan["initial_pair"][1]: "R02",
    }
    active = {}
    completed = []
    for candidate_id, drone_id in drone_at.items():
        link_complete = (
            float(relay_type["工位固定准备时间（s）"])
            + profiles[candidate_id]["outbound_seconds"]
            + float(relay_type["建链时间（s）"])
        )
        active[drone_id] = {
            "drone_id": drone_id,
            "candidate_id": candidate_id,
            "start_seconds": 0.0,
            "link_complete_seconds": link_complete,
        }

    turnaround = float(relay_type["架次周转时间（s）"])
    preparation = float(relay_type["工位固定准备时间（s）"])
    link_time = float(relay_type["建链时间（s）"])
    for action in plan["actions"]:
        if action["action"] == "relocate":
            moves = [(action["origin"], action["destination"])]
        elif action["action"] == "dual_relocate":
            origins = action["origin_pair"]
            destinations = action["destination_pair"]
            alternatives = (
                [(origins[0], destinations[0]), (origins[1], destinations[1])],
                [(origins[0], destinations[1]), (origins[1], destinations[0])],
            )
            moves = min(
                alternatives,
                key=lambda mapping: max(
                    profiles[origin]["inbound_seconds"]
                    + turnaround
                    + preparation
                    + profiles[destination]["outbound_seconds"]
                    + link_time
                    for origin, destination in mapping
                ),
            )
        else:
            raise ValueError("unknown relay action {}".format(action["action"]))

        arrivals = []
        for origin, destination in moves:
            drone_id = drone_at.pop(origin)
            current = active.pop(drone_id)
            current["service_end_seconds"] = float(action["start_seconds"])
            current["return_seconds"] = current["service_end_seconds"] + profiles[origin][
                "inbound_seconds"
            ]
            completed.append(current)
            next_start = current["return_seconds"] + turnaround
            link_complete = (
                next_start
                + preparation
                + profiles[destination]["outbound_seconds"]
                + link_time
            )
            arrivals.append(link_complete)
            active[drone_id] = {
                "drone_id": drone_id,
                "candidate_id": destination,
                "start_seconds": next_start,
                "link_complete_seconds": link_complete,
            }
            drone_at[destination] = drone_id
        if abs(max(arrivals) - float(action["arrival_seconds"])) > 1e-6:
            raise AssertionError("relay transition replay mismatch")

    for drone_id, current in active.items():
        current["service_end_seconds"] = float(plan["coverage_end_seconds"])
        current["return_seconds"] = (
            current["service_end_seconds"]
            + profiles[current["candidate_id"]]["inbound_seconds"]
        )
        completed.append(current)

    completed.sort(key=lambda row: (row["start_seconds"], row["drone_id"]))
    output = []
    violations = []
    for index, row in enumerate(completed, start=1):
        candidate = candidates[row["candidate_id"]]
        service_seconds = row["service_end_seconds"] - row["link_complete_seconds"]
        metrics = relay_sortie_metrics(
            profiles[row["candidate_id"]], service_seconds, relay_type
        )
        if not metrics["energy_feasible"]:
            violations.append("relay_energy_{}".format(index))
        output.append(
            {
                "中继架次编号": "RLY-{:03d}".format(index),
                "中继无人机编号": row["drone_id"],
                "能源组件编号": "R-MOD-{:02d}".format(index),
                "候选点编号": row["candidate_id"],
                "开始时刻（s）": row["start_seconds"],
                "悬停经度（°）": candidate["lon"],
                "悬停纬度（°）": candidate["lat"],
                "悬停海拔（m）": candidate["alt_m"],
                "建链完成时刻（s）": row["link_complete_seconds"],
                "服务结束时刻（s）": row["service_end_seconds"],
                "返回O01时刻（s）": row["return_seconds"],
                "架次能耗（kWh）": metrics["energy_kwh"],
                "返航SOC（%）": metrics["remaining_soc_percent"],
            }
        )
    by_drone = defaultdict(list)
    for row in output:
        by_drone[row["中继无人机编号"]].append(row)
    for drone_id, rows in by_drone.items():
        rows.sort(key=lambda row: row["开始时刻（s）"])
        for left, right in zip(rows, rows[1:]):
            earliest = left["返回O01时刻（s）"] + float(relay_type["架次周转时间（s）"])
            if right["开始时刻（s）"] < earliest - EPSILON:
                violations.append("relay_turnaround_{}".format(drone_id))
    if len(output) > int(data["relay"]["modules"][0]["共享能源组件总数（组）"]):
        violations.append("relay_module_inventory")
    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "sorties": output,
        "relay_energy_kwh": sum(row["架次能耗（kWh）"] for row in output),
        "relay_makespan_seconds": max(row["返回O01时刻（s）"] for row in output),
        "minimum_relay_soc_percent": min(row["返航SOC（%）"] for row in output),
        "candidates": candidates,
    }


def audit_continuous_communication(project_root, transport, relay):
    """Replay shifted 1-second samples against active relay sorties."""

    project_root = Path(project_root).resolve()
    data = transport["data"]
    dem = RasterDEM(data["paths"]["dem_tif"])
    model = CommunicationModel(data["communication"])
    samples = read_csv(project_root / "results" / "q3_direct_1s" / "direct_link_samples.csv")
    sorties = relay["sorties"]
    candidates = relay["candidates"]
    backhaul_cache = {}
    gateway_center = data["nodes"]["centers"][0]
    gateway = {
        "lon": float(gateway_center["经度（°）"]),
        "lat": float(gateway_center["纬度（°）"]),
        "alt_m": float(gateway_center["海拔（m）"]) + model.gateway_agl_m,
    }
    for row in sorties:
        candidate_id = row["候选点编号"]
        candidate = candidates[candidate_id]
        relay_position = {
            "lon": candidate["lon"],
            "lat": candidate["lat"],
            "alt_m": candidate["alt_m"],
        }
        backhaul_cache[candidate_id] = audit_link(
            dem, model, relay_position, gateway, "relay_backhaul", "gateway"
        )

    assigned = []
    violations = []
    relay_margins = []
    direct_margins = []
    minimum_margin_evidence = None
    for row in samples:
        trip_id = row["trip_id"]
        time_seconds = float(row["time_seconds"]) + _shift(trip_id)
        direct = str(row["available"]).lower() == "true"
        if direct:
            direct_margins.append(float(row["margin_db"]))
        assignment = {
            "运输架次编号": trip_id,
            "通信阶段": row["phase"],
            "开始时刻（s）": time_seconds,
            "结束时刻（s）": time_seconds,
            "保障方式": "直连" if direct else "",
            "中继架次编号": "",
        }
        if not direct:
            position = {
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "alt_m": float(row["alt_m"]),
            }
            feasible = []
            for sortie in sorties:
                if not (
                    float(sortie["建链完成时刻（s）"]) - EPSILON
                    <= time_seconds
                    <= float(sortie["服务结束时刻（s）"]) + EPSILON
                ):
                    continue
                candidate_id = sortie["候选点编号"]
                if not backhaul_cache[candidate_id]["available"]:
                    continue
                candidate = candidates[candidate_id]
                relay_position = {
                    "lon": candidate["lon"],
                    "lat": candidate["lat"],
                    "alt_m": candidate["alt_m"],
                }
                access = audit_link(
                    dem, model, position, relay_position, "transport", "relay_access"
                )
                if access["available"]:
                    feasible.append((access["margin_db"], sortie))
            if not feasible:
                violations.append(
                    {"trip_id": trip_id, "time_seconds": time_seconds, "phase": row["phase"]}
                )
                assignment["保障方式"] = "中断"
            else:
                margin, sortie = max(feasible, key=lambda item: item[0])
                relay_margins.append(margin)
                if (
                    minimum_margin_evidence is None
                    or margin < minimum_margin_evidence["margin_db"]
                ):
                    minimum_margin_evidence = {
                        "trip_id": trip_id,
                        "time_seconds": time_seconds,
                        "phase": row["phase"],
                        "relay_sortie_id": sortie["中继架次编号"],
                        "candidate_id": sortie["候选点编号"],
                        "margin_db": margin,
                    }
                assignment["保障方式"] = "中继"
                assignment["中继架次编号"] = sortie["中继架次编号"]
        assigned.append(assignment)

    assigned.sort(key=lambda row: (row["运输架次编号"], row["开始时刻（s）"]))
    merged = []
    for row in assigned:
        key = (
            row["运输架次编号"],
            row["通信阶段"],
            row["保障方式"],
            row["中继架次编号"],
        )
        if merged:
            previous = merged[-1]
            previous_key = (
                previous["运输架次编号"],
                previous["通信阶段"],
                previous["保障方式"],
                previous["中继架次编号"],
            )
            if key == previous_key and row["开始时刻（s）"] - previous["结束时刻（s）"] <= 1.000001:
                previous["结束时刻（s）"] = row["结束时刻（s）"]
                continue
        merged.append(dict(row))
    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "sample_count": len(samples),
        "relay_sample_count": sum(row["保障方式"] == "中继" for row in assigned),
        "minimum_relay_access_margin_db": min(relay_margins) if relay_margins else None,
        "minimum_relay_backhaul_margin_db": min(
            item["margin_db"] for item in backhaul_cache.values()
        ),
        "minimum_adopted_direct_margin_db": min(direct_margins) if direct_margins else None,
        "minimum_margin_evidence": minimum_margin_evidence,
        "intervals": merged,
    }


def run_q3_joint(project_root, output_dir):
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    transport = build_adjusted_transport(project_root)
    if transport["status"] != "PASS":
        raise AssertionError("adjusted transport audit failed: {}".format(transport["violations"]))
    relay = build_relay_sorties(project_root, transport)
    if relay["status"] != "PASS":
        raise AssertionError("relay sortie audit failed: {}".format(relay["violations"]))
    communication = audit_continuous_communication(project_root, transport, relay)
    if communication["status"] != "PASS":
        raise AssertionError(
            "communication audit failed at {} samples".format(len(communication["violations"]))
        )

    write_csv(output_dir / "transport_trips.csv", transport["trips"])
    write_csv(output_dir / "transport_box_deliveries.csv", transport["deliveries"])
    write_csv(output_dir / "transport_segments.csv", transport["segments"])
    write_csv(output_dir / "transport_timeline.csv", transport["timeline"])
    write_csv(output_dir / "transport_battery_timeline.csv", transport["battery_rows"])
    write_csv(output_dir / "relay_sorties.csv", relay["sorties"])
    write_csv(output_dir / "communication_guarantee.csv", communication["intervals"])

    summary = {
        "status": "PASS",
        "solution_version": "S3-v1.0.0-provisional_v0",
        "parent_version": "S2-v1.1.0-provisional_v0",
        "energy_model_version": "provisional_v0",
        "transport_trip_shifts_seconds": TRIP_SHIFTS_SECONDS,
        "cargo_swap": CARGO_SWAP,
        "transport_metrics": {
            "timeliness_loss_weighted_seconds": transport["timeliness_loss_weighted_seconds"],
            "makespan_seconds": transport["transport_makespan_seconds"],
            "energy_kwh": transport["transport_energy_kwh"],
            "trip_count": len(transport["trips"]),
            "minimum_hard_deadline_margin_seconds": transport[
                "minimum_hard_deadline_margin_seconds"
            ],
        },
        "relay_metrics": {
            "makespan_seconds": relay["relay_makespan_seconds"],
            "energy_kwh": relay["relay_energy_kwh"],
            "sortie_count": len(relay["sorties"]),
            "drones_used": len({row["中继无人机编号"] for row in relay["sorties"]}),
            "modules_used": len({row["能源组件编号"] for row in relay["sorties"]}),
            "minimum_return_soc_percent": relay["minimum_relay_soc_percent"],
        },
        "joint_metrics": {
            "makespan_seconds": max(
                transport["transport_makespan_seconds"], relay["relay_makespan_seconds"]
            ),
            "total_energy_kwh": transport["transport_energy_kwh"]
            + relay["relay_energy_kwh"],
            "transport_and_relay_sorties": len(transport["trips"]) + len(relay["sorties"]),
        },
        "communication_audit": {
            "status": communication["status"],
            "step_seconds": 1.0,
            "sample_count": communication["sample_count"],
            "relay_sample_count": communication["relay_sample_count"],
            "minimum_relay_access_margin_db": communication[
                "minimum_relay_access_margin_db"
            ],
            "minimum_relay_backhaul_margin_db": communication[
                "minimum_relay_backhaul_margin_db"
            ],
            "minimum_adopted_relay_link_margin_db": min(
                communication["minimum_relay_access_margin_db"],
                communication["minimum_relay_backhaul_margin_db"],
            ),
            "minimum_adopted_direct_margin_db": communication[
                "minimum_adopted_direct_margin_db"
            ],
            "minimum_margin_evidence": communication["minimum_margin_evidence"],
            "violations": len(communication["violations"]),
        },
        "constraint_audit": {
            "status": "PASS",
            "transport_violations": len(transport["violations"]),
            "relay_violations": len(relay["violations"]),
            "communication_violations": len(communication["violations"]),
        },
    }
    (output_dir / "q3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
