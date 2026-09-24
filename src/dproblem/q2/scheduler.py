"""Explicit heterogeneous drone/battery list scheduler used by Q2 solvers."""

from copy import deepcopy

from dproblem.domain.charging import charge_time_to_full_seconds
from dproblem.q2.baseline import _hard_deadline


EPSILON = 1e-8


def route_deadline_key(task):
    hard = [deadline for deadline in (_hard_deadline(box) for box in task["boxes"]) if deadline is not None]
    return (
        min(hard) if hard else float("inf"),
        min(float(box["期望送达时间（s）"]) for box in task["boxes"]),
        -max(float(box["应急优先系数"]) for box in task["boxes"]),
        tuple(task["visit_order"]),
        tuple(sorted(box["货箱编号"] for box in task["boxes"])),
    )


def schedule_routes(route_tasks, data, config, evaluator, preserve_order=False):
    """Schedule routes and choose the earliest-delivery feasible drone type.

    This is a deterministic decoder: route composition/order are supplied by
    the outer search, while concrete type, drone, battery and start time are
    selected here with full physical replay.
    """

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

    scheduled = []
    battery_rows = []
    ordered_tasks = list(route_tasks) if preserve_order else sorted(route_tasks, key=route_deadline_key)
    for sequence, task in enumerate(ordered_tasks, start=1):
        choices = []
        allowed_types = task.get("allowed_types", sorted(types))
        for type_id in allowed_types:
            compatible_drones = sorted(
                drone_id for drone_id, state in drones.items() if state["type_id"] == type_id
            )
            compatible_batteries = sorted(
                battery_id for battery_id, state in batteries.items() if state["type_id"] == type_id
            )
            if not compatible_drones or not compatible_batteries:
                continue
            resource_choices = [
                (
                    max(drones[drone_id]["available"], batteries[battery_id]["available"]),
                    drone_id,
                    battery_id,
                )
                for drone_id in compatible_drones
                for battery_id in compatible_batteries
            ]
            start, drone_id, battery_id = min(resource_choices)
            route = evaluator.evaluate(
                task["boxes"], task["visit_order"], types[type_id], start_seconds=start
            )
            if route is None:
                continue
            latenesses = []
            for box in task["boxes"]:
                deadline = _hard_deadline(box)
                if deadline is not None:
                    latenesses.append(route["逐箱交付时刻"][box["货箱编号"]] - deadline)
            maximum_lateness = max(latenesses) if latenesses else float("-inf")
            desired_loss = sum(
                float(box["应急优先系数"])
                * max(
                    0.0,
                    route["逐箱交付时刻"][box["货箱编号"]]
                    - float(box["期望送达时间（s）"]),
                )
                for box in task["boxes"]
                if _hard_deadline(box) is None
            )
            choices.append(
                (
                    maximum_lateness,
                    desired_loss,
                    route["返回O01时刻（s）"],
                    route["架次能耗（kWh）"],
                    type_id,
                    drone_id,
                    battery_id,
                    route,
                )
            )
        if not choices:
            return {"status": "FAIL", "reason": "no_physical_route", "task": task}
        _, _, _, _, type_id, drone_id, battery_id, route = min(choices)
        trip_id = "Q2-{:03d}".format(sequence)
        route.update(
            {
                "架次编号": trip_id,
                "无人机编号": drone_id,
                "电池编号": battery_id,
                "货箱编号列表": [box["货箱编号"] for box in task["boxes"]],
                "_boxes": task["boxes"],
            }
        )
        scheduled.append(route)
        drones[drone_id]["available"] = (
            route["返回O01时刻（s）"]
            + config["transport_timeline"]["transport_turnaround_seconds"]
        )
        soc_after = route["返航SOC（%）"] / 100.0
        charge_start = route["返回O01时刻（s）"]
        charge_end = charge_start + charge_time_to_full_seconds(
            soc_after, charge_full[type_id]
        )
        batteries[battery_id]["available"] = charge_end
        battery_rows.append(
            {
                "电池编号": battery_id,
                "机型编号": type_id,
                "架次编号": trip_id,
                "任务开始时刻（s）": route["开始时刻（s）"],
                "任务结束时刻（s）": charge_start,
                "任务后SOC（%）": route["返航SOC（%）"],
                "充电开始时刻（s）": charge_start,
                "充电结束时刻（s）": charge_end,
                "再次可用时刻（s）": charge_end,
            }
        )

    hard_violations = []
    timeliness_loss = 0.0
    deliveries = {}
    for trip in scheduled:
        for box in trip["_boxes"]:
            cargo_id = box["货箱编号"]
            delivery = trip["逐箱交付时刻"][cargo_id]
            deliveries[cargo_id] = delivery
            deadline = _hard_deadline(box)
            if deadline is not None and delivery > deadline + EPSILON:
                hard_violations.append(
                    {"货箱编号": cargo_id, "交付时刻（s）": delivery, "硬截止（s）": deadline}
                )
            if deadline is None:
                timeliness_loss += float(box["应急优先系数"]) * max(
                    0.0, delivery - float(box["期望送达时间（s）"])
                )
    return {
        "status": "PASS" if not hard_violations else "FAIL",
        "trips": scheduled,
        "battery_rows": battery_rows,
        "deliveries": deliveries,
        "hard_violations": hard_violations,
        "timeliness_loss": timeliness_loss,
        "makespan": max(trip["返回O01时刻（s）"] for trip in scheduled),
        "energy": sum(trip["架次能耗（kWh）"] for trip in scheduled),
        "trip_count": len(scheduled),
    }


def q1_batches_as_flexible_routes(project_root, boxes):
    from dproblem.q2.baseline import build_q1_tasks

    tasks = build_q1_tasks(project_root, boxes)
    return [
        {
            "boxes": deepcopy(task["boxes"]),
            "visit_order": [task["service_id"]],
            "allowed_types": ["A", "B", "C"],
        }
        for task in tasks
    ]
