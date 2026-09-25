"""Solve Q4 from the frozen S3 transport and relay schedule."""

import csv
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

from dproblem.common.hashing import sha256_file
from dproblem.domain.charging import charge_time_to_full_seconds
from dproblem.io.dataset import load_project_data


EPSILON = 1e-7
RESOURCE_NAMES = (
    "A型运输无人机数",
    "B型运输无人机数",
    "C型运输无人机数",
    "A型电池组数",
    "B型电池组数",
    "C型电池组数",
    "中继无人机数",
    "中继能源组件数",
)
WORKLOAD_NAMES = ("货物质量（kg）", "运输任务时长（s）", "中继任务时长（s）", "总能耗（kWh）")


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


def verify_s3_freeze(project_root):
    root = Path(project_root).resolve()
    freeze = json.loads((root / "results/q3/S3_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_PASS":
        raise ValueError("S3 is not frozen with PASS status")
    mismatches = []
    for relative, expected in freeze["sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError("S3 freeze hash mismatch: {}".format(mismatches))
    return freeze


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _union_all(union_find, items):
    items = sorted(set(items))
    for item in items[1:]:
        union_find.union(items[0], item)


def _minimum_identical_resources(intervals):
    """Minimum interval coloring count; intervals are half-open [start, end)."""

    active_ends = []
    maximum = 0
    for start, end in sorted(intervals):
        while active_ends and active_ends[0] <= start + EPSILON:
            heapq.heappop(active_ends)
        heapq.heappush(active_ends, end)
        maximum = max(maximum, len(active_ends))
    return maximum


def _restricted_growth_assignments(count, k):
    assignment = [0] * count

    def visit(index, maximum):
        if index == count:
            if maximum == k - 1:
                yield tuple(assignment)
            return
        for label in range(min(maximum + 1, k - 1) + 1):
            assignment[index] = label
            yield from visit(index + 1, max(maximum, label))

    if count:
        yield from visit(1, 0)


def _coefficient_of_variation(values):
    mean = sum(values) / len(values)
    if mean <= EPSILON:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def _dominates(left, right):
    return all(a <= b + EPSILON for a, b in zip(left, right)) and any(
        a < b - EPSILON for a, b in zip(left, right)
    )


def _pareto(rows, vector_key):
    return [
        row
        for index, row in enumerate(rows)
        if not any(
            index != other_index and _dominates(other[vector_key], row[vector_key])
            for other_index, other in enumerate(rows)
        )
    ]


def _geographic_sse(groups, service_by_id):
    mean_lat = sum(float(row["纬度（°）"]) for row in service_by_id.values()) / len(service_by_id)
    lon_scale = 111320.0 * math.cos(math.radians(mean_lat))
    lat_scale = 110540.0
    total = 0.0
    for services in groups:
        points = [
            (
                float(service_by_id[item]["经度（°）"]) * lon_scale,
                float(service_by_id[item]["纬度（°）"]) * lat_scale,
            )
            for item in services
        ]
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        total += sum((x - center_x) ** 2 + (y - center_y) ** 2 for x, y in points)
    return total


def _canonical_group_key(groups):
    return tuple(tuple(sorted(group)) for group in sorted(groups, key=lambda group: tuple(sorted(group))))


def solve_q4(project_root, output_dir="results/q4"):
    root = Path(project_root).resolve()
    verify_s3_freeze(root)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_project_data(root)
    trips = read_csv(root / "results/q3/transport_trips.csv")
    deliveries = read_csv(root / "results/q3/transport_box_deliveries.csv")
    batteries = read_csv(root / "results/q3/transport_battery_timeline.csv")
    sorties = read_csv(root / "results/q3/relay_sorties.csv")
    communication = read_csv(root / "results/q3/communication_guarantee.csv")
    service_ids = sorted(row["服务区编号"] for row in data["nodes"]["service_areas"])
    service_by_id = {row["服务区编号"]: row for row in data["nodes"]["service_areas"]}
    box_by_id = {row["货箱编号"]: row for row in data["demands"]["boxes"]}

    trip_services = {
        row["架次编号"]: set(filter(None, row["访问服务区顺序"].split(";"))) for row in trips
    }
    relay_trips = defaultdict(set)
    for row in communication:
        if row["保障方式"] == "中继" and row["中继架次编号"]:
            relay_trips[row["中继架次编号"]].add(row["运输架次编号"])
    relay_services = {
        sortie_id: set().union(*(trip_services[trip_id] for trip_id in trip_ids))
        for sortie_id, trip_ids in relay_trips.items()
    }

    union_find = UnionFind(service_ids)
    for services in trip_services.values():
        _union_all(union_find, services)
    for services in relay_services.values():
        _union_all(union_find, services)
    components_by_root = defaultdict(set)
    for service_id in service_ids:
        components_by_root[union_find.find(service_id)].add(service_id)
    components = sorted(components_by_root.values(), key=lambda item: tuple(sorted(item)))
    component_index = {
        service_id: index for index, component in enumerate(components) for service_id in component
    }

    trip_by_id = {row["架次编号"]: row for row in trips}
    battery_by_trip = {row["架次编号"]: row for row in batteries}
    sortie_by_id = {row["中继架次编号"]: row for row in sorties}
    delivery_trip_by_box = {row["货箱编号"]: row["架次编号"] for row in deliveries}
    relay_module = data["relay"]["modules"][0]
    relay_turnaround = float(data["relay"]["types"][0]["架次周转时间（s）"])
    relay_charge_full = float(relay_module["等效完全充电时间（s）"])

    stock = (
        sum(row["机型编号"] == "A" for row in data["transport"]["drones"]),
        sum(row["机型编号"] == "B" for row in data["transport"]["drones"]),
        sum(row["机型编号"] == "C" for row in data["transport"]["drones"]),
        next(int(row["共享电池组总数（组）"]) for row in data["transport"]["batteries"] if row["机型编号"] == "A"),
        next(int(row["共享电池组总数（组）"]) for row in data["transport"]["batteries"] if row["机型编号"] == "B"),
        next(int(row["共享电池组总数（组）"]) for row in data["transport"]["batteries"] if row["机型编号"] == "C"),
        len(data["relay"]["drones"]),
        int(relay_module["共享能源组件总数（组）"]),
    )

    candidates_by_k = {}
    selected_by_k = {}
    baseline_by_k = {}
    for k in (2, 3):
        candidates = []
        for assignment in _restricted_growth_assignments(len(components), k):
            groups = [set() for _ in range(k)]
            for index, label in enumerate(assignment):
                groups[label].update(components[index])
            groups.sort(key=lambda group: tuple(sorted(group)))

            group_rows = []
            for group_index, services in enumerate(groups, start=1):
                group_trip_ids = sorted(
                    trip_id for trip_id, trip_set in trip_services.items() if trip_set <= services
                )
                if any(
                    trip_set & services and not trip_set <= services
                    for trip_set in trip_services.values()
                ):
                    raise AssertionError("transport must-link edge was cut")
                group_sortie_ids = sorted(
                    sortie_id
                    for sortie_id, service_set in relay_services.items()
                    if service_set <= services
                )
                if any(
                    service_set & services and not service_set <= services
                    for service_set in relay_services.values()
                ):
                    raise AssertionError("relay must-link edge was cut")

                drone_counts = []
                battery_counts = []
                for type_id in ("A", "B", "C"):
                    typed_trips = [trip_by_id[item] for item in group_trip_ids if trip_by_id[item]["机型编号"] == type_id]
                    drone_counts.append(
                        _minimum_identical_resources(
                            [
                                (float(row["开始时刻（s）"]), float(row["返回O01时刻（s）"]))
                                for row in typed_trips
                            ]
                        )
                    )
                    battery_counts.append(
                        _minimum_identical_resources(
                            [
                                (
                                    float(battery_by_trip[row["架次编号"]]["任务开始时刻（s）"]),
                                    float(battery_by_trip[row["架次编号"]]["再次可用时刻（s）"]),
                                )
                                for row in typed_trips
                            ]
                        )
                    )

                group_sorties = [sortie_by_id[item] for item in group_sortie_ids]
                relay_drones = _minimum_identical_resources(
                    [
                        (
                            float(row["开始时刻（s）"]),
                            float(row["返回O01时刻（s）"]) + relay_turnaround,
                        )
                        for row in group_sorties
                    ]
                )
                relay_modules = _minimum_identical_resources(
                    [
                        (
                            float(row["开始时刻（s）"]),
                            float(row["返回O01时刻（s）"])
                            + charge_time_to_full_seconds(
                                float(row["返航SOC（%）"]) / 100.0, relay_charge_full
                            ),
                        )
                        for row in group_sorties
                    ]
                )
                resource_vector = tuple(drone_counts + battery_counts + [relay_drones, relay_modules])
                cargo_ids = [
                    box_id
                    for box_id, trip_id in delivery_trip_by_box.items()
                    if trip_id in group_trip_ids
                ]
                mass = sum(float(box_by_id[item]["单箱质量（kg）"]) for item in cargo_ids)
                transport_duration = sum(
                    float(trip_by_id[item]["返回O01时刻（s）"])
                    - float(trip_by_id[item]["开始时刻（s）"])
                    for item in group_trip_ids
                )
                relay_duration = sum(
                    float(sortie_by_id[item]["返回O01时刻（s）"])
                    - float(sortie_by_id[item]["开始时刻（s）"])
                    for item in group_sortie_ids
                )
                energy = sum(float(trip_by_id[item]["架次能耗（kWh）"]) for item in group_trip_ids) + sum(
                    float(sortie_by_id[item]["架次能耗（kWh）"]) for item in group_sortie_ids
                )
                group_rows.append(
                    {
                        "group": group_index,
                        "services": tuple(sorted(services)),
                        "trip_ids": tuple(group_trip_ids),
                        "relay_sortie_ids": tuple(group_sortie_ids),
                        "resources": resource_vector,
                        "workload": (mass, transport_duration, relay_duration, energy),
                    }
                )

            totals = tuple(sum(group["resources"][index] for group in group_rows) for index in range(8))
            shortage = tuple(max(total - available, 0) for total, available in zip(totals, stock))
            redundancy = tuple(max(available - total, 0) for total, available in zip(totals, stock))
            cvs = tuple(
                _coefficient_of_variation([group["workload"][index] for group in group_rows])
                for index in range(4)
            )
            candidates.append(
                {
                    "groups": group_rows,
                    "configuration": totals,
                    "shortage": shortage,
                    "redundancy": redundancy,
                    "cv": cvs,
                    "max_cv": max(cvs),
                    "geo_sse_m2": _geographic_sse(groups, service_by_id),
                    "key": _canonical_group_key(groups),
                }
            )

        shortage_front = _pareto(candidates, "shortage")
        for row in shortage_front:
            row["joint_vector"] = row["configuration"] + row["cv"]
        joint_front = _pareto(shortage_front, "joint_vector")
        minimum_max_cv = min(row["max_cv"] for row in joint_front)
        balance_finalists = [
            row for row in joint_front if abs(row["max_cv"] - minimum_max_cv) <= EPSILON
        ]
        # User-approved display-only tie break: if incomparable resource vectors
        # remain at the same max-CV value, show the plan with fewer deficient
        # resource categories.  This does not assert cross-type Pareto superiority.
        selected = min(
            balance_finalists,
            key=lambda row: (
                sum(value > 0 for value in row["shortage"]),
                row["key"],
            ),
        )
        baseline = min(candidates, key=lambda row: (row["geo_sse_m2"], row["key"]))
        candidates_by_k[k] = candidates
        selected_by_k[k] = selected
        baseline_by_k[k] = baseline

    component_rows = [
        {"连通分量编号": "C{:02d}".format(index), "服务区列表": ";".join(sorted(component))}
        for index, component in enumerate(components, start=1)
    ]
    selected_rows = []
    for k in (2, 3):
        for group in selected_by_k[k]["groups"]:
            row = {
                "K（2或3）": k,
                "任务组编号": "K{}-G{}".format(k, group["group"]),
                "服务区列表": ";".join(group["services"]),
            }
            row.update(dict(zip(RESOURCE_NAMES, group["resources"])))
            row.update(dict(zip(WORKLOAD_NAMES, group["workload"])))
            row["运输架次列表"] = ";".join(group["trip_ids"])
            row["中继架次列表"] = ";".join(group["relay_sortie_ids"])
            selected_rows.append(row)

    candidate_rows = []
    for k in (2, 3):
        shortage_front_keys = {row["key"] for row in _pareto(candidates_by_k[k], "shortage")}
        for row in candidates_by_k[k]:
            candidate_rows.append(
                {
                    "K": k,
                    "分组": " | ".join(";".join(group["services"]) for group in row["groups"]),
                    "资源配置向量": json.dumps(row["configuration"], ensure_ascii=False),
                    "资源缺口向量": json.dumps(row["shortage"], ensure_ascii=False),
                    "资源冗余向量": json.dumps(row["redundancy"], ensure_ascii=False),
                    "工作量CV向量": json.dumps(row["cv"], ensure_ascii=False),
                    "最大CV": row["max_cv"],
                    "地理SSE（m²）": row["geo_sse_m2"],
                    "缺口Pareto": row["key"] in shortage_front_keys,
                    "最终选中": row["key"] == selected_by_k[k]["key"],
                    "地理baseline": row["key"] == baseline_by_k[k]["key"],
                }
            )

    write_csv(output_dir / "must_link_components.csv", component_rows)
    write_csv(output_dir / "selected_partition.csv", selected_rows)
    write_csv(output_dir / "partition_candidates.csv", candidate_rows)

    summary = {
        "status": "PASS",
        "solution_version": "S4-v1.0.0-provisional_v0",
        "parent_version": "S3-v1.0.0-provisional_v0",
        "must_link_rule": "same transport trip or same indivisible relay sortie implies same group",
        "representative_tie_break": "after Pareto filtering and minimum max-CV, minimize the number of deficient resource categories for display only; then use service-list lexicographic order",
        "component_count": len(components),
        "components": [sorted(component) for component in components],
        "stock_vector": dict(zip(RESOURCE_NAMES, stock)),
        "results": {},
    }
    for k in (2, 3):
        chosen = selected_by_k[k]
        baseline = baseline_by_k[k]
        summary["results"][str(k)] = {
            "candidate_count": len(candidates_by_k[k]),
            "groups": [sorted(group["services"]) for group in chosen["groups"]],
            "configuration_vector": dict(zip(RESOURCE_NAMES, chosen["configuration"])),
            "shortage_vector": dict(zip(RESOURCE_NAMES, chosen["shortage"])),
            "redundancy_vector": dict(zip(RESOURCE_NAMES, chosen["redundancy"])),
            "workload_cv_vector": dict(zip(WORKLOAD_NAMES, chosen["cv"])),
            "maximum_cv": chosen["max_cv"],
            "geographic_baseline_groups": [sorted(group["services"]) for group in baseline["groups"]],
            "geographic_baseline_sse_m2": baseline["geo_sse_m2"],
        }
    (output_dir / "q4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
