import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from dproblem.common.hashing import stable_json_hash
from dproblem.config import energy_config_hash, load_model_config
from dproblem.domain.energy import transport_segment_energy_kwh
from dproblem.domain.geometry import RasterDEM, segment_terrain_profile
from dproblem.io.dataset import load_project_data
from dproblem.q1.work_time import per_trip_work_time_seconds


EPSILON = 1e-10


@dataclass(frozen=True)
class TripOption:
    mask: int
    type_id: str
    mass_kg: float
    volume_m3: float
    box_count: int
    flight_seconds: float
    work_seconds: float
    energy_kwh: float
    energy_limit_kwh: float
    remaining_soc_fraction: float


@dataclass(frozen=True)
class PlanState:
    trip_count: int
    energy_kwh: float
    work_seconds: float
    trips: tuple


def _flight_seconds(profile, drone_type):
    forward = (
        profile["forward_climb_m"] / drone_type["最大爬升速度（m/s）"]
        + profile["horizontal_distance_m"] / drone_type["计划巡航速度（m/s）"]
        + profile["forward_descent_m"] / drone_type["最大下降速度（m/s）"]
    )
    reverse = (
        profile["reverse_climb_m"] / drone_type["最大爬升速度（m/s）"]
        + profile["horizontal_distance_m"] / drone_type["计划巡航速度（m/s）"]
        + profile["reverse_descent_m"] / drone_type["最大下降速度（m/s）"]
    )
    return forward + reverse


def evaluate_trip(mask, mass_kg, volume_m3, box_count, drone_type, profile, energy_config, reserve_override=None):
    if mass_kg > drone_type["最大载货质量（kg）"] + EPSILON:
        return None
    if volume_m3 > drone_type["可用装载体积（m³）"] + EPSILON:
        return None
    outbound = transport_segment_energy_kwh(
        profile["horizontal_distance_m"],
        profile["forward_climb_m"],
        mass_kg,
        drone_type,
        energy_config,
    )
    inbound = transport_segment_energy_kwh(
        profile["horizontal_distance_m"],
        profile["reverse_climb_m"],
        0.0,
        drone_type,
        energy_config,
    )
    energy_kwh = outbound["total_kwh"] + inbound["total_kwh"]
    reserve_fraction = (
        reserve_override
        if reserve_override is not None
        else drone_type["返航电量下限（%）"] / 100.0
    )
    energy_limit_kwh = drone_type["电池可用能量（kWh）"] * (1.0 - reserve_fraction)
    if energy_kwh > energy_limit_kwh + EPSILON:
        return None
    flight_seconds = _flight_seconds(profile, drone_type)
    work_seconds = per_trip_work_time_seconds(drone_type, box_count, flight_seconds)
    return TripOption(
        mask=mask,
        type_id=drone_type["机型编号"],
        mass_kg=mass_kg,
        volume_m3=volume_m3,
        box_count=box_count,
        flight_seconds=flight_seconds,
        work_seconds=work_seconds,
        energy_kwh=energy_kwh,
        energy_limit_kwh=energy_limit_kwh,
        remaining_soc_fraction=1.0 - energy_kwh / drone_type["电池可用能量（kWh）"],
    )


def maximum_safe_payload_kg(drone_type, profile, energy_config, reserve_fraction):
    capacity = float(drone_type["最大载货质量（kg）"])

    def energy(payload):
        outbound = transport_segment_energy_kwh(
            profile["horizontal_distance_m"],
            profile["forward_climb_m"],
            payload,
            drone_type,
            energy_config,
        )["total_kwh"]
        inbound = transport_segment_energy_kwh(
            profile["horizontal_distance_m"],
            profile["reverse_climb_m"],
            0.0,
            drone_type,
            energy_config,
        )["total_kwh"]
        return outbound + inbound

    sampled = [energy(capacity * index / 100.0) for index in range(101)]
    if any(right + EPSILON < left for left, right in zip(sampled, sampled[1:])):
        raise AssertionError("round-trip energy is not monotone in payload")
    limit = drone_type["电池可用能量（kWh）"] * (1.0 - reserve_fraction)
    if sampled[0] > limit + EPSILON:
        return 0.0
    if sampled[-1] <= limit + EPSILON:
        return capacity
    low, high = 0.0, capacity
    for _ in range(60):
        middle = (low + high) / 2.0
        if energy(middle) <= limit:
            low = middle
        else:
            high = middle
    return low


def _subset_totals(boxes):
    size = 1 << len(boxes)
    masses = [0.0] * size
    volumes = [0.0] * size
    counts = [0] * size
    for mask in range(1, size):
        bit = mask & -mask
        index = bit.bit_length() - 1
        previous = mask ^ bit
        masses[mask] = masses[previous] + boxes[index]["单箱质量（kg）"]
        volumes[mask] = volumes[previous] + boxes[index]["单箱体积（m³）"]
        counts[mask] = counts[previous] + 1
    return masses, volumes, counts


def _pareto_trip_options(options):
    kept = []
    for candidate in sorted(options, key=lambda item: (item.energy_kwh, item.work_seconds, item.type_id)):
        if any(
            existing.energy_kwh <= candidate.energy_kwh + EPSILON
            and existing.work_seconds <= candidate.work_seconds + EPSILON
            for existing in kept
        ):
            continue
        kept = [
            existing
            for existing in kept
            if not (
                candidate.energy_kwh <= existing.energy_kwh + EPSILON
                and candidate.work_seconds <= existing.work_seconds + EPSILON
            )
        ]
        kept.append(candidate)
    return tuple(kept)


def build_candidate_options(boxes, drone_types, profile, energy_config):
    masses, volumes, counts = _subset_totals(boxes)
    options = {}
    for mask in range(1, 1 << len(boxes)):
        feasible = []
        for drone_type in drone_types:
            option = evaluate_trip(
                mask,
                masses[mask],
                volumes[mask],
                counts[mask],
                drone_type,
                profile,
                energy_config,
            )
            if option is not None:
                feasible.append(option)
        if feasible:
            options[mask] = _pareto_trip_options(feasible)
    for index, box in enumerate(boxes):
        if 1 << index not in options:
            raise RuntimeError("no feasible drone type for cargo {}".format(box["货箱编号"]))
    return options


def _pareto_states(states):
    if not states:
        return tuple()
    minimum_trips = min(state.trip_count for state in states)
    candidates = [state for state in states if state.trip_count == minimum_trips]
    candidates.sort(
        key=lambda state: (
            state.energy_kwh,
            state.work_seconds,
            tuple((trip.mask, trip.type_id) for trip in state.trips),
        )
    )
    kept = []
    best_time = float("inf")
    for state in candidates:
        if state.work_seconds >= best_time - EPSILON:
            continue
        kept.append(state)
        best_time = state.work_seconds
    return tuple(kept)


def pareto_partition_mask(target_mask, candidate_options):
    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return (PlanState(0, 0.0, 0.0, tuple()),)
        anchor = remaining & -remaining
        states = []
        subset = remaining
        while subset:
            if subset & anchor and subset in candidate_options:
                tails = solve(remaining ^ subset)
                for option in candidate_options[subset]:
                    for tail in tails:
                        states.append(
                            PlanState(
                                1 + tail.trip_count,
                                option.energy_kwh + tail.energy_kwh,
                                option.work_seconds + tail.work_seconds,
                                (option,) + tail.trips,
                            )
                        )
            subset = (subset - 1) & remaining
        return _pareto_states(states)

    result = solve(target_mask)
    if not result:
        raise RuntimeError("no feasible partition")
    return result


def exact_pareto_partition(boxes, candidate_options):
    return pareto_partition_mask((1 << len(boxes)) - 1, candidate_options)


def _state_from_trips(trips):
    trips = tuple(trips)
    return PlanState(
        len(trips),
        sum(item.energy_kwh for item in trips),
        sum(item.work_seconds for item in trips),
        tuple(sorted(trips, key=lambda item: (item.mask, item.type_id))),
    )


def _strictly_better(candidate, incumbent):
    if candidate.trip_count != incumbent.trip_count:
        return candidate.trip_count < incumbent.trip_count
    if candidate.energy_kwh < incumbent.energy_kwh - EPSILON:
        return True
    if abs(candidate.energy_kwh - incumbent.energy_kwh) <= EPSILON:
        return candidate.work_seconds < incumbent.work_seconds - EPSILON
    return False


def local_search_from_baseline(baseline, candidate_options):
    """FFD-seeded exact neighborhood search using required Q1 operators.

    A two-bin neighborhood implements moves, swaps, merges and type changes.
    A three-bin neighborhood is used when the pair neighborhood is locally
    optimal. Every replacement is re-solved under the common hard constraints.
    """

    current = _state_from_trips(
        min(candidate_options[trip.mask], key=lambda item: (item.energy_kwh, item.work_seconds, item.type_id))
        for trip in baseline.trips
    )
    iterations = 0
    type_replacements = sum(
        original.type_id != updated.type_id
        for original, updated in zip(
            sorted(baseline.trips, key=lambda item: item.mask),
            sorted(current.trips, key=lambda item: item.mask),
        )
    )
    operator_counts = {
        "type_replacement": type_replacements,
        "pair_repartition": 0,
        "triple_repartition": 0,
    }
    while True:
        best = current
        best_operator = None
        for neighborhood_size, operator_name in (
            (2, "pair_repartition"),
            (3, "triple_repartition"),
        ):
            if len(current.trips) < neighborhood_size:
                continue
            for indices in combinations(range(len(current.trips)), neighborhood_size):
                union_mask = 0
                selected = set(indices)
                for index in indices:
                    union_mask |= current.trips[index].mask
                replacement = min(
                    pareto_partition_mask(union_mask, candidate_options),
                    key=lambda state: (state.trip_count, state.energy_kwh, state.work_seconds),
                )
                new_trips = [
                    trip for index, trip in enumerate(current.trips) if index not in selected
                ] + list(replacement.trips)
                candidate = _state_from_trips(new_trips)
                if _strictly_better(candidate, best):
                    best = candidate
                    best_operator = operator_name
            if best_operator is not None:
                break
        if best_operator is None:
            break
        current = best
        operator_counts[best_operator] += 1
        iterations += 1
        if iterations > 1000:
            raise RuntimeError("Q1 local search exceeded iteration guard")
    return current, {"iterations": iterations, "operator_counts": operator_counts}


def baseline_ffd(boxes, drone_types, profile, energy_config):
    ordered_types = sorted(
        drone_types,
        key=lambda row: (row["最大载货质量（kg）"], row["可用装载体积（m³）"], row["机型编号"]),
    )
    ordered_indices = sorted(
        range(len(boxes)),
        key=lambda index: (-boxes[index]["单箱质量（kg）"], -boxes[index]["单箱体积（m³）"], boxes[index]["货箱编号"]),
    )
    bins = []
    for index in ordered_indices:
        bit = 1 << index
        placed = False
        for bin_index, current in enumerate(bins):
            mask = current.mask | bit
            members = [boxes[i] for i in range(len(boxes)) if mask & (1 << i)]
            option = None
            for drone_type in ordered_types:
                option = evaluate_trip(
                    mask,
                    sum(row["单箱质量（kg）"] for row in members),
                    sum(row["单箱体积（m³）"] for row in members),
                    len(members),
                    drone_type,
                    profile,
                    energy_config,
                )
                if option is not None:
                    break
            if option is not None:
                bins[bin_index] = option
                placed = True
                break
        if not placed:
            single = None
            for drone_type in ordered_types:
                single = evaluate_trip(
                    bit,
                    boxes[index]["单箱质量（kg）"],
                    boxes[index]["单箱体积（m³）"],
                    1,
                    drone_type,
                    profile,
                    energy_config,
                )
                if single is not None:
                    break
            if single is None:
                raise RuntimeError("baseline cannot assign {}".format(boxes[index]["货箱编号"]))
            bins.append(single)
    return PlanState(
        len(bins),
        sum(item.energy_kwh for item in bins),
        sum(item.work_seconds for item in bins),
        tuple(bins),
    )


def _combine_site_fronts(site_fronts):
    states = [(0.0, 0.0, {})]
    for service_id in sorted(site_fronts):
        candidates = []
        for energy, work, plans in states:
            for site_state in site_fronts[service_id]:
                updated = dict(plans)
                updated[service_id] = site_state
                candidates.append((energy + site_state.energy_kwh, work + site_state.work_seconds, updated))
        candidates.sort(key=lambda item: (item[0], item[1]))
        states = []
        best_work = float("inf")
        for candidate in candidates:
            if candidate[1] >= best_work - EPSILON:
                continue
            states.append(candidate)
            best_work = candidate[1]
    return states


def solve_q1(project_root):
    project_root = Path(project_root).resolve()
    data = load_project_data(project_root)
    model_config = load_model_config(project_root)
    energy_config = model_config["energy_model"]
    center = data["nodes"]["centers"][0]
    drone_types = sorted(data["transport"]["types"], key=lambda row: row["机型编号"])
    dem = RasterDEM(data["paths"]["dem_tif"])

    profiles = {}
    safe_payload_rows = []
    sensitivity_rows = []
    for service in data["nodes"]["service_areas"]:
        service_id = service["服务区编号"]
        profile = segment_terrain_profile(
            dem, center, service, origin_work_offset_m=0.0, destination_work_offset_m=30.0
        )
        profiles[service_id] = profile
        for drone_type in drone_types:
            standard_reserve = drone_type["返航电量下限（%）"] / 100.0
            safe = maximum_safe_payload_kg(drone_type, profile, energy_config, standard_reserve)
            safe_payload_rows.append(
                {
                    "服务区编号": service_id,
                    "机型编号": drone_type["机型编号"],
                    "返航余量": standard_reserve,
                    "最大安全载荷（kg）": safe,
                }
            )
            for reserve in (0.10, 0.15, 0.20, 0.25, 0.30):
                sensitivity_rows.append(
                    {
                        "服务区编号": service_id,
                        "机型编号": drone_type["机型编号"],
                        "返航余量": reserve,
                        "最大安全载荷（kg）": maximum_safe_payload_kg(
                            drone_type, profile, energy_config, reserve
                        ),
                    }
                )

    boxes_by_service = {}
    for box in data["demands"]["boxes"]:
        boxes_by_service.setdefault(box["服务区编号"], []).append(box)
    for boxes in boxes_by_service.values():
        boxes.sort(key=lambda row: row["货箱编号"])

    baseline_by_service = {}
    local_by_service = {}
    local_search_logs = {}
    front_by_service = {}
    for service_id in sorted(boxes_by_service):
        boxes = boxes_by_service[service_id]
        profile = profiles[service_id]
        baseline_by_service[service_id] = baseline_ffd(
            boxes, drone_types, profile, energy_config
        )
        candidates = build_candidate_options(boxes, drone_types, profile, energy_config)
        front_by_service[service_id] = exact_pareto_partition(boxes, candidates)
        local_by_service[service_id], local_search_logs[service_id] = local_search_from_baseline(
            baseline_by_service[service_id], candidates
        )
        exact_best = min(
            front_by_service[service_id],
            key=lambda state: (state.trip_count, state.energy_kwh, state.work_seconds),
        )
        local_best = local_by_service[service_id]
        if local_best.trip_count != exact_best.trip_count:
            raise AssertionError(
                "local search trip-count gap at {}: {} vs exact {}".format(
                    service_id, local_best.trip_count, exact_best.trip_count
                )
            )

    global_front = _combine_site_fronts(front_by_service)
    exact_energy, exact_work, exact_plans = global_front[0]
    representative_plans = local_by_service
    representative_energy = sum(state.energy_kwh for state in representative_plans.values())
    representative_work = sum(state.work_seconds for state in representative_plans.values())
    baseline_trip_count = sum(state.trip_count for state in baseline_by_service.values())
    main_trip_count = sum(state.trip_count for state in representative_plans.values())
    if main_trip_count > baseline_trip_count:
        raise AssertionError("exact plan uses more trips than baseline")

    manifest_path = project_root / "results" / "00_audit" / "input_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run P0 audit before Q1: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "data": data,
        "model_config": model_config,
        "boxes_by_service": boxes_by_service,
        "profiles": profiles,
        "safe_payload_rows": safe_payload_rows,
        "sensitivity_rows": sensitivity_rows,
        "baseline_by_service": baseline_by_service,
        "local_by_service": local_by_service,
        "local_search_logs": local_search_logs,
        "front_by_service": front_by_service,
        "global_front": global_front,
        "representative_plans": representative_plans,
        "summary": {
            "status": "PASS",
            "solution_version": "S1-v1.0.0-provisional_v0",
            "parent_version": "P1-common-physics-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "deterministic": True,
            "random_seed": None,
            "input_manifest_sha256": manifest["manifest_sha256"],
            "model_config_sha256": stable_json_hash(model_config),
            "solver": "ffd_seeded_pair_triple_local_search_with_exact_certificate",
            "energy_model_version": energy_config["version"],
            "energy_config_sha256": energy_config_hash(model_config),
            "q1_work_time_version": model_config["q1_cumulative_work_time"]["version"],
            "baseline_trip_count": baseline_trip_count,
            "baseline_total_energy_kwh": sum(state.energy_kwh for state in baseline_by_service.values()),
            "baseline_cumulative_work_seconds": sum(state.work_seconds for state in baseline_by_service.values()),
            "main_trip_count": main_trip_count,
            "main_total_energy_kwh": representative_energy,
            "main_cumulative_work_seconds": representative_work,
            "exact_certificate_total_energy_kwh": exact_energy,
            "exact_certificate_cumulative_work_seconds": exact_work,
            "exact_certificate_trip_count": sum(state.trip_count for state in exact_plans.values()),
            "local_search_trip_count_gap": main_trip_count
            - sum(state.trip_count for state in exact_plans.values()),
            "local_search_iterations": sum(
                item["iterations"] for item in local_search_logs.values()
            ),
            "local_search_operator_counts": {
                name: sum(log["operator_counts"][name] for log in local_search_logs.values())
                for name in ("type_replacement", "pair_repartition", "triple_repartition")
            },
            "global_pareto_points": len(global_front),
        },
    }


def _trip_rows(plans_by_service, boxes_by_service, prefix):
    rows = []
    sequence = 1
    for service_id in sorted(plans_by_service):
        boxes = boxes_by_service[service_id]
        state = plans_by_service[service_id]
        ordered_trips = sorted(
            state.trips,
            key=lambda trip: tuple(
                boxes[index]["货箱编号"]
                for index in range(len(boxes))
                if trip.mask & (1 << index)
            ),
        )
        for trip in ordered_trips:
            cargo_ids = [
                boxes[index]["货箱编号"]
                for index in range(len(boxes))
                if trip.mask & (1 << index)
            ]
            rows.append(
                {
                    "架次编号": "{}-{:03d}".format(prefix, sequence),
                    "服务区编号": service_id,
                    "机型编号": trip.type_id,
                    "货箱编号列表": ";".join(cargo_ids),
                    "总质量（kg）": trip.mass_kg,
                    "总体积（m³）": trip.volume_m3,
                    "往返时间（s）": trip.flight_seconds,
                    "累计作业时间（s）": trip.work_seconds,
                    "架次能耗（kWh）": trip.energy_kwh,
                    "返航SOC（%）": trip.remaining_soc_fraction * 100.0,
                }
            )
            sequence += 1
    return rows


def validate_q1_rows(rows, boxes_by_service, transport_types):
    expected = {
        box["货箱编号"]: box
        for boxes in boxes_by_service.values()
        for box in boxes
    }
    seen = {}
    types = {row["机型编号"]: row for row in transport_types}
    violations = []
    for row in rows:
        cargo_ids = row["货箱编号列表"].split(";")
        drone_type = types[row["机型编号"]]
        if row["总质量（kg）"] > drone_type["最大载货质量（kg）"] + EPSILON:
            violations.append("{} mass".format(row["架次编号"]))
        if row["总体积（m³）"] > drone_type["可用装载体积（m³）"] + EPSILON:
            violations.append("{} volume".format(row["架次编号"]))
        if row["返航SOC（%）"] + EPSILON < drone_type["返航电量下限（%）"]:
            violations.append("{} reserve".format(row["架次编号"]))
        for cargo_id in cargo_ids:
            if cargo_id in seen:
                violations.append("duplicate {}".format(cargo_id))
            seen[cargo_id] = row["架次编号"]
            if cargo_id not in expected:
                violations.append("unknown {}".format(cargo_id))
            elif expected[cargo_id]["服务区编号"] != row["服务区编号"]:
                violations.append("wrong service {}".format(cargo_id))
    missing = sorted(set(expected) - set(seen))
    if missing:
        violations.append("missing {}".format(",".join(missing)))
    if violations:
        raise AssertionError("; ".join(violations))
    return {"status": "PASS", "violations": 0, "cargo_boxes": len(seen), "trips": len(rows)}


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_q1_results(project_root, output_dir="results/q1"):
    project_root = Path(project_root).resolve()
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    solution = solve_q1(project_root)
    data = solution["data"]
    baseline_rows = _trip_rows(
        solution["baseline_by_service"], solution["boxes_by_service"], "Q1B"
    )
    main_rows = _trip_rows(
        solution["representative_plans"], solution["boxes_by_service"], "Q1"
    )
    baseline_audit = validate_q1_rows(
        baseline_rows, solution["boxes_by_service"], data["transport"]["types"]
    )
    main_audit = validate_q1_rows(
        main_rows, solution["boxes_by_service"], data["transport"]["types"]
    )
    summary = dict(solution["summary"])
    summary["baseline_constraint_audit"] = baseline_audit
    summary["main_constraint_audit"] = main_audit

    pareto_rows = [
        {
            "序号": index + 1,
            "总架次数": sum(state.trip_count for state in plans.values()),
            "总能耗（kWh）": energy,
            "累计作业时间（s）": work,
            "是否选用": index == 0,
        }
        for index, (energy, work, plans) in enumerate(solution["global_front"])
    ]
    _write_csv(output_dir / "safe_payload_matrix.csv", solution["safe_payload_rows"])
    _write_csv(output_dir / "safe_payload_sensitivity.csv", solution["sensitivity_rows"])
    _write_csv(output_dir / "baseline_trips.csv", baseline_rows)
    _write_csv(output_dir / "main_trips.csv", main_rows)
    _write_csv(output_dir / "pareto_front.csv", pareto_rows)
    (output_dir / "q1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from dproblem.compat import import_openpyxl_load_workbook

    load_workbook = import_openpyxl_load_workbook()
    workbook = load_workbook(data["paths"]["submission_template"])
    sheet = workbook["Q1_单点组批"]
    template_headers = [cell.value for cell in sheet[1] if cell.value is not None]
    for row_index, row in enumerate(main_rows, start=2):
        for column_index, header in enumerate(template_headers, start=1):
            sheet.cell(row=row_index, column=column_index, value=row[header])
    workbook.save(output_dir / "结果提交_Q1.xlsx")
    workbook.close()
    return summary
