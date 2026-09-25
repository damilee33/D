import argparse
import bisect
import csv
import heapq
import json
import sys
from itertools import combinations
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def floats(rows, fields):
    for row in rows:
        for field in fields:
            row[field] = float(row[field])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--candidate-limit", type=int, default=36)
    parser.add_argument(
        "--minimum-margin-db",
        type=float,
        default=0.0,
        help="Require each retained candidate coverage interval (and recorded backhaul) to meet this margin.",
    )
    parser.add_argument("--shift-trip", default=None)
    parser.add_argument("--shift-seconds", type=float, default=0.0)
    parser.add_argument(
        "--shifts",
        default="",
        help="Comma-separated trip=seconds adjustments used for Q3 probes.",
    )
    parser.add_argument("--output", default="results/q3_relay_path_plan.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.domain.geometry import RasterDEM
    from dproblem.io.dataset import load_project_data
    from dproblem.q3.cover import build_gap_cells, candidate_masks
    from dproblem.q3.relay import relay_travel_profile

    samples = floats(read_csv(root / "results/q3_direct_5s/direct_link_samples.csv"), ("time_seconds",))
    gaps = floats(read_csv(root / "results/q3_direct_5s/direct_gaps.csv"), ("start_seconds", "end_seconds"))
    interval_paths = [
        root / "results/q3_candidate_probe_5s/coverage_intervals.csv",
        root / "results/q3_grid_refinement/refined_coverage_intervals.csv",
        root / "results/q3_multicover_refinement/multicover_coverage_intervals.csv",
    ]
    intervals = []
    for path in interval_paths:
        rows = floats(read_csv(path), ("start_seconds", "end_seconds"))
        intervals.extend(
            row
            for row in rows
            if float(row["minimum_access_margin_db"]) + 1e-12
            >= args.minimum_margin_db
            and (
                not row.get("backhaul_margin_db")
                or float(row["backhaul_margin_db"]) + 1e-12
                >= args.minimum_margin_db
            )
        )
    shifts = {}
    if args.shift_trip:
        shifts[args.shift_trip] = args.shift_seconds
    if args.shifts:
        for item in args.shifts.split(","):
            trip_id, seconds = item.split("=", 1)
            shifts[trip_id] = float(seconds)
    for trip_id, shift_seconds in shifts.items():
        shifted_gap_ids = {
            row["gap_id"] for row in gaps if row["trip_id"] == trip_id
        }
        for row in samples:
            if row["trip_id"] == trip_id:
                row["time_seconds"] += shift_seconds
        for row in gaps:
            if row["trip_id"] == trip_id:
                row["start_seconds"] += shift_seconds
                row["end_seconds"] += shift_seconds
        for row in intervals:
            if row["gap_id"] in shifted_gap_ids:
                row["start_seconds"] += shift_seconds
                row["end_seconds"] += shift_seconds
    candidate_paths = [
        root / "results/q3_candidate_probe_5s/candidates.csv",
        root / "results/q3_grid_refinement/refined_candidates.csv",
        root / "results/q3_multicover_refinement/multicover_candidates.csv",
    ]
    candidate_rows = {}
    for path in candidate_paths:
        for row in read_csv(path):
            for field in ("lon", "lat", "alt_m", "height_agl_m", "ground_elevation_m"):
                row[field] = float(row[field])
            candidate_rows[row["candidate_id"]] = row

    cells = build_gap_cells(samples, gaps)
    all_masks = candidate_masks(cells, intervals)
    required_full = (1 << len(cells)) - 1
    sequence = json.loads((root / "results/q3_pair_sequence.json").read_text(encoding="utf-8"))
    mandatory = {
        candidate_id
        for segment in sequence["segments"]
        for candidate_id in segment["candidate_ids"]
    }
    mandatory.update(
        {
            "C004-H300",
            "C011-H300",
            "C012-H300",
            "C013-H300",
            "C033-H300",
            "C039-H300",
        }
    )
    ranked = sorted(
        all_masks,
        key=lambda candidate_id: bin(all_masks[candidate_id] & required_full).count("1"),
        reverse=True,
    )
    selected = []
    for candidate_id in list(sorted(mandatory)) + ranked:
        if candidate_id in all_masks and candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= args.candidate_limit:
            break
    masks = {candidate_id: all_masks[candidate_id] for candidate_id in selected}

    boundaries = sorted(
        {value for cell in cells for value in (cell["start_seconds"], cell["end_seconds"])}
    )
    full_boundaries = [boundaries[0]]
    for left, right in zip(boundaries, boundaries[1:]):
        full_boundaries.append(right)
    required_by_interval = []
    for left, right in zip(full_boundaries, full_boundaries[1:]):
        midpoint = 0.5 * (left + right)
        required_by_interval.append(
            sum(
                1 << index
                for index, cell in enumerate(cells)
                if cell["start_seconds"] <= midpoint <= cell["end_seconds"]
            )
        )
    data = load_project_data(root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    relay_type = data["relay"]["types"][0]
    energy_config = {"gravity_m_s2": 9.80665}
    # Each row in the official Q3 submission template is one relay sortie with
    # one fixed hover point.  A change of hover point therefore requires the
    # current sortie to return to O01 and a new sortie to complete turnaround,
    # preparation, outbound flight, and link establishment.
    base_profiles = {
        candidate_id: relay_travel_profile(
            dem,
            data["nodes"]["centers"][0],
            candidate_rows[candidate_id],
            relay_type,
            energy_config,
        )
        for candidate_id in selected
    }
    travel = {}
    for origin in selected:
        for destination in selected:
            if origin == destination:
                continue
            travel[(origin, destination)] = (
                base_profiles[origin]["inbound_seconds"]
                + float(relay_type["架次周转时间（s）"])
                + float(relay_type["工位固定准备时间（s）"])
                + base_profiles[destination]["outbound_seconds"]
                + float(relay_type["建链时间（s）"])
            )

    pairs = [tuple(sorted(pair)) for pair in combinations(selected, 2)]
    initial_pairs = []
    first_time = full_boundaries[0]
    for pair in pairs:
        ready = []
        for candidate_id in pair:
            profile = base_profiles[candidate_id]
            ready.append(
                float(relay_type["工位固定准备时间（s）"])
                + profile["outbound_seconds"]
                + float(relay_type["建链时间（s）"])
            )
        if max(ready) <= first_time + 1e-7:
            initial_pairs.append(pair)

    count = len(required_by_interval)

    def pair_covers(pair, interval_index):
        required = required_by_interval[interval_index]
        return ((masks[pair[0]] | masks[pair[1]]) & required) == required

    queue = []
    previous = {}
    distance = {}
    for pair in initial_pairs:
        if pair_covers(pair, 0):
            state = (0, pair)
            distance[state] = 0
            previous[state] = None
            heapq.heappush(queue, (0, 0, pair))
    goal = None
    while queue:
        cost, index, pair = heapq.heappop(queue)
        state = (index, pair)
        if distance.get(state) != cost:
            continue
        if index >= count:
            goal = state
            break
        if pair_covers(pair, index):
            next_state = (index + 1, pair)
            if cost < distance.get(next_state, 10 ** 9):
                distance[next_state] = cost
                previous[next_state] = (state, {"action": "wait"})
                heapq.heappush(queue, (cost, index + 1, pair))
        start_time = full_boundaries[index]
        for moving_index in (0, 1):
            stationary = pair[1 - moving_index]
            origin = pair[moving_index]
            for destination in selected:
                if destination in pair:
                    continue
                duration = travel[(origin, destination)]
                arrival_time = start_time + duration
                arrival_index = bisect.bisect_left(full_boundaries, arrival_time - 1e-9)
                arrival_index = min(arrival_index, count)
                stationary_mask = masks[stationary]
                if any(
                    stationary_mask & required_by_interval[scan]
                    != required_by_interval[scan]
                    for scan in range(index, arrival_index)
                ):
                    continue
                new_pair = tuple(sorted((stationary, destination)))
                if arrival_index < count and not pair_covers(new_pair, arrival_index):
                    continue
                next_state = (arrival_index, new_pair)
                next_cost = cost + 1
                if next_cost < distance.get(next_state, 10 ** 9):
                    distance[next_state] = next_cost
                    previous[next_state] = (
                        state,
                        {
                            "action": "relocate",
                            "origin": origin,
                            "destination": destination,
                            "stationary": stationary,
                            "start_seconds": start_time,
                            "arrival_seconds": arrival_time,
                        },
                    )
                    heapq.heappush(queue, (next_cost, arrival_index, new_pair))
        if required_by_interval[index] == 0:
            next_required_index = index
            while (
                next_required_index < count
                and required_by_interval[next_required_index] == 0
            ):
                next_required_index += 1
            gap_end = full_boundaries[next_required_index]
            for destination_pair in pairs:
                if destination_pair == pair:
                    continue
                first_duration = max(
                    0.0 if pair[0] == destination_pair[0] else travel[(pair[0], destination_pair[0])],
                    0.0 if pair[1] == destination_pair[1] else travel[(pair[1], destination_pair[1])],
                )
                second_duration = max(
                    0.0 if pair[0] == destination_pair[1] else travel[(pair[0], destination_pair[1])],
                    0.0 if pair[1] == destination_pair[0] else travel[(pair[1], destination_pair[0])],
                )
                duration = min(first_duration, second_duration)
                arrival_time = start_time + duration
                if arrival_time > gap_end + 1e-7:
                    continue
                arrival_index = bisect.bisect_left(full_boundaries, arrival_time - 1e-9)
                arrival_index = min(arrival_index, count)
                next_state = (arrival_index, destination_pair)
                next_cost = cost + 2
                if next_cost < distance.get(next_state, 10 ** 9):
                    distance[next_state] = next_cost
                    previous[next_state] = (
                        state,
                        {
                            "action": "dual_relocate",
                            "origin_pair": list(pair),
                            "destination_pair": list(destination_pair),
                            "start_seconds": start_time,
                            "arrival_seconds": arrival_time,
                        },
                    )
                    heapq.heappush(
                        queue, (next_cost, arrival_index, destination_pair)
                    )
    if goal is None:
        maximum_index = max(state[0] for state in distance)

        def active_gap_ids(interval_index):
            if interval_index < 0 or interval_index >= len(required_by_interval):
                return []
            required = required_by_interval[interval_index]
            return sorted(
                {
                    cell["gap_id"]
                    for cell_index, cell in enumerate(cells)
                    if required & (1 << cell_index)
                }
            )

        reachable_pairs = sorted(
            {state[1] for state in distance if state[0] == maximum_index}
        )
        covering_pairs_after = (
            [pair for pair in pairs if pair_covers(pair, maximum_index)]
            if maximum_index < count
            else []
        )

        diagnostic_actions = []
        diagnostic_state = next(
            state for state in distance if state[0] == maximum_index
        )
        diagnostic_cursor = diagnostic_state
        while previous[diagnostic_cursor] is not None:
            parent, action = previous[diagnostic_cursor]
            if action["action"] != "wait":
                diagnostic_actions.append(action)
            diagnostic_cursor = parent
        diagnostic_actions.reverse()

        def candidate_coverage_by_gap(interval_index):
            if interval_index < 0 or interval_index >= len(required_by_interval):
                return {}
            required = required_by_interval[interval_index]
            result = {}
            for cell_index, cell in enumerate(cells):
                bit = 1 << cell_index
                if not (required & bit):
                    continue
                result[cell["gap_id"]] = sorted(
                    candidate_id
                    for candidate_id, mask in masks.items()
                    if mask & bit
                )
            return result
        summary = {
            "status": "FAIL",
            "candidate_count": len(selected),
            "reachable_state_count": len(distance),
            "maximum_reached_seconds": full_boundaries[maximum_index],
            "active_gap_ids_before_failure": active_gap_ids(maximum_index - 1),
            "active_gap_ids_after_failure": active_gap_ids(maximum_index),
            "reachable_pair_count_at_failure": len(reachable_pairs),
            "reachable_pairs_at_failure": [list(pair) for pair in reachable_pairs[:20]],
            "diagnostic_initial_pair": list(diagnostic_cursor[1]),
            "diagnostic_actions_to_failure": diagnostic_actions,
            "covering_pair_count_after_failure": len(covering_pairs_after),
            "covering_pairs_after_failure": [
                list(pair) for pair in covering_pairs_after[:20]
            ],
            "candidate_coverage_before_failure": candidate_coverage_by_gap(maximum_index - 1),
            "candidate_coverage_after_failure": candidate_coverage_by_gap(maximum_index),
        }
    else:
        actions = []
        cursor = goal
        while previous[cursor] is not None:
            parent, action = previous[cursor]
            if action["action"] != "wait":
                actions.append(action)
            cursor = parent
        actions.reverse()
        summary = {
            "status": "PASS",
            "candidate_count": len(selected),
            "relocation_count": distance[goal],
            "initial_pair": list(cursor[1]),
            "actions": actions,
            "final_pair": list(goal[1]),
            "coverage_end_seconds": full_boundaries[-1],
        }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
