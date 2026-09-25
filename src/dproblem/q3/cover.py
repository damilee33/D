"""Discrete time-cell coverage utilities for relay candidate selection."""

from itertools import combinations


TIME_EPSILON = 1e-7


def build_gap_cells(samples, gaps):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["trip_id"], []).append(sample)
    cells = []
    for gap in gaps:
        rows = sorted(grouped[gap["trip_id"]], key=lambda row: row["time_seconds"])
        rows = [
            row
            for row in rows
            if gap["start_seconds"] - TIME_EPSILON
            <= row["time_seconds"]
            <= gap["end_seconds"] + TIME_EPSILON
        ]
        for index, (left, right) in enumerate(zip(rows, rows[1:]), start=1):
            cells.append(
                {
                    "cell_id": "{}-T{:04d}".format(gap["gap_id"], index),
                    "gap_id": gap["gap_id"],
                    "trip_id": gap["trip_id"],
                    "start_seconds": left["time_seconds"],
                    "end_seconds": right["time_seconds"],
                }
            )
    return cells


def candidate_masks(cells, coverage_intervals):
    by_candidate = {}
    for row in coverage_intervals:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    masks = {}
    for candidate_id, intervals in by_candidate.items():
        mask = 0
        by_gap = {}
        for interval in intervals:
            by_gap.setdefault(interval["gap_id"], []).append(interval)
        for index, cell in enumerate(cells):
            if any(
                interval["start_seconds"] <= cell["start_seconds"] + TIME_EPSILON
                and interval["end_seconds"] >= cell["end_seconds"] - TIME_EPSILON
                for interval in by_gap.get(cell["gap_id"], [])
            ):
                mask |= 1 << index
        if mask:
            masks[candidate_id] = mask
    return masks


def temporal_components(gaps):
    ordered = sorted(gaps, key=lambda row: (row["start_seconds"], row["end_seconds"]))
    components = []
    current = []
    current_end = None
    for gap in ordered:
        if current and gap["start_seconds"] > current_end + TIME_EPSILON:
            components.append(current)
            current = []
            current_end = None
        current.append(gap)
        current_end = max(current_end or gap["end_seconds"], gap["end_seconds"])
    if current:
        components.append(current)
    return components


def minimum_candidate_cover(masks, required_mask, maximum_candidates=2):
    eligible = [
        (candidate_id, mask & required_mask)
        for candidate_id, mask in masks.items()
        if mask & required_mask
    ]
    for size in range(1, maximum_candidates + 1):
        feasible = []
        for chosen in combinations(eligible, size):
            covered = 0
            for _, mask in chosen:
                covered |= mask
            if covered == required_mask:
                feasible.append(tuple(candidate_id for candidate_id, _ in chosen))
        if feasible:
            return feasible
    return []
