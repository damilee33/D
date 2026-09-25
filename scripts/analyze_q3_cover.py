import argparse
import csv
import json
import sys
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
    parser.add_argument("--output", default="results/q3_cover_analysis.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.q3.cover import (
        build_gap_cells,
        candidate_masks,
        minimum_candidate_cover,
        temporal_components,
    )

    samples = floats(
        read_csv(root / "results/q3_direct_5s" / "direct_link_samples.csv"),
        ("time_seconds",),
    )
    gaps = floats(
        read_csv(root / "results/q3_direct_5s" / "direct_gaps.csv"),
        ("start_seconds", "end_seconds"),
    )
    intervals = floats(
        read_csv(root / "results/q3_candidate_probe_5s" / "coverage_intervals.csv"),
        ("start_seconds", "end_seconds"),
    )
    intervals.extend(
        floats(
            read_csv(root / "results/q3_grid_refinement" / "refined_coverage_intervals.csv"),
            ("start_seconds", "end_seconds"),
        )
    )
    intervals.extend(
        floats(
            read_csv(
                root
                / "results"
                / "q3_multicover_refinement"
                / "multicover_coverage_intervals.csv"
            ),
            ("start_seconds", "end_seconds"),
        )
    )
    cells = build_gap_cells(samples, gaps)
    masks = candidate_masks(cells, intervals)
    rows = []
    for component_index, component in enumerate(temporal_components(gaps), start=1):
        gap_ids = {row["gap_id"] for row in component}
        required_mask = sum(
            1 << index for index, cell in enumerate(cells) if cell["gap_id"] in gap_ids
        )
        covers = minimum_candidate_cover(masks, required_mask, maximum_candidates=2)
        rows.append(
            {
                "component": component_index,
                "start_seconds": min(row["start_seconds"] for row in component),
                "end_seconds": max(row["end_seconds"] for row in component),
                "gap_ids": sorted(gap_ids),
                "cell_count": bin(required_mask).count("1"),
                "minimum_candidate_count": len(covers[0]) if covers else None,
                "feasible_cover_count": len(covers),
                "example_cover": list(covers[0]) if covers else [],
            }
        )
    full_mask = (1 << len(cells)) - 1
    global_covers = minimum_candidate_cover(masks, full_mask, maximum_candidates=2)
    boundaries = sorted(
        {value for cell in cells for value in (cell["start_seconds"], cell["end_seconds"])}
    )
    atomic_rows = []
    cache = {}
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = 0.5 * (start + end)
        active_indices = [
            index
            for index, cell in enumerate(cells)
            if cell["start_seconds"] <= midpoint <= cell["end_seconds"]
        ]
        if not active_indices:
            continue
        required = sum(1 << index for index in active_indices)
        if required not in cache:
            covers = minimum_candidate_cover(masks, required, maximum_candidates=2)
            cache[required] = covers[0] if covers else None
        cover = cache[required]
        atomic_rows.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "active_gap_count": len(active_indices),
                "minimum_candidate_count": len(cover) if cover else None,
                "example_cover": list(cover) if cover else [],
            }
        )
    summary = {
        "cell_count": len(cells),
        "candidate_count_with_coverage": len(masks),
        "components": rows,
        "global_cover_with_at_most_two_fixed_candidates": (
            list(global_covers[0]) if global_covers else None
        ),
        "atomic_time_interval_count": len(atomic_rows),
        "maximum_simultaneous_gap_count": max(row["active_gap_count"] for row in atomic_rows),
        "atomic_intervals_without_two_candidate_cover": [
            row for row in atomic_rows if row["minimum_candidate_count"] is None
        ],
        "maximum_atomic_minimum_candidate_count": max(
            row["minimum_candidate_count"] or 3 for row in atomic_rows
        ),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
