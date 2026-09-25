import argparse
import csv
import json
import sys
from itertools import combinations_with_replacement
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
    parser.add_argument("--output", default="results/q3_pair_sequence.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.q3.cover import build_gap_cells, candidate_masks

    samples = floats(read_csv(root / "results/q3_direct_5s/direct_link_samples.csv"), ("time_seconds",))
    gaps = floats(read_csv(root / "results/q3_direct_5s/direct_gaps.csv"), ("start_seconds", "end_seconds"))
    interval_paths = [
        root / "results/q3_candidate_probe_5s/coverage_intervals.csv",
        root / "results/q3_grid_refinement/refined_coverage_intervals.csv",
        root / "results/q3_multicover_refinement/multicover_coverage_intervals.csv",
    ]
    intervals = []
    for path in interval_paths:
        intervals.extend(floats(read_csv(path), ("start_seconds", "end_seconds")))
    cells = build_gap_cells(samples, gaps)
    masks = candidate_masks(cells, intervals)
    boundaries = sorted({value for cell in cells for value in (cell["start_seconds"], cell["end_seconds"])})
    atomic = []
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = 0.5 * (start + end)
        indices = [index for index, cell in enumerate(cells) if cell["start_seconds"] <= midpoint <= cell["end_seconds"]]
        if indices:
            atomic.append({"start_seconds": start, "end_seconds": end, "required_mask": sum(1 << index for index in indices)})
    candidates = sorted(masks)
    pairs = []
    for left, right in combinations_with_replacement(candidates, 2):
        if left == right:
            combined = masks[left]
            pair = (left,)
        else:
            combined = masks[left] | masks[right]
            pair = (left, right)
        pairs.append((pair, combined))

    sequence = []
    index = 0
    while index < len(atomic):
        required = atomic[index]["required_mask"]
        feasible = [(pair, mask) for pair, mask in pairs if mask & required == required]
        if not feasible:
            raise RuntimeError("no relay pair covers atomic interval {}".format(index))
        best = None
        for pair, mask in feasible:
            horizon = index
            while horizon + 1 < len(atomic):
                next_row = atomic[horizon + 1]
                if next_row["start_seconds"] > atomic[horizon]["end_seconds"] + 1e-7:
                    break
                if mask & next_row["required_mask"] != next_row["required_mask"]:
                    break
                horizon += 1
            score = (horizon, -len(pair), pair)
            if best is None or score > best[0]:
                best = (score, pair, mask, horizon)
        _, pair, _, horizon = best
        sequence.append(
            {
                "segment_index": len(sequence) + 1,
                "start_seconds": atomic[index]["start_seconds"],
                "end_seconds": atomic[horizon]["end_seconds"],
                "candidate_ids": list(pair),
                "duration_seconds": atomic[horizon]["end_seconds"] - atomic[index]["start_seconds"],
            }
        )
        index = horizon + 1
    summary = {
        "status": "PASS",
        "segment_count": len(sequence),
        "segments": sequence,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
