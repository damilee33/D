import argparse
import csv
import json
import sys
from pathlib import Path


def write_csv(path, rows):
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Replay frozen S2 and audit G01 direct links.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/q3_direct")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.q3.direct import extract_conservative_gaps, sample_direct_links

    samples = sample_direct_links(root, args.step_seconds)
    gaps = extract_conservative_gaps(samples)
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "direct_link_samples.csv", samples)
    write_csv(output / "direct_gaps.csv", gaps)
    summary = {
        "status": "PASS",
        "parent_version": "S2-v1.1.0-provisional_v0",
        "step_seconds": args.step_seconds,
        "sample_count": len(samples),
        "gap_count": len(gaps),
        "trips_with_gaps": len({row["trip_id"] for row in gaps}),
        "total_conservative_gap_seconds": sum(row["duration_seconds"] for row in gaps),
        "minimum_direct_margin_db": min(row["margin_db"] for row in samples),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
