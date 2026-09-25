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
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/q3_candidate_probe")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))

    from dproblem.domain.communication import CommunicationModel
    from dproblem.domain.geometry import RasterDEM
    from dproblem.q3.direct import extract_conservative_gaps, sample_direct_links
    from dproblem.q3.relay import (
        candidate_link_series,
        conservative_cover_intervals,
        generate_candidates,
        relay_travel_profile,
    )

    samples = sample_direct_links(root, args.step_seconds)
    gaps = extract_conservative_gaps(samples)
    from dproblem.io.dataset import load_project_data

    data = load_project_data(root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    model = CommunicationModel(data["communication"])
    center = data["nodes"]["centers"][0]
    gateway = {
        "lon": center["经度（°）"],
        "lat": center["纬度（°）"],
        "alt_m": center["海拔（m）"] + model.gateway_agl_m,
    }
    relay_type = data["relay"]["types"][0]
    candidates = generate_candidates(dem, data, samples, gaps)
    gap_samples = {}
    for gap in gaps:
        gap_samples[gap["gap_id"]] = [
            row
            for row in samples
            if row["trip_id"] == gap["trip_id"]
            and gap["start_seconds"] - 1e-8 <= row["time_seconds"] <= gap["end_seconds"] + 1e-8
        ]
    candidate_rows = []
    coverage_rows = []
    interval_rows = []
    for candidate in candidates:
        travel = relay_travel_profile(
            dem, center, candidate, relay_type, {"gravity_m_s2": 9.80665}
        )
        candidate_rows.append(dict(candidate, **travel))
        for gap in gaps:
            backhaul, access_rows = candidate_link_series(
                dem, model, candidate, gateway, gap_samples[gap["gap_id"]]
            )
            evidence = {
                "covered": bool(access_rows)
                and all(row["available"] for row in access_rows),
                "backhaul_available": backhaul["available"],
                "backhaul_margin_db": backhaul["margin_db"],
                "minimum_access_margin_db": (
                    min(row["margin_db"] for row in access_rows)
                    if access_rows
                    else None
                ),
                "failed_access_samples": (
                    sum(not row["available"] for row in access_rows)
                    if access_rows
                    else len(gap_samples[gap["gap_id"]])
                ),
            }
            coverage_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "gap_id": gap["gap_id"],
                    **evidence,
                }
            )
            if backhaul["available"]:
                for interval_index, interval in enumerate(
                    conservative_cover_intervals(access_rows), start=1
                ):
                    interval_rows.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "gap_id": gap["gap_id"],
                            "interval_index": interval_index,
                            "backhaul_margin_db": backhaul["margin_db"],
                            **interval,
                        }
                    )
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidates.csv", candidate_rows)
    write_csv(output / "coverage.csv", coverage_rows)
    write_csv(output / "coverage_intervals.csv", interval_rows)
    uncovered = [
        gap["gap_id"]
        for gap in gaps
        if not any(
            row["gap_id"] == gap["gap_id"] and row["covered"]
            for row in coverage_rows
        )
    ]
    summary = {
        "status": "PASS" if not uncovered else "FAIL",
        "step_seconds": args.step_seconds,
        "candidate_count": len(candidates),
        "gap_count": len(gaps),
        "fully_coverable_gap_count": len(gaps) - len(uncovered),
        "uncovered_gap_ids": uncovered,
        "candidate_gap_cover_pairs": sum(row["covered"] for row in coverage_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
