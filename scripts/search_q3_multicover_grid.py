import argparse
import csv
import json
import sys
from itertools import combinations
from math import floor
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--grid-stride-cells", type=int, default=5)
    parser.add_argument("--expand-cells", type=int, default=20)
    parser.add_argument("--output-dir", default="results/q3_multicover_refinement")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.domain.communication import CommunicationModel, audit_link
    from dproblem.domain.geometry import RasterDEM
    from dproblem.io.dataset import load_project_data
    from dproblem.q3.relay import candidate_link_series, conservative_cover_intervals

    data = load_project_data(root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    model = CommunicationModel(data["communication"])
    center = data["nodes"]["centers"][0]
    gateway = {
        "lon": center["经度（°）"],
        "lat": center["纬度（°）"],
        "alt_m": center["海拔（m）"] + model.gateway_agl_m,
    }
    samples = read_csv(root / "results" / "q3_direct_5s" / "direct_link_samples.csv")
    for row in samples:
        for field in ("time_seconds", "lon", "lat", "alt_m"):
            row[field] = float(row[field])
    gaps = read_csv(root / "results" / "q3_direct_5s" / "direct_gaps.csv")
    for row in gaps:
        row["start_seconds"] = float(row["start_seconds"])
        row["end_seconds"] = float(row["end_seconds"])
    targets = (
        ("early", 1063.0, 1287.0),
        ("middle", 2582.0, 2593.0),
    )
    retained = []
    chosen_pairs = []
    interval_rows = []
    for label, window_start, window_end in targets:
        target_time = 0.5 * (window_start + window_end)
        active_gaps = [
            row
            for row in gaps
            if row["start_seconds"] <= window_start
            and row["end_seconds"] >= window_end
        ]
        endpoints = []
        for gap in active_gaps:
            trip_samples = [row for row in samples if row["trip_id"] == gap["trip_id"]]
            for sample_time in (window_start, target_time, window_end):
                endpoints.append(
                    (
                        gap,
                        min(
                            trip_samples,
                            key=lambda row, sample_time=sample_time: abs(
                                row["time_seconds"] - sample_time
                            ),
                        ),
                    )
                )
        raster_points = [dem.fractional_cell(gateway["lon"], gateway["lat"])]
        raster_points.extend(dem.fractional_cell(row["lon"], row["lat"]) for _, row in endpoints)
        row_low = max(0, int(floor(min(row for row, _ in raster_points))) - args.expand_cells)
        row_high = min(dem.height - 1, int(floor(max(row for row, _ in raster_points))) + args.expand_cells)
        col_low = max(0, int(floor(min(col for _, col in raster_points))) - args.expand_cells)
        col_high = min(dem.width - 1, int(floor(max(col for _, col in raster_points))) + args.expand_cells)
        candidates = []
        full_mask = (1 << len(endpoints)) - 1
        for raster_row in range(row_low, row_high + 1, args.grid_stride_cells):
            for raster_col in range(col_low, col_high + 1, args.grid_stride_cells):
                ground = float(dem.values[raster_row, raster_col])
                if dem.nodata is not None and ground == dem.nodata:
                    continue
                lon = dem.lon0 + (raster_col + 0.5) * dem.lon_step
                lat = dem.lat0 - (raster_row + 0.5) * dem.lat_step
                for height_m in (100.0, 200.0, 300.0):
                    relay = {"lon": lon, "lat": lat, "alt_m": ground + height_m}
                    backhaul = audit_link(
                        dem, model, relay, gateway, "relay_backhaul", "gateway"
                    )
                    if not backhaul["available"]:
                        continue
                    mask = 0
                    margins = [backhaul["margin_db"]]
                    for index, (_, sample) in enumerate(endpoints):
                        access = audit_link(
                            dem,
                            model,
                            {"lon": sample["lon"], "lat": sample["lat"], "alt_m": sample["alt_m"]},
                            relay,
                            "transport",
                            "relay_access",
                        )
                        if access["available"]:
                            mask |= 1 << index
                            margins.append(access["margin_db"])
                    if mask:
                        candidates.append(
                            {
                                "candidate_id": "MC-{}-R{:04d}C{:04d}H{:03d}".format(label, raster_row, raster_col, int(height_m)),
                                "target_label": label,
                                "row": raster_row,
                                "col": raster_col,
                                "lon": lon,
                                "lat": lat,
                                "ground_elevation_m": ground,
                                "height_agl_m": height_m,
                                "alt_m": ground + height_m,
                                "coverage_mask": mask,
                                "representative_minimum_margin_db": min(margins),
                                "representative_backhaul_margin_db": backhaul["margin_db"],
                            }
                        )
        pairs = [
            (left, right)
            for left, right in combinations(candidates, 2)
            if (left["coverage_mask"] | right["coverage_mask"]) == full_mask
        ]
        pairs.sort(
            key=lambda pair: (
                min(pair[0]["representative_minimum_margin_db"], pair[1]["representative_minimum_margin_db"]),
                pair[0]["representative_minimum_margin_db"] + pair[1]["representative_minimum_margin_db"],
            ),
            reverse=True,
        )
        if not pairs:
            chosen_pairs.append({"target_label": label, "status": "FAIL"})
            continue
        chosen = pairs[0]
        chosen_pairs.append(
            {
                "target_label": label,
                "status": "PASS",
                "candidate_1": chosen[0]["candidate_id"],
                "candidate_2": chosen[1]["candidate_id"],
                "active_gap_ids": ";".join(row["gap_id"] for row in active_gaps),
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
            }
        )
        for candidate in chosen:
            retained.append(candidate)
            for gap in gaps:
                gap_samples = [
                    row
                    for row in samples
                    if row["trip_id"] == gap["trip_id"]
                    and gap["start_seconds"] - 1e-8 <= row["time_seconds"] <= gap["end_seconds"] + 1e-8
                ]
                backhaul, series = candidate_link_series(
                    dem, model, candidate, gateway, gap_samples
                )
                if backhaul["available"]:
                    for interval_index, interval in enumerate(
                        conservative_cover_intervals(series), start=1
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
    write_csv(output / "multicover_candidates.csv", retained)
    write_csv(output / "multicover_pairs.csv", chosen_pairs)
    write_csv(output / "multicover_coverage_intervals.csv", interval_rows)
    summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in chosen_pairs) else "FAIL",
        "grid_stride_cells": args.grid_stride_cells,
        "nominal_grid_spacing_m": args.grid_stride_cells * 30,
        "targets": chosen_pairs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
