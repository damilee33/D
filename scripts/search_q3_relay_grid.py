import argparse
import csv
import json
import sys
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
    parser.add_argument("--expand-cells", type=int, default=30)
    parser.add_argument("--output-dir", default="results/q3_grid_refinement")
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
        for field in ("time_seconds", "lon", "lat", "alt_m", "margin_db"):
            row[field] = float(row[field])
    direct_gaps = {
        row["gap_id"]: row
        for row in read_csv(root / "results" / "q3_direct_5s" / "direct_gaps.csv")
    }
    holes = [
        ("Q2-013-G01", "Q2-013", 4769.032511133426, 5142.361945337528),
        ("Q2-016-G01", "Q2-016", 6514.75454169773, 6920.724124827612),
    ]
    gateway_row, gateway_col = dem.fractional_cell(gateway["lon"], gateway["lat"])
    all_candidates = []
    interval_rows = []
    for gap_id, trip_id, hole_start, hole_end in holes:
        hole_samples = [
            row
            for row in samples
            if row["trip_id"] == trip_id
            and hole_start - 5.0 <= row["time_seconds"] <= hole_end + 5.0
        ]
        full_start = float(direct_gaps[gap_id]["start_seconds"])
        full_end = float(direct_gaps[gap_id]["end_seconds"])
        gap_samples = [
            row
            for row in samples
            if row["trip_id"] == trip_id
            and full_start - 1e-8 <= row["time_seconds"] <= full_end + 1e-8
        ]
        target_time = 0.5 * (hole_start + hole_end)
        target = min(hole_samples, key=lambda row: abs(row["time_seconds"] - target_time))
        target_row, target_col = dem.fractional_cell(target["lon"], target["lat"])
        row_low = max(0, int(floor(min(gateway_row, target_row))) - args.expand_cells)
        row_high = min(dem.height - 1, int(floor(max(gateway_row, target_row))) + args.expand_cells)
        col_low = max(0, int(floor(min(gateway_col, target_col))) - args.expand_cells)
        col_high = min(dem.width - 1, int(floor(max(gateway_col, target_col))) + args.expand_cells)
        coarse = []
        for raster_row in range(row_low, row_high + 1, args.grid_stride_cells):
            for raster_col in range(col_low, col_high + 1, args.grid_stride_cells):
                ground = float(dem.values[raster_row, raster_col])
                if dem.nodata is not None and ground == dem.nodata:
                    continue
                lon = dem.lon0 + (raster_col + 0.5) * dem.lon_step
                lat = dem.lat0 - (raster_row + 0.5) * dem.lat_step
                for height_m in (100.0, 200.0, 300.0):
                    candidate = {
                        "candidate_id": "{}-R{:04d}C{:04d}H{:03d}".format(
                            gap_id, raster_row, raster_col, int(height_m)
                        ),
                        "gap_id": gap_id,
                        "row": raster_row,
                        "col": raster_col,
                        "lon": lon,
                        "lat": lat,
                        "ground_elevation_m": ground,
                        "height_agl_m": height_m,
                        "alt_m": ground + height_m,
                    }
                    relay = {"lon": lon, "lat": lat, "alt_m": ground + height_m}
                    backhaul = audit_link(
                        dem, model, relay, gateway, "relay_backhaul", "gateway"
                    )
                    if not backhaul["available"]:
                        continue
                    access = audit_link(
                        dem,
                        model,
                        {"lon": target["lon"], "lat": target["lat"], "alt_m": target["alt_m"]},
                        relay,
                        "transport",
                        "relay_access",
                    )
                    if access["available"]:
                        candidate["representative_backhaul_margin_db"] = backhaul["margin_db"]
                        candidate["representative_access_margin_db"] = access["margin_db"]
                        candidate["representative_minimum_margin_db"] = min(
                            backhaul["margin_db"], access["margin_db"]
                        )
                        coarse.append(candidate)
        coarse.sort(
            key=lambda row: row["representative_minimum_margin_db"], reverse=True
        )
        finalists = coarse[:80]
        for candidate in finalists:
            backhaul, series = candidate_link_series(
                dem, model, candidate, gateway, gap_samples
            )
            intervals = conservative_cover_intervals(series)
            candidate["backhaul_margin_db"] = backhaul["margin_db"]
            candidate["covered_hole_seconds"] = sum(
                max(0.0, min(row["end_seconds"], hole_end) - max(row["start_seconds"], hole_start))
                for row in intervals
            )
            all_candidates.append(candidate)
            for index, interval in enumerate(intervals, start=1):
                interval_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "gap_id": gap_id,
                        "interval_index": index,
                        **interval,
                    }
                )
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "refined_candidates.csv", all_candidates)
    write_csv(output / "refined_coverage_intervals.csv", interval_rows)
    summary = {
        "status": "PASS" if all(any(row["gap_id"] == gap_id and row["covered_hole_seconds"] >= end - start - 5.1 for row in all_candidates) for gap_id, _, start, end in holes) else "PARTIAL",
        "grid_stride_cells": args.grid_stride_cells,
        "nominal_grid_spacing_m": args.grid_stride_cells * 30,
        "retained_candidate_count": len(all_candidates),
        "maximum_covered_hole_seconds": {
            gap_id: max((row["covered_hole_seconds"] for row in all_candidates if row["gap_id"] == gap_id), default=0.0)
            for gap_id, _, _, _ in holes
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
