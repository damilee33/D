import argparse
import csv
import json
import sys
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--radius-cells", type=int, default=80)
    parser.add_argument("--stride-cells", type=int, default=5)
    parser.add_argument("--minimum-margin-db", type=float, default=1.0)
    parser.add_argument("--output", default="results/q3_margin_refinement.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))

    from dproblem.domain.communication import CommunicationModel, audit_link
    from dproblem.domain.geometry import RasterDEM
    from dproblem.io.dataset import load_project_data
    from dproblem.q3.joint import TRIP_SHIFTS_SECONDS, _candidate_rows

    data = load_project_data(root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    model = CommunicationModel(data["communication"])
    center = data["nodes"]["centers"][0]
    gateway = {
        "lon": float(center["经度（°）"]),
        "lat": float(center["纬度（°）"]),
        "alt_m": float(center["海拔（m）"]) + model.gateway_agl_m,
    }
    candidate_rows = _candidate_rows(root)
    sorties = read_csv(root / "results" / "q3" / "relay_sorties.csv")
    target_sortie = next(row for row in sorties if row["中继架次编号"] == "RLY-002")
    service_start = float(target_sortie["建链完成时刻（s）"])
    service_end = float(target_sortie["服务结束时刻（s）"])
    other_sorties = [row for row in sorties if row["中继架次编号"] != "RLY-002"]

    samples = read_csv(root / "results" / "q3_direct_5s" / "direct_link_samples.csv")
    target_samples = []
    for row in samples:
        if str(row["available"]).lower() == "true":
            continue
        time_seconds = float(row["time_seconds"]) + float(
            TRIP_SHIFTS_SECONDS.get(row["trip_id"], 0.0)
        )
        if not (service_start - 1e-7 <= time_seconds <= service_end + 1e-7):
            continue
        position = {
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "alt_m": float(row["alt_m"]),
        }
        robustly_covered_elsewhere = False
        for sortie in other_sorties:
            if not (
                float(sortie["建链完成时刻（s）"]) - 1e-7
                <= time_seconds
                <= float(sortie["服务结束时刻（s）"]) + 1e-7
            ):
                continue
            candidate = candidate_rows[sortie["候选点编号"]]
            relay_position = {
                "lon": candidate["lon"],
                "lat": candidate["lat"],
                "alt_m": candidate["alt_m"],
            }
            backhaul = audit_link(
                dem, model, relay_position, gateway, "relay_backhaul", "gateway"
            )
            if backhaul["margin_db"] < args.minimum_margin_db:
                continue
            access = audit_link(
                dem, model, position, relay_position, "transport", "relay_access"
            )
            if access["margin_db"] >= args.minimum_margin_db:
                robustly_covered_elsewhere = True
                break
        if not robustly_covered_elsewhere:
            target_samples.append(
                {
                    "trip_id": row["trip_id"],
                    "time_seconds": time_seconds,
                    "phase": row["phase"],
                    **position,
                }
            )

    original = candidate_rows[target_sortie["候选点编号"]]
    center_row = int(original["row"])
    center_col = int(original["col"])
    retained = []
    tested = 0
    for raster_row in range(
        max(0, center_row - args.radius_cells),
        min(dem.height - 1, center_row + args.radius_cells) + 1,
        args.stride_cells,
    ):
        for raster_col in range(
            max(0, center_col - args.radius_cells),
            min(dem.width - 1, center_col + args.radius_cells) + 1,
            args.stride_cells,
        ):
            ground = float(dem.values[raster_row, raster_col])
            if dem.nodata is not None and ground == dem.nodata:
                continue
            tested += 1
            candidate = {
                "candidate_id": "MR-R{:04d}C{:04d}H300".format(raster_row, raster_col),
                "row": raster_row,
                "col": raster_col,
                "lon": dem.lon0 + (raster_col + 0.5) * dem.lon_step,
                "lat": dem.lat0 - (raster_row + 0.5) * dem.lat_step,
                "ground_elevation_m": ground,
                "height_agl_m": 300.0,
                "alt_m": ground + 300.0,
            }
            relay_position = {
                "lon": candidate["lon"],
                "lat": candidate["lat"],
                "alt_m": candidate["alt_m"],
            }
            backhaul = audit_link(
                dem, model, relay_position, gateway, "relay_backhaul", "gateway"
            )
            if backhaul["margin_db"] < args.minimum_margin_db:
                continue
            minimum_access = float("inf")
            feasible = True
            for sample in target_samples:
                access = audit_link(
                    dem,
                    model,
                    sample,
                    relay_position,
                    "transport",
                    "relay_access",
                )
                minimum_access = min(minimum_access, access["margin_db"])
                if minimum_access < args.minimum_margin_db:
                    feasible = False
                    break
            if feasible:
                candidate["backhaul_margin_db"] = backhaul["margin_db"]
                candidate["minimum_required_access_margin_db"] = minimum_access
                candidate["required_sample_count"] = len(target_samples)
                retained.append(candidate)

    retained.sort(
        key=lambda row: min(
            row["backhaul_margin_db"], row["minimum_required_access_margin_db"]
        ),
        reverse=True,
    )
    summary = {
        "status": "PASS" if retained else "FAIL",
        "minimum_margin_db": args.minimum_margin_db,
        "target_sample_count": len(target_samples),
        "tested_candidate_count": tested,
        "retained_candidate_count": len(retained),
        "best_candidates": retained[:20],
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
