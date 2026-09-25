"""Replay frozen S2 trajectories and extract direct-to-G01 communication gaps."""

import csv
import hashlib
import json
from pathlib import Path

from dproblem.domain.communication import CommunicationModel, audit_link
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data
from dproblem.q3.trajectory import (
    build_flight_phases,
    build_handover_phase,
    sample_phase,
)


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_s2_freeze(project_root):
    project_root = Path(project_root).resolve()
    freeze_path = project_root / "results" / "q2" / "S2_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_PASS":
        raise ValueError("S2 is not frozen with PASS status")
    mismatches = []
    for relative_path, expected in freeze["sha256"].items():
        actual = _sha256(project_root / relative_path)
        if actual != expected:
            mismatches.append(
                {"path": relative_path, "expected": expected, "actual": actual}
            )
    if mismatches:
        raise ValueError("S2 freeze hash mismatch: {}".format(mismatches))
    return freeze


def build_frozen_transport_phases(project_root):
    project_root = Path(project_root).resolve()
    verify_s2_freeze(project_root)
    data = load_project_data(project_root)
    nodes = {"O01": data["nodes"]["centers"][0]}
    nodes.update({row["服务区编号"]: row for row in data["nodes"]["service_areas"]})
    types = {row["机型编号"]: row for row in data["transport"]["types"]}
    trip_rows = {
        row["架次编号"]: row
        for row in _read_csv(project_root / "results" / "q2" / "main_trips.csv")
    }
    segments = {}
    for row in _read_csv(project_root / "results" / "q2" / "main_segments.csv"):
        for field in (
            "爬升（m）",
            "水平距离（m）",
            "下降（m）",
            "开始时刻（s）",
            "结束时刻（s）",
        ):
            row[field] = float(row[field])
        segments.setdefault(row["架次编号"], []).append(row)
    events = {}
    for row in _read_csv(project_root / "results" / "q2" / "main_timeline.csv"):
        row["开始时刻（s）"] = float(row["开始时刻（s）"])
        row["结束时刻（s）"] = float(row["结束时刻（s）"])
        events.setdefault(row["架次编号"], []).append(row)

    result = {}
    for trip_id, trip in trip_rows.items():
        phases = []
        for segment in sorted(segments[trip_id], key=lambda row: int(row["航段序号"])):
            phases.extend(build_flight_phases(segment, types[trip["机型编号"]], nodes))
        for event in events[trip_id]:
            if event["阶段"] == "交接":
                phases.append(build_handover_phase(event, nodes[event["服务区编号"]]))
        phases.sort(key=lambda phase: (phase["start_seconds"], phase["end_seconds"]))
        result[trip_id] = {
            "trip": trip,
            "phases": phases,
        }
    return data, result


def sample_direct_links(project_root, step_seconds):
    data, trips = build_frozen_transport_phases(project_root)
    dem = RasterDEM(data["paths"]["dem_tif"])
    model = CommunicationModel(data["communication"])
    center = data["nodes"]["centers"][0]
    gateway = {
        "lon": float(center["经度（°）"]),
        "lat": float(center["纬度（°）"]),
        "alt_m": float(center["海拔（m）"]) + model.gateway_agl_m,
    }
    rows = []
    for trip_id in sorted(trips):
        by_time = {}
        for phase in trips[trip_id]["phases"]:
            for time_seconds, position in sample_phase(phase, step_seconds):
                link = audit_link(
                    dem,
                    model,
                    position,
                    gateway,
                    "transport",
                    "gateway",
                )
                by_time[round(time_seconds, 9)] = {
                    "trip_id": trip_id,
                    "drone_id": trips[trip_id]["trip"]["无人机编号"],
                    "time_seconds": time_seconds,
                    "phase": phase["phase"],
                    "lon": position["lon"],
                    "lat": position["lat"],
                    "alt_m": position["alt_m"],
                    **link,
                }
        rows.extend(by_time[key] for key in sorted(by_time))
    return rows


def extract_conservative_gaps(samples):
    """Treat a time cell as unavailable if either endpoint is unavailable."""

    grouped = {}
    for row in samples:
        grouped.setdefault(row["trip_id"], []).append(row)
    gaps = []
    for trip_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["time_seconds"])
        current = None
        gap_index = 0
        for left, right in zip(rows, rows[1:]):
            unavailable = not (left["available"] and right["available"])
            if unavailable:
                if current is None:
                    gap_index += 1
                    current = {
                        "gap_id": "{}-G{:02d}".format(trip_id, gap_index),
                        "trip_id": trip_id,
                        "drone_id": left["drone_id"],
                        "start_seconds": left["time_seconds"],
                        "end_seconds": right["time_seconds"],
                        "minimum_direct_margin_db": min(left["margin_db"], right["margin_db"]),
                        "sample_intervals": 1,
                    }
                else:
                    current["end_seconds"] = right["time_seconds"]
                    current["minimum_direct_margin_db"] = min(
                        current["minimum_direct_margin_db"],
                        left["margin_db"],
                        right["margin_db"],
                    )
                    current["sample_intervals"] += 1
            elif current is not None:
                current["duration_seconds"] = (
                    current["end_seconds"] - current["start_seconds"]
                )
                gaps.append(current)
                current = None
        if current is not None:
            current["duration_seconds"] = current["end_seconds"] - current["start_seconds"]
            gaps.append(current)
    return gaps
