"""Continuous 3-D replay of frozen Q2 transport flights and handovers."""

from math import isclose


TIME_TOLERANCE_SECONDS = 1e-6


def _point(node, altitude_m):
    return {
        "lon": float(node["经度（°）"]),
        "lat": float(node["纬度（°）"]),
        "alt_m": float(altitude_m),
    }


def _phase(kind, start_seconds, end_seconds, first, second, origin_id, destination_id):
    return {
        "phase": kind,
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "first": first,
        "second": second,
        "origin_id": origin_id,
        "destination_id": destination_id,
    }


def build_flight_phases(segment, drone_type, nodes):
    """Split a Q2 aggregate flight event into vertical/cruise/vertical phases."""

    origin_id = segment["起点"]
    destination_id = segment["终点"]
    origin = nodes[origin_id]
    destination = nodes[destination_id]
    origin_work_altitude = float(origin["海拔（m）"]) + (
        0.0 if origin_id == "O01" else 30.0
    )
    destination_work_altitude = float(destination["海拔（m）"]) + (
        0.0 if destination_id == "O01" else 30.0
    )
    cruise_altitude_from_origin = origin_work_altitude + float(segment["爬升（m）"])
    cruise_altitude_from_destination = (
        destination_work_altitude + float(segment["下降（m）"])
    )
    if not isclose(
        cruise_altitude_from_origin,
        cruise_altitude_from_destination,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("segment climb/descent do not define one cruise altitude")
    cruise_altitude = 0.5 * (
        cruise_altitude_from_origin + cruise_altitude_from_destination
    )

    climb_seconds = float(segment["爬升（m）"]) / float(
        drone_type["最大爬升速度（m/s）"]
    )
    cruise_seconds = float(segment["水平距离（m）"]) / float(
        drone_type["计划巡航速度（m/s）"]
    )
    descent_seconds = float(segment["下降（m）"]) / float(
        drone_type["最大下降速度（m/s）"]
    )
    start = float(segment["开始时刻（s）"])
    recorded_end = float(segment["结束时刻（s）"])
    computed_end = start + climb_seconds + cruise_seconds + descent_seconds
    if not isclose(
        computed_end,
        recorded_end,
        rel_tol=0.0,
        abs_tol=TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError("Q2 flight duration cannot be replayed from segment fields")

    phases = []
    cursor = start
    if climb_seconds > 0:
        phases.append(
            _phase(
                "climb",
                cursor,
                cursor + climb_seconds,
                _point(origin, origin_work_altitude),
                _point(origin, cruise_altitude),
                origin_id,
                destination_id,
            )
        )
        cursor += climb_seconds
    phases.append(
        _phase(
            "cruise",
            cursor,
            cursor + cruise_seconds,
            _point(origin, cruise_altitude),
            _point(destination, cruise_altitude),
            origin_id,
            destination_id,
        )
    )
    cursor += cruise_seconds
    if descent_seconds > 0:
        phases.append(
            _phase(
                "descent",
                cursor,
                recorded_end,
                _point(destination, cruise_altitude),
                _point(destination, destination_work_altitude),
                origin_id,
                destination_id,
            )
        )
    return phases


def build_handover_phase(event, node):
    altitude_m = float(node["海拔（m）"]) + 30.0
    position = _point(node, altitude_m)
    return _phase(
        "handover",
        event["开始时刻（s）"],
        event["结束时刻（s）"],
        position,
        dict(position),
        event["服务区编号"],
        event["服务区编号"],
    )


def position_at(phase, time_seconds):
    start = phase["start_seconds"]
    end = phase["end_seconds"]
    if time_seconds < start - TIME_TOLERANCE_SECONDS or time_seconds > end + TIME_TOLERANCE_SECONDS:
        raise ValueError("time lies outside phase")
    if end <= start:
        fraction = 0.0
    else:
        fraction = min(1.0, max(0.0, (time_seconds - start) / (end - start)))
    return {
        key: phase["first"][key]
        + fraction * (phase["second"][key] - phase["first"][key])
        for key in ("lon", "lat", "alt_m")
    }


def sample_phase(phase, step_seconds):
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    start = phase["start_seconds"]
    end = phase["end_seconds"]
    times = [start]
    cursor = start + step_seconds
    while cursor < end - TIME_TOLERANCE_SECONDS:
        times.append(cursor)
        cursor += step_seconds
    if end > start + TIME_TOLERANCE_SECONDS:
        times.append(end)
    return [(time_seconds, position_at(phase, time_seconds)) for time_seconds in times]
