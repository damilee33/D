"""Relay candidate generation, coverage evidence, travel, and sortie energy."""

from math import floor

from dproblem.domain.communication import audit_link
from dproblem.domain.energy import climb_energy_kwh, relay_power_energy_kwh
from dproblem.domain.geometry import local_wgs84_distance_m


def _vertical_time_seconds(start_alt_m, end_alt_m, relay_type):
    difference = end_alt_m - start_alt_m
    if difference >= 0:
        return difference / float(relay_type["最大爬升速度（m/s）"])
    return -difference / float(relay_type["最大下降速度（m/s）"])


def relay_travel_profile(dem, center, candidate, relay_type, energy_config, clearance_m=50.0):
    distance_m = local_wgs84_distance_m(
        center["经度（°）"],
        center["纬度（°）"],
        candidate["lon"],
        candidate["lat"],
    )
    maximum_ground_m = dem.maximum_elevation_on_line(
        center["经度（°）"],
        center["纬度（°）"],
        candidate["lon"],
        candidate["lat"],
    )
    cruise_altitude_m = maximum_ground_m + clearance_m
    origin_altitude_m = float(center["海拔（m）"])
    hover_altitude_m = candidate["alt_m"]
    horizontal_seconds = distance_m / float(relay_type["计划巡航速度（m/s）"])
    outbound_seconds = (
        _vertical_time_seconds(origin_altitude_m, cruise_altitude_m, relay_type)
        + horizontal_seconds
        + _vertical_time_seconds(cruise_altitude_m, hover_altitude_m, relay_type)
    )
    inbound_seconds = (
        _vertical_time_seconds(hover_altitude_m, cruise_altitude_m, relay_type)
        + horizontal_seconds
        + _vertical_time_seconds(cruise_altitude_m, origin_altitude_m, relay_type)
    )
    positive_climb_m = (
        max(0.0, cruise_altitude_m - origin_altitude_m)
        + max(0.0, hover_altitude_m - cruise_altitude_m)
        + max(0.0, cruise_altitude_m - hover_altitude_m)
        + max(0.0, origin_altitude_m - cruise_altitude_m)
    )
    climb_kwh = climb_energy_kwh(
        empty_mass_kg=float(relay_type["计划起飞总质量（kg）"]),
        payload_kg=0.0,
        positive_height_gain_m=positive_climb_m,
        efficiency=float(relay_type["爬升能耗效率"]),
        gravity_m_s2=float(energy_config["gravity_m_s2"]),
    )
    return {
        "distance_one_way_m": distance_m,
        "maximum_ground_elevation_m": maximum_ground_m,
        "cruise_altitude_m": cruise_altitude_m,
        "outbound_seconds": outbound_seconds,
        "inbound_seconds": inbound_seconds,
        "round_trip_flight_seconds": outbound_seconds + inbound_seconds,
        "round_trip_positive_climb_m": positive_climb_m,
        "round_trip_climb_energy_kwh": climb_kwh,
    }


def relay_transition_profile(dem, origin, destination, relay_type, energy_config, clearance_m=50.0):
    """Terrain-cleared relocation between two airborne hover candidates."""

    distance_m = local_wgs84_distance_m(
        origin["lon"], origin["lat"], destination["lon"], destination["lat"]
    )
    maximum_ground_m = dem.maximum_elevation_on_line(
        origin["lon"], origin["lat"], destination["lon"], destination["lat"]
    )
    cruise_altitude_m = maximum_ground_m + clearance_m
    horizontal_seconds = distance_m / float(relay_type["计划巡航速度（m/s）"])
    flight_seconds = (
        _vertical_time_seconds(origin["alt_m"], cruise_altitude_m, relay_type)
        + horizontal_seconds
        + _vertical_time_seconds(cruise_altitude_m, destination["alt_m"], relay_type)
    )
    positive_climb_m = max(0.0, cruise_altitude_m - origin["alt_m"]) + max(
        0.0, destination["alt_m"] - cruise_altitude_m
    )
    climb_kwh = climb_energy_kwh(
        empty_mass_kg=float(relay_type["计划起飞总质量（kg）"]),
        payload_kg=0.0,
        positive_height_gain_m=positive_climb_m,
        efficiency=float(relay_type["爬升能耗效率"]),
        gravity_m_s2=float(energy_config["gravity_m_s2"]),
    )
    base_kwh = relay_power_energy_kwh(
        cruise_power_kw=float(relay_type["巡航功率（kW）"]),
        cruise_seconds=flight_seconds,
        hover_power_kw=0.0,
        communication_power_kw=0.0,
        service_seconds=0.0,
    )
    return {
        "distance_m": distance_m,
        "maximum_ground_elevation_m": maximum_ground_m,
        "cruise_altitude_m": cruise_altitude_m,
        "flight_seconds": flight_seconds,
        "positive_climb_m": positive_climb_m,
        "climb_energy_kwh": climb_kwh,
        "base_energy_kwh": base_kwh,
        "total_energy_kwh": base_kwh + climb_kwh,
    }


def relay_sortie_metrics(travel, service_seconds, relay_type):
    base_and_service = relay_power_energy_kwh(
        cruise_power_kw=float(relay_type["巡航功率（kW）"]),
        cruise_seconds=travel["round_trip_flight_seconds"],
        hover_power_kw=float(relay_type["悬停功率（kW）"]),
        communication_power_kw=float(relay_type["通信附加功率（kW）"]),
        service_seconds=service_seconds,
    )
    total_kwh = base_and_service + travel["round_trip_climb_energy_kwh"]
    usable_kwh = float(relay_type["能源组件可用能量（kWh）"])
    energy_limit_kwh = usable_kwh * (
        1.0 - float(relay_type["返航电量下限（%）"]) / 100.0
    )
    return {
        "energy_kwh": total_kwh,
        "energy_limit_kwh": energy_limit_kwh,
        "remaining_soc_percent": 100.0 * (1.0 - total_kwh / usable_kwh),
        "energy_feasible": total_kwh <= energy_limit_kwh + 1e-10,
    }


def generate_candidates(dem, data, direct_samples, gaps, height_levels_m=(100.0, 200.0, 300.0)):
    """Create bounded DEM-cell candidates from nodes and direct-gap evidence."""

    source_points = []
    center = data["nodes"]["centers"][0]
    source_points.append(("O01", center["经度（°）"], center["纬度（°）"]))
    source_points.extend(
        (row["服务区编号"], row["经度（°）"], row["纬度（°）"])
        for row in data["nodes"]["service_areas"]
    )
    by_trip = {}
    for sample in direct_samples:
        by_trip.setdefault(sample["trip_id"], []).append(sample)
    for gap in gaps:
        selected = [
            row
            for row in by_trip[gap["trip_id"]]
            if gap["start_seconds"] - 1e-8
            <= row["time_seconds"]
            <= gap["end_seconds"] + 1e-8
        ]
        targets = [
            gap["start_seconds"],
            0.5 * (gap["start_seconds"] + gap["end_seconds"]),
            gap["end_seconds"],
        ]
        representatives = [
            min(selected, key=lambda row, target=target: abs(row["time_seconds"] - target))
            for target in targets
        ]
        representatives.append(min(selected, key=lambda row: row["margin_db"]))
        source_points.extend(
            (
                "{}-P{:02d}".format(gap["gap_id"], index),
                row["lon"],
                row["lat"],
            )
            for index, row in enumerate(representatives, start=1)
        )

    cells = {}
    for source, lon, lat in source_points:
        row, col = dem.fractional_cell(float(lon), float(lat))
        row_index = min(dem.height - 1, max(0, int(floor(row))))
        col_index = min(dem.width - 1, max(0, int(floor(col))))
        value = float(dem.values[row_index, col_index])
        if dem.nodata is not None and value == dem.nodata:
            continue
        cells.setdefault((row_index, col_index), []).append(source)

    candidates = []
    for sequence, ((row, col), sources) in enumerate(sorted(cells.items()), start=1):
        lon = dem.lon0 + (col + 0.5) * dem.lon_step
        lat = dem.lat0 - (row + 0.5) * dem.lat_step
        ground = float(dem.values[row, col])
        for height_m in height_levels_m:
            candidates.append(
                {
                    "candidate_id": "C{:03d}-H{:03d}".format(sequence, int(height_m)),
                    "row": row,
                    "col": col,
                    "lon": lon,
                    "lat": lat,
                    "ground_elevation_m": ground,
                    "height_agl_m": float(height_m),
                    "alt_m": ground + float(height_m),
                    "sources": ";".join(sorted(set(sources))),
                }
            )
    return candidates


def candidate_link_evidence(dem, model, candidate, gateway, gap_samples):
    backhaul, access_rows = candidate_link_series(
        dem, model, candidate, gateway, gap_samples
    )
    if not backhaul["available"]:
        return {
            "covered": False,
            "backhaul_available": False,
            "backhaul_margin_db": backhaul["margin_db"],
            "minimum_access_margin_db": None,
            "failed_access_samples": len(gap_samples),
        }
    return {
        "covered": all(row["available"] for row in access_rows),
        "backhaul_available": True,
        "backhaul_margin_db": backhaul["margin_db"],
        "minimum_access_margin_db": min(row["margin_db"] for row in access_rows),
        "failed_access_samples": sum(not row["available"] for row in access_rows),
    }


def candidate_link_series(dem, model, candidate, gateway, gap_samples):
    relay_position = {
        "lon": candidate["lon"],
        "lat": candidate["lat"],
        "alt_m": candidate["alt_m"],
    }
    backhaul = audit_link(
        dem,
        model,
        relay_position,
        gateway,
        "relay_backhaul",
        "gateway",
    )
    if not backhaul["available"]:
        return backhaul, []
    access_rows = []
    for sample in gap_samples:
        transport_position = {
            "lon": sample["lon"],
            "lat": sample["lat"],
            "alt_m": sample["alt_m"],
        }
        link = audit_link(
                dem,
                model,
                transport_position,
                relay_position,
                "transport",
                "relay_access",
            )
        access_rows.append(
            {
                "time_seconds": sample["time_seconds"],
                "available": link["available"],
                "margin_db": link["margin_db"],
            }
        )
    return backhaul, access_rows


def conservative_cover_intervals(access_rows):
    """Return maximal time cells whose two endpoint access links are available."""

    intervals = []
    current = None
    for left, right in zip(access_rows, access_rows[1:]):
        covered = left["available"] and right["available"]
        if covered:
            if current is None:
                current = {
                    "start_seconds": left["time_seconds"],
                    "end_seconds": right["time_seconds"],
                    "minimum_access_margin_db": min(
                        left["margin_db"], right["margin_db"]
                    ),
                    "sample_intervals": 1,
                }
            else:
                current["end_seconds"] = right["time_seconds"]
                current["minimum_access_margin_db"] = min(
                    current["minimum_access_margin_db"],
                    left["margin_db"],
                    right["margin_db"],
                )
                current["sample_intervals"] += 1
        elif current is not None:
            current["duration_seconds"] = (
                current["end_seconds"] - current["start_seconds"]
            )
            intervals.append(current)
            current = None
    if current is not None:
        current["duration_seconds"] = current["end_seconds"] - current["start_seconds"]
        intervals.append(current)
    return intervals
