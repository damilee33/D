import json
from pathlib import Path

from dproblem.config import energy_config_hash, load_model_config
from dproblem.domain.energy import transport_segment_energy_kwh
from dproblem.domain.geometry import RasterDEM, segment_terrain_profile
from dproblem.io.dataset import load_project_data
from dproblem.q1.work_time import per_trip_work_time_seconds


def run_single_box_smoke(project_root, service_id="S001", drone_type_id="A", cargo_id="S001-WAT-01"):
    data = load_project_data(project_root)
    model_config = load_model_config(project_root)
    energy_config = model_config["energy_model"]
    work_time_config = model_config["q1_cumulative_work_time"]
    center = data["nodes"]["centers"][0]
    service = next(row for row in data["nodes"]["service_areas"] if row["服务区编号"] == service_id)
    drone_type = next(row for row in data["transport"]["types"] if row["机型编号"] == drone_type_id)
    box = next(row for row in data["demands"]["boxes"] if row["货箱编号"] == cargo_id)
    if box["服务区编号"] != service_id:
        raise ValueError("cargo does not belong to selected service area")

    dem = RasterDEM(data["paths"]["dem_tif"])
    profile = segment_terrain_profile(
        dem,
        center,
        service,
        origin_work_offset_m=0.0,
        destination_work_offset_m=30.0,
    )
    payload_kg = box["单箱质量（kg）"]
    payload_volume_m3 = box["单箱体积（m³）"]
    outbound = transport_segment_energy_kwh(
        profile["horizontal_distance_m"],
        profile["forward_climb_m"],
        payload_kg,
        drone_type,
        energy_config,
    )
    inbound = transport_segment_energy_kwh(
        profile["horizontal_distance_m"],
        profile["reverse_climb_m"],
        0.0,
        drone_type,
        energy_config,
    )
    outbound_time = (
        profile["forward_climb_m"] / drone_type["最大爬升速度（m/s）"]
        + profile["horizontal_distance_m"] / drone_type["计划巡航速度（m/s）"]
        + profile["forward_descent_m"] / drone_type["最大下降速度（m/s）"]
    )
    inbound_time = (
        profile["reverse_climb_m"] / drone_type["最大爬升速度（m/s）"]
        + profile["horizontal_distance_m"] / drone_type["计划巡航速度（m/s）"]
        + profile["reverse_descent_m"] / drone_type["最大下降速度（m/s）"]
    )
    flight_seconds = outbound_time + inbound_time
    total_energy = outbound["total_kwh"] + inbound["total_kwh"]
    usable_energy = drone_type["电池可用能量（kWh）"]
    reserve_fraction = drone_type["返航电量下限（%）"] / 100.0
    energy_limit = usable_energy * (1.0 - reserve_fraction)
    constraints = {
        "mass_capacity": {
            "lhs_kg": payload_kg,
            "rhs_kg": drone_type["最大载货质量（kg）"],
            "pass": payload_kg <= drone_type["最大载货质量（kg）"],
        },
        "volume_capacity": {
            "lhs_m3": payload_volume_m3,
            "rhs_m3": drone_type["可用装载体积（m³）"],
            "pass": payload_volume_m3 <= drone_type["可用装载体积（m³）"],
        },
        "energy_with_return_reserve": {
            "lhs_kwh": total_energy,
            "rhs_kwh": energy_limit,
            "pass": total_energy <= energy_limit,
        },
    }
    if not all(item["pass"] for item in constraints.values()):
        raise AssertionError("smoke trip violates a hard capacity or energy constraint")

    return {
        "status": "PASS",
        "purpose": "P1 structural smoke test; not a Q1 final solution",
        "service_id": service_id,
        "drone_type": drone_type_id,
        "cargo_id": cargo_id,
        "payload_kg": payload_kg,
        "payload_volume_m3": payload_volume_m3,
        "profile": profile,
        "outbound_energy": outbound,
        "return_energy": inbound,
        "total_energy_kwh": total_energy,
        "energy_limit_kwh": energy_limit,
        "remaining_soc_fraction": 1.0 - total_energy / usable_energy,
        "round_trip_flight_seconds": flight_seconds,
        "q1_work_time_seconds": per_trip_work_time_seconds(drone_type, 1, flight_seconds),
        "q1_work_time_definition": work_time_config["version"],
        "energy_config_sha256": energy_config_hash(model_config),
        "constraints": constraints,
    }


def write_single_box_smoke(project_root, output_path):
    result = run_single_box_smoke(project_root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path(project_root) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
