"""Replayable multi-stop transport-trip evaluator for Q2."""

from collections import defaultdict

from dproblem.domain.energy import transport_segment_energy_kwh
from dproblem.domain.geometry import segment_terrain_profile


EPSILON = 1e-10


class RouteEvaluator:
    def __init__(self, dem, center, service_areas, energy_config, clearance_m=50.0):
        self.dem = dem
        self.center = center
        self.energy_config = energy_config
        self.clearance_m = clearance_m
        self.nodes = {center["调度中心编号"]: center}
        self.nodes.update({row["服务区编号"]: row for row in service_areas})
        self.profile_cache = {}

    def profile(self, origin_id, destination_id):
        key = (origin_id, destination_id)
        if key not in self.profile_cache:
            origin = self.nodes[origin_id]
            destination = self.nodes[destination_id]
            self.profile_cache[key] = segment_terrain_profile(
                self.dem,
                origin,
                destination,
                origin_work_offset_m=0.0 if origin_id == "O01" else 30.0,
                destination_work_offset_m=0.0 if destination_id == "O01" else 30.0,
                clearance_m=self.clearance_m,
            )
        return self.profile_cache[key]

    def evaluate(self, boxes, visit_order, drone_type, start_seconds=0.0):
        if start_seconds < 0:
            raise ValueError("start_seconds must be nonnegative")
        grouped = defaultdict(list)
        for box in boxes:
            grouped[box["服务区编号"]].append(box)
        if len(visit_order) != len(set(visit_order)):
            raise ValueError("visit_order contains a repeated service")
        if set(visit_order) != set(grouped):
            raise ValueError("visit_order must contain exactly the cargo service areas")

        total_mass = sum(box["单箱质量（kg）"] for box in boxes)
        total_volume = sum(box["单箱体积（m³）"] for box in boxes)
        if total_mass > drone_type["最大载货质量（kg）"] + EPSILON:
            return None
        if total_volume > drone_type["可用装载体积（m³）"] + EPSILON:
            return None

        cursor = float(start_seconds)
        events = []

        def add_event(stage, duration, **fields):
            nonlocal cursor
            start = cursor
            cursor += duration
            event = {"阶段": stage, "开始时刻（s）": start, "结束时刻（s）": cursor}
            event.update(fields)
            events.append(event)
            return event

        add_event("准备", drone_type["工位固定准备时间（s）"])
        add_event("装载", len(boxes) * drone_type["每箱装载时间（s）"])

        current_mass = total_mass
        current_node = "O01"
        total_energy = 0.0
        total_flight = 0.0
        segments = []
        delivery_times = {}
        for destination_id in list(visit_order) + ["O01"]:
            profile = self.profile(current_node, destination_id)
            energy = transport_segment_energy_kwh(
                profile["horizontal_distance_m"],
                profile["forward_climb_m"],
                current_mass,
                drone_type,
                self.energy_config,
            )
            flight_seconds = (
                profile["forward_climb_m"] / drone_type["最大爬升速度（m/s）"]
                + profile["horizontal_distance_m"] / drone_type["计划巡航速度（m/s）"]
                + profile["forward_descent_m"] / drone_type["最大下降速度（m/s）"]
            )
            flight_event = add_event(
                "飞行",
                flight_seconds,
                起点=current_node,
                终点=destination_id,
                **{"航段载荷（kg）": current_mass},
            )
            segments.append(
                {
                    "起点": current_node,
                    "终点": destination_id,
                    "载荷（kg）": current_mass,
                    "水平距离（m）": profile["horizontal_distance_m"],
                    "爬升（m）": profile["forward_climb_m"],
                    "下降（m）": profile["forward_descent_m"],
                    "开始时刻（s）": flight_event["开始时刻（s）"],
                    "结束时刻（s）": flight_event["结束时刻（s）"],
                    "水平能耗（kWh）": energy["horizontal_kwh"],
                    "爬升能耗（kWh）": energy["climb_kwh"],
                    "总能耗（kWh）": energy["total_kwh"],
                }
            )
            total_energy += energy["total_kwh"]
            total_flight += flight_seconds
            current_node = destination_id
            if destination_id != "O01":
                delivered = grouped[destination_id]
                handover_seconds = (
                    drone_type["接收点基础交接时间（s）"]
                    + len(delivered) * drone_type["每箱增加交接时间（s）"]
                )
                handover_event = add_event(
                    "交接",
                    handover_seconds,
                    服务区编号=destination_id,
                    **{"货箱编号列表": [box["货箱编号"] for box in delivered]},
                )
                for box in delivered:
                    delivery_times[box["货箱编号"]] = handover_event["结束时刻（s）"]
                current_mass -= sum(box["单箱质量（kg）"] for box in delivered)

        if abs(current_mass) > 1e-8:
            raise AssertionError("route replay ended with nonzero payload")
        energy_limit = drone_type["电池可用能量（kWh）"] * (
            1.0 - drone_type["返航电量下限（%）"] / 100.0
        )
        if total_energy > energy_limit + EPSILON:
            return None
        remaining_soc = 1.0 - total_energy / drone_type["电池可用能量（kWh）"]
        return {
            "机型编号": drone_type["机型编号"],
            "开始时刻（s）": float(start_seconds),
            "返回O01时刻（s）": cursor,
            "持续时间（s）": cursor - float(start_seconds),
            "访问服务区顺序": list(visit_order),
            "总质量（kg）": total_mass,
            "总体积（m³）": total_volume,
            "总飞行时间（s）": total_flight,
            "架次能耗（kWh）": total_energy,
            "能量上限（kWh）": energy_limit,
            "返航SOC（%）": remaining_soc * 100.0,
            "逐箱交付时刻": delivery_times,
            "航段": segments,
            "时间线": events,
        }

