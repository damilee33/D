"""Feasibility-preserving ALNS for the Q2 transport schedule."""

import random
from copy import deepcopy
from itertools import combinations

from dproblem.q2.baseline import _hard_deadline
from dproblem.q2.scheduler import q1_batches_as_flexible_routes, route_deadline_key, schedule_routes


def objective(result):
    return (
        result["timeliness_loss"],
        result["makespan"],
        result["energy"],
        result["trip_count"],
    )


def _unique_in_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def urgency_repack_split_services(routes, drone_types):
    """Concentrate hard boxes without changing Q1 batch mass/volume profiles.

    Only like-for-like boxes (same mass and volume) are exchanged. This keeps
    every Q1-certified batch physically feasible while bringing first-batch or
    medical cargo into the route with the most hard-deadline boxes.
    """

    by_service = {}
    for route in routes:
        if len(route["visit_order"]) != 1:
            raise ValueError("urgency repack expects single-service Q1 routes")
        by_service.setdefault(route["visit_order"][0], []).append(route)
    repaired = []
    for service_id in sorted(by_service):
        service_routes = by_service[service_id]
        if len(service_routes) == 1:
            repaired.extend(service_routes)
            continue
        target = max(
            service_routes,
            key=lambda route: sum(_hard_deadline(box) is not None for box in route["boxes"]),
        )
        for source in service_routes:
            if source is target:
                continue
            for urgent in list(source["boxes"]):
                if _hard_deadline(urgent) is None:
                    continue
                replacement = next(
                    (
                        box
                        for box in target["boxes"]
                        if _hard_deadline(box) is None
                        and box["单箱质量（kg）"] == urgent["单箱质量（kg）"]
                        and box["单箱体积（m³）"] == urgent["单箱体积（m³）"]
                    ),
                    None,
                )
                if replacement is not None:
                    source["boxes"].remove(urgent)
                    target["boxes"].remove(replacement)
                    source["boxes"].append(replacement)
                    target["boxes"].append(urgent)
        repaired.extend(service_routes)
    return sorted(repaired, key=route_deadline_key)


def _weighted_operator(rng, weights):
    total = sum(weights.values())
    point = rng.random() * total
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if point <= cumulative:
            return name
    return next(reversed(weights))


def _state_signature(routes):
    return tuple(
        (
            tuple(route["visit_order"]),
            tuple(sorted(box["货箱编号"] for box in route["boxes"])),
        )
        for route in routes
    )


def _normalise_route(route):
    services = {box["服务区编号"] for box in route["boxes"]}
    route["visit_order"] = [
        service_id for service_id in route["visit_order"] if service_id in services
    ]
    route["visit_order"].extend(sorted(services - set(route["visit_order"])))


def _move_boxes_to_route(candidate, source_index, boxes, target_index, rng):
    """Remove selected boxes and reinsert them into a different route."""

    moved_ids = {box["货箱编号"] for box in boxes}
    candidate[source_index]["boxes"] = [
        box
        for box in candidate[source_index]["boxes"]
        if box["货箱编号"] not in moved_ids
    ]
    _normalise_route(candidate[source_index])
    if not candidate[source_index]["boxes"]:
        candidate.pop(source_index)
        if target_index > source_index:
            target_index -= 1
    candidate[target_index]["boxes"].extend(boxes)
    for service_id in _unique_in_order(box["服务区编号"] for box in boxes):
        if service_id not in candidate[target_index]["visit_order"]:
            position = rng.randrange(len(candidate[target_index]["visit_order"]) + 1)
            candidate[target_index]["visit_order"].insert(position, service_id)
    _normalise_route(candidate[target_index])


def _service_destroy_repair(candidate, rng):
    """Remove one service block from a route and insert it into another route."""

    if len(candidate) < 2:
        return
    source_index = rng.randrange(len(candidate))
    services = _unique_in_order(
        box["服务区编号"] for box in candidate[source_index]["boxes"]
    )
    service_id = rng.choice(services)
    boxes = [
        box
        for box in candidate[source_index]["boxes"]
        if box["服务区编号"] == service_id
    ]
    targets = [index for index in range(len(candidate)) if index != source_index]
    _move_boxes_to_route(
        candidate, source_index, boxes, rng.choice(targets), rng
    )


def _late_box_destroy_repair(candidate, rng, current_result):
    """Remove the box with greatest delivery delay and reinsert it elsewhere."""

    if len(candidate) < 2:
        return
    boxes = {
        box["货箱编号"]: box
        for trip in current_result["trips"]
        for box in trip["_boxes"]
    }
    deliveries = current_result["deliveries"]
    cargo_id = max(
        boxes,
        key=lambda box_id: (
            deliveries[box_id] - float(boxes[box_id]["期望送达时间（s）"]),
            deliveries[box_id],
            box_id,
        ),
    )
    source_index = next(
        index
        for index, route in enumerate(candidate)
        if any(box["货箱编号"] == cargo_id for box in route["boxes"])
    )
    targets = [index for index in range(len(candidate)) if index != source_index]
    _move_boxes_to_route(
        candidate, source_index, [boxes[cargo_id]], rng.choice(targets), rng
    )


def _high_energy_route_destroy_regroup(candidate, rng, current_result):
    """Destroy the highest-energy trip and regroup its cargo with another trip."""

    if len(candidate) < 2:
        return
    trip = max(current_result["trips"], key=lambda row: row["架次能耗（kWh）"])
    target_ids = set(trip["货箱编号列表"])
    source_index = next(
        (
            index
            for index, route in enumerate(candidate)
            if {box["货箱编号"] for box in route["boxes"]} == target_ids
        ),
        None,
    )
    if source_index is None:
        return
    source = candidate.pop(source_index)
    target_index = rng.randrange(len(candidate))
    candidate[target_index]["boxes"].extend(source["boxes"])
    visits = _unique_in_order(
        candidate[target_index]["visit_order"] + source["visit_order"]
    )
    if rng.random() < 0.5:
        visits.reverse()
    candidate[target_index]["visit_order"] = visits
    _normalise_route(candidate[target_index])


def _related_services_destroy_repair(candidate, rng, data):
    """Jointly remove routes for a geographically related service pair and reinsert."""

    from dproblem.domain.geometry import local_wgs84_distance_m

    present = sorted(
        {service_id for route in candidate for service_id in route["visit_order"]}
    )
    if len(present) < 2:
        return
    nodes = {row["服务区编号"]: row for row in data["nodes"]["service_areas"]}
    first = rng.choice(present)
    first_node = nodes[first]
    second = min(
        (service_id for service_id in present if service_id != first),
        key=lambda service_id: local_wgs84_distance_m(
            first_node["经度（°）"],
            first_node["纬度（°）"],
            nodes[service_id]["经度（°）"],
            nodes[service_id]["纬度（°）"],
        ),
    )
    related = {first, second}
    removed = [
        route
        for route in candidate
        if related.intersection(route["visit_order"])
    ]
    if not removed or len(removed) == len(candidate):
        return
    candidate[:] = [
        route
        for route in candidate
        if not related.intersection(route["visit_order"])
    ]
    removed.reverse()
    insertion = rng.randrange(len(candidate) + 1)
    candidate[insertion:insertion] = removed


def _mutate(routes, name, rng, current_result, data=None):
    candidate = deepcopy(routes)
    if name == "service_destroy_repair":
        _service_destroy_repair(candidate, rng)
    elif name == "route_swap" and len(candidate) >= 2:
        left, right = rng.sample(range(len(candidate)), 2)
        candidate[left], candidate[right] = candidate[right], candidate[left]
    elif name == "late_box_destroy_repair":
        _late_box_destroy_repair(candidate, rng, current_result)
    elif name == "high_energy_route_destroy_regroup":
        _high_energy_route_destroy_regroup(candidate, rng, current_result)
    elif name == "related_services_destroy_repair":
        if data is None:
            raise ValueError("related-services operator requires project data")
        _related_services_destroy_repair(candidate, rng, data)
    elif name == "merge" and len(candidate) >= 2:
        left, right = sorted(rng.sample(range(len(candidate)), 2))
        first, second = candidate[left], candidate[right]
        visits = _unique_in_order(first["visit_order"] + second["visit_order"])
        if rng.random() < 0.5:
            visits = list(reversed(visits))
        merged = {
            "boxes": first["boxes"] + second["boxes"],
            "visit_order": visits,
            "allowed_types": ["A", "B", "C"],
        }
        candidate.pop(right)
        candidate[left] = merged
    elif name == "split":
        choices = [index for index, route in enumerate(candidate) if len(route["visit_order"]) > 1]
        if choices:
            index = rng.choice(choices)
            route = candidate.pop(index)
            split_routes = [
                {
                    "boxes": [box for box in route["boxes"] if box["服务区编号"] == service_id],
                    "visit_order": [service_id],
                    "allowed_types": ["A", "B", "C"],
                }
                for service_id in route["visit_order"]
            ]
            for offset, split_route in enumerate(split_routes):
                candidate.insert(index + offset, split_route)
    elif name in ("box_move", "box_exchange"):
        groups = {}
        for index, route in enumerate(candidate):
            if len(route["visit_order"]) == 1:
                groups.setdefault(route["visit_order"][0], []).append(index)
        services = [service for service, indices in groups.items() if len(indices) >= 2]
        if services:
            left, right = rng.sample(groups[rng.choice(services)], 2)
            if name == "box_move" and len(candidate[left]["boxes"]) > 1:
                box = rng.choice(candidate[left]["boxes"])
                candidate[left]["boxes"].remove(box)
                candidate[right]["boxes"].append(box)
            elif name == "box_exchange":
                box_left = rng.choice(candidate[left]["boxes"])
                box_right = rng.choice(candidate[right]["boxes"])
                candidate[left]["boxes"].remove(box_left)
                candidate[right]["boxes"].remove(box_right)
                candidate[left]["boxes"].append(box_right)
                candidate[right]["boxes"].append(box_left)
    elif name == "route_reverse":
        choices = [index for index, route in enumerate(candidate) if len(route["visit_order"]) > 1]
        if choices:
            index = rng.choice(choices)
            candidate[index]["visit_order"].reverse()
    return candidate


def run_alns(initial_routes, data, config, evaluator, seed, iterations=400):
    rng = random.Random(seed)
    operators = (
        "service_destroy_repair",
        "route_swap",
        "late_box_destroy_repair",
        "high_energy_route_destroy_regroup",
        "related_services_destroy_repair",
        "merge",
        "split",
        "box_move",
        "box_exchange",
        "route_reverse",
    )
    weights = {name: 1.0 for name in operators}
    counts = {name: 0 for name in operators}
    accepted = {name: 0 for name in operators}
    improved = {name: 0 for name in operators}
    rejections = {
        name: {"noop": 0, "physical": 0, "hard_deadline": 0, "not_accepted": 0}
        for name in operators
    }
    current_routes = deepcopy(initial_routes)
    current = schedule_routes(current_routes, data, config, evaluator, preserve_order=True)
    if current["status"] != "PASS":
        raise RuntimeError("ALNS initial solution is not feasible: {}".format(current["hard_violations"]))
    best_routes = deepcopy(current_routes)
    best = current
    history = []
    for iteration in range(1, iterations + 1):
        operator = _weighted_operator(rng, weights)
        counts[operator] += 1
        candidate_routes = _mutate(current_routes, operator, rng, current, data=data)
        if _state_signature(candidate_routes) == _state_signature(current_routes):
            candidate = {"status": "NOOP"}
            rejections[operator]["noop"] += 1
        else:
            candidate = schedule_routes(
                candidate_routes, data, config, evaluator, preserve_order=True
            )
        reward = 0.2
        accepted_candidate = False
        if candidate.get("status") == "PASS":
            if objective(candidate) < objective(current):
                accepted_candidate = True
                reward = 3.0
            else:
                exploration = max(0.01, 0.18 * (1.0 - iteration / float(iterations)))
                if rng.random() < exploration:
                    accepted_candidate = True
                    reward = 0.8
            if accepted_candidate:
                current_routes = candidate_routes
                current = candidate
                accepted[operator] += 1
            else:
                rejections[operator]["not_accepted"] += 1
            if objective(candidate) < objective(best):
                best_routes = deepcopy(candidate_routes)
                best = candidate
                improved[operator] += 1
                reward = 6.0
        elif candidate.get("status") != "NOOP":
            reason = (
                "hard_deadline"
                if candidate.get("hard_violations")
                else "physical"
            )
            rejections[operator][reason] += 1
        weights[operator] = 0.9 * weights[operator] + 0.1 * reward
        candidate_matches_best = (
            candidate.get("status") == "PASS" and objective(best) == objective(candidate)
        )
        if iteration == 1 or iteration % 10 == 0 or candidate_matches_best:
            history.append(
                {
                    "iteration": iteration,
                    "best_timeliness_loss": best["timeliness_loss"],
                    "best_makespan": best["makespan"],
                    "best_energy": best["energy"],
                    "best_trip_count": best["trip_count"],
                }
            )
    return {
        "seed": seed,
        "iterations": iterations,
        "best_routes": best_routes,
        "best": best,
        "history": history,
        "operator_counts": counts,
        "operator_accepted": accepted,
        "operator_improvements": improved,
        "operator_rejections": rejections,
        "final_operator_weights": weights,
    }


def pair_merge_descent(routes, result, data, config, evaluator):
    """Exact best-improvement sweep over all two-route multi-stop merges."""

    current_routes = deepcopy(routes)
    current = result
    accepted = []
    while True:
        best_routes = None
        best_result = current
        best_record = None
        for left, right in combinations(range(len(current_routes)), 2):
            first = current_routes[left]
            second = current_routes[right]
            base_visits = _unique_in_order(first["visit_order"] + second["visit_order"])
            for visits in (base_visits, list(reversed(base_visits))):
                candidate_routes = deepcopy(current_routes)
                merged = {
                    "boxes": candidate_routes[left]["boxes"] + candidate_routes[right]["boxes"],
                    "visit_order": visits,
                    "allowed_types": ["A", "B", "C"],
                }
                candidate_routes.pop(right)
                candidate_routes[left] = merged
                candidate = schedule_routes(
                    candidate_routes, data, config, evaluator, preserve_order=True
                )
                if candidate.get("status") == "PASS" and objective(candidate) < objective(best_result):
                    best_routes = candidate_routes
                    best_result = candidate
                    best_record = {
                        "left_index": left,
                        "right_index": right,
                        "visit_order": visits,
                        "objective_before": objective(current),
                        "objective_after": objective(candidate),
                    }
        if best_routes is None:
            break
        current_routes = best_routes
        current = best_result
        accepted.append(best_record)
    return current_routes, current, accepted


def build_initial_routes(project_root, data):
    routes = q1_batches_as_flexible_routes(project_root, data["demands"]["boxes"])
    return urgency_repack_split_services(routes, data["transport"]["types"])
