import unittest

from dproblem.q2.alns import (
    _high_energy_route_destroy_regroup,
    _late_box_destroy_repair,
    _related_services_destroy_repair,
    _service_destroy_repair,
)


class FirstRng:
    def randrange(self, size):
        return 0

    def choice(self, values):
        return values[0]

    def random(self):
        return 0.0


def box(box_id, service, desired=100.0):
    return {
        "货箱编号": box_id,
        "服务区编号": service,
        "期望送达时间（s）": desired,
    }


def route(*boxes):
    services = []
    for item in boxes:
        if item["服务区编号"] not in services:
            services.append(item["服务区编号"])
    return {"boxes": list(boxes), "visit_order": services, "allowed_types": ["A", "B", "C"]}


def cargo_ids(routes):
    return sorted(item["货箱编号"] for task in routes for item in task["boxes"])


class Q2AlnsOperatorTests(unittest.TestCase):
    def test_service_destroy_repair_changes_route_composition(self):
        routes = [route(box("A", "S001")), route(box("B", "S002"))]
        _service_destroy_repair(routes, FirstRng())
        self.assertEqual(cargo_ids(routes), ["A", "B"])
        self.assertEqual(len(routes), 1)
        self.assertEqual(set(routes[0]["visit_order"]), {"S001", "S002"})

    def test_late_box_operator_targets_box_delay_not_return_time(self):
        slow = box("LATE", "S001", desired=100.0)
        early = box("EARLY", "S002", desired=900.0)
        routes = [route(slow), route(early)]
        current = {
            "trips": [
                {"_boxes": [slow], "返回O01时刻（s）": 500.0},
                {"_boxes": [early], "返回O01时刻（s）": 1000.0},
            ],
            "deliveries": {"LATE": 450.0, "EARLY": 950.0},
        }
        _late_box_destroy_repair(routes, FirstRng(), current)
        self.assertEqual(cargo_ids(routes), ["EARLY", "LATE"])
        self.assertEqual(len(routes), 1)
        self.assertIn("LATE", [item["货箱编号"] for item in routes[0]["boxes"]])

    def test_high_energy_trip_is_destroyed_and_regrouped(self):
        first = box("A", "S001")
        second = box("B", "S002")
        routes = [route(first), route(second)]
        current = {
            "trips": [
                {"货箱编号列表": ["A"], "架次能耗（kWh）": 9.0},
                {"货箱编号列表": ["B"], "架次能耗（kWh）": 1.0},
            ]
        }
        _high_energy_route_destroy_regroup(routes, FirstRng(), current)
        self.assertEqual(cargo_ids(routes), ["A", "B"])
        self.assertEqual(len(routes), 1)

    def test_related_services_are_removed_and_reinserted_as_group(self):
        routes = [
            route(box("A", "S001")),
            route(box("B", "S002")),
            route(box("C", "S003")),
        ]
        data = {
            "nodes": {
                "service_areas": [
                    {"服务区编号": "S001", "经度（°）": 100.0, "纬度（°）": 20.0},
                    {"服务区编号": "S002", "经度（°）": 100.001, "纬度（°）": 20.0},
                    {"服务区编号": "S003", "经度（°）": 101.0, "纬度（°）": 21.0},
                ]
            }
        }
        _related_services_destroy_repair(routes, FirstRng(), data)
        self.assertEqual(cargo_ids(routes), ["A", "B", "C"])
        self.assertEqual(
            [task["visit_order"][0] for task in routes],
            ["S002", "S001", "S003"],
        )


if __name__ == "__main__":
    unittest.main()
