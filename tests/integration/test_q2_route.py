import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.config import load_model_config  # noqa: E402
from dproblem.domain.geometry import RasterDEM  # noqa: E402
from dproblem.io.dataset import load_project_data  # noqa: E402
from dproblem.q1.solver import evaluate_trip  # noqa: E402
from dproblem.q2.route import RouteEvaluator  # noqa: E402


class Q2RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_project_data(PROJECT_ROOT)
        cls.config = load_model_config(PROJECT_ROOT)
        cls.center = cls.data["nodes"]["centers"][0]
        cls.services = cls.data["nodes"]["service_areas"]
        cls.evaluator = RouteEvaluator(
            RasterDEM(cls.data["paths"]["dem_tif"]),
            cls.center,
            cls.services,
            cls.config["energy_model"],
        )
        cls.types = {row["机型编号"]: row for row in cls.data["transport"]["types"]}

    def test_one_service_route_matches_q1_trip_physics(self):
        boxes = [
            box
            for box in self.data["demands"]["boxes"]
            if box["货箱编号"] == "S001-WAT-01"
        ]
        drone_type = self.types["A"]
        route = self.evaluator.evaluate(boxes, ["S001"], drone_type)
        profile = self.evaluator.profile("O01", "S001")
        q1 = evaluate_trip(1, 14.0, 0.027, 1, drone_type, profile, self.config["energy_model"])
        self.assertIsNotNone(route)
        self.assertAlmostEqual(route["架次能耗（kWh）"], q1.energy_kwh)
        self.assertAlmostEqual(route["总飞行时间（s）"], q1.flight_seconds)
        self.assertAlmostEqual(route["持续时间（s）"], q1.work_seconds)

    def test_multistop_payload_decreases_after_delivery(self):
        wanted = {"S001-MED-01", "S006-MED-01"}
        boxes = [box for box in self.data["demands"]["boxes"] if box["货箱编号"] in wanted]
        route = self.evaluator.evaluate(boxes, ["S001", "S006"], self.types["A"])
        self.assertIsNotNone(route)
        self.assertEqual([segment["载荷（kg）"] for segment in route["航段"]], [6, 3, 0])
        self.assertLess(
            route["逐箱交付时刻"]["S001-MED-01"],
            route["逐箱交付时刻"]["S006-MED-01"],
        )

    def test_visit_order_must_match_cargo_services(self):
        box = next(box for box in self.data["demands"]["boxes"] if box["货箱编号"] == "S001-MED-01")
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([box], ["S002"], self.types["A"])


if __name__ == "__main__":
    unittest.main()
