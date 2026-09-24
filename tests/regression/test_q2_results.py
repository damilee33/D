import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.compat import import_openpyxl_load_workbook  # noqa: E402
from dproblem.config import load_model_config  # noqa: E402
from dproblem.domain.charging import charge_time_to_full_seconds  # noqa: E402
from dproblem.domain.geometry import RasterDEM  # noqa: E402
from dproblem.io.dataset import load_project_data  # noqa: E402
from dproblem.q2.baseline import _hard_deadline  # noqa: E402
from dproblem.q2.route import RouteEvaluator  # noqa: E402


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Q2ResultRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT_ROOT / "results" / "q2"
        cls.summary = json.loads((cls.result_dir / "q2_summary.json").read_text(encoding="utf-8"))
        cls.trips = read_csv(cls.result_dir / "main_trips.csv")
        cls.box_rows = read_csv(cls.result_dir / "main_box_deliveries.csv")
        cls.battery_rows = read_csv(cls.result_dir / "main_battery_timeline.csv")
        cls.data = load_project_data(PROJECT_ROOT)
        cls.config = load_model_config(PROJECT_ROOT)
        cls.boxes = {box["货箱编号"]: box for box in cls.data["demands"]["boxes"]}
        cls.types = {row["机型编号"]: row for row in cls.data["transport"]["types"]}
        cls.evaluator = RouteEvaluator(
            RasterDEM(cls.data["paths"]["dem_tif"]),
            cls.data["nodes"]["centers"][0],
            cls.data["nodes"]["service_areas"],
            cls.config["energy_model"],
        )

    def test_summary_and_unique_cargo_coverage(self):
        self.assertEqual(self.summary["status"], "PASS")
        self.assertEqual(self.summary["constraint_audit"]["violations"], 0)
        self.assertEqual(self.summary["baseline_status"], "FAIL")
        cargo_ids = [cargo_id for row in self.trips for cargo_id in row["货箱编号列表"].split(";")]
        self.assertEqual(len(cargo_ids), 80)
        self.assertEqual(len(set(cargo_ids)), 80)
        self.assertEqual(set(cargo_ids), set(self.boxes))
        self.assertEqual(len(self.summary["seeds"]), 5)
        self.assertGreaterEqual(self.summary["pareto_seed_count"], 1)

    def test_independent_route_replay_matches_outputs(self):
        deliveries = {row["货箱编号"]: float(row["交付完成时刻（s）"]) for row in self.box_rows}
        for row in self.trips:
            cargo_ids = row["货箱编号列表"].split(";")
            route = self.evaluator.evaluate(
                [self.boxes[cargo_id] for cargo_id in cargo_ids],
                row["访问服务区顺序"].split(";"),
                self.types[row["机型编号"]],
                start_seconds=float(row["开始时刻（s）"]),
            )
            with self.subTest(trip=row["架次编号"]):
                self.assertIsNotNone(route)
                self.assertAlmostEqual(route["返回O01时刻（s）"], float(row["返回O01时刻（s）"]), places=7)
                self.assertAlmostEqual(route["架次能耗（kWh）"], float(row["架次能耗（kWh）"]), places=9)
                self.assertAlmostEqual(route["返航SOC（%）"], float(row["返航SOC（%）"]), places=7)
                for cargo_id in cargo_ids:
                    self.assertAlmostEqual(route["逐箱交付时刻"][cargo_id], deliveries[cargo_id], places=7)

    def test_deadlines_and_resource_intervals(self):
        for row in self.box_rows:
            box = self.boxes[row["货箱编号"]]
            deadline = _hard_deadline(box)
            if deadline is not None:
                self.assertLessEqual(float(row["交付完成时刻（s）"]), deadline + 1e-8)
        for field in ("无人机编号", "电池编号"):
            groups = {}
            for row in self.trips:
                groups.setdefault(row[field], []).append(
                    (float(row["开始时刻（s）"]), float(row["返回O01时刻（s）"]))
                )
            for intervals in groups.values():
                intervals.sort()
                self.assertTrue(all(left[1] <= right[0] + 1e-8 for left, right in zip(intervals, intervals[1:])))
        battery_groups = {}
        for row in self.battery_rows:
            battery_groups.setdefault(row["电池编号"], []).append(row)
        full_times = {
            row["机型编号"]: float(row["等效完全充电时间（s）"])
            for row in self.data["transport"]["batteries"]
        }
        for rows in battery_groups.values():
            rows.sort(key=lambda row: float(row["任务开始时刻（s）"]))
            for row in rows:
                expected = charge_time_to_full_seconds(
                    float(row["任务后SOC（%）"]) / 100.0,
                    full_times[row["机型编号"]],
                )
                self.assertAlmostEqual(
                    float(row["充电结束时刻（s）"]) - float(row["充电开始时刻（s）"]),
                    expected,
                    places=7,
                )
            self.assertTrue(
                all(
                    float(left["再次可用时刻（s）"]) <= float(right["任务开始时刻（s）"]) + 1e-8
                    for left, right in zip(rows, rows[1:])
                )
            )

    def test_submission_workbook_matches_csv(self):
        workbook = import_openpyxl_load_workbook()(
            self.result_dir / "结果提交_Q2.xlsx", read_only=False, data_only=False
        )
        try:
            trip_sheet = workbook["Q2_运输架次"]
            box_sheet = workbook["Q2_逐箱交付"]
            self.assertEqual(trip_sheet.max_row - 1, len(self.trips))
            self.assertEqual(box_sheet.max_row - 1, len(self.box_rows))
            self.assertEqual(trip_sheet["A2"].value, self.trips[0]["架次编号"])
            self.assertEqual(box_sheet["A2"].value, self.box_rows[0]["货箱编号"])
        finally:
            workbook.close()


if __name__ == "__main__":
    unittest.main()
