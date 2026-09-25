import csv
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.io.dataset import load_project_data  # noqa: E402


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Q3ResultRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT_ROOT / "results" / "q3"
        cls.summary = json.loads(
            (cls.result_dir / "q3_summary.json").read_text(encoding="utf-8")
        )
        cls.trips = read_csv(cls.result_dir / "transport_trips.csv")
        cls.deliveries = read_csv(cls.result_dir / "transport_box_deliveries.csv")
        cls.batteries = read_csv(cls.result_dir / "transport_battery_timeline.csv")
        cls.sorties = read_csv(cls.result_dir / "relay_sorties.csv")
        cls.communication = read_csv(cls.result_dir / "communication_guarantee.csv")
        cls.data = load_project_data(PROJECT_ROOT)

    def test_summary_meets_robust_communication_gate(self):
        self.assertEqual(self.summary["status"], "PASS")
        self.assertEqual(self.summary["constraint_audit"]["status"], "PASS")
        audit = self.summary["communication_audit"]
        self.assertEqual(audit["violations"], 0)
        self.assertGreaterEqual(audit["minimum_adopted_relay_link_margin_db"], 1.0)
        self.assertGreaterEqual(audit["minimum_adopted_direct_margin_db"], 0.0)

    def test_transport_cargo_deadlines_and_resources(self):
        cargo_ids = [
            cargo_id
            for row in self.trips
            for cargo_id in row["货箱编号列表"].split(";")
        ]
        self.assertEqual(len(cargo_ids), 80)
        self.assertEqual(len(set(cargo_ids)), 80)
        self.assertFalse(
            any(row["硬截止违例"].lower() == "true" for row in self.deliveries)
        )
        for field in ("无人机编号", "电池编号"):
            groups = defaultdict(list)
            for row in self.trips:
                groups[row[field]].append(
                    (float(row["开始时刻（s）"]), float(row["返回O01时刻（s）"]))
                )
            for intervals in groups.values():
                intervals.sort()
                self.assertTrue(
                    all(
                        left[1] <= right[0] + 1e-8
                        for left, right in zip(intervals, intervals[1:])
                    )
                )
        battery_groups = defaultdict(list)
        for row in self.batteries:
            battery_groups[row["电池编号"]].append(row)
        for rows in battery_groups.values():
            rows.sort(key=lambda row: float(row["任务开始时刻（s）"]))
            self.assertTrue(
                all(
                    float(left["再次可用时刻（s）"])
                    <= float(right["任务开始时刻（s）"]) + 1e-8
                    for left, right in zip(rows, rows[1:])
                )
            )

    def test_relay_inventory_energy_and_continuity(self):
        relay_type = self.data["relay"]["types"][0]
        module_limit = int(self.data["relay"]["modules"][0]["共享能源组件总数（组）"])
        self.assertLessEqual(len({row["能源组件编号"] for row in self.sorties}), module_limit)
        self.assertLessEqual(
            len({row["中继无人机编号"] for row in self.sorties}),
            len(self.data["relay"]["drones"]),
        )
        self.assertTrue(
            all(
                float(row["返航SOC（%）"])
                >= float(relay_type["返航电量下限（%）"]) - 1e-8
                for row in self.sorties
            )
        )
        self.assertFalse(any(row["保障方式"] == "中断" for row in self.communication))


if __name__ == "__main__":
    unittest.main()
