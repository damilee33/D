import csv
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Q1ResultRegressionTests(unittest.TestCase):
    def test_q1_summary_and_cargo_coverage(self):
        result_dir = PROJECT_ROOT / "results" / "q1"
        summary = json.loads((result_dir / "q1_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["main_trip_count"], 18)
        self.assertEqual(summary["baseline_trip_count"], 18)
        self.assertEqual(summary["local_search_trip_count_gap"], 0)
        self.assertEqual(summary["main_constraint_audit"]["violations"], 0)
        self.assertTrue(summary["deterministic"])
        self.assertIsNone(summary["random_seed"])
        self.assertEqual(len(summary["input_manifest_sha256"]), 64)
        self.assertEqual(len(summary["model_config_sha256"]), 64)

        with (result_dir / "main_trips.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        cargo_ids = [
            cargo_id
            for row in rows
            for cargo_id in row["货箱编号列表"].split(";")
        ]
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(cargo_ids), 80)
        self.assertEqual(len(set(cargo_ids)), 80)

    def test_safe_payload_decreases_with_reserve(self):
        path = PROJECT_ROOT / "results" / "q1" / "safe_payload_sensitivity.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        grouped = {}
        for row in rows:
            key = (row["服务区编号"], row["机型编号"])
            grouped.setdefault(key, []).append(
                (float(row["返航余量"]), float(row["最大安全载荷（kg）"]))
            )
        self.assertEqual(len(grouped), 45)
        self.assertEqual(len(rows), 225)
        for key, values in grouped.items():
            sorted_values = sorted(values)
            self.assertEqual(
                [round(reserve, 2) for reserve, _ in sorted_values],
                [0.10, 0.15, 0.20, 0.25, 0.30],
            )
            ordered = [payload for _, payload in sorted_values]
            with self.subTest(key=key):
                for left, right in zip(ordered, ordered[1:]):
                    self.assertGreaterEqual(left + 1e-9, right)


if __name__ == "__main__":
    unittest.main()
