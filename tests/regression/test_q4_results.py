import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Q4ResultRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT_ROOT / "results/q4"
        cls.summary = json.loads(
            (cls.result_dir / "q4_summary.json").read_text(encoding="utf-8")
        )
        cls.rows = read_csv(cls.result_dir / "selected_partition.csv")
        cls.components = read_csv(cls.result_dir / "must_link_components.csv")
        cls.q3 = json.loads(
            (PROJECT_ROOT / "results/q3/q3_summary.json").read_text(encoding="utf-8")
        )

    def test_candidate_enumeration_and_expected_representatives(self):
        self.assertEqual(self.summary["status"], "PASS")
        self.assertEqual(self.summary["component_count"], 6)
        self.assertEqual(self.summary["results"]["2"]["candidate_count"], 31)
        self.assertEqual(self.summary["results"]["3"]["candidate_count"], 90)
        self.assertEqual(self.summary["results"]["2"]["groups"][1], ["S006"])
        self.assertEqual(self.summary["results"]["3"]["groups"][1:], [["S006"], ["S008"]])

    def test_each_k_is_complete_disjoint_and_preserves_components(self):
        expected = {"S{:03d}".format(index) for index in range(1, 16)}
        components = [set(row["服务区列表"].split(";")) for row in self.components]
        for k in (2, 3):
            groups = [
                set(row["服务区列表"].split(";"))
                for row in self.rows
                if int(row["K（2或3）"]) == k
            ]
            self.assertEqual(len(groups), k)
            self.assertTrue(all(groups))
            self.assertEqual(set().union(*groups), expected)
            self.assertEqual(sum(len(group) for group in groups), 15)
            for component in components:
                self.assertEqual(sum(component <= group for group in groups), 1)

    def test_workloads_recompose_frozen_s3(self):
        for k in (2, 3):
            rows = [row for row in self.rows if int(row["K（2或3）"]) == k]
            self.assertAlmostEqual(
                sum(float(row["总能耗（kWh）"]) for row in rows),
                self.q3["joint_metrics"]["total_energy_kwh"],
                places=9,
            )
            self.assertEqual(sum(float(row["货物质量（kg）"]) for row in rows), 758.0)

    def test_reported_shortage_vectors_match_selected_rows(self):
        resource_fields = [
            "A型运输无人机数",
            "B型运输无人机数",
            "C型运输无人机数",
            "A型电池组数",
            "B型电池组数",
            "C型电池组数",
            "中继无人机数",
            "中继能源组件数",
        ]
        stock = self.summary["stock_vector"]
        for k in (2, 3):
            rows = [row for row in self.rows if int(row["K（2或3）"]) == k]
            for field in resource_fields:
                total = sum(int(row[field]) for row in rows)
                self.assertEqual(
                    self.summary["results"][str(k)]["shortage_vector"][field],
                    max(total - int(stock[field]), 0),
                )


if __name__ == "__main__":
    unittest.main()

