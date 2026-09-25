import csv
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.compat import import_openpyxl_load_workbook  # noqa: E402


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FinalSubmissionWorkbookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workbook = import_openpyxl_load_workbook()(
            PROJECT_ROOT / "results/结果提交_Q1-Q4.xlsx", data_only=False
        )

    @classmethod
    def tearDownClass(cls):
        cls.workbook.close()

    def test_sheet_names_and_row_counts(self):
        expected = {
            "Q1_单点组批": 18,
            "Q2_运输架次": 23,
            "Q2_逐箱交付": 80,
            "Q3_中继架次": 6,
            "Q3_通信保障": 199,
            "Q4_分区配置": 5,
        }
        self.assertEqual(list(expected), self.workbook.sheetnames)
        for sheet_name, count in expected.items():
            sheet = self.workbook[sheet_name]
            populated = sum(
                any(cell.value is not None for cell in row)
                for row in sheet.iter_rows(min_row=2)
            )
            self.assertEqual(populated, count)

    def test_final_q2_uses_q3_adjusted_joint_schedule(self):
        csv_rows = read_csv(PROJECT_ROOT / "results/q3/transport_trips.csv")
        sheet = self.workbook["Q2_运输架次"]
        by_id = {sheet.cell(row, 1).value: row for row in range(2, sheet.max_row + 1)}
        self.assertEqual(set(by_id), {row["架次编号"] for row in csv_rows})
        for row in csv_rows:
            excel_row = by_id[row["架次编号"]]
            self.assertAlmostEqual(
                float(sheet.cell(excel_row, 5).value), float(row["开始时刻（s）"]), places=8
            )
            self.assertAlmostEqual(
                float(sheet.cell(excel_row, 7).value), float(row["返回O01时刻（s）"]), places=8
            )

    def test_q1_volume_visibility_and_no_formulas(self):
        sheet = self.workbook["Q1_单点组批"]
        for row in range(2, 20):
            self.assertEqual(sheet.cell(row, 6).number_format, "0.000000")
        formulas = [
            cell.coordinate
            for sheet in self.workbook.worksheets
            for row in sheet.iter_rows()
            for cell in row
            if isinstance(cell.value, str) and cell.value.startswith("=")
        ]
        self.assertEqual(formulas, [])


if __name__ == "__main__":
    unittest.main()
