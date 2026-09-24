import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.q1.work_time import per_trip_work_time_seconds  # noqa: E402


class WorkTimeTests(unittest.TestCase):
    def setUp(self):
        self.drone_type = {
            "工位固定准备时间（s）": 300,
            "每箱装载时间（s）": 30,
            "接收点基础交接时间（s）": 150,
            "每箱增加交接时间（s）": 30,
        }

    def test_type_a_two_boxes(self):
        self.assertEqual(
            per_trip_work_time_seconds(self.drone_type, 2, 600), 1170
        )

    def test_negative_flight_time_rejected(self):
        with self.assertRaises(ValueError):
            per_trip_work_time_seconds(self.drone_type, 1, -1)


if __name__ == "__main__":
    unittest.main()
