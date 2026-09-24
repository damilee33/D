import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.q1.smoke import run_single_box_smoke  # noqa: E402


class Q1SmokeTests(unittest.TestCase):
    def test_s001_water_box_type_a_is_feasible(self):
        result = run_single_box_smoke(PROJECT_ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertLessEqual(result["total_energy_kwh"], result["energy_limit_kwh"])
        self.assertGreater(result["q1_work_time_seconds"], result["round_trip_flight_seconds"])
        self.assertTrue(all(item["pass"] for item in result["constraints"].values()))
        self.assertEqual(len(result["energy_config_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
