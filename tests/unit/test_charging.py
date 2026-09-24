import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.domain.charging import charge_time_to_full_seconds  # noqa: E402


class ChargingTests(unittest.TestCase):
    def test_required_boundaries(self):
        full = 2000.0
        self.assertAlmostEqual(charge_time_to_full_seconds(0.0, full), full)
        self.assertAlmostEqual(charge_time_to_full_seconds(0.9, full), 0.35 * full)
        self.assertAlmostEqual(charge_time_to_full_seconds(1.0, full), 0.0)

    def test_crossing_ninety_percent(self):
        full = 2000.0
        expected = full * (0.65 * (0.9 - 0.5) / 0.9 + 0.35)
        self.assertAlmostEqual(charge_time_to_full_seconds(0.5, full), expected)
        self.assertGreater(
            charge_time_to_full_seconds(0.899, full),
            charge_time_to_full_seconds(0.901, full),
        )

    def test_time_is_monotone_nonincreasing_in_soc(self):
        values = [charge_time_to_full_seconds(index / 100.0, 1800.0) for index in range(101)]
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_invalid_input_rejected(self):
        for soc in (-0.01, 1.01, float("nan")):
            with self.subTest(soc=soc), self.assertRaises(ValueError):
                charge_time_to_full_seconds(soc, 1800.0)
        with self.assertRaises(ValueError):
            charge_time_to_full_seconds(0.5, -1.0)


if __name__ == "__main__":
    unittest.main()
