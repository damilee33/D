import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.domain.energy import (  # noqa: E402
    climb_energy_kwh,
    equivalent_range_m,
    horizontal_energy_kwh,
    relay_power_energy_kwh,
)


class EnergyTests(unittest.TestCase):
    def test_equivalent_range_endpoints(self):
        self.assertAlmostEqual(equivalent_range_m(0, 25, 25000, 20000, 1.5), 25000)
        self.assertAlmostEqual(equivalent_range_m(25, 25, 25000, 20000, 1.5), 20000)

    def test_horizontal_energy_at_equivalent_range(self):
        self.assertAlmostEqual(
            horizontal_energy_kwh(25000, 0, 25, 25000, 20000, 4.5, 1.5), 4.5
        )

    def test_climb_energy_is_zero_without_gain(self):
        self.assertEqual(climb_energy_kwh(70, 20, 0, 0.72, 9.80665), 0)

    def test_relay_power_integrates_kw_seconds(self):
        self.assertAlmostEqual(
            relay_power_energy_kwh(1.15, 3600, 1.05, 0.05, 3600), 2.25
        )

    def test_payload_over_capacity_rejected(self):
        with self.assertRaises(ValueError):
            equivalent_range_m(26, 25, 25000, 20000, 1.5)

    def test_nonfinite_parameters_rejected(self):
        invalid_calls = [
            lambda: equivalent_range_m(1, float("nan"), 25000, 20000, 1.5),
            lambda: equivalent_range_m(1, 25, float("nan"), 20000, 1.5),
            lambda: equivalent_range_m(1, 25, 25000, 20000, float("nan")),
            lambda: climb_energy_kwh(70, 1, 10, 0.72, float("nan")),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
