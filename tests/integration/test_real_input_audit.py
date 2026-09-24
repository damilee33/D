import sys
import unittest
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.io.dataset import load_project_data  # noqa: E402
from dproblem.validation.audit import (  # noqa: E402
    _validate_demand_consistency,
    build_data_audit,
)


class RealInputAuditTests(unittest.TestCase):
    def test_supplied_dataset_passes_contract(self):
        audit = build_data_audit(PROJECT_ROOT)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["counts"]["cargo_boxes"], 80)
        self.assertEqual(
            audit["model_contract"]["energy_model_version"], "provisional_v0"
        )

    def test_mutated_deadlines_are_rejected(self):
        demands = deepcopy(load_project_data(PROJECT_ROOT)["demands"])
        demands["boxes"][0]["期望送达时间（s）"] = 999
        with self.assertRaises(AssertionError):
            _validate_demand_consistency(demands["summary"], demands["boxes"])

        demands = deepcopy(load_project_data(PROJECT_ROOT)["demands"])
        demands["boxes"][0]["首批截止时间（s）"] = 999
        with self.assertRaises(AssertionError):
            _validate_demand_consistency(demands["summary"], demands["boxes"])


if __name__ == "__main__":
    unittest.main()
