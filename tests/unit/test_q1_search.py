import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.q1.solver import (  # noqa: E402
    PlanState,
    TripOption,
    exact_pareto_partition,
    local_search_from_baseline,
)


def option(mask, energy, work, type_id="T"):
    return TripOption(mask, type_id, 1, 1, 1, 1, work, energy, 10, 0.9)


class Q1SearchTests(unittest.TestCase):
    def test_pair_repartition_matches_exact_trip_count(self):
        candidates = {
            1: (option(1, 1, 1),),
            2: (option(2, 1, 1),),
            4: (option(4, 1, 1),),
            3: (option(3, 1.5, 1.5),),
            5: (option(5, 1.5, 1.5),),
            6: (option(6, 1.5, 1.5),),
        }
        baseline = PlanState(3, 3, 3, (candidates[1][0], candidates[2][0], candidates[4][0]))
        local, log = local_search_from_baseline(baseline, candidates)
        exact = exact_pareto_partition([{}, {}, {}], candidates)[0]
        self.assertEqual(local.trip_count, 2)
        self.assertEqual(local.trip_count, exact.trip_count)
        self.assertGreaterEqual(log["operator_counts"]["pair_repartition"], 1)

    def test_type_replacement_improves_same_batch(self):
        baseline_option = option(1, 2.0, 2.0, "A")
        improved_option = option(1, 1.0, 1.0, "B")
        candidates = {1: (baseline_option, improved_option)}
        baseline = PlanState(1, 2.0, 2.0, (baseline_option,))
        local, log = local_search_from_baseline(baseline, candidates)
        self.assertEqual(local.trips[0].type_id, "B")
        self.assertEqual(log["operator_counts"]["type_replacement"], 1)

    def test_pair_repartition_covers_one_box_move(self):
        candidates = {
            3: (option(3, 2.0, 2.0),),
            4: (option(4, 2.0, 2.0),),
            1: (option(1, 1.0, 1.0),),
            6: (option(6, 1.0, 1.0),),
        }
        baseline = PlanState(2, 4.0, 4.0, (candidates[3][0], candidates[4][0]))
        local, log = local_search_from_baseline(baseline, candidates)
        self.assertEqual({trip.mask for trip in local.trips}, {1, 6})
        self.assertGreaterEqual(log["operator_counts"]["pair_repartition"], 1)

    def test_pair_repartition_covers_box_exchange(self):
        candidates = {
            3: (option(3, 2.0, 2.0),),
            12: (option(12, 2.0, 2.0),),
            5: (option(5, 1.0, 1.0),),
            10: (option(10, 1.0, 1.0),),
        }
        baseline = PlanState(2, 4.0, 4.0, (candidates[3][0], candidates[12][0]))
        local, log = local_search_from_baseline(baseline, candidates)
        self.assertEqual({trip.mask for trip in local.trips}, {5, 10})
        self.assertGreaterEqual(log["operator_counts"]["pair_repartition"], 1)

    def test_triple_repartition_escapes_pair_local_optimum(self):
        candidates = {
            3: (option(3, 1.0, 1.0),),
            12: (option(12, 1.0, 1.0),),
            48: (option(48, 1.0, 1.0),),
            21: (option(21, 1.4, 1.4),),
            42: (option(42, 1.4, 1.4),),
        }
        # Incumbent batches are {0,1}, {2,3}, {4,5}. No pair union has a
        # one-batch candidate, while the triple union can be repartitioned as
        # {0,2,4} and {1,3,5}.
        incumbent = (candidates[3][0], candidates[12][0], candidates[48][0])
        baseline = PlanState(3, 3.0, 3.0, incumbent)
        local, log = local_search_from_baseline(baseline, candidates)
        self.assertEqual(local.trip_count, 2)
        self.assertGreaterEqual(log["operator_counts"]["triple_repartition"], 1)


if __name__ == "__main__":
    unittest.main()
