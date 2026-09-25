import unittest

from dproblem.q3.direct import extract_conservative_gaps


class DirectGapTests(unittest.TestCase):
    def test_conservative_gap_uses_adjacent_sample_cells(self):
        samples = [
            {"trip_id": "T", "drone_id": "U", "time_seconds": 0.0, "available": True, "margin_db": 1.0},
            {"trip_id": "T", "drone_id": "U", "time_seconds": 5.0, "available": False, "margin_db": -2.0},
            {"trip_id": "T", "drone_id": "U", "time_seconds": 10.0, "available": True, "margin_db": 0.5},
            {"trip_id": "T", "drone_id": "U", "time_seconds": 15.0, "available": True, "margin_db": 2.0},
        ]
        gaps = extract_conservative_gaps(samples)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start_seconds"], 0.0)
        self.assertEqual(gaps[0]["end_seconds"], 10.0)
        self.assertEqual(gaps[0]["duration_seconds"], 10.0)
        self.assertEqual(gaps[0]["minimum_direct_margin_db"], -2.0)


if __name__ == "__main__":
    unittest.main()
