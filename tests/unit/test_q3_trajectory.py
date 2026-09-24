import unittest

from dproblem.q3.trajectory import build_flight_phases, position_at, sample_phase


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.nodes = {
            "O01": {"经度（°）": 100.0, "纬度（°）": 20.0, "海拔（m）": 100.0},
            "S001": {"经度（°）": 100.01, "纬度（°）": 20.01, "海拔（m）": 200.0},
        }
        self.drone_type = {
            "最大爬升速度（m/s）": 2.0,
            "计划巡航速度（m/s）": 10.0,
            "最大下降速度（m/s）": 5.0,
        }
        self.segment = {
            "起点": "O01",
            "终点": "S001",
            "爬升（m）": 200.0,
            "水平距离（m）": 1000.0,
            "下降（m）": 70.0,
            "开始时刻（s）": 10.0,
            "结束时刻（s）": 224.0,
        }

    def test_phase_boundaries_and_positions(self):
        phases = build_flight_phases(self.segment, self.drone_type, self.nodes)
        self.assertEqual([phase["phase"] for phase in phases], ["climb", "cruise", "descent"])
        self.assertEqual(phases[0]["start_seconds"], 10.0)
        self.assertEqual(phases[0]["end_seconds"], 110.0)
        self.assertEqual(phases[1]["end_seconds"], 210.0)
        self.assertEqual(phases[2]["end_seconds"], 224.0)
        midpoint = position_at(phases[1], 160.0)
        self.assertAlmostEqual(midpoint["lon"], 100.005)
        self.assertAlmostEqual(midpoint["lat"], 20.005)
        self.assertAlmostEqual(midpoint["alt_m"], 300.0)

    def test_sampling_includes_both_boundaries(self):
        phase = build_flight_phases(self.segment, self.drone_type, self.nodes)[2]
        samples = sample_phase(phase, 5.0)
        self.assertEqual([row[0] for row in samples], [210.0, 215.0, 220.0, 224.0])


if __name__ == "__main__":
    unittest.main()
