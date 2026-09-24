import math
import unittest

import numpy as np

from dproblem.domain.communication import (
    CommunicationModel,
    audit_link,
    free_space_path_loss_db,
    terrain_obstructed,
)


def communication_rows():
    values = [
        ("传播参数", "载波频率（MHz）", 2400),
        ("传播参数", "系统损耗（dB）", 3),
        ("传播参数", "地形遮挡附加损耗（dB）", 10),
        ("接收参数", "接收灵敏度（dBm）", -98),
        ("接收参数", "衰落裕量（dB）", 8),
        ("运输无人机", "发射功率（dBm）", 20),
        ("运输无人机", "天线增益（dBi）", 3),
        ("中继接入端", "发射功率（dBm）", 20),
        ("中继接入端", "天线增益（dBi）", 6),
        ("中继回传端", "发射功率（dBm）", 19),
        ("中继回传端", "天线增益（dBi）", 8),
        ("固定网关 G01", "发射功率（dBm）", 27),
        ("固定网关 G01", "天线增益（dBi）", 12),
        ("固定网关 G01", "天线离地高度（m）", 20),
    ]
    return [
        {"参数类别": category, "参数名称": name, "参数值": value}
        for category, name, value in values
    ]


class FlatDem:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.height, self.width = self.values.shape
        self.nodata = None

    def fractional_cell(self, lon, lat):
        return lat, lon

    def line_cells(self, lon1, lat1, lon2, lat2):
        # The unit tests use a horizontal line through these cells.
        return [(0, index) for index in range(self.width)]


class CommunicationTest(unittest.TestCase):
    def test_official_link_budgets(self):
        model = CommunicationModel(communication_rows())
        self.assertEqual(model.effective_receive_threshold_dbm, -90.0)
        self.assertEqual(
            model.bidirectional_loss_limit_db("transport", "gateway"), 122.0
        )
        self.assertEqual(
            model.bidirectional_loss_limit_db("transport", "relay_access"), 116.0
        )
        self.assertEqual(
            model.bidirectional_loss_limit_db("relay_backhaul", "gateway"), 126.0
        )

    def test_fspl_uses_mhz_and_km(self):
        expected = 32.45 + 20.0 * math.log10(2400.0)
        self.assertAlmostEqual(free_space_path_loss_db(2400.0, 1.0), expected)

    def test_los_clear_and_obstructed(self):
        first = {"lon": 0.1, "lat": 0.5, "alt_m": 100.0}
        second = {"lon": 2.9, "lat": 0.5, "alt_m": 100.0}
        self.assertFalse(terrain_obstructed(FlatDem([[90, 90, 90]]), first, second))
        self.assertTrue(terrain_obstructed(FlatDem([[90, 110, 90]]), first, second))

    def test_obstruction_loss_is_applied_once(self):
        model = CommunicationModel(communication_rows())
        first = {"lon": 0.1, "lat": 0.5, "alt_m": 100.0}
        second = {"lon": 2.9, "lat": 0.5, "alt_m": 100.0}
        clear = audit_link(
            FlatDem([[90, 90, 90]]),
            model,
            first,
            second,
            "transport",
            "gateway",
        )
        blocked = audit_link(
            FlatDem([[90, 110, 90]]),
            model,
            first,
            second,
            "transport",
            "gateway",
        )
        self.assertAlmostEqual(blocked["path_loss_db"] - clear["path_loss_db"], 10.0)


if __name__ == "__main__":
    unittest.main()
