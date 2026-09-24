"""Official appendix-3 radio-link calculations for Q3.

The transport energy model is provisional, but the communication equations in
this module are taken directly from the problem statement.  Keeping them in a
separate module makes the two definition sources impossible to confuse.
"""

from math import inf, log10, sqrt

from dproblem.domain.geometry import local_wgs84_distance_m


EPSILON = 1e-10


def _parameter(rows, category, name):
    matches = [
        row["参数值"]
        for row in rows
        if row["参数类别"] == category and row["参数名称"] == name
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one communication parameter: {}/{}".format(
                category, name
            )
        )
    return float(matches[0])


class CommunicationModel:
    """Link-budget constants and endpoint radio interfaces."""

    def __init__(self, rows):
        self.frequency_mhz = _parameter(rows, "传播参数", "载波频率（MHz）")
        self.system_loss_db = _parameter(rows, "传播参数", "系统损耗（dB）")
        self.obstruction_loss_db = _parameter(
            rows, "传播参数", "地形遮挡附加损耗（dB）"
        )
        self.receiver_sensitivity_dbm = _parameter(
            rows, "接收参数", "接收灵敏度（dBm）"
        )
        self.fade_margin_db = _parameter(rows, "接收参数", "衰落裕量（dB）")
        self.gateway_agl_m = _parameter(
            rows, "固定网关 G01", "天线离地高度（m）"
        )
        self.interfaces = {
            "transport": self._interface(rows, "运输无人机"),
            "relay_access": self._interface(rows, "中继接入端"),
            "relay_backhaul": self._interface(rows, "中继回传端"),
            "gateway": self._interface(rows, "固定网关 G01"),
        }

    @staticmethod
    def _interface(rows, category):
        return {
            "transmit_power_dbm": _parameter(rows, category, "发射功率（dBm）"),
            "antenna_gain_dbi": _parameter(rows, category, "天线增益（dBi）"),
        }

    @property
    def effective_receive_threshold_dbm(self):
        return self.receiver_sensitivity_dbm + self.fade_margin_db

    def directional_loss_limit_db(self, transmitter, receiver):
        source = self.interfaces[transmitter]
        target = self.interfaces[receiver]
        return (
            source["transmit_power_dbm"]
            + source["antenna_gain_dbi"]
            + target["antenna_gain_dbi"]
            - self.system_loss_db
            - self.effective_receive_threshold_dbm
        )

    def bidirectional_loss_limit_db(self, first, second):
        return min(
            self.directional_loss_limit_db(first, second),
            self.directional_loss_limit_db(second, first),
        )


def distance_3d_km(first, second):
    horizontal_m = local_wgs84_distance_m(
        first["lon"], first["lat"], second["lon"], second["lat"]
    )
    vertical_m = second["alt_m"] - first["alt_m"]
    return sqrt(horizontal_m * horizontal_m + vertical_m * vertical_m) / 1000.0


def free_space_path_loss_db(frequency_mhz, distance_km):
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    if distance_km < 0:
        raise ValueError("distance_km must be nonnegative")
    if distance_km == 0:
        return -inf
    return 32.45 + 20.0 * log10(frequency_mhz) + 20.0 * log10(distance_km)


def _cell_segment_interval(row1, col1, row2, col2, row, col):
    """Liang-Barsky interval where a raster-coordinate segment touches a cell."""

    t_low = 0.0
    t_high = 1.0
    for start, delta, lower, upper in (
        (col1, col2 - col1, float(col), float(col + 1)),
        (row1, row2 - row1, float(row), float(row + 1)),
    ):
        if abs(delta) <= EPSILON:
            if start < lower - EPSILON or start > upper + EPSILON:
                return None
            continue
        enter = (lower - start) / delta
        leave = (upper - start) / delta
        if enter > leave:
            enter, leave = leave, enter
        t_low = max(t_low, enter)
        t_high = min(t_high, leave)
        if t_low > t_high + EPSILON:
            return None
    return max(0.0, t_low), min(1.0, t_high)


def terrain_obstructed(dem, first, second, altitude_tolerance_m=1e-7):
    """Conservative supercover LOS test against every touched 30 m DEM cell.

    A DEM cell is treated as terrain at its recorded elevation across the full
    cell.  For a touched interval we compare that elevation with the minimum
    line-of-sight altitude in the interval.  This deliberately errs on the safe
    side at pixel boundaries and exact grid corners.
    """

    row1, col1 = dem.fractional_cell(first["lon"], first["lat"])
    row2, col2 = dem.fractional_cell(second["lon"], second["lat"])
    for row, col in dem.line_cells(
        first["lon"], first["lat"], second["lon"], second["lat"]
    ):
        value = float(dem.values[row, col])
        if dem.nodata is not None and value == dem.nodata:
            continue
        interval = _cell_segment_interval(row1, col1, row2, col2, row, col)
        if interval is None:
            continue
        low, high = interval
        altitude_low = first["alt_m"] + (second["alt_m"] - first["alt_m"]) * low
        altitude_high = first["alt_m"] + (second["alt_m"] - first["alt_m"]) * high
        if value >= min(altitude_low, altitude_high) - altitude_tolerance_m:
            return True
    return False


def audit_link(dem, model, first, second, first_interface, second_interface):
    distance_km = distance_3d_km(first, second)
    obstructed = terrain_obstructed(dem, first, second)
    fspl_db = free_space_path_loss_db(model.frequency_mhz, distance_km)
    path_loss_db = fspl_db + (model.obstruction_loss_db if obstructed else 0.0)
    limit_db = model.bidirectional_loss_limit_db(first_interface, second_interface)
    return {
        "distance_3d_km": distance_km,
        "terrain_obstructed": obstructed,
        "fspl_db": fspl_db,
        "obstruction_loss_db": model.obstruction_loss_db if obstructed else 0.0,
        "path_loss_db": path_loss_db,
        "bidirectional_loss_limit_db": limit_db,
        "margin_db": limit_db - path_loss_db,
        "available": path_loss_db <= limit_db + EPSILON,
    }
