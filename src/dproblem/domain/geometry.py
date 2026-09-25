"""Local WGS-84 distance and DEM line-profile utilities."""

from math import cos, floor, hypot, inf, isfinite, pi, sin, sqrt

import numpy as np
from PIL import Image


WGS84_A_M = 6378137.0
WGS84_E2 = 6.69437999014e-3


def local_wgs84_distance_m(lon1, lat1, lon2, lat2):
    """Distance in a local WGS-84 tangent metric, suitable for this small area."""

    mean_lat = (lat1 + lat2) * 0.5 * pi / 180.0
    sin_lat = sin(mean_lat)
    denominator = sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    prime_vertical_radius = WGS84_A_M / denominator
    meridional_radius = WGS84_A_M * (1.0 - WGS84_E2) / denominator ** 3
    dx = prime_vertical_radius * cos(mean_lat) * (lon2 - lon1) * pi / 180.0
    dy = meridional_radius * (lat2 - lat1) * pi / 180.0
    return hypot(dx, dy)


class RasterDEM:
    def __init__(self, path):
        with Image.open(path) as image:
            self.values = np.asarray(image).copy()
            tags = image.tag_v2
            scale = tags.get(33550)
            tiepoint = tags.get(33922)
            if not scale or not tiepoint:
                raise ValueError("DEM lacks GeoTIFF pixel scale or tiepoint")
            self.lon0 = float(tiepoint[3])
            self.lat0 = float(tiepoint[4])
            self.lon_step = float(scale[0])
            self.lat_step = float(scale[1])
            nodata_raw = tags.get(42113)
            self.nodata = float(nodata_raw) if nodata_raw is not None else None
        self.height, self.width = self.values.shape

    def fractional_cell(self, lon, lat):
        col = (lon - self.lon0) / self.lon_step
        row = (self.lat0 - lat) / self.lat_step
        return row, col

    def contains(self, lon, lat):
        row, col = self.fractional_cell(lon, lat)
        tolerance = 1e-12
        return (
            -tolerance <= row <= self.height + tolerance
            and -tolerance <= col <= self.width + tolerance
        )

    def elevation_at_cell(self, lon, lat):
        """Return the recorded elevation of the containing DEM cell."""

        if not self.contains(lon, lat):
            raise ValueError("point lies outside DEM")
        row, col = self.fractional_cell(lon, lat)
        row_index = min(self.height - 1, max(0, int(floor(row))))
        col_index = min(self.width - 1, max(0, int(floor(col))))
        value = float(self.values[row_index, col_index])
        if self.nodata is not None and value == self.nodata:
            raise ValueError("point lies in a DEM nodata cell")
        return value

    def line_cells(self, lon1, lat1, lon2, lat2):
        """Return a conservative supercover of every grid cell touched by a segment."""
        row1, col1 = self.fractional_cell(lon1, lat1)
        row2, col2 = self.fractional_cell(lon2, lat2)
        if not (self.contains(lon1, lat1) and self.contains(lon2, lat2)):
            raise ValueError("segment endpoint lies outside DEM")

        delta_col = col2 - col1
        delta_row = row2 - row1
        step_col = 1 if delta_col > 0 else (-1 if delta_col < 0 else 0)
        step_row = 1 if delta_row > 0 else (-1 if delta_row < 0 else 0)
        def cell_index(coordinate, size):
            return min(size - 1, max(0, int(floor(coordinate))))

        col = cell_index(col1, self.width)
        row = cell_index(row1, self.height)
        end_col = cell_index(col2, self.width)
        end_row = cell_index(row2, self.height)

        t_delta_col = abs(1.0 / delta_col) if delta_col else inf
        t_delta_row = abs(1.0 / delta_row) if delta_row else inf
        next_col_boundary = col + 1 if step_col > 0 else col
        next_row_boundary = row + 1 if step_row > 0 else row
        t_max_col = (next_col_boundary - col1) / delta_col if delta_col else inf
        t_max_row = (next_row_boundary - row1) / delta_row if delta_row else inf

        cells = []
        seen = set()

        def add(cell_row, cell_col):
            if 0 <= cell_row < self.height and 0 <= cell_col < self.width:
                cell = (cell_row, cell_col)
                if cell not in seen:
                    seen.add(cell)
                    cells.append(cell)

        add(row, col)
        while row != end_row or col != end_col:
            # Events at t=1 belong to the endpoint and are handled by
            # add_endpoint_touches below. Parameter-based termination prevents
            # outer-edge endpoints from stepping outside the raster forever.
            if min(t_max_col, t_max_row) >= 1.0 - 1e-12:
                break
            event_tolerance = 1e-12 * max(
                1.0, abs(t_max_col), abs(t_max_row)
            )
            event_difference = t_max_col - t_max_row
            if event_difference < -event_tolerance:
                col += step_col
                t_max_col += t_delta_col
                add(row, col)
            elif event_difference > event_tolerance:
                row += step_row
                t_max_row += t_delta_row
                add(row, col)
            else:
                # At an exact corner, include both side cells as well as the
                # diagonal cell so a zero-width corner contact cannot hide a peak.
                add(row, col + step_col)
                add(row + step_row, col)
                col += step_col
                row += step_row
                t_max_col += t_delta_col
                t_max_row += t_delta_row
                add(row, col)

        tolerance = 1e-12
        horizontal_boundary = (
            abs(delta_row) <= tolerance
            and abs(row1 - round(row1)) <= tolerance
        )
        vertical_boundary = (
            abs(delta_col) <= tolerance
            and abs(col1 - round(col1)) <= tolerance
        )
        horizontal_boundary_index = int(round(row1))
        vertical_boundary_index = int(round(col1))
        if horizontal_boundary and 0 < horizontal_boundary_index < self.height:
            for cell_row, cell_col in list(cells):
                add(cell_row - 1, cell_col)
        if vertical_boundary and 0 < vertical_boundary_index < self.width:
            for cell_row, cell_col in list(cells):
                add(cell_row, cell_col - 1)

        def add_endpoint_touches(endpoint_row, endpoint_col):
            base_row = cell_index(endpoint_row, self.height)
            base_col = cell_index(endpoint_col, self.width)
            rows = [base_row]
            cols = [base_col]
            if (
                abs(endpoint_row - round(endpoint_row)) <= tolerance
                and 0 < int(round(endpoint_row)) < self.height
            ):
                rows.append(base_row - 1)
            if (
                abs(endpoint_col - round(endpoint_col)) <= tolerance
                and 0 < int(round(endpoint_col)) < self.width
            ):
                cols.append(base_col - 1)
            for endpoint_cell_row in rows:
                for endpoint_cell_col in cols:
                    add(endpoint_cell_row, endpoint_cell_col)

        add_endpoint_touches(row1, col1)
        add_endpoint_touches(row2, col2)
        return cells

    def maximum_elevation_on_line(self, lon1, lat1, lon2, lat2):
        elevations = []
        for row, col in self.line_cells(lon1, lat1, lon2, lat2):
            value = float(self.values[row, col])
            if self.nodata is None or value != self.nodata:
                elevations.append(value)
        if not elevations:
            raise ValueError("DEM segment contains no valid elevation cells")
        return max(elevations)


def segment_terrain_profile(
    dem,
    origin,
    destination,
    origin_work_offset_m,
    destination_work_offset_m,
    clearance_m=50.0,
):
    """Compute a directed segment profile with explicit endpoint work heights."""

    for name, value in (
        ("origin_work_offset_m", origin_work_offset_m),
        ("destination_work_offset_m", destination_work_offset_m),
        ("clearance_m", clearance_m),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(name))
    distance_m = local_wgs84_distance_m(
        origin["经度（°）"],
        origin["纬度（°）"],
        destination["经度（°）"],
        destination["纬度（°）"],
    )
    maximum_ground_m = dem.maximum_elevation_on_line(
        origin["经度（°）"],
        origin["纬度（°）"],
        destination["经度（°）"],
        destination["纬度（°）"],
    )
    cruise_altitude_m = maximum_ground_m + clearance_m
    origin_work_altitude_m = origin["海拔（m）"] + origin_work_offset_m
    destination_work_altitude_m = destination["海拔（m）"] + destination_work_offset_m
    return {
        "horizontal_distance_m": distance_m,
        "maximum_ground_elevation_m": maximum_ground_m,
        "cruise_altitude_m": cruise_altitude_m,
        "origin_work_altitude_m": origin_work_altitude_m,
        "destination_work_altitude_m": destination_work_altitude_m,
        "forward_climb_m": max(0.0, cruise_altitude_m - origin_work_altitude_m),
        "forward_descent_m": max(0.0, cruise_altitude_m - destination_work_altitude_m),
        "reverse_climb_m": max(0.0, cruise_altitude_m - destination_work_altitude_m),
        "reverse_descent_m": max(0.0, cruise_altitude_m - origin_work_altitude_m),
    }
