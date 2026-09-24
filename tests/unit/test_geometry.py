import sys
import unittest
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dproblem.domain.geometry import (  # noqa: E402
    RasterDEM,
    local_wgs84_distance_m,
    segment_terrain_profile,
)
from dproblem.io.dataset import load_project_data  # noqa: E402


class GeometryTests(unittest.TestCase):
    @staticmethod
    def synthetic_dem():
        dem = RasterDEM.__new__(RasterDEM)
        dem.width = 6
        dem.height = 5
        dem.lon0 = 0.0
        dem.lat0 = 5.0
        dem.lon_step = 1.0
        dem.lat_step = 1.0
        dem.nodata = None
        return dem

    @staticmethod
    def closed_rectangle_oracle(row1, col1, row2, col2, height, width):
        cells = set()
        delta_row = row2 - row1
        delta_col = col2 - col1
        tolerance = 1e-12
        for row in range(height):
            for col in range(width):
                lower = 0.0
                upper = 1.0
                intersects = True
                for coordinate, delta, minimum, maximum in (
                    (row1, delta_row, row, row + 1),
                    (col1, delta_col, col, col + 1),
                ):
                    if abs(delta) <= tolerance:
                        if coordinate < minimum - tolerance or coordinate > maximum + tolerance:
                            intersects = False
                            break
                    else:
                        first = (minimum - coordinate) / delta
                        second = (maximum - coordinate) / delta
                        enter, leave = sorted((first, second))
                        lower = max(lower, enter)
                        upper = min(upper, leave)
                        if lower > upper + tolerance:
                            intersects = False
                            break
                if intersects:
                    cells.add((row, col))
        return cells

    def test_zero_distance(self):
        self.assertEqual(local_wgs84_distance_m(109.2, 23.0, 109.2, 23.0), 0)

    def test_distance_is_symmetric(self):
        forward = local_wgs84_distance_m(109.2, 23.0, 109.25, 23.05)
        reverse = local_wgs84_distance_m(109.25, 23.05, 109.2, 23.0)
        self.assertAlmostEqual(forward, reverse, places=9)

    def test_strict_supercover_catches_s003_peak(self):
        data = load_project_data(PROJECT_ROOT)
        origin = data["nodes"]["centers"][0]
        destination = next(
            row
            for row in data["nodes"]["service_areas"]
            if row["服务区编号"] == "S003"
        )
        dem = RasterDEM(data["paths"]["dem_tif"])
        cells = dem.line_cells(
            origin["经度（°）"],
            origin["纬度（°）"],
            destination["经度（°）"],
            destination["纬度（°）"],
        )
        self.assertEqual(len(cells), 339)
        self.assertAlmostEqual(
            max(float(dem.values[row, col]) for row, col in cells),
            511.80413818359375,
            places=5,
        )
        reverse = dem.line_cells(
            destination["经度（°）"],
            destination["纬度（°）"],
            origin["经度（°）"],
            origin["纬度（°）"],
        )
        self.assertEqual(set(cells), set(reverse))

    def test_all_supplied_node_pairs_have_symmetric_supercovers(self):
        data = load_project_data(PROJECT_ROOT)
        nodes = data["nodes"]["centers"] + data["nodes"]["service_areas"]
        dem = RasterDEM(data["paths"]["dem_tif"])
        for first_index, first in enumerate(nodes):
            self.assertTrue(dem.contains(first["经度（°）"], first["纬度（°）"]))
            for second in nodes[first_index + 1 :]:
                forward = dem.line_cells(
                    first["经度（°）"], first["纬度（°）"], second["经度（°）"], second["纬度（°）"]
                )
                reverse = dem.line_cells(
                    second["经度（°）"], second["纬度（°）"], first["经度（°）"], first["纬度（°）"]
                )
                self.assertEqual(set(forward), set(reverse))

    def test_service_to_service_profile_uses_explicit_work_offsets(self):
        data = load_project_data(PROJECT_ROOT)
        first, second = data["nodes"]["service_areas"][:2]
        dem = RasterDEM(data["paths"]["dem_tif"])
        profile = segment_terrain_profile(
            dem,
            first,
            second,
            origin_work_offset_m=30,
            destination_work_offset_m=30,
        )
        self.assertAlmostEqual(profile["origin_work_altitude_m"], first["海拔（m）"] + 30)
        self.assertAlmostEqual(profile["destination_work_altitude_m"], second["海拔（m）"] + 30)
        self.assertEqual(profile["forward_climb_m"], profile["reverse_descent_m"])
        self.assertEqual(profile["forward_descent_m"], profile["reverse_climb_m"])

    def test_horizontal_internal_boundary_includes_both_sides(self):
        dem = self.synthetic_dem()
        cells = dem.line_cells(0.2, 3.0, 4.8, 3.0)
        self.assertEqual(set(cells), {(row, col) for row in (1, 2) for col in range(5)})
        reverse = dem.line_cells(4.8, 3.0, 0.2, 3.0)
        self.assertEqual(set(cells), set(reverse))

    def test_vertical_internal_boundary_includes_both_sides(self):
        dem = self.synthetic_dem()
        cells = dem.line_cells(3.0, 4.8, 3.0, 1.2)
        self.assertEqual(set(cells), {(row, col) for row in range(4) for col in (2, 3)})
        reverse = dem.line_cells(3.0, 1.2, 3.0, 4.8)
        self.assertEqual(set(cells), set(reverse))

    def test_outer_edge_and_zero_length_boundary(self):
        dem = self.synthetic_dem()
        top = dem.line_cells(0.2, 5.0, 4.8, 5.0)
        bottom = dem.line_cells(0.2, 0.0, 4.8, 0.0)
        left = dem.line_cells(0.0, 4.8, 0.0, 1.2)
        right = dem.line_cells(6.0, 4.8, 6.0, 1.2)
        self.assertEqual(set(top), {(0, col) for col in range(5)})
        self.assertEqual(set(bottom), {(4, col) for col in range(5)})
        self.assertEqual(set(left), {(row, 0) for row in range(4)})
        self.assertEqual(set(right), {(row, 5) for row in range(4)})
        point = dem.line_cells(3.0, 3.0, 3.0, 3.0)
        self.assertEqual(set(point), {(1, 2), (1, 3), (2, 2), (2, 3)})

    def test_noncollinear_boundary_endpoints_include_all_touched_cells(self):
        dem = self.synthetic_dem()
        cells = dem.line_cells(3.0, 4.2, 4.8, 2.2)
        self.assertIn((0, 2), cells)
        reverse = dem.line_cells(4.8, 2.2, 3.0, 4.2)
        self.assertEqual(set(cells), set(reverse))

        corner_start = dem.line_cells(3.0, 3.0, 4.8, 1.2)
        for touched in ((1, 2), (1, 3), (2, 2), (2, 3)):
            self.assertIn(touched, corner_start)
        corner_end = dem.line_cells(4.8, 1.2, 3.0, 3.0)
        self.assertEqual(set(corner_start), set(corner_end))

    def test_noncollinear_outer_edge_endpoints_terminate_and_are_symmetric(self):
        dem = self.synthetic_dem()
        cases = [
            ((5.7, 5.0), (6.0, 1.0)),
            ((0.3, 5.0), (0.0, 1.0)),
            ((0.0, 4.7), (4.0, 5.0)),
            ((0.0, 0.3), (4.0, 0.0)),
        ]
        for start, end in cases:
            with self.subTest(start=start, end=end):
                forward = dem.line_cells(start[0], start[1], end[0], end[1])
                reverse = dem.line_cells(end[0], end[1], start[0], start[1])
                self.assertTrue(forward)
                self.assertEqual(set(forward), set(reverse))

    def test_internal_corner_rounding_is_symmetric(self):
        dem = self.synthetic_dem()
        forward = dem.line_cells(1.2, 3.8, 4.0, 1.0)
        reverse = dem.line_cells(4.0, 1.0, 1.2, 3.8)
        self.assertIn((1, 2), forward)
        self.assertIn((2, 3), forward)
        self.assertEqual(set(forward), set(reverse))

    def test_supercover_matches_closed_rectangle_oracle(self):
        dem = self.synthetic_dem()
        generator = random.Random(20260924)
        for _ in range(500):
            row1 = generator.choice([0.0, 0.2, 1.0, 1.2, 2.0, 2.5, 3.0, 4.0, 4.8, 5.0])
            col1 = generator.choice([0.0, 0.2, 1.0, 1.2, 2.0, 3.0, 4.0, 4.8, 5.0, 6.0])
            row2 = generator.choice([0.0, 0.3, 1.0, 1.8, 2.0, 3.0, 4.0, 4.7, 5.0])
            col2 = generator.choice([0.0, 0.3, 1.0, 1.8, 2.0, 3.0, 4.0, 5.0, 5.7, 6.0])
            lon1, lat1 = col1, dem.lat0 - row1
            lon2, lat2 = col2, dem.lat0 - row2
            actual = set(dem.line_cells(lon1, lat1, lon2, lat2))
            expected = self.closed_rectangle_oracle(
                row1, col1, row2, col2, dem.height, dem.width
            )
            with self.subTest(start=(row1, col1), end=(row2, col2)):
                self.assertEqual(actual, expected)
                reverse = set(dem.line_cells(lon2, lat2, lon1, lat1))
                self.assertEqual(actual, reverse)


if __name__ == "__main__":
    unittest.main()
