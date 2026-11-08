import math
import unittest

import numpy as np

from terrain_planner import a_star_terrain, compute_slope, cost_surface


class TerrainRoutingTests(unittest.TestCase):
    def test_route_reconsiders_node_when_cheaper_approach_is_found(self):
        slopes = np.array([[20., 20., 10.], [15., 25., 0.]])
        path = a_star_terrain(cost_surface(slopes), (2, 1), (0, 1), slopes, cell_size_m=1)
        # The direct two-edge route costs 6.25. The upper detour costs 4.5 + sqrt(2).
        self.assertEqual(path, [(2, 1), (2, 0), (1, 0), (0, 1)])

    def test_blocked_endpoints_and_diagonal_corner_are_rejected(self):
        costs = np.array([[1., np.inf], [np.inf, 1.]])
        slopes = np.zeros_like(costs)
        self.assertIsNone(a_star_terrain(costs, (0, 0), (1, 1), slopes))
        self.assertIsNone(a_star_terrain(costs, (1, 0), (1, 0), slopes))
        self.assertIsNone(a_star_terrain(costs, (-1, 0), (0, 0), slopes))
        self.assertEqual(a_star_terrain(costs, (0, 0), (0, 0), slopes), [(0, 0)])

    def test_uniform_steep_grade_remains_steep_at_the_boundary(self):
        heights = np.tile([0., 30., 60.], (3, 1))
        slope, _ = compute_slope(heights, cell_size_m=25)
        np.testing.assert_allclose(slope, math.degrees(math.atan(30 / 25)))
        self.assertTrue(np.isinf(cost_surface(slope)).all())

    def test_missing_elevation_is_not_a_passable_cell(self):
        heights = np.zeros((3, 3))
        heights[1, 1] = np.nan
        slope, _ = compute_slope(heights)
        self.assertTrue(np.isinf(cost_surface(slope)[1, 1]))


if __name__ == '__main__':
    unittest.main()
