import unittest
import numpy as np
from robotics.local_navigation import LocalGrid,WaypointFollower


class LocalNavigationTests(unittest.TestCase):
    def test_unknown_map_does_not_plan(self):
        self.assertEqual(LocalGrid().route([0,0],[4,0]),[])

    def test_lidar_updates_free_and_hit(self):
        grid=LocalGrid();grid.update([0,0],[0],[1.],[True],.1)
        self.assertTrue(grid.known[grid.cell([.4,0])])
        self.assertTrue(grid.occupied[grid.cell([1,0])])
        with self.assertRaises(ValueError):grid.update([0,0],[0],[1],[False],0)

    def test_lidar_rejects_nan(self):
        with self.assertRaises(ValueError):LocalGrid().update([0,0],[float('nan')],[1],[False],0)

    def test_follow_requires_path_and_stops_at_goal(self):
        follower=WaypointFollower()
        np.testing.assert_array_equal(follower.command([0,0],0,[],[3,0]),[0,0,0])
        np.testing.assert_array_equal(follower.command([2.9,0],0,[[3,0]],[3,0]),[0,0,0])
        self.assertGreater(follower.command([0,0],0,[[0,0],[1,0]],[3,0])[0],0)

    def test_local_route_never_enters_unknown(self):
        grid=LocalGrid();angles=np.linspace(-np.pi,np.pi,120,endpoint=False)
        grid.update([0,0],angles,np.full(120,3.2),np.zeros(120,bool),0)
        path=grid.route([0,0],[4.5,0])
        self.assertTrue(path)
        for point in path:
            cell=grid.cell(point)
            self.assertTrue(grid.known[cell]);self.assertFalse(grid.occupied[cell])

    def test_global_guide_routes_around_local_minimum(self):
        grid=LocalGrid();grid.known[:]=True
        for row in range(15,25):grid.occupied[row,19:22]=True
        path=grid.route([0,0],[4.5,0])
        self.assertTrue(path)
        self.assertGreater(max(abs(p[1]) for p in path),.5)

    def test_geodesic_target_advances_toward_wall_opening(self):
        grid = LocalGrid(origin=(0.,0.),shape=(22,24),resolution=.2)
        grid.known.fill(True)
        grid.occupied[:15,10] = True
        start,goal = grid.point((7,7)),grid.point((7,13))
        for options in ({'geodesic':True}, {'weighted_local':True}):
            with self.subTest(options=options):
                path = grid.route(start,goal,radius=0.,**options)
                self.assertTrue(path)
                self.assertGreaterEqual(grid.cell(path[-1])[0],14)
                self.assertTrue(all(not grid.occupied[grid.cell(point)] for point in path))
