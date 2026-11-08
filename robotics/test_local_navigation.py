import unittest
import numpy as np
from robotics.local_navigation import LocalGrid,WaypointFollower
from robotics.terrain_planner import inflate


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
                self.assertEqual(grid.cell(path[-1]),grid.cell(goal))
                self.assertGreaterEqual(max(grid.cell(point)[0] for point in path),15)
                self.assertTrue(all(not grid.occupied[grid.cell(point)] for point in path))

    def test_local_frontier_advances_through_observed_wall_opening(self):
        grid = LocalGrid(origin=(0.,0.),shape=(22,24),resolution=.2)
        grid.known[:,:11] = True
        grid.known[15:,:13] = True
        grid.occupied[:15,10] = True
        start,goal = grid.point((7,7)),grid.point((7,13))
        self.assertFalse(grid.known[grid.cell(goal)])
        for options in ({'geodesic':True}, {'weighted_local':True}):
            with self.subTest(options=options):
                path = grid.route(start,goal,radius=0.,**options)
                self.assertTrue(path)
                self.assertGreaterEqual(grid.cell(path[-1])[0],15)
                self.assertTrue(all(grid.known[grid.cell(point)] and not grid.occupied[grid.cell(point)]
                                    for point in path))

    def test_weighted_complete_route_increases_clearance_with_same_footprint(self):
        grid = LocalGrid()
        grid.known.fill(True)
        obstacle = (20,21)
        grid.occupied[obstacle] = True
        start,goal = grid.point((20,10)),grid.point((20,32))
        radius = .48
        ordinary = grid.route(start,goal,radius=radius)
        weighted = grid.route(start,goal,radius=radius,weighted_local=True)
        blocked = inflate(grid.occupied,grid.resolution,radius)
        for path in (ordinary,weighted):
            self.assertTrue(path)
            np.testing.assert_allclose(path[0],start)
            np.testing.assert_allclose(path[-1],goal)
            self.assertTrue(all(not blocked[grid.cell(point)] for point in path))
        ordinary_clearance = np.linalg.norm(np.asarray(ordinary)-grid.point(obstacle),axis=1).min()
        weighted_clearance = np.linalg.norm(np.asarray(weighted)-grid.point(obstacle),axis=1).min()
        self.assertGreater(weighted_clearance,ordinary_clearance+grid.resolution*.5)
