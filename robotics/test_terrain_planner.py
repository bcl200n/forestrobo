import unittest
import numpy as np
from robotics.terrain_planner import plan


class PlannerTests(unittest.TestCase):
    def test_avoids_inflated_obstacle(self):
        h=np.zeros((30,30));o=np.zeros_like(h);o[10:20,14:16]=1
        result=plan(h,o,(15,5),(15,25),footprint_radius_m=.2)
        self.assertTrue(result['found'])
        self.assertTrue(all(not(8<=r<=21 and 12<=c<=17) for r,c in result['path']))

    def test_unknown_band_is_not_free_space(self):
        h=np.zeros((20,20));h[10,:]=np.nan
        self.assertFalse(plan(h,np.zeros_like(h),(5,10),(15,10),footprint_radius_m=.1)['found'])

    def test_steep_step_is_blocked(self):
        h=np.zeros((20,20));h[10:,:]=.5
        self.assertFalse(plan(h,np.zeros_like(h),(5,10),(15,10),footprint_radius_m=.1)['found'])

    def test_no_diagonal_corner_cut(self):
        h=np.zeros((3,3));o=np.zeros_like(h);o[0,1]=1;o[1,0]=1
        self.assertFalse(plan(h,o,(0,0),(2,2),footprint_radius_m=0)['found'])

    def test_uncertain_goal_is_rejected(self):
        h=np.zeros((10,10));u=h.copy();u[5,5]=.4
        self.assertFalse(plan(h,h,(3,3),(5,5),footprint_radius_m=.1,height_uncertainty_m=u)['found'])


if __name__=='__main__':unittest.main()
