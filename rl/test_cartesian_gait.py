import unittest
import math
import numpy as np
from cartesian_gait import leg_ik,targets


class CartesianGaitTests(unittest.TestCase):
    def test_ik_matches_actual_link_geometry(self):
        for front in (True,False):
            for side in (-1,1):
                target=np.array([.03,.07*side+.02,-.42])
                roll,hip,knee=leg_ik(target,front,side)
                a=.03 if front else -.03
                def ry(t):return np.array([[math.cos(t),0,math.sin(t)],[0,1,0],[-math.sin(t),0,math.cos(t)]])
                p=ry(hip)@np.array([a,0,-.30])+ry(hip+knee)@np.array([-.02,0,-.32])+[0,.07*side,0]
                rx=np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
                np.testing.assert_allclose(rx@p,target,atol=1e-7)

    def test_stance_stationary_when_command_zero(self):
        np.testing.assert_allclose(targets(0,[0,0,0]),targets(.11,[0,0,0]))
