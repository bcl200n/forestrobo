"""Regression checks for v2 state ordering and episode semantics."""
import unittest
import numpy as np
from forest_robust import RobustForestEnv


class ForestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = RobustForestEnv(faults=False)
        cls.env = cls.wrapper.unwrapped

    @classmethod
    def tearDownClass(cls):
        cls.wrapper.close()

    def test_seed(self):
        a, _ = self.wrapper.reset(seed=21)
        b, _ = self.wrapper.reset(seed=21)
        np.testing.assert_array_equal(a, b)

    def test_joint_addresses(self):
        e = self.env
        e.reset(seed=1)
        e.data.qpos[e.wheel_q] = [1, 2, 3, 4]
        np.testing.assert_allclose(e.data.qpos[e.leg_q], np.tile([0, .55, -1.15], 4))
        self.assertFalse(set(e.leg_v) & set(e.wheel_v))

    def test_timeout_bootstraps(self):
        e = self.env
        e.reset(seed=3)
        e._k = 499
        _, _, terminated, truncated, info = e.step(np.zeros(16))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info['is_success'])

    def test_success_is_not_truncation(self):
        e = self.env
        e.reset(seed=3)
        e.goal = e.data.qpos[:2].copy()
        _, _, terminated, truncated, info = e.step(np.zeros(16))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['is_success'])


if __name__ == '__main__':
    unittest.main()
