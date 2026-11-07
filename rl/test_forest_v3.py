import unittest
import numpy as np
from forest_v3 import ForestHistoryEnv
from run_forest_experiment import wilson


class HistoryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=ForestHistoryEnv(embodiment='wheel',scenario='blackout',max_steps=20)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reproducible_domain_and_history(self):
        a,ia=self.env.reset(seed=52);b,ib=self.env.reset(seed=52)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ia['domain'],ib['domain'])
        self.assertEqual(a.shape,(340,))

    def test_blackout_exposes_age_and_validity(self):
        e=self.env
        e.reset(seed=12)
        e.blackout_remaining=10
        obs,_,_,_,info=e.step(np.zeros(16,np.float32))
        frame=obs[-e.frame_dim:]
        np.testing.assert_array_equal(frame[69:81],np.zeros(12))
        self.assertGreater(info['range_age_s'],0)
        self.assertGreater(frame[-4],0)

    def test_nonfinite_action_rejected_before_physics(self):
        self.env.reset(seed=2)
        t=self.env.unwrapped.data.time
        with self.assertRaises(ValueError):self.env.step(np.full(16,np.nan))
        self.assertEqual(self.env.unwrapped.data.time,t)

    def test_physical_metrics_finite(self):
        self.env.reset(seed=2)
        _,_,_,_,info=self.env.step(np.zeros(16))
        self.assertGreaterEqual(info['energy_j'],0)
        self.assertGreaterEqual(info['path_m'],0)
        self.assertTrue(np.isfinite(info['energy_j']))

    def test_confidence_interval_handles_small_sample(self):
        lo,hi=wilson(0,6)
        self.assertEqual(lo,0)
        self.assertGreater(hi,.3)
        self.assertLess(hi,.5)


if __name__=='__main__':unittest.main()
