import unittest

import numpy as np

from rescue_coverage import CoverageSearchEnv


class CueCoverageTests(unittest.TestCase):
    def setUp(self):
        self.env = CoverageSearchEnv(scenario='corridor')
        self.env.reset(options={'map_seed': 35012})

    def test_fallback_routes_to_viewpoint_beyond_adjacent_cell(self):
        env = self.env
        env.known[:] = -1
        env.known[2:8, 2] = 1
        env.swept[:] = False
        env.swept[2:7, 2] = True
        env.frontier_cache = None
        self.assertEqual(env._frontier(), (0, 2))
        env.swept[7, 2] = True
        env.frontier_cache = None
        self.assertEqual(env._frontier(), (0, 1))

    def test_view_rejects_unknown_or_blocked_line_of_sight(self):
        env = self.env
        known = np.ones_like(env.known)
        target = np.zeros_like(env.swept)
        target[5, 2] = True
        np.testing.assert_array_equal(env._view(env.pos, known == 1, target), [[5, 2]])
        for occlusion in (0, -1):
            known[3, 2] = occlusion
            self.assertEqual(env._view(env.pos, known == 1, target).size, 0)

    def test_outage_does_not_mark_cue_coverage(self):
        env = self.env
        for link, sensor in ((False, True), (True, False)):
            env.swept[:] = False
            env.link_valid, env.sensor_valid = link, sensor
            env._observe_map()
            self.assertFalse(env.swept.any())
        env.link_valid = env.sensor_valid = True
        env._observe_map()
        self.assertTrue(env.swept[tuple(env.pos)])

    def test_street_sweep_continues_after_cues_and_resets(self):
        from evaluate_rescue_coverage import CASE_PATH, StreetMap, StreetSearchEnv

        env = StreetSearchEnv(street_map=StreetMap(CASE_PATH), required_coverage=.98)
        initial_obs, initial_info = env.reset(options={'map_seed':37000})
        env.scanned[env.real] = True
        _, _, terminated, _, info = env.step(9)
        self.assertFalse(terminated)
        self.assertFalse(info['success'])
        env.swept[env.searchable] = True
        _, _, terminated, _, info = env.step(9)
        self.assertTrue(terminated)
        self.assertTrue(info['success'])
        obs, info = env.reset(options={'map_seed':37000})
        np.testing.assert_array_equal(obs, initial_obs)
        self.assertEqual(info, initial_info)
        env.close()


if __name__ == '__main__':
    unittest.main()
