import unittest
import numpy as np
from rescue_search_env import RescueSearchEnv


class SearchContractTests(unittest.TestCase):
    def test_reproducible_maps_and_disjoint_map_identity(self):
        a,b=RescueSearchEnv(),RescueSearchEnv()
        oa,ia=a.reset(seed=7,options={'map_seed':101})
        ob,ib=b.reset(seed=7,options={'map_seed':101})
        np.testing.assert_array_equal(oa,ob)
        self.assertEqual(ia['map_hash'],ib['map_hash'])
        _,ic=b.reset(options={'map_seed':10001})
        self.assertNotEqual(ia['map_hash'],ic['map_hash'])

    def test_hidden_target_not_in_policy_observation(self):
        env=RescueSearchEnv()
        env.reset(seed=0)
        env.targets=np.array([[14,14],[15,15],[16,16],[17,17]])
        before=env._obs()
        env.targets=np.array([[13,14],[14,15],[15,16],[16,17]])
        np.testing.assert_array_equal(before,env._obs())
        self.assertFalse(env._visible())

    def test_time_limit_is_truncation(self):
        env=RescueSearchEnv(max_steps=1)
        env.reset(seed=3)
        _,_,terminated,truncated,_=env.step(8)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_dropout_holds_position(self):
        env=RescueSearchEnv(scenario='dropout')
        env.reset(seed=4)
        env.t=49
        p=env.pos.copy()
        env.step(0)
        np.testing.assert_array_equal(p,env.pos)
        self.assertFalse(env.link_valid)

    def test_actor_contract_during_random_rollout(self):
        for scenario in ('corridor','rubble','dropout'):
            env=RescueSearchEnv(scenario=scenario)
            obs,_=env.reset(seed=12)
            for _ in range(100):
                self.assertTrue(env.observation_space.contains(obs))
                obs,_,term,trunc,_=env.step(env.action_space.sample())
                if term or trunc:
                    obs,_=env.reset()


if __name__=='__main__':
    unittest.main()
