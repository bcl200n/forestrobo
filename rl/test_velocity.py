import unittest
import numpy as np
from velocity_env import VelocityEnv
from velocity_primitive import targets
from foot_teacher_env import FootTeacherEnv


class VelocityTests(unittest.TestCase):
    def test_zero_command_is_stance(self):
        q0=np.tile([0,.9,-1.72],4)
        np.testing.assert_allclose(targets(.13,[0,0,0],q0),q0)

    def test_teacher_contact_observation_and_seed(self):
        env=FootTeacherEnv()
        try:
            a,_=env.reset(seed=9);b,_=env.reset(seed=9)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(a.shape,(64,))
            self.assertGreater(a[-8:-4].sum(),0)
            self.assertEqual(env.action_space.shape,(16,))
        finally:env.close()

    def test_measured_forward_direction(self):
        env=VelocityEnv('foot')
        try:
            env.reset(seed=11,options={'command':[.3,0,0]})
            speeds=[]
            for step in range(150):
                a=np.clip((targets(env.data.time,env.command,env.q0)-env.q0)/.4,-1,1)
                _,_,term,_,info=env.step(a)
                self.assertFalse(term)
                if step>=25:speeds.append(info['actual'][0])
            self.assertGreater(np.mean(speeds),.1)
        finally:env.close()

    def test_wheel_dimensions_and_timeout(self):
        env=VelocityEnv('wheel',max_steps=1)
        try:
            obs,_=env.reset(seed=1)
            self.assertEqual(obs.shape,(64,))
            _,_,term,trunc,_=env.step(np.zeros(16))
            self.assertFalse(term);self.assertTrue(trunc)
        finally:env.close()


if __name__=='__main__':unittest.main()
