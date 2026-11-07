import dataclasses
import unittest
import numpy as np
from robotics.manipulator import chain, static_check, support_check, position_ik
from robotics.navigation import Health, CommandGate


class PlanningChecks(unittest.TestCase):
    def setUp(self):
        self.health=Health(.02,.02,.03,1.0,0.)

    def test_stale_sensor_stops(self):
        decision=CommandGate().filter(.5,.2,dataclasses.replace(self.health,range_age_s=.21))
        self.assertEqual(decision.linear_m_s,0)
        self.assertFalse(decision.tool_enabled)

    def test_communications_only_affect_teleop_and_tools(self):
        health=dataclasses.replace(self.health,communication_age_s=5)
        self.assertGreater(CommandGate().filter(.5,0,health,autonomous=True).linear_m_s,0)
        self.assertEqual(CommandGate().filter(.5,0,health,autonomous=False).linear_m_s,0)

    def test_tool_needs_manual_rearm_after_link_loss(self):
        gate=CommandGate()
        tool=dataclasses.replace(self.health,tool_active=True,tool_locked=True,communication_age_s=1)
        self.assertFalse(gate.filter(.1,0,tool).tool_enabled)
        self.assertEqual(gate.filter(.1,0,self.health).linear_m_s,0)
        self.assertFalse(gate.rearm(self.health))
        self.assertTrue(gate.rearm(self.health,operator_confirmed=True))

    def test_no_motion_while_cutting(self):
        tool=dataclasses.replace(self.health,tool_active=True,tool_locked=True)
        decision=CommandGate().filter(.4,.3,tool)
        self.assertEqual((decision.linear_m_s,decision.yaw_rad_s),(0,0))
        self.assertTrue(decision.tool_enabled)

    def test_braking_distance_and_invalid_measurements(self):
        decision=CommandGate().filter(.6,0,dataclasses.replace(self.health,minimum_clearance_m=.32))
        self.assertLess(decision.linear_m_s,.15)
        self.assertEqual(CommandGate().filter(.5,0,dataclasses.replace(self.health,range_age_s=float('nan'))).linear_m_s,0)

    def test_joint_torque_matches_numeric_jacobian(self):
        q=np.array([.2,-.7,.8,-.1,.2]);force=np.array([0,0,500.])
        result=static_check(q,force)
        zero=static_check(q)
        for i in range(5):
            a=q.copy();b=q.copy();a[i]+=1e-6;b[i]-=1e-6
            derivative=(chain(a)[0]-chain(b)[0])/2e-6
            self.assertAlmostEqual(result['torque_nm'][i]-zero['torque_nm'][i],float(derivative@force),places=5)

    def test_height_and_overload(self):
        result=static_check(np.zeros(5),force_n=(0,0,500))
        self.assertAlmostEqual(result['theoretical_max_height_m'],1.52)
        self.assertFalse(result['within_rated'])

    def test_foot_force_cannot_create_ground_anchor(self):
        result=support_check(30,[0,0],.35,.25,[.6,0,.2],[0,0,500])
        self.assertFalse(result['feasible'])
        self.assertEqual(result['reason'],'ground_unloaded')
        centered=support_check(30,[0,0],.35,.25,[0,0,0],[0,0,0])
        np.testing.assert_allclose(centered['center_of_pressure_m'],[0,0])

    def test_position_ik_rejects_unreachable_task(self):
        result=position_ik([.4,.1,.3])
        self.assertTrue(result['reachable'])
        self.assertFalse(result['orientation_checked'])
        impossible=position_ik([0,0,1.75])
        self.assertFalse(impossible['reachable'])


if __name__=='__main__':
    unittest.main()
