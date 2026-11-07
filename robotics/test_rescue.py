from dataclasses import replace
import unittest
from robotics.rescue import RescueState,RescueInterlock


class RescueTests(unittest.TestCase):
    def setUp(self):
        self.good=RescueState(.1,.1,True,True,True,True,True,True,True,True)

    def test_all_cut_conditions_required(self):
        self.assertTrue(RescueInterlock().evaluate(self.good,True)['cut_enabled'])
        for name in ('gas_clear','stationary','support_confirmed','tool_locked','tool_id_confirmed',
                     'operator_hold','human_zone_clear','structural_clearance'):
            self.assertFalse(RescueInterlock().evaluate(replace(self.good,**{name:False}),True)['cut_enabled'],name)

    def test_communication_recovery_does_not_restart_cut(self):
        gate=RescueInterlock();gate.evaluate(replace(self.good,heartbeat_age_s=2),True)
        self.assertFalse(gate.evaluate(self.good,True)['cut_enabled'])
        self.assertFalse(gate.rearm(self.good,True))
        self.assertTrue(gate.rearm(replace(self.good,operator_hold=False),True))

    def test_unknown_gas_and_nan_fail_closed(self):
        self.assertFalse(RescueInterlock().evaluate(replace(self.good,gas_clear=None),True)['cut_enabled'])
        self.assertFalse(RescueInterlock().evaluate(replace(self.good,gas_age_s=float('nan')),True)['cut_enabled'])

    def test_sensing_is_not_a_life_detection_claim(self):
        result=RescueInterlock().evaluate(replace(self.good,vibration_quiet=True))
        self.assertTrue(result['quiet_sensing_ready'])
        self.assertEqual(result['life_detection_status'],'not_implemented')
