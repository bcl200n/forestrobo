import unittest
from robotics.rescue_workflow import RescueMission
from robotics.rescue_selection import selection,arm_preview


class WorkflowTests(unittest.TestCase):
    def test_review_does_not_enable_tool_or_confirm_life(self):
        m=RescueMission();m.dispatch('start')
        m.dispatch('candidate',id='c1',source='simulated cue')
        with self.assertRaises(ValueError):m.dispatch('resume')
        m.dispatch('review',actor='reviewer',id='c1',decision='accept_cue')
        self.assertFalse(m.snapshot()['cut_enabled'])
        self.assertFalse(m.snapshot()['candidates'][0]['life_confirmed'])

    def test_disconnect_recovery_requires_manual_rearm(self):
        m=RescueMission();m.dispatch('start');m.dispatch('disconnect')
        with self.assertRaises(ValueError):m.dispatch('resume')
        good=dict(heartbeat_age_s=.1,gas_age_s=.1,gas_clear=True,stationary=True,
                  support_confirmed=True,tool_locked=True,tool_id_confirmed=True,
                  operator_hold=False,human_zone_clear=True,structural_clearance=True)
        m.dispatch('rearm',confirmed=True,state=good)
        self.assertEqual(m.mode,'paused')
        m.dispatch('resume');self.assertEqual(m.mode,'autonomy')

    def test_unknown_or_overweight_configuration_not_approved(self):
        self.assertIsNone(selection('hazmat')['variant'])
        self.assertFalse(selection('void',object_mass_kg=2)['within_nominal_payload'])
        result=arm_preview('clear',[.45,0,.15])
        self.assertFalse(result['execution_approved'])
        self.assertFalse(result['collision_checked'])

    def test_tool_requires_review_and_all_interlocks(self):
        m=RescueMission();m.dispatch('start');m.dispatch('candidate',id='c1')
        m.dispatch('review',actor='reviewer',id='c1',decision='accept_cue')
        state=dict(heartbeat_age_s=.1,gas_age_s=.1,gas_clear=True,stationary=True,
                   support_confirmed=True,tool_locked=True,tool_id_confirmed=True,
                   operator_hold=True,human_zone_clear=True,structural_clearance=True)
        self.assertTrue(m.dispatch('tool_check',id='c1',state=state)['tool_result']['cut_enabled'])
        state['gas_clear']=False
        self.assertFalse(m.dispatch('tool_check',id='c1',state=state)['tool_result']['cut_enabled'])

    def test_pending_review_blocks_other_tool_operation(self):
        m=RescueMission();m.dispatch('start')
        m.dispatch('candidate',id='c1');m.dispatch('candidate',id='c2')
        m.dispatch('review',actor='reviewer',id='c1',decision='accept_cue')
        m.dispatch('pause')
        with self.assertRaisesRegex(ValueError,'pending human reviews'):
            m.dispatch('tool_check',id='c1',state={})

    def test_string_flags_are_not_accepted_as_hardware_booleans(self):
        good=dict(heartbeat_age_s=.1,gas_age_s=.1,gas_clear=True,stationary=True,
                  support_confirmed=True,tool_locked=True,tool_id_confirmed=True,
                  operator_hold=False,human_zone_clear=True,structural_clearance=True)
        good['tool_locked']='false'
        with self.assertRaises(ValueError):RescueMission._state(good)


if __name__=='__main__':unittest.main()
