"""Fail-closed concept interlocks; no hardware or life-detection implementation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RescueState:
    heartbeat_age_s: float
    gas_age_s: float
    gas_clear: bool
    stationary: bool
    support_confirmed: bool
    tool_locked: bool
    tool_id_confirmed: bool
    operator_hold: bool
    human_zone_clear: bool
    structural_clearance: bool
    estop: bool=False
    vibration_quiet: bool=False


class RescueInterlock:
    def __init__(self):self.latched=False

    def evaluate(self,state,cut_requested=False):
        s=state
        valid=all(math.isfinite(t) and t>=0 for t in (s.heartbeat_age_s,s.gas_age_s))
        if s.estop or (cut_requested and (not valid or s.heartbeat_age_s>.5 or not s.tool_locked)):
            self.latched=True
        checks=dict(telemetry=valid and s.heartbeat_age_s<=.5,
            gas=valid and s.gas_age_s<=1. and s.gas_clear is True,
            stationary=s.stationary is True,support=s.support_confirmed is True,
            tool=s.tool_locked is True and s.tool_id_confirmed is True,
            operator=s.operator_hold is True,zone=s.human_zone_clear is True,
            structure=s.structural_clearance is True,estop=not s.estop,latch=not self.latched)
        return dict(cut_enabled=bool(cut_requested and all(checks.values())),
            motion_enabled=not cut_requested and checks['telemetry'] and not s.estop and not self.latched,
            quiet_sensing_ready=bool(s.stationary and s.vibration_quiet and not cut_requested and not s.estop),
            blocked_by=[name for name,value in checks.items() if not value],
            latched=self.latched,life_detection_status='not_implemented')

    def rearm(self,state,operator_confirmed=False):
        if operator_confirmed and not state.estop and not state.operator_hold and 0<=state.heartbeat_age_s<=.5 and state.tool_locked:
            self.latched=False;return True
        return False
