"""Conservative command gate, separate from a learned locomotion policy.

All ages must come from the same monotonic clock. This module does not replace
hardware e-stop, localization, SLAM, Nav2 or a motor controller.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Health:
    range_age_s: float
    localization_age_s: float
    localization_std_m: float
    minimum_clearance_m: float
    tilt_rad: float
    communication_age_s: float = 0.
    estop: bool = False
    tool_locked: bool = False
    tool_active: bool = False


@dataclass(frozen=True)
class Decision:
    linear_m_s: float
    yaw_rad_s: float
    tool_enabled: bool
    reason: str


class CommandGate:
    def __init__(self, max_speed=.6, brake_acceleration=.8, margin=.30):
        if not all(math.isfinite(v) and v > 0 for v in (max_speed, brake_acceleration, margin)):
            raise ValueError('positive limits required')
        self.max_speed, self.brake, self.margin = max_speed, brake_acceleration, margin
        self.latched = False

    def rearm(self, h: Health, operator_confirmed=False):
        if operator_confirmed and self._healthy(h) and not h.tool_active:
            self.latched = False
            return True
        return False

    def _healthy(self, h):
        values = [h.range_age_s, h.localization_age_s, h.localization_std_m,
                  h.minimum_clearance_m, h.tilt_rad, h.communication_age_s]
        return (all(math.isfinite(x) for x in values)
                and min(h.range_age_s, h.localization_age_s, h.localization_std_m, h.communication_age_s) >= 0
                and not h.estop and h.range_age_s <= .2
                and h.localization_age_s <= .3 and h.localization_std_m <= .25
                and abs(h.tilt_rad) < math.radians(35) and h.minimum_clearance_m > self.margin)

    def filter(self, linear, yaw, h: Health, autonomous=True):
        if h.estop or h.tool_active and (h.communication_age_s > .5 or not h.tool_locked):
            self.latched = True
        if not math.isfinite(linear) or not math.isfinite(yaw):
            return Decision(0, 0, False, 'invalid_command')
        if self.latched:
            return Decision(0, 0, False, 'latched_stop')
        if not self._healthy(h):
            return Decision(0, 0, False, 'stale_or_unsafe_sensing')
        if not autonomous and h.communication_age_s > .5:
            return Decision(0, 0, False, 'teleoperation_lost')
        # Clearance is swept-body clearance, not distance from the robot origin.
        available = max(0., h.minimum_clearance_m-self.margin)
        # Include reaction distance for 200 ms sensor age threshold.
        reaction = .2
        cap = -self.brake*reaction + math.sqrt((self.brake*reaction)**2+2*self.brake*available)
        speed = min(self.max_speed, cap, abs(linear))
        if h.tool_active:
            return Decision(0, 0, h.tool_locked and h.communication_age_s <= .5, 'tool_stationary')
        return Decision(math.copysign(speed, linear), max(-.6, min(.6, yaw)), False,
                        'onboard_autonomy' if autonomous else 'teleoperation')
