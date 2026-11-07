"""Supplement map frontiers with reachable cue-detection viewpoints."""
import numpy as np

from rescue_search_env import CUE_RANGE, CUE_RAY_FRACTIONS
from train_rescue_long_horizon import DetourSearchEnv


OFFSETS = np.array([(x, y) for x in range(-CUE_RANGE, CUE_RANGE+1)
                    for y in range(-CUE_RANGE, CUE_RANGE+1) if x*x+y*y <= CUE_RANGE**2])
RAYS = np.rint(OFFSETS[:, None, :] * CUE_RAY_FRACTIONS).astype(int)


class CoverageSearchEnv(DetourSearchEnv):
    def __init__(self, *args, supplement=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.supplement = supplement

    def reset(self, *, seed=None, options=None):
        self.swept = np.zeros((self.size, self.size), bool)
        return super().reset(seed=seed, options=options)

    def _view(self, position, free, unswept):
        ends = position + OFFSETS
        valid = ((ends >= 0) & (ends < self.size)).all(axis=1)
        valid[valid] = unswept[tuple(ends[valid].T)]
        rays = position + RAYS[valid]
        visible = free[rays[:, :, 0], rays[:, :, 1]].all(axis=1)
        return ends[valid][visible]

    def _observe_map(self):
        super()._observe_map()
        if self.link_valid and self.sensor_valid:
            free = self.known == 1
            visible = self._view(self.pos, free, free & ~self.swept)
            self.swept[tuple(visible.T)] = True

    def _frontier_fallback(self, free):
        if self.supplement:
            unswept = free & ~self.swept
            if unswept.any():
                for position, first, distance in self._reachable_routes(free):
                    if distance and self._view(position, free, unswept).size:
                        return first, distance
        return super()._frontier_fallback(free)

    def _info(self):
        swept = int(self.swept.sum())
        return dict(super()._info(), swept_cells=swept,
                    cue_coverage=swept / int((~self.blocked).sum()))
