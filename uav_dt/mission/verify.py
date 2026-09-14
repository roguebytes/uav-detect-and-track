"""Verify strategies: how an airframe gets a close look at the candidate tracks.

The mission state machine hands the strategy the ordered list of targets and gets back one flat
list of steps for the whole low-altitude pass: ('goto', x, y, z, yaw) or ('dwell', seconds, track_id).
The quad descends once at the first target, then flies between targets at the verify altitude,
as in the paper's survey-high, verify-low profile. A fixed wing would loiter at the verify
altitude instead, which is a later strategy on the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VerifyStrategy(ABC):
    @abstractmethod
    def plan(self, targets, survey_alt: float, verify_alt: float) -> list[tuple]:
        """targets: ordered list of (track_id, x, y). Returns the step list for the whole pass."""


class QuadDescendVerify(VerifyStrategy):
    """Descend once over the first target, then hop between targets at the verify altitude, dwelling at each."""

    def __init__(self, dwell_s: float = 4.0):
        self.dwell_s = dwell_s

    def plan(self, targets, survey_alt, verify_alt):
        steps = []
        for i, (tid, x, y) in enumerate(targets):
            if i == 0:
                steps.append(("goto", x, y, survey_alt, None))     # transit at survey altitude, then one descent
            steps.append(("goto", x, y, verify_alt, None))
            steps.append(("dwell", self.dwell_s, tid))
        return steps


class FixedWingLoiterVerify(VerifyStrategy):
    """Placeholder for the fixed-wing variant: loiter around each target at the verify altitude."""

    def plan(self, targets, survey_alt, verify_alt):
        raise NotImplementedError("fixed-wing loiter verify is a later step; see the spec")
