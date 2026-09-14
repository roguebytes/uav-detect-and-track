"""Verify strategies: how an airframe gets a close look at a candidate track.

The mission state machine asks the strategy for a list of steps for one track. Each step is
('goto', x, y, z, yaw) or ('dwell', seconds). The quad descends and hovers. A fixed wing would
loiter at the verify altitude instead, which is a later strategy on the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VerifyStrategy(ABC):
    @abstractmethod
    def steps(self, x: float, y: float, survey_alt: float, verify_alt: float) -> list[tuple]: ...


class QuadDescendVerify(VerifyStrategy):
    """Fly over the track at survey altitude, descend to the verify altitude, dwell, climb back."""

    def __init__(self, dwell_s: float = 4.0):
        self.dwell_s = dwell_s

    def steps(self, x, y, survey_alt, verify_alt):
        return [("goto", x, y, survey_alt, None), ("goto", x, y, verify_alt, None), ("dwell", self.dwell_s),
                ("goto", x, y, survey_alt, None)]


class FixedWingLoiterVerify(VerifyStrategy):
    """Placeholder for the fixed-wing variant: loiter around the track at the verify altitude."""

    def steps(self, x, y, survey_alt, verify_alt):
        raise NotImplementedError("fixed-wing loiter verify is a later step; see the spec")
