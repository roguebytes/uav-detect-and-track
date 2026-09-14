"""Survey-then-verify mission as a polled state machine, independent of ROS.

    mission = SurveyVerifyMission(controller, waypoints, survey_alt=40, verify_alt=11, strategy=QuadDescendVerify())
    every 0.2 s: mission.tick(now, confirmed_tracks, observations)

`confirmed_tracks` is a list of (id, x, y) from the perception node. During a dwell the caller
passes `observations`, the ground points of the detections in the latest frame, and the mission
counts frames in which one lies within `verify_radius` of the track. The verdict is
hits / frames >= verify_ratio. Verdicts are exposed in `verdicts` for the perception log.
"""
from __future__ import annotations

import math
from enum import Enum


class State(Enum):
    INIT = "init"
    TAKEOFF = "takeoff"
    SURVEY = "survey"
    VERIFY = "verify"
    RETURN = "return"
    LAND = "land"
    DONE = "done"


class SurveyVerifyMission:
    def __init__(self, controller, waypoints, survey_alt: float, verify_alt: float, strategy,
                 wp_tol: float = 1.5, verify_radius: float = 0.75, verify_ratio: float = 0.5,
                 max_verify: int | None = None, home=(0.0, 0.0), settle_s: float = 1.0):
        self.ctl, self.waypoints, self.survey_alt, self.verify_alt = controller, list(waypoints), survey_alt, verify_alt
        self.strategy, self.wp_tol, self.verify_radius, self.verify_ratio = strategy, wp_tol, verify_radius, verify_ratio
        self.max_verify, self.home, self.settle_s = max_verify, home, settle_s
        self.state, self.wp_index = State.INIT, 0
        self.queue: list = []                  # verify targets: (id, x, y)
        self.steps: list = []                  # remaining steps for the current target
        self.current = None                    # (id, x, y) being verified
        self.dwell_until, self.dwell_frames, self.dwell_hits = None, 0, 0
        self.verdicts: dict[int, bool] = {}
        self.transitions: list[tuple[float, str]] = []
        self._reached_since = None
        self._last_frame_key = None

    # ---- helpers ------------------------------------------------------------------------
    def _set(self, state: State, now: float):
        self.state = state
        self.transitions.append((now, state.value))

    def _at_target(self, now: float) -> bool:
        """Reached and settled for settle_s seconds."""
        if not self.ctl.reached(self.wp_tol):
            self._reached_since = None
            return False
        if self._reached_since is None:
            self._reached_since = now
        return now - self._reached_since >= self.settle_s

    def _plan_verify(self, tracks):
        """Order confirmed tracks nearest-neighbour from the current position."""
        pos = self.ctl.position() or (0.0, 0.0, 0.0)
        todo = [(tid, x, y) for tid, x, y in tracks]
        if self.max_verify is not None:
            todo = todo[: self.max_verify]
        ordered, px, py = [], pos[0], pos[1]
        while todo:
            d, best = min((math.hypot(x - px, y - py), (tid, x, y)) for tid, x, y in todo)
            ordered.append(best); todo.remove(best); px, py = best[1], best[2]
        return ordered

    # ---- main loop ------------------------------------------------------------------------
    def tick(self, now: float, tracks=(), observations=()):
        if self.state is State.INIT:
            if self.ctl.connected():
                self.ctl.takeoff(self.survey_alt)
                self._set(State.TAKEOFF, now)
        elif self.state is State.TAKEOFF:
            if self.ctl.armed() and self._at_target(now):
                self._goto_wp(); self._set(State.SURVEY, now)
        elif self.state is State.SURVEY:
            if self._at_target(now):
                self.wp_index += 1
                if self.wp_index < len(self.waypoints):
                    self._goto_wp()
                else:
                    self.queue = self._plan_verify(tracks)
                    self._set(State.VERIFY, now)
                    self._next_target(now)
        elif self.state is State.VERIFY:
            self._verify_tick(now, observations)
        elif self.state is State.RETURN:
            if self._at_target(now):
                self.ctl.land(); self._set(State.LAND, now)
        elif self.state is State.LAND:
            if not self.ctl.armed():
                self._set(State.DONE, now)

    def _goto_wp(self):
        x, y, z, yaw = self.waypoints[self.wp_index]
        self.ctl.goto(x, y, z, yaw)
        self._reached_since = None

    def _next_target(self, now: float):
        if not self.queue:
            self.ctl.goto(self.home[0], self.home[1], self.survey_alt)
            self._reached_since = None
            self._set(State.RETURN, now)
            return
        self.current = self.queue.pop(0)
        _, x, y = self.current
        self.steps = self.strategy.steps(x, y, self.survey_alt, self.verify_alt)
        self._next_step(now)

    def _next_step(self, now: float):
        if not self.steps:
            self._next_target(now)
            return
        step = self.steps.pop(0)
        if step[0] == "goto":
            _, x, y, z, yaw = step
            self.ctl.goto(x, y, z, yaw)
            self._reached_since = None
        elif step[0] == "dwell":
            self.dwell_until = now + step[1]
            self.dwell_frames = self.dwell_hits = 0
            self._last_frame_key = None

    def _verify_tick(self, now: float, observations):
        if self.dwell_until is not None:
            tid, x, y = self.current
            key = tuple(round(v, 3) for pt in observations for v in pt) if observations is not None else None
            if observations is not None and key != self._last_frame_key:   # count each new frame once
                self._last_frame_key = key
                self.dwell_frames += 1
                if any(math.hypot(px - x, py - y) <= self.verify_radius for px, py in observations):
                    self.dwell_hits += 1
            if now >= self.dwell_until:
                self.verdicts[tid] = self.dwell_frames > 0 and self.dwell_hits / self.dwell_frames >= self.verify_ratio
                self.dwell_until = None
                self._next_step(now)
        elif self._at_target(now):
            self._next_step(now)

    @property
    def done(self) -> bool:
        return self.state is State.DONE
