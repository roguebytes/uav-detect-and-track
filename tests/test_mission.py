import math

from uav_dt.mission.state_machine import State, SurveyVerifyMission
from uav_dt.mission.verify import QuadDescendVerify


class FakeController:
    """Teleports to the target on the next tick; tracks calls."""

    def __init__(self):
        self.pos, self.tgt, self._armed, self.calls = (0.0, 0.0, 0.0), None, False, []

    def connected(self): return True
    def position(self): return self.pos
    def armed(self): return self._armed
    def target(self): return self.tgt
    def takeoff(self, alt): self._armed = True; self.tgt = (self.pos[0], self.pos[1], alt); self.calls.append(("takeoff", alt))
    def goto(self, x, y, z, yaw=None): self.tgt = (x, y, z); self.calls.append(("goto", x, y, z))
    def land(self): self._armed = False; self.tgt = None; self.calls.append(("land",))
    def reached(self, tol=1.0):
        if self.tgt is not None:
            self.pos = self.tgt   # teleport
        return True


def run(mission, tracks, obs_for, max_ticks=5000):
    t = 0.0
    for _ in range(max_ticks):
        obs = obs_for(mission) if mission.state is State.VERIFY else ()
        mission.tick(t, tracks, obs)
        t += 0.5
        if mission.done:
            return t
    raise AssertionError(f"mission stuck in {mission.state}")


def test_full_sequence_with_two_tracks():
    ctl = FakeController()
    wps = [(-10, 0, 40, 0.0), (10, 0, 40, 0.0), (10, 5, 40, math.pi), (-10, 5, 40, math.pi)]
    m = SurveyVerifyMission(ctl, wps, 40, 11, QuadDescendVerify(dwell_s=2.0), settle_s=0.0, dwell_min_frames=1)
    tracks = [(1, 5.0, 1.0), (2, -8.0, 4.0)]

    def obs(mission):   # bowl 1 is really there, bowl 2 is not
        if mission.current is None or mission.dwell_until is None:
            return ()
        tid, x, y = mission.current
        return [(x + 0.1, y - 0.1)] if tid == 1 else [(x + 5.0, y)]

    run(m, tracks, obs)
    assert [s for _, s in m.transitions] == ["takeoff", "survey", "verify", "return", "land", "done"]
    assert m.verdicts == {1: True, 2: False}
    gotos = [c for c in ctl.calls if c[0] == "goto"]
    assert gotos[:4] == [("goto", -10, 0, 40), ("goto", 10, 0, 40), ("goto", 10, 5, 40), ("goto", -10, 5, 40)]
    # verify order is nearest-neighbour from the last waypoint (-10, 5): track 2 first, one descent,
    # then track 1 at the verify altitude, then home at the verify altitude
    assert gotos[4:] == [("goto", -8.0, 4.0, 40), ("goto", -8.0, 4.0, 11), ("goto", 5.0, 1.0, 11), ("goto", 0.0, 0.0, 11)]
    assert ctl.calls[-1] == ("land",)


def test_no_tracks_goes_straight_home():
    ctl = FakeController()
    m = SurveyVerifyMission(ctl, [(0, 0, 40, 0.0), (10, 0, 40, 0.0)], 40, 11, QuadDescendVerify(1.0), settle_s=0.0, dwell_min_frames=1)
    run(m, [], lambda mission: ())
    assert m.verdicts == {} and [s for _, s in m.transitions] == ["takeoff", "survey", "verify", "return", "land", "done"]


def test_dwell_waits_for_frames_then_gives_up():
    ctl = FakeController()
    m = SurveyVerifyMission(ctl, [(0, 0, 40, 0.0)], 40, 11, QuadDescendVerify(dwell_s=1.0), settle_s=0.0,
                            dwell_min_frames=3, dwell_max_s=6.0)
    tracks = [(1, 5.0, 1.0)]
    t = 0.0
    while m.state is not State.VERIFY or m.dwell_until is None:
        m.tick(t, tracks, ()); t += 0.5
    start = t
    while m.dwell_until is not None and t < start + 4.0:      # no frames arrive: the 1 s timer alone must not end the dwell
        m.tick(t, tracks, None); t += 0.5
    assert m.dwell_until is not None
    while m.dwell_until is not None:                          # ... but the deadline does
        m.tick(t, tracks, None); t += 0.5
    assert m.verdicts == {1: False} and t - start <= 6.5
