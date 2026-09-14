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
    m = SurveyVerifyMission(ctl, wps, 40, 11, QuadDescendVerify(dwell_s=2.0), settle_s=0.0)
    tracks = [(1, 5.0, 1.0), (2, -8.0, 4.0)]

    def obs(mission):   # bowl 1 is really there, bowl 2 is not
        tid, x, y = mission.current
        return [(x + 0.1, y - 0.1)] if tid == 1 else [(x + 5.0, y)]

    run(m, tracks, obs)
    assert [s for _, s in m.transitions] == ["takeoff", "survey", "verify", "return", "land", "done"]
    assert m.verdicts == {1: True, 2: False}
    gotos = [c for c in ctl.calls if c[0] == "goto"]
    assert gotos[:4] == [("goto", -10, 0, 40), ("goto", 10, 0, 40), ("goto", 10, 5, 40), ("goto", -10, 5, 40)]
    # verify order is nearest-neighbour from the last waypoint (-10, 5): track 2 first
    assert gotos[4:6] == [("goto", -8.0, 4.0, 40), ("goto", -8.0, 4.0, 11)]
    assert gotos[-1] == ("goto", 0.0, 0.0, 40) and ctl.calls[-1] == ("land",)


def test_no_tracks_goes_straight_home():
    ctl = FakeController()
    m = SurveyVerifyMission(ctl, [(0, 0, 40, 0.0), (10, 0, 40, 0.0)], 40, 11, QuadDescendVerify(1.0), settle_s=0.0)
    run(m, [], lambda mission: ())
    assert m.verdicts == {} and [s for _, s in m.transitions] == ["takeoff", "survey", "verify", "return", "land", "done"]
