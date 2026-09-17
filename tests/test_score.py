__author__ = "Frank Loewenich"

import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import score  # noqa: E402


def test_scoring_matches_one_track_per_bowl():
    bowls = [{"id": 0, "x": 0.0, "y": 0.0}, {"id": 1, "x": 10.0, "y": 0.0}, {"id": 2, "x": 50.0, "y": 50.0}]
    tracks = [{"id": 1, "x": 0.3, "y": 0.1, "verified": True}, {"id": 2, "x": 0.9, "y": 0.0, "verified": False},
              {"id": 3, "x": 10.4, "y": -0.2, "verified": True}, {"id": 4, "x": 30.0, "y": 30.0, "verified": None}]
    frames = [{"t": 0.0, "tracks": [], "inference_s": 0.5}, {"t": 100.0, "tracks": tracks, "inference_s": 0.5}]
    res = score.score(frames, tracks, bowls, 2.0, 0.75)
    assert res["survey"]["tp"] == 2 and res["survey"]["fp"] == 2 and res["survey"]["fn"] == 1
    assert res["survey"]["recall"] == 2 / 3 and res["missed_bowls"] == [2]
    assert res["verify"]["tp"] == 2 and res["verify"]["fp"] == 0 and res["verify"]["rejected"] == 1
    assert math.isclose(res["mission_time_s"], 100.0)
