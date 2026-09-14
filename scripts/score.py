#!/usr/bin/env python3
"""Score a perception log against the world's bowl manifest.

    python scripts/score.py runs/perception.jsonl sim/worlds/bowl_field_sparse.json [--survey-radius 2.0] [--verify-radius 0.75]

Matching: each confirmed track is matched to the nearest unmatched bowl within the survey
radius (greedy, closest pairs first). Verified tracks are matched again within the verify
radius. A bowl matches at most one track, so duplicate tracks on one bowl count as false
positives. Reports recall, precision, geolocation error and mission time, and writes an
optional Markdown table for docs/results.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def load_log(path):
    frames = [json.loads(line) for line in open(path) if line.strip()]
    if not frames:
        sys.exit(f"no frames in {path}")
    # final state of every track id (position converges as observations accumulate)
    tracks = {}
    for fr in frames:
        for tr in fr["tracks"]:
            tracks[tr["id"]] = tr
    return frames, list(tracks.values())


def greedy_match(tracks, bowls, radius):
    pairs = sorted(
        ((math.hypot(t["x"] - b["x"], t["y"] - b["y"]), ti, bi) for ti, t in enumerate(tracks) for bi, b in enumerate(bowls))
    )
    used_t, used_b, matches = set(), set(), []
    for d, ti, bi in pairs:
        if d > radius:
            break
        if ti in used_t or bi in used_b:
            continue
        used_t.add(ti); used_b.add(bi); matches.append((ti, bi, d))
    return matches


def score(frames, tracks, bowls, survey_radius, verify_radius):
    out = {"bowls": len(bowls), "frames": len(frames), "mission_time_s": frames[-1]["t"] - frames[0]["t"]}
    survey = greedy_match(tracks, bowls, survey_radius)
    out["survey"] = {
        "tracks": len(tracks), "tp": len(survey), "fp": len(tracks) - len(survey), "fn": len(bowls) - len(survey),
        "recall": len(survey) / len(bowls) if bowls else 0.0,
        "precision": len(survey) / len(tracks) if tracks else 0.0,
        "mean_error_m": sum(d for _, _, d in survey) / len(survey) if survey else float("nan"),
        "max_error_m": max((d for _, _, d in survey), default=float("nan")),
    }
    verified = [t for t in tracks if t.get("verified") is True]
    rejected = [t for t in tracks if t.get("verified") is False]
    if verified or rejected:
        vm = greedy_match(verified, bowls, verify_radius)
        out["verify"] = {
            "verified": len(verified), "rejected": len(rejected), "tp": len(vm), "fp": len(verified) - len(vm),
            "fn": len(bowls) - len(vm),
            "recall": len(vm) / len(bowls) if bowls else 0.0,
            "precision": len(vm) / len(verified) if verified else 0.0,
            "mean_error_m": sum(d for _, _, d in vm) / len(vm) if vm else float("nan"),
        }
    matched_b = {bi for _, bi, _ in survey}
    out["missed_bowls"] = [bowls[i]["id"] for i in range(len(bowls)) if i not in matched_b]
    inf = [fr["inference_s"] for fr in frames if fr.get("inference_s")]
    out["mean_inference_s"] = sum(inf) / len(inf) if inf else 0.0
    return out


def fmt(x):
    return "n/a" if isinstance(x, float) and math.isnan(x) else (f"{x:.2f}" if isinstance(x, float) else str(x))


def markdown(res, world):
    rows = [("Bowls in world", res["bowls"]), ("Frames processed", res["frames"]),
            ("Survey tracks (TP / FP / FN)", f"{res['survey']['tracks']} ({res['survey']['tp']} / {res['survey']['fp']} / {res['survey']['fn']})"),
            ("Survey recall", res["survey"]["recall"]), ("Survey precision", res["survey"]["precision"]),
            ("Survey geolocation error, mean / max (m)", f"{fmt(res['survey']['mean_error_m'])} / {fmt(res['survey']['max_error_m'])}")]
    if "verify" in res:
        v = res["verify"]
        rows += [("Verified tracks (TP / FP / FN)", f"{v['verified']} ({v['tp']} / {v['fp']} / {v['fn']})"),
                 ("Verified recall", v["recall"]), ("Verified precision", v["precision"]),
                 ("Verify geolocation error, mean (m)", v["mean_error_m"]), ("Rejected by verification", v["rejected"])]
    rows += [("Mission time (s)", res["mission_time_s"]), ("Mean inference per frame (s)", res["mean_inference_s"])]
    lines = [f"### {world}", "", "| Metric | Value |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in rows]
    if res["missed_bowls"]:
        lines.append(f"\nMissed bowl ids: {res['missed_bowls']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log"); ap.add_argument("manifest")
    ap.add_argument("--survey-radius", type=float, default=2.0)
    ap.add_argument("--verify-radius", type=float, default=0.75)
    ap.add_argument("--markdown", help="write a results table to this file")
    ap.add_argument("--min-recall", type=float, default=None, help="exit 1 if survey recall is below this")
    a = ap.parse_args()
    frames, tracks = load_log(a.log)
    manifest = json.load(open(a.manifest))
    res = score(frames, tracks, manifest["bowls"], a.survey_radius, a.verify_radius)
    print(json.dumps(res, indent=2))
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(markdown(res, manifest["world"]))
        print(f"wrote {a.markdown}")
    if a.min_recall is not None and res["survey"]["recall"] < a.min_recall:
        sys.exit(f"survey recall {res['survey']['recall']:.2f} below {a.min_recall}")


if __name__ == "__main__":
    main()
