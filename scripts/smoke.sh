#!/usr/bin/env bash
# Headless end-to-end smoke run: Gazebo + PX4 SITL + MAVROS + perception (oracle detector, so no
# weights or GPU) + mission (survey at 40 m, verify at 11 m) + scoring. Exits 0 only if the survey
# found every bowl. Takes about 4 minutes on the sim machine with software rendering.
#
#   scripts/smoke.sh                 # sparse world, 2 verifications
#   MAX_VERIFY=0 scripts/smoke.sh    # verify every track (slower)
#   DETECTOR=yolo POSE=mavros scripts/smoke.sh   # real detector on the full-resolution camera (needs GPU + weights)
#   RECORD=true scripts/smoke.sh     # also write follow-camera and annotated MP4s into the run dir
set -o pipefail   # no -u: the ROS setup scripts reference unset variables
cd "$(dirname "$0")/.."
source scripts/env.sh
deactivate 2>/dev/null || true            # ROS nodes run on the system Python
[ -f ros2_ws/install/setup.bash ] || (cd ros2_ws && colcon build --symlink-install >/dev/null)
[ -x build/gz_set_pose/gz_set_pose ] || scripts/build_tools.sh
source ros2_ws/install/setup.bash
export UAV_DT_REPO="$PWD"

WORLD="${WORLD:-bowl_field_sparse}"
DETECTOR="${DETECTOR:-gt}"
POSE="${POSE:-gz}"
MAX_VERIFY="${MAX_VERIFY:-2}"
DWELL="${DWELL:-3}"
TIMEOUT_S="${TIMEOUT_S:-600}"
MODEL="${MODEL:-$([ "$DETECTOR" = gt ] && echo x500_nadir_cam_lite || echo x500_nadir_cam)}"
RUN="${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="runs/$RUN"; mkdir -p "$LOGDIR"

cleanup() {
  echo "smoke: cleaning up"
  # stop the recorder first and give ffmpeg time to flush before anything else goes down
  for p in $(pgrep -f 'scripts/record_video.py' 2>/dev/null); do kill -INT "$p" 2>/dev/null; done; sleep 3
  [ -n "${MP:-}" ] && kill -INT "$MP" 2>/dev/null
  [ -n "${LP:-}" ] && kill -INT "$LP" 2>/dev/null
  sleep 5
  pkill -x px4 2>/dev/null; pkill -x ruby 2>/dev/null
  for p in $(pgrep -f 'parameter_bridge|perception_node|mission_node|mavros_node|ros_gz_sim/create' 2>/dev/null); do kill "$p" 2>/dev/null; done
}
trap cleanup EXIT
# a stale PX4 makes the new one exit with "already running"
pkill -x px4 2>/dev/null; pkill -x ruby 2>/dev/null; sleep 1

echo "smoke: starting sim ($WORLD, $MODEL, headless)"
ros2 launch uav_dt_ros sim.launch.py world:="$WORLD" model:="$MODEL" headless:=true mavros:=true > "$LOGDIR/sim.log" 2>&1 &
LP=$!
for i in $(seq 1 120); do grep -q "Got HEARTBEAT" "$LOGDIR/sim.log" 2>/dev/null && break; sleep 1; done
grep -q "Got HEARTBEAT" "$LOGDIR/sim.log" || { echo "smoke: MAVROS never connected, see $LOGDIR/sim.log"; exit 1; }
echo "smoke: MAVROS connected after ${i}s"; sleep 5

echo "smoke: starting perception ($DETECTOR, pose from $POSE) and mission (max_verify=$MAX_VERIFY)"
ros2 launch uav_dt_ros mission.launch.py world:="$WORLD" model:="$MODEL" detector:="$DETECTOR" pose_source:="$POSE" \
  max_verify:="$MAX_VERIFY" dwell_s:="$DWELL" run_name:="$RUN" record:="${RECORD:-false}" > "$LOGDIR/mission.log" 2>&1 &
MP=$!
T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "$TIMEOUT_S" ]; do
  grep -q '"state": "done"' "$LOGDIR/mission.jsonl" 2>/dev/null && break
  kill -0 "$MP" 2>/dev/null || break
  sleep 5
done
grep -q '"state": "done"' "$LOGDIR/mission.jsonl" 2>/dev/null || { echo "smoke: mission did not finish in ${TIMEOUT_S}s, see $LOGDIR/mission.log"; exit 1; }
echo "smoke: mission done in $(( $(date +%s) - T0 ))s wall"

python3 scripts/score.py "$LOGDIR/perception.jsonl" "sim/worlds/$WORLD.json" --markdown "$LOGDIR/results.md" --min-recall 1.0 \
  | python3 -c "import json,sys; t=sys.stdin.read(); r=json.loads(t[:t.rindex('}')+1]); s=r['survey']; v=r.get('verify',{}); print(f\"smoke: survey recall {s['recall']:.2f} precision {s['precision']:.2f} mean error {s['mean_error_m']:.3f} m, verified {v.get('tp','-')}/{v.get('verified','-')}, mission {r['mission_time_s']:.0f} s sim\")" \
  && { echo "SMOKE RUN PASSED ($LOGDIR)"; exit 0; } || { echo "SMOKE RUN FAILED ($LOGDIR)"; exit 1; }
