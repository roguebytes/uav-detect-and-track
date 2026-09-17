#!/usr/bin/env bash
# Author: Frank Loewenich
# Build the small native helpers under tools/ into build/ (needs the Gazebo Garden dev packages).
set -e
cd "$(dirname "$0")/.."
mkdir -p build/gz_set_pose
(cd build/gz_set_pose && cmake ../../tools/gz_set_pose > /dev/null && make --no-print-directory)
echo "built build/gz_set_pose/gz_set_pose"
