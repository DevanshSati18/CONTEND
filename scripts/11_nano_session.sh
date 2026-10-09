#!/usr/bin/env bash
# Platform control before EVERY measurement session (Execution Plan, Section 4 checklist).
#   bash scripts/11_nano_session.sh
set -euo pipefail
sudo nvpmodel -m 0                  # MAXN (10 W) - needs the 5 V/4 A barrel supply + J48 jumper
sudo jetson_clocks                  # lock CPU/GPU/EMC clocks at max
sudo jetson_clocks --fan || true    # fan to max where supported
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' 2>/dev/null || true
sudo systemctl stop docker 2>/dev/null || true
sudo nvpmodel -q
sudo jetson_clocks --show | head -20
echo "Idle 5 minutes before the first run. Write the room temperature into docs/session_log.md."
