#!/usr/bin/env bash
# RASPBERRY PI: platform control before EVERY measurement session.
set -euo pipefail
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$c" >/dev/null; done
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "freq kHz: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
vcgencmd measure_temp || true
vcgencmd get_throttled || true   # must be throttled=0x0 (else: better PSU / cooling)
echo "Use the official PSU and an active cooler. Idle 5 min before the first run."
