#!/usr/bin/env bash
set -e
OUT="/data/telemetry/current_signals.txt"
mkdir -p "$(dirname "$OUT")"

cpu_raw=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 50000)
gpu_raw=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
gpu_raw=${gpu_raw:-50}

safe_cpu=$(awk -v t=$cpu_raw 'BEGIN{t/=1000;v=1-(t-40)/40;if(v<0)v=0;if(v>1)v=1;print v}')
safe_gpu=$(awk -v t=$gpu_raw 'BEGIN{v=1-(t-40)/40;if(v<0)v=0;if(v>1)v=1;print v}')

build_log="/root/Qallow/build.log"
if [[ -f "$build_log" ]]; then
  errors=$(grep -c "error" "$build_log" || true)
else
  errors=0
fi
clarity=$(awk -v e=$errors 'BEGIN{v=1-e/10;if(v<0)v=0;if(v>1)v=1;print v}')

human_file="/data/human_feedback.txt"
if [[ -f "$human_file" ]]; then
  human=$(cat "$human_file")
else
  human=0.7
fi

printf '%s %s %s  %s %s %s %s  %s %s %s\n' \
  "$safe_cpu" "$safe_gpu" "$safe_cpu" \
  "$clarity" "$clarity" "$clarity" "$clarity" \
  "$human" "$human" "$human" > "$OUT"

echo "[collect] Updated $OUT"
