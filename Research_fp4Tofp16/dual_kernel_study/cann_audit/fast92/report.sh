#!/bin/bash
set -eo pipefail
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
id=$1
archive=$(find "$root/runs/$id/archive" -mindepth 1 -maxdepth 1 -type d | head -1)
timeout -k 3 55 npusim report -e "$archive" -o "$root/reports/$id" -n all > "$root/logs/${id}_report.log" 2>&1
tail -12 "$root/logs/${id}_report.log"
