#!/bin/bash
set -o pipefail
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
e=$1; b=$2; v=$3; tag=${4:-isolated}
build="$root/build/dav-3510_e2m1_b${b}_e${e}"
id="dav-3510_e2m1_b${b}_e${e}_${v}_ca_${tag}"
mkdir -p "$root/runs/$id" "$root/logs"
cd "$root/runs/$id"
# npusim uses the executable's parent as cwd. A physical copy isolates traces.
cp "$build/harness_$v" ./app
export FP4_ELEMENTS=$e
mode=A; [ "$v" != cube ] || mode=B
start=$SECONDS
timeout -k 5 85 npusim record -s Ascend950 -o "$PWD/archive" -f "$build/${v}_kernel.o" -u "$mode $build/${v}_kernel.o" "$PWD/app" > "$root/logs/$id.log" 2>&1
status=$?
echo "$id exit=$status wall_seconds=$((SECONDS-start))" | tee "$root/logs/$id.status"
tail -7 "$root/logs/$id.log"
bash "$root/report.sh" "$id"
exit "$status"
