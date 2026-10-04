#!/bin/bash
set -o pipefail
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
arch=${1:-dav-2201}; soc=${2:-Ascend910B1}; elems=${3:-512}; block=${4:-16}; variant=${5:-simd}; model=${6:-ca}; fmt=${7:-e2m1}
build="$root/build/${arch}_${fmt}_b${block}_e${elems}"
id="${arch}_${fmt}_b${block}_e${elems}_${variant}_${model}"
[ -z "${8:-}" ] || id="${id}_${8}"
mkdir -p "$root/logs" "$root/runs/$id"
cd "$root/runs/$id"
export FP4_ELEMENTS=$elems
mode=A; [ "$variant" != cube ] || mode=B
start=$SECONDS
if [ "$model" = pv ]; then
  sim="$ASCEND_HOME_PATH/x86_64-linux/simulator/${arch/-/_}/lib"
  export LD_LIBRARY_PATH="$sim:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH"
  timeout -k 3 55 "$build/harness_${variant}_pv" "$mode" "$build/${variant}_kernel.o" > "$root/logs/$id.log" 2>&1
else
  timeout -k 3 55 npusim record -s "$soc" -o "$PWD/archive" -f "$build/${variant}_kernel.o" -u "$mode $build/${variant}_kernel.o" "$build/harness_$variant" > "$root/logs/$id.log" 2>&1
fi
status=$?
echo "$id exit=$status wall_seconds=$((SECONDS-start))" | tee "$root/logs/$id.status"
tail -22 "$root/logs/$id.log"
exit "$status"
