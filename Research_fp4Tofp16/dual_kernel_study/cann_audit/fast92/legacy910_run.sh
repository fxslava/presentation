#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
e=${1:-512}; b=${2:-16}; v=${3:-simd}
build="$root/build/dav-2201_e2m1_b${b}_e${e}"
sim="$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib"
id="dav-2201_e2m1_b${b}_e${e}_${v}_legacy_ca"
out="$root/runs/$id"
mkdir -p "$out/lib" "$out/log/ub_log" "$out/run_log"
ln -sf "$sim/libruntime_camodel.so" "$out/lib/libruntime.so"
g++ -std=c++17 -O2 -DRAW_BINARY -DFAST_E2M1 -DFP4_BLOCK="$b" -DFAST_TILE="$e" -I"$ASCEND_HOME_PATH/include" "$root/src/harness.cpp" -o "$build/harness_${v}_legacy_ca" -L"$sim" -L"$ASCEND_HOME_PATH/lib64" -Wl,--no-as-needed -lruntime_camodel -lnpu_drv_camodel -lnpu_drv -lascendcl -Wl,--disable-new-dtags -Wl,-rpath,"$out/lib:$sim:$ASCEND_HOME_PATH/lib64" > "$root/logs/${id}_compile.log" 2>&1
export LD_LIBRARY_PATH="$out/lib:$sim:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH"
export CAMODEL_LOG_PATH="$out/log" STARS_LOG_PATH="$out/log" FP4_ELEMENTS="$e"
cd "$out"
mode=A; [ "$v" != cube ] || mode=B
start=$SECONDS
timeout -k 3 55 msprof op simulator --application="$build/harness_${v}_legacy_ca $mode $build/${v}_kernel.o" --output="$root/reports/$id" --soc-version=Ascend910B4 --core-id=0 > "$root/logs/$id.log" 2>&1
status=$?
echo "$id exit=$status wall_seconds=$((SECONDS-start))" | tee "$root/logs/$id.status"
tail -22 "$root/logs/$id.log"
exit "$status"
