#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
sim="$ASCEND_HOME_PATH/tools/simulator/dav_2201/lib"
out="$root/runs/legacy910_ca"
mkdir -p "$out/lib" "$out/log" "$out/run_log" "$out/log/ub_log"
ln -sf "$sim/libruntime_camodel.so" "$out/lib/libruntime.so"
export LD_LIBRARY_PATH="$out/lib:$sim:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH"
export CAMODEL_LOG_PATH="$out/log" STARS_LOG_PATH="$out/log"
g++ -O2 -I"$ASCEND_HOME_PATH/include" "$root/smoke.cpp" -L"$sim" -L"$ASCEND_HOME_PATH/lib64" -Wl,--no-as-needed -lruntime_camodel -lnpu_drv_camodel -lnpu_drv -lascendcl -Wl,--disable-new-dtags -Wl,-rpath,"$out/lib:$sim:$ASCEND_HOME_PATH/lib64" -o "$root/build/smoke_legacy_ca"
cd "$out"
timeout -k 3 25 "$root/build/smoke_legacy_ca" > "$root/logs/smoke_legacy_ca.log" 2>&1
echo "legacy CA smoke exit=$?"
tail -18 "$root/logs/smoke_legacy_ca.log"
command -v msopprof msprof
LD_LIBRARY_PATH="$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH" timeout 10 msprof op simulator --help > "$root/logs/msprof_help.log" 2>&1
tail -8 "$root/logs/msprof_help.log"
