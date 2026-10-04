#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
root=/tmp/fp4-cann-cmake
sim=/usr/local/Ascend/cann-8.5.0/tools/simulator/Ascend910B1/lib
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release > "$root/build.log" 2>&1 && cmake --build "$root/build" -j2 >> "$root/build.log" 2>&1 || { tail -50 "$root/build.log"; exit 2; }
export LD_LIBRARY_PATH="$sim:/usr/local/Ascend/cann-8.5.0/lib64:$LD_LIBRARY_PATH"
export CAMODEL_LOG_PATH="$root/model_logs"
mkdir -p "$CAMODEL_LOG_PATH"
for k in A B; do
  mkdir -p "$root/run_$k"
  cd "$root/run_$k"
  timeout 60 "$root/build/fp4_harness" "$k" > run.log 2>&1
  echo "kernel=$k exit=$?"
  tail -40 run.log
done
