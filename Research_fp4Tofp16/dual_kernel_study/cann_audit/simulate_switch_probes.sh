#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
mkdir -p /tmp/fp4-switch-probes
cd /tmp/fp4-switch-probes
for kind in scalar simt; do
  timeout -k 10 180 npusim record -s Ascend950 -o "$out/sim_${kind}_switch" -f "$out/${kind}_switch_probe_kernel.o" -u "A $out/${kind}_switch_probe_kernel.o" "$out/harness_${kind}_switch" > "$out/sim_${kind}_switch.log" 2>&1
  status=$?
  echo "$kind probe exit=$status"
  tail -18 "$out/sim_${kind}_switch.log"
  if [ "$status" != 0 ]; then exit "$status"; fi
done
