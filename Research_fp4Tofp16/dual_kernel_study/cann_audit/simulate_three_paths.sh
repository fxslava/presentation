#!/bin/bash
set -o pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
path=${1:-simt_direct_gm_unpack}
block=${2:-16}
elements=${3:-8192}
case "$path" in
  simd_naive_unpack) h=harness_simd_b$block; mode=A;;
  simt_direct_gm_unpack) h=harness_simt_direct_b$block; mode=A;;
  simt_cube_hybrid) h=harness_simt_hybrid_b$block; mode=B;;
  *) echo "invalid path" >&2; exit 2;;
esac
export FP4_ELEMENTS=$elements
mkdir -p /tmp/fp4-three-paths
cd /tmp/fp4-three-paths
id=${path}_b${block}_n${elements}
timeout -k 10 360 npusim record -s Ascend950 -o "$out/sim_$id" -f "$out/${path}_b${block}_kernel.o" -u "$mode $out/${path}_b${block}_kernel.o" "$out/$h" > "$out/sim_${id}.log" 2>&1
status=$?
tail -22 "$out/sim_${id}.log"
exit "$status"
