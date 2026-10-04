#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
src=/work/dual_kernel_study/cann_audit
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils asc/include/simt_api tikcpp/tikcfw; do inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p"); done
for p in simt_switch_probe scalar_switch_probe; do
  bisheng -std=c++17 -O2 -DNDEBUG --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -c "$src/$p.cpp" -o "$out/$p.o" > "$out/${p}_compile.log" 2>&1 || { cat "$out/${p}_compile.log"; exit 1; }
  "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/$p.o" -static -o "$out/${p}_kernel.o"
done
g++ -std=c++17 -O2 -DRAW_BINARY -DSIMT_SWITCH_PROBE -I"$ASCEND_HOME_PATH/include" "$src/harness.cpp" -o "$out/harness_simt_switch" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
g++ -std=c++17 -O2 -DRAW_BINARY -DSCALAR_SWITCH_PROBE -I"$ASCEND_HOME_PATH/include" "$src/harness.cpp" -o "$out/harness_scalar_switch" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
