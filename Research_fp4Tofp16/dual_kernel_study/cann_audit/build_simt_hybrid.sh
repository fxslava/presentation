#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils asc/include/simt_api tikcpp/tikcfw; do inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p"); done
bisheng -std=c++17 -O2 -DNDEBUG --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -c /work/dual_kernel_study/cann_audit/simt_cube_hybrid.cpp -o "$out/simt_cube_hybrid.o" > "$out/simt_cube_hybrid_compile.log" 2>&1
status=$?
cat "$out/simt_cube_hybrid_compile.log"
if [ "$status" = 0 ]; then "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/simt_cube_hybrid.o" -static -o "$out/simt_cube_hybrid_kernel.o"; fi
g++ -std=c++17 -O2 -DRAW_BINARY -DSIMT_HYBRID -I"$ASCEND_HOME_PATH/include" /work/dual_kernel_study/cann_audit/harness.cpp -o "$out/raw_harness_simt_hybrid" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
exit "$status"
