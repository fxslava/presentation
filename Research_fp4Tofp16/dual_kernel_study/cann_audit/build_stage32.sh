#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils tikcpp/tikcfw; do inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p"); done
out=/work/dual_kernel_study/cann_audit/native950
bisheng -std=c++17 -O2 -DNDEBUG --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -c /work/dual_kernel_study/cann_audit/cube_trick_stage32.cpp -o "$out/stage32.o" > "$out/stage32_compile.log" 2>&1 || { cat "$out/stage32_compile.log"; exit 1; }
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/stage32.o" -static -o "$out/stage32_kernel.o"
g++ -std=c++17 -O2 -DRAW_BINARY -I"$ASCEND_HOME_PATH/include" /work/dual_kernel_study/cann_audit/harness.cpp -o "$out/raw_harness_stage32" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
