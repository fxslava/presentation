#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
out=/work/dual_kernel_study/cann_audit/native950
src=/work/dual_kernel_study/cann_audit
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils asc/include/simt_api tikcpp/tikcfw; do inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p"); done
for block in 16 32 64; do
  for path in simd_naive_unpack simt_direct_gm_unpack simt_cube_hybrid; do
    bisheng -std=c++17 -O2 -DNDEBUG -DFP4_BLOCK="$block" --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -c "$src/$path.cpp" -o "$out/${path}_b${block}.o" > "$out/${path}_b${block}_compile.log" 2>&1 || { cat "$out/${path}_b${block}_compile.log"; exit 1; }
    "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/${path}_b${block}.o" -static -o "$out/${path}_b${block}_kernel.o"
  done
  g++ -std=c++17 -O2 -DRAW_BINARY -DFP4_BLOCK="$block" -I"$ASCEND_HOME_PATH/include" "$src/harness.cpp" -o "$out/harness_simd_b${block}" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
  g++ -std=c++17 -O2 -DRAW_BINARY -DSIMT_DIRECT -DFP4_BLOCK="$block" -I"$ASCEND_HOME_PATH/include" "$src/harness.cpp" -o "$out/harness_simt_direct_b${block}" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
  g++ -std=c++17 -O2 -DRAW_BINARY -DSIMT_HYBRID -DFP4_BLOCK="$block" -I"$ASCEND_HOME_PATH/include" "$src/harness.cpp" -o "$out/harness_simt_hybrid_b${block}" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
done
bisheng -std=c++17 -O2 -DNDEBUG --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -c "$src/simt_direct_gm_only.cpp" -o "$out/simt_direct_gm_only.o" > "$out/simt_direct_gm_only_compile.log" 2>&1 || { cat "$out/simt_direct_gm_only_compile.log"; exit 1; }
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/simt_direct_gm_only.o" -static -o "$out/simt_direct_gm_only_kernel.o"
