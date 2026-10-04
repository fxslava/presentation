#!/bin/bash
set -eo pipefail
source /opt/Ascend92/cann/set_env.sh
set -u
root=/work
out=$root/dual_kernel_study/cann_audit/native950
mkdir -p "$out"
inc=("-I$root/nvfp4_dequant" "-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils tikcpp/tikcfw; do
  inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p")
done
flags=(-std=c++17 -O2 -DNDEBUG --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510)
bisheng "${flags[@]}" "${inc[@]}" -c "$root/nvfp4_dequant/nvfp4_dequant_main.cpp" -o "$out/nvfp4_main.o" > "$out/native_compile.log" 2>&1
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/nvfp4_main.o" -static -o "$out/nvfp4_kernel.o"
g++ -std=c++17 -O2 -g -I"$root/nvfp4_dequant" -I"$root/host" -I"$ASCEND_HOME_PATH/include" "$root/host/main.cpp" -o "$out/nvfp4_host" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" > "$out/host_compile.log" 2>&1
for k in naive_aiv cube_trick; do
  set +e
  bisheng "${flags[@]}" "${inc[@]}" -c "$root/dual_kernel_study/cann_audit/${k}_fp4_to_fp16.cpp" -o "$out/${k}.o" > "$out/${k}_compile.log" 2>&1
  status=$?
  set -e
  echo "$k native950 compile exit=$status"
  head -35 "$out/${k}_compile.log"
  if [ "$status" = 0 ]; then
    "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/${k}.o" -static -o "$out/${k}_kernel.o"
  fi
done
g++ -std=c++17 -O2 -DRAW_BINARY -I"$ASCEND_HOME_PATH/include" "$root/dual_kernel_study/cann_audit/harness.cpp" -o "$out/raw_harness" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" > "$out/raw_host_compile.log" 2>&1
