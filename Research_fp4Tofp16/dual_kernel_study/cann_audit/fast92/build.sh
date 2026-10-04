#!/bin/bash
set -eo pipefail
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
arch=${1:-dav-2201}; elems=${2:-512}; block=${3:-16}; fmt=${4:-e2m1}
out="$root/build/${arch}_${fmt}_b${block}_e${elems}"
mkdir -p "$out"
extra=(); [ "$fmt" != e2m1 ] || extra=(-DFAST_E2M1)
for variant in simd cube simt; do
  start=$SECONDS
  if bisheng -std=c++17 -O2 -DNDEBUG -DFP4_BLOCK="$block" -DFAST_TILE="$elems" "${extra[@]}" --asc-aicore-lang --cce-aicore-only --npu-arch="$arch" "${inc[@]}" -c "$root/src/$variant.cpp" -o "$out/$variant.o" > "$out/${variant}_compile.log" 2>&1; then
    "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld" -m aicorelinux -Ttext=0 "$out/$variant.o" -static -o "$out/${variant}_kernel.o" >> "$out/${variant}_compile.log" 2>&1
    hx=(); [ "$variant" != simt ] || hx=(-DSIMT_DIRECT)
    g++ -std=c++17 -O2 -DRAW_BINARY -DFP4_BLOCK="$block" -DFAST_TILE="$elems" "${extra[@]}" "${hx[@]}" -I"$ASCEND_HOME_PATH/include" "$root/src/harness.cpp" -o "$out/harness_$variant" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" >> "$out/${variant}_compile.log" 2>&1
    sim="$ASCEND_HOME_PATH/tools/simulator/${arch/-/_}/lib"
    g++ -std=c++17 -O2 -DRAW_BINARY -DFP4_BLOCK="$block" -DFAST_TILE="$elems" "${extra[@]}" "${hx[@]}" -I"$ASCEND_HOME_PATH/include" "$root/src/harness.cpp" -o "$out/harness_${variant}_pv" -L"$sim" -L"$ASCEND_HOME_PATH/lib64" -Wl,--no-as-needed -lruntime_cmodel -lnpu_drv_pvmodel -lnpu_drv -lascendcl -Wl,--disable-new-dtags -Wl,-rpath,"$sim:$ASCEND_HOME_PATH/lib64" >> "$out/${variant}_compile.log" 2>&1
    touch "$out/${variant}.ready"
    echo "$arch $fmt E=$elems B=$block $variant BUILD_PASS $((SECONDS-start))s"
  else
    echo "$arch $fmt E=$elems B=$block $variant BUILD_FAIL $((SECONDS-start))s"
    head -12 "$out/${variant}_compile.log"
  fi
done
