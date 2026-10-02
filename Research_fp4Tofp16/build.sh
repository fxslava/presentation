#!/usr/bin/env bash
# build.sh -- compile the NVFP4 dequant kernel (ccec/bisheng, dav-3510) and the
# ACL host harness. Run inside WSL/Linux with CANN 9.x installed:
#
#   ./build.sh [project_root]
#
# Environment:
#   ASCEND_HOME_PATH  (sourced from ascend-toolkit/set_env.sh if unset)
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
OUT="$ROOT/build"
mkdir -p "$OUT"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
echo "[build] ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

CCEC="$ASCEND_HOME_PATH/x86_64-linux/bin/ccec"
LLD="$ASCEND_HOME_PATH/x86_64-linux/bin/ld.lld"
if [ ! -x "$CCEC" ]; then CCEC="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ccec"; fi
if [ ! -x "$LLD" ]; then LLD="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld"; fi
echo "[build] ccec=$CCEC"
echo "[build] lld=$LLD"

KERNEL_CXXFLAGS=(
    -std=c++17 -O2 -DNDEBUG
    --asc-aicore-lang
    --cce-aicore-only
    --npu-arch=dav-3510
    -I"$ROOT/nvfp4_dequant"
    -I"$ASCEND_HOME_PATH/x86_64-linux/tikcpp/tikcfw"
    -I"$ASCEND_HOME_PATH/include"
)

# ---- device object: single entry TU; Cube/Vector pipeline units are the
# header-inline nvfp4_dequant_aic.h / nvfp4_dequant_aiv.h (arch35 MIX pattern)
"$CCEC" "${KERNEL_CXXFLAGS[@]}" -c "$ROOT/nvfp4_dequant/nvfp4_dequant_main.cpp" -o "$OUT/nvfp4_dequant_main.o"

"$LLD" -m aicorelinux -Ttext=0 \
    "$OUT/nvfp4_dequant_main.o" \
    -static -o "$OUT/nvfp4_dequant_kernel.o"
echo "[build] kernel image: $OUT/nvfp4_dequant_kernel.o"

# ---- host harness ----------------------------------------------------------
g++ -std=c++17 -O2 -g \
    -I"$ROOT/nvfp4_dequant" -I"$ROOT/host" -I"$ASCEND_HOME_PATH/include" \
    "$ROOT/host/main.cpp" \
    -o "$OUT/nvfp4_host" \
    -L"$ASCEND_HOME_PATH/lib64" -lascendcl -Wl,-rpath,"$ASCEND_HOME_PATH/lib64"
echo "[build] host harness: $OUT/nvfp4_host"
echo "[build] done"
