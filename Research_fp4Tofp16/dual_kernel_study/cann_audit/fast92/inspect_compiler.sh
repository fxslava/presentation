#!/bin/bash
root=/work/dual_kernel_study/cann_audit/fast92
source "$root/env.sh"
grep -R -n 'define ASCEND_IS_AIC\|define ASCEND_IS_AIV' "$ASCEND_HOME_PATH/x86_64-linux/asc" | head -15
bisheng --help | grep -E 'mix|core-type|arch|run-mode|kernel' | head -50
readelf -S "$root/build/dav-2201_e2m1_b16_e512/cube.o"
readelf -S "$root/build/dav-3510_e2m1_b16_e512/cube.o"
strings "$root/build/dav-2201_e2m1_b16_e512/cube.o" | tail -18
ls /usr/local/Ascend/cann-9.2.0-beta.2/lib64/libascend_hal.so
