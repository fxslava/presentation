#!/bin/bash
source /opt/Ascend92/cann/set_env.sh
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils tikcpp/tikcfw; do inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p"); done
cd /work/dual_kernel_study/cann_audit/native950
bisheng -std=c++17 -O2 --asc-aicore-lang --cce-aicore-only --npu-arch=dav-3510 "${inc[@]}" -S /work/dual_kernel_study/cann_audit/cube_trick_fp4_to_fp16.cpp
