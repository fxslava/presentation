#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /mnt/d/Projects/Presentation/Research_fp4Tofp16/dual_kernel_study
inc=()
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils tikcpp/tikcfw; do
  inc+=("-I/usr/local/Ascend/cann-8.5.0/x86_64-linux/$p")
done
for k in naive_aiv cube_trick; do
  bisheng -std=c++17 -O2 --cce-aicore-lang --cce-aicore-only --cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-aicore-arch=dav-c220-vec --cce-auto-sync -DTILING_KEY_VAR=0 "${inc[@]}" -I/usr/local/Ascend/cann-8.5.0/include -c ${k}_fp4_to_fp16.cpp -o cann_audit/${k}.o > cann_audit/${k}_compile.log 2>&1
  echo "$k exit=$?"
  head -80 cann_audit/${k}_compile.log
done
