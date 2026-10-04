#!/bin/bash
root=/usr/local/Ascend/cann-9.2.0-beta.2
grep -R -n 'CrossCoreSetFlag\|SetSyncBaseAddr' "$root/x86_64-linux/asc/impl/basic_api/kernel_operator_common_impl.h" "$root/x86_64-linux/asc/impl/basic_api/dav_c220" | head -35
sed -n '55,92p' "$root/x86_64-linux/asc/impl/basic_api/utils/sys_macros.h"
find "$root" -name '*Launch*' -o -name '*launch*' | head -15
