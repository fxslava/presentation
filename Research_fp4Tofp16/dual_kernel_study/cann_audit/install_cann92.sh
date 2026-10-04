#!/bin/bash
set -euo pipefail
toolkit=/packages/Ascend-cann-toolkit_9.2.0-beta.2_linux-x86_64.run
ops=/packages/Ascend-cann-950-ops_9.2.0-beta.2_linux-x86_64.run
out=/work/dual_kernel_study/cann_audit
bash "$toolkit" --check > "$out/toolkit92_integrity.log" 2>&1
bash "$ops" --check > "$out/ops950_integrity.log" 2>&1
bash "$toolkit" --install --quiet --install-for-all --install-path=/opt/Ascend92 > "$out/toolkit92_install.log" 2>&1
source /opt/Ascend92/ascend-toolkit/set_env.sh
bash "$ops" --install --quiet --install-for-all --install-path=/opt/Ascend92 > "$out/ops950_install.log" 2>&1
bisheng --version
find /opt/Ascend92 -maxdepth 7 -name '*950*' -o -name npusim
