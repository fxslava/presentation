#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export PATH="$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
inc=("-I$ASCEND_HOME_PATH/include")
for p in asc/impl/adv_api asc/impl/basic_api asc/impl/utils asc/include asc/include/adv_api asc/include/basic_api asc/include/aicpu_api asc/include/utils asc/include/simt_api tikcpp/tikcfw; do
  inc+=("-I$ASCEND_HOME_PATH/x86_64-linux/$p")
done
