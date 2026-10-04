#!/bin/bash
out=/work/dual_kernel_study/cann_audit/native950/analyzer_discovery.log
{
  echo 'Requested command: ascend_analyzer --chip ascend351x'
  if command -v ascend_analyzer; then
    ascend_analyzer --help
  else
    echo 'ascend_analyzer: not in PATH'
  fi
  find /opt/Ascend92 -type f -name 'ascend_analyzer*' -print
  echo 'Compiler diagnostics for 16/32/64 builds:'
  for f in /work/dual_kernel_study/cann_audit/native950/*_b{16,32,64}_compile.log; do
    if [ -s "$f" ]; then echo "$f: nonempty"; else echo "$f: empty"; fi
  done
} > "$out" 2>&1
cat "$out"
