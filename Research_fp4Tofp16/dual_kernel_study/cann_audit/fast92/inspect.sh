#!/bin/bash
root=/usr/local/Ascend/cann-9.2.0-beta.2
grep -R -n 'LD_PRELOAD\|SUPPORTED_SOC\|Ascend950DT\|libnpu_drv_camodel' "$root/python/site-packages/cannsim/core" | head -60
for dir in dav_2201 dav_3510; do
  echo "SYMBOLS $dir"
  for f in "$root/tools/simulator/$dir/lib/"*.so "$root/tools/simulator/$dir/camodel/"*.so; do
    nm -D "$f" 2>/dev/null | grep -E ' [TW] .*get_stars_interrupt' && echo "$f"
  done
  ls -la "$root/tools/simulator/$dir"
done
find "$root/tools/simulator" -name '*pv*' -o -name '*model*' | tail -35
