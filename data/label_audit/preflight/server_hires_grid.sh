#!/bin/sh
cd "$(dirname "$0")"
for a in "640" "800" "1024" "640 tta" "1024 tta"; do python server_hires.py $a gpu >> server_hires_gpu.log 2>&1; done
echo ALLDONE >> server_hires_gpu.log
