#!/usr/bin/env bash
set -euo pipefail

set -a
source /mnt/c/Users/intern/Documents/SEP490/train/.env
set +a

cache=/mnt/c/Users/intern/Documents/SEP490/train/benchmakring_transmission_line_defect_model/data/v6p2e_cache
mkdir -p "$cache"

camber_bin="$HOME/.camber/bin/camber"
data_root=stash://bunpmc/projects/yolov11sdi/datasets

"$camber_bin" stash cp \
  "$data_root/curated_v6_rs1280_56b7ea194ee5.tar" \
  "$cache/base.tar"

"$camber_bin" stash cp \
  "$data_root/v6p1/v6p1_5d0b1149891c/v6p1_fotl_overlay_ee2d3c4497fd.tar.gz" \
  "$cache/overlay.tar.gz"
