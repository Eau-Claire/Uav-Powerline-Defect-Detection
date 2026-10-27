#!/usr/bin/env bash
set -euo pipefail

root=/mnt/c/Users/intern/Documents/SEP490/train/benchmakring_transmission_line_defect_model/data
cache="$root/v6p2e_cache"
sources="$root/v6p2e_sources"

mkdir -p "$sources/base" "$sources/overlay"

tar -xf "$cache/base.tar" -C "$sources/base" \
  --wildcards '*/images/test/*' '*/labels/test/*' '*/data.yaml'

tar -xzf "$cache/overlay.tar.gz" -C "$sources/overlay" \
  --wildcards '*/images/test/*' '*/labels/test/*' '*/data.yaml'
