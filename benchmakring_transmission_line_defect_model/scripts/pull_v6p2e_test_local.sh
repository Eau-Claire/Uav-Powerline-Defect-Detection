#!/usr/bin/env bash
# Pull V6.2E sources from Camber, extract only TEST, rebuild v6p2e_frozen_test
# (2012 imgs) and test_main_v1 (1931 imgs) locally. Needs `camber login` or
# CAMBER_API_KEY (env or repo-root .env).
set -euo pipefail

bench="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd -- "$bench/.." && pwd)"
data="$bench/data"
cache="$data/v6p2e_cache"
sources="$data/v6p2e_sources"
records="$cache/records"
py="${PYTHON:-$repo/.venv/bin/python}"

if [[ -f "$repo/.env" ]]; then set -a; . "$repo/.env"; set +a; fi
camber="${CAMBER_BIN:-$(command -v camber || true)}"
[[ -x "$camber" ]] || camber="$HOME/.camber/bin/camber"
[[ -x "$camber" ]] || camber="$repo/deployment/modal/.tools/camber"
"$camber" me >/dev/null 2>&1 || { echo "Camber not authenticated: run '$camber login' or set CAMBER_API_KEY" >&2; exit 1; }

root=stash://bunpmc/projects/yolov11sdi/datasets
tag=v6p2e_23562a45b343
mkdir -p "$cache" "$records" "$sources/base" "$sources/overlay"

fetch() { [[ -s "$2" ]] && { echo "reuse $2"; return; }; echo "get $1"; "$camber" stash cp "$1" "$2.part" && mv "$2.part" "$2"; }

# Records (manifest + reconciled labels) sit under the tag folder; locate them.
listing="$("$camber" stash ls -r "$root" 2>&1 || true)"
for f in v6p2e_split_manifest.csv v6p2e_reconciled_labels.zip v6p2e.freeze.json v6p2e_final_distribution.csv perceptual_hash_manifest.csv; do
    rel="$(grep -oE "[^[:space:]]*$tag[^[:space:]]*/$f|[^[:space:]]*/$tag/$f" <<<"$listing" | head -1 || true)"
    [[ -n "$rel" ]] || rel="$tag/$f"
    [[ "$rel" == stash://* ]] || rel="$root/${rel#/}"
    fetch "$rel" "$records/$f"
done

fetch "$root/curated_v6_rs1280_56b7ea194ee5.tar" "$cache/base.tar"
fetch "$root/v6p1/v6p1_5d0b1149891c/v6p1_fotl_overlay_ee2d3c4497fd.tar.gz" "$cache/overlay.tar.gz"

[[ -d "$sources/base/curated_v6/images/test" ]] || tar -xf "$cache/base.tar" -C "$sources/base" \
    --wildcards '*/images/test/*' '*/labels/test/*' '*/data.yaml'
[[ -d "$sources/overlay/v6p1_fotl_overlay/images/test" ]] || tar -xzf "$cache/overlay.tar.gz" -C "$sources/overlay" \
    --wildcards '*/images/test/*' '*/labels/test/*' '*/data.yaml'

# Rebuild into a scratch dir so the committed records stay untouched, then
# compare its freeze against the committed one before linking images over.
rebuilt="$data/v6p2e_rebuilt"
"$py" "$bench/scripts/reconstruct_v6p2e_test.py" --records-dir "$records" --sources-dir "$sources" --out-dir "$rebuilt"
"$py" - "$rebuilt" "$data/v6p2e_frozen_test" <<'EOF'
import json, sys
from pathlib import Path
new, old = (json.loads((Path(p) / "freeze.json").read_text()) for p in sys.argv[1:])
diff = {k: (old.get(k), new.get(k)) for k in old if old.get(k) != new.get(k)}
if diff:
    sys.exit(f"freeze.json mismatch vs committed record: {diff}")
print("v6p2e_frozen_test matches committed freeze.json")
EOF
for d in images labels; do
    rm -rf "$data/v6p2e_frozen_test/$d"; mv "$rebuilt/$d" "$data/v6p2e_frozen_test/$d"
done
rm -rf "$rebuilt"

# test_main_v1 = frozen TEST deduplicated minus reviewed person-as-FO images.
local_yaml="$cache/v6p2e_frozen_test.local.yaml"
sed "s#^path:.*#path: $data/v6p2e_frozen_test#" "$data/v6p2e_frozen_test/dataset.yaml" > "$local_yaml"
tmp_main="$cache/test_main_v1_rebuild"
rm -rf "$tmp_main"
(cd "$bench" && "$py" scripts/build_clean_test.py full --dataset-yaml "$local_yaml" \
    --exclude-csv data/clean_test_v2/fo_person_review.csv --out-dir "$tmp_main")
cmp -s <(tail -n +2 "$tmp_main/manifest.csv" | sort) <(tail -n +2 "$data/test_main_v1/manifest.csv" | sort) \
    || { echo "test_main_v1 manifest differs from committed one; kept rebuild at $tmp_main" >&2; exit 1; }
for d in images labels; do
    rm -rf "$data/test_main_v1/$d"; mv "$tmp_main/$d" "$data/test_main_v1/$d"
done
rm -rf "$tmp_main"
echo "Done: $(ls "$data/v6p2e_frozen_test/images/test" | wc -l) frozen, $(ls "$data/test_main_v1/images/test" | wc -l) test_main_v1"
