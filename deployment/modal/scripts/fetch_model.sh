#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$project_dir/models"
target="$model_dir/baseline_v5_seed42_best.pt"
mkdir -p "$model_dir"
if [[ -s "$target" ]]; then
    echo "Using existing checkpoint: $target"
    if [[ -f "$target.source" ]]; then cat "$target.source"; fi
    exit 0
fi
if command -v camber >/dev/null 2>&1; then
    camber_bin="$(command -v camber)"
elif [[ -x "$HOME/.camber/bin/camber" ]]; then
    camber_bin="$HOME/.camber/bin/camber"
elif [[ -x "$project_dir/.tools/camber" ]]; then
    camber_bin="$project_dir/.tools/camber"
else
    # Same official release endpoint as install-v2.sh; install project-locally,
    # without changing shell startup files or requiring sudo.
    case "$(uname -m)" in
        x86_64) arch=x86_64 ;;
        aarch64|arm64) arch=arm64 ;;
        *) echo "Unsupported architecture" >&2; exit 1 ;;
    esac
    version="$(curl -fsSL https://cli.cambercloud.com/latest.txt)"
    [[ "$version" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.-]+)?$ ]] || { echo "Invalid CLI release version" >&2; exit 1; }
    install_tmp="$(mktemp -d)"
    curl -fsSL "https://cli.cambercloud.com/releases/$version/cambercli_${version}_Linux_${arch}.tar.gz" -o "$install_tmp/camber.tar.gz"
    tar -xzf "$install_tmp/camber.tar.gz" -C "$install_tmp" camber
    mkdir -p "$project_dir/.tools"
    install -m 755 "$install_tmp/camber" "$project_dir/.tools/camber"
    camber_bin="$project_dir/.tools/camber"
    echo "Installed Camber CLI: $camber_bin"
fi
"$camber_bin" version
if [[ -z "${CAMBER_API_KEY:-}" ]]; then
    echo "CAMBER_API_KEY is unset; trying existing Camber login credentials." >&2
fi
base="stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class"
for suffix in \
    stage12_v5_clean/baseline_v5_seed42_best.pt \
    stage12/baseline_v5_seed42_best.pt \
    baseline_v5_seed42_best.pt; do
    source_path="$base/$suffix"
    attempt_dir="$(mktemp -d "$model_dir/.fetch-XXXXXX")"
    echo "Trying $source_path"
    if "$camber_bin" stash cp "$source_path" "$attempt_dir/model.pt" && [[ -s "$attempt_dir/model.pt" ]]; then
        mv -- "$attempt_dir/model.pt" "$target"
        printf '%s\n' "$source_path" > "$target.source"
        sha256sum "$target"
        echo "Saved checkpoint: $target"
        exit 0
    fi
done
echo "Checkpoint unavailable. Set CAMBER_API_KEY or run $camber_bin login, then retry." >&2
exit 1
