#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_ROOT/.venv"
MONGO_DATA="${TMPDIR:-/tmp}/fiftyone-mongodb"
MONGO_LOG="$PROJECT_ROOT/.fiftyone/mongod.log"
MONGO_URI="mongodb://127.0.0.1:27017"

cd "$PROJECT_ROOT"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Chưa có .venv. Đang tạo môi trường Python..."
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"

if ! python - <<'PY'
try:
    import fiftyone
    import dotenv
except ImportError:
    raise SystemExit(1)
PY
then
    echo "Đang cài dependencies..."
    python -m pip install -r requirements.txt
fi

if ! command -v mongod >/dev/null 2>&1; then
    echo "Không tìm thấy mongod. Hãy cài MongoDB trước rồi chạy lại script."
    exit 1
fi

mkdir -p "$MONGO_DATA" "$(dirname "$MONGO_LOG")"

mongo_started_by_script=0
if ! (echo > /dev/tcp/127.0.0.1/27017) >/dev/null 2>&1; then
    echo "Đang khởi động MongoDB..."
    if ! mongod --dbpath "$MONGO_DATA" --bind_ip 127.0.0.1 --port 27017 \
        --logpath "$MONGO_LOG" --fork >/dev/null; then
        echo "Không thể khởi động MongoDB. Chi tiết log: $MONGO_LOG"
        exit 1
    fi
    mongo_started_by_script=1
    trap 'if [[ "$mongo_started_by_script" == "1" ]]; then mongod --shutdown --dbpath "$MONGO_DATA" >/dev/null 2>&1 || true; fi' EXIT
fi

export FIFTYONE_DATABASE_URI="$MONGO_URI"
export RAW_ROOT="$PROJECT_ROOT/Power-equipment-image-dataset/transmission line object detection"
export MPLCONFIGDIR="$PROJECT_ROOT/.fiftyone/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Mở FiftyOne tại http://127.0.0.1:5151"
python view_fiftyone.py "$@"
