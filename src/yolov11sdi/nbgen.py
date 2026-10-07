"""Generate thin .ipynb files (no nbformat dependency).

Used by scripts/build_notebooks.py (local notebooks/) and scripts/export_kaggle.py
(dist/kaggle/). Notebook cells only call package functions.
"""
from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path

LOCAL_BOOTSTRAP = '''\
# Bootstrap: make the project package importable from notebooks/.
import os, sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists() and (p / "configs").is_dir())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLOV11SDI_PROJECT_ROOT", str(ROOT))

import pandas as pd
from IPython.display import display, Image as IPImage
from yolov11sdi import pipeline
from yolov11sdi.config import load_config
from yolov11sdi.paths import ProjectPaths
from yolov11sdi.state import StateStore

FORCE_REBUILD = False   # True = rebuild even if a matching complete result exists
DRY_RUN = False
paths = ProjectPaths.detect().ensure()
store = StateStore(paths.state)
print("env:", paths.env.value, "| home:", paths.home, "| data:", paths.data_root)'''


def _src(text: str) -> list[str]:
    lines = textwrap.dedent(text).strip("\n").splitlines(keepends=True)
    return lines


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(text)}


def notebook(cells: list[dict], kaggle: bool = False) -> dict:
    meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if kaggle:
        meta["kaggle"] = {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}
    # nbformat 4.5 requires a unique id per cell; deterministic ids keep diffs stable.
    cells = [{"id": f"cell-{i:02d}", **c} for i, c in enumerate(cells)]
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def write_notebook(path: Path, nb: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def check_compiles(nb: dict, name: str) -> None:
    """Every code cell must at least compile (shell/magic lines excluded)."""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        py = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))
        compile(py, f"{name}[cell {i}]", "exec")


def bundle_project(project_root: Path) -> str:
    """Base64 zip of src/yolov11sdi + configs + pyproject (the single source of truth)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        files = [project_root / "pyproject.toml"]
        files += sorted((project_root / "configs").glob("*.yaml"))
        files += sorted(p for p in (project_root / "src/yolov11sdi").rglob("*.py"))
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(project_root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")
