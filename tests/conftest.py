import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolov11sdi.config import load_config  # noqa: E402
from yolov11sdi.environment import RuntimeEnv  # noqa: E402
from yolov11sdi.paths import ProjectPaths  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("yolo11n_ablation", ROOT / "configs")


@pytest.fixture()
def tmp_paths(tmp_path):
    return ProjectPaths(project_root=ROOT, home=tmp_path, data_root=tmp_path / "data", env=RuntimeEnv.LOCAL).ensure()


def write_img(path: Path, color=(120, 120, 120), size=(64, 48)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path
