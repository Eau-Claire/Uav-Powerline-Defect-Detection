"""Resolve extracted dataset roots with explicit taxonomy diagnostics."""
from pathlib import Path

import yaml

from ..config import CANONICAL_TAXONOMY


def find_sixclass_root(root):
    root = Path(root)
    candidates = sorted(root.rglob("data.yaml"))
    problems = []
    for path in candidates:
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = cfg.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda k: int(k))]
            if tuple(names) != CANONICAL_TAXONOMY:
                raise ValueError(f"Unexpected taxonomy: {names!r}")
            if not (path.parent / "images").is_dir():
                raise ValueError("Missing images directory")
            return path.parent
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            problems.append(f"{path}: {exc}")
    details = "\n".join(problems) or "No data.yaml files found"
    raise RuntimeError(f"Cannot resolve dataset under {root}:\n{details}")
