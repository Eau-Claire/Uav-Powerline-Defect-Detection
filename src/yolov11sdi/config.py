"""YAML config loading with `inherits:` merging and stable config hashing."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .hashing import stable_json_hash

# Guard constant: the configured taxonomy must equal this, in this order.
CANONICAL_TAXONOMY: tuple[str, ...] = (
    "insulator",
    "insulator_broken",
    "insulator_flashover",
    "bird_nest",
    "foreign_object",
    "broken_strand",
)

# Sections that never participate in hashes (machine-specific behaviour only).
NON_SEMANTIC_KEYS = frozenset({"runtime", "inherits"})


class ConfigError(RuntimeError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (dicts merge, rest replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(name: str, config_dir: Path, _seen: tuple[str, ...] = ()) -> dict:
    if name in _seen:
        raise ConfigError(f"Config inheritance cycle: {' -> '.join(_seen + (name,))}")
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for parent in raw.get("inherits", []) or []:
        merged = deep_merge(merged, _load_raw(parent, config_dir, _seen + (name,)))
    return deep_merge(merged, {k: v for k, v in raw.items() if k != "inherits"})


@dataclass
class Config:
    name: str
    data: dict
    config_dir: Path
    sources: list[str] = field(default_factory=list)

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is ...:
                raise KeyError(f"Missing config key '{dotted}' in config '{self.name}'")
            else:
                return default
        return node

    def has(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    @property
    def names(self) -> list[str]:
        return list(self.get("taxonomy.names"))

    def semantic(self) -> dict:
        return {k: v for k, v in self.data.items() if k not in NON_SEMANTIC_KEYS}

    def config_hash(self) -> str:
        """Hash of every semantic section (whole experiment identity)."""
        return stable_json_hash(self.semantic())

    def section_hash(self, keys: Iterable[str]) -> str:
        """Hash of selected dotted keys. Used for per-stage staleness."""
        payload = {k: self.get(k, None) for k in sorted(set(keys))}
        return stable_json_hash(payload)

    def runtime(self, key: str, default: Any = None) -> Any:
        return self.get(f"runtime.{key}", default)


def default_config_dir() -> Path:
    from .paths import find_project_root

    return find_project_root() / "configs"


def load_config(
    name: str,
    config_dir: str | Path | None = None,
    overrides: dict | None = None,
) -> Config:
    """Load `configs/<name>.yaml` with inheritance; validate the taxonomy."""
    cdir = Path(config_dir) if config_dir else default_config_dir()
    data = _load_raw(name, cdir)
    if overrides:
        data = deep_merge(data, overrides)
    cfg = Config(name=name, data=data, config_dir=cdir)
    validate_taxonomy(cfg.names)
    return cfg


def validate_taxonomy(names: list[str]) -> None:
    if tuple(names) != CANONICAL_TAXONOMY:
        raise ConfigError(
            f"Taxonomy changed. Expected {list(CANONICAL_TAXONOMY)}, got {names}. "
            "The six-class order is part of the dataset contract."
        )
