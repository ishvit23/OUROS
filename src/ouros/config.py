"""Configuration loading for Ouros."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_CONFIG = PROJECT_ROOT / "config" / "system.yaml"
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "config" / "models.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""

    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_system_config(path: str | Path = DEFAULT_SYSTEM_CONFIG) -> dict[str, Any]:
    """Load system configuration."""

    return load_yaml(path)


def load_models_config(path: str | Path = DEFAULT_MODELS_CONFIG) -> dict[str, str]:
    """Load per-agent model strings."""

    data = load_yaml(path)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models config must contain a 'models' mapping")

    return {str(key): str(value) for key, value in models.items()}
