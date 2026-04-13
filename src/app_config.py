from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


DEFAULT_CONFIG = {
    "paths": {
        "input_dir": str(BASE_DIR / "input"),
        "output_dir": str(BASE_DIR / "output"),
    },
    "servers": {
        "fs": "http://192.168.30.237:8000",
        "s2m2": "http://192.168.30.237:8001",
    },
    "model_selection": "fs",
    "session": {
        "sample_name": "sample_001",
        "left_image": "",
        "right_image": "",
        "intrinsic_text": "754.66 0.0 489.37 0.0 754.66 265.16 0.0 0.0 1.0\n0.063",
        "save_input_copy": True,
    },
    "fs_params": {
        "scale": 1.0,
        "iters": 32,
        "hiera": False,
        "get_pc": True,
        "z_far": 10.0,
        "rm_inv": False,
        "denoise": False,
        "d_nb": 30,
        "d_rad": 0.03,
    },
    "s2m2_params": {
        "model_choice": "X-Large (XL)",
        "get_pc": True,
        "z_far": 10.0,
        "conf_thresh": 0.0,
        "denoise": False,
        "d_nb": 30,
        "d_rad": 0.03,
    },
}


def _merge_dict(defaults: dict, override: dict) -> dict:
    merged = deepcopy(defaults)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_default_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
    return load_config()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return deepcopy(DEFAULT_CONFIG)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        current = yaml.safe_load(file) or {}
    return _merge_dict(DEFAULT_CONFIG, current)


def save_config(config_data: dict) -> None:
    merged = _merge_dict(DEFAULT_CONFIG, config_data or {})
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(merged, file, allow_unicode=True, sort_keys=False)

