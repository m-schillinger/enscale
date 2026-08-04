import sys
import tempfile
from pathlib import Path
import pytest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "enscale"))

from load_config import load_config


def test_load_config_supports_inheritance(tmp_path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"

    base_config = {
        "general": {"save_name": "base-run", "seed": 123},
        "data": {
            "validation_source": "folder",
            "validation_mode": "valid",
            "test_mode": "test",
            "preprocessing": {"norm_method_input": "none"},
        },
    }
    child_config = {
        "inherits_from": str(base_path),
        "general": {"save_name": "child-run"},
        "data": {"validation_mode": "test_interpolation"},
    }

    base_path.write_text(yaml.safe_dump(base_config))
    child_path.write_text(yaml.safe_dump(child_config))

    cfg = load_config(str(child_path))

    assert cfg.general.save_name == "child-run"
    assert cfg.general.seed == 123
    assert cfg.data.validation_source == "folder"
    assert cfg.data.validation_mode == "test_interpolation"
    assert cfg.data.test_mode == "test"
    assert cfg.data.preprocessing.norm_method_input == "none"


def test_load_config_accepts_supported_ensemble_encoding_scheme(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_doc = {
        "data": {
            "ensemble_encoding": {
                "enabled": True,
                "scheme": "rcm",
            }
        }
    }
    cfg_path.write_text(yaml.safe_dump(cfg_doc))

    cfg = load_config(str(cfg_path))
    assert cfg.data.ensemble_encoding["enabled"] is True
    assert cfg.data.ensemble_encoding["scheme"] == "rcm"


def test_load_config_rejects_legacy_ignore_one_hot_keys(tmp_path):
    cfg_path = tmp_path / "cfg_legacy.yaml"
    cfg_doc = {
        "data": {
            "ignore_one_hot_gcm": True,
        }
    }
    cfg_path.write_text(yaml.safe_dump(cfg_doc))

    with pytest.raises(TypeError):
        load_config(str(cfg_path))
