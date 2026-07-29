import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "enscale"))

from load_config_v2 import load_config_v2, load_train_and_inference_config_v2


def test_load_config_v2_supports_inheritance(tmp_path):
    base = tmp_path / "train_base.yaml"
    child = tmp_path / "train_child.yaml"

    base_doc = {
        "general": {"save_name": "base"},
        "data": {
            "type": "pattern",
            "variables": ["tas"],
            "variables_lr": ["tas"],
            "data_dir": "/tmp/data",
            "pattern": {
                "lr_pattern": "{root}/{folder}/{var}_lr{file_suffix}.nc",
                "hr_pattern": "{root}/{folder}/{var}_hr{file_suffix}.nc",
            },
            "train": {"folder": "train", "file_suffix": ""},
            "preprocessing": {
                "norm_method_input": "none",
                "norm_method_output": "none",
            },
        },
    }

    child_doc = {
        "inherits_from": str(base),
        "general": {"save_name": "child"},
        "inference": {
            "checkpoint_source": "pretrained",
            "pretrained_checkpoints": {"dummy": "/tmp/model.pt"},
            "data_modes": {
                "test": {
                    "submodes": {
                        "historical": {"folder": "test/historical", "file_suffix": ""},
                        "future": {"folder": "test/future", "file_suffix": ""}
                    }
                }
            }
        },
    }

    base.write_text(yaml.safe_dump(base_doc))
    child.write_text(yaml.safe_dump(child_doc))

    cfg = load_config_v2(str(child))
    assert cfg.general.save_name == "child"
    assert cfg.inference.data_modes.test.submodes["future"].folder == "test/future"
    assert cfg.inference.data_modes.test.submodes["historical"].folder == "test/historical"


def test_load_train_and_inference_config_v2_merges_docs(tmp_path):
    train = tmp_path / "train.yaml"
    infer = tmp_path / "inference.yaml"

    train_doc = {
        "general": {"save_name": "train"},
        "data": {
            "type": "pattern",
            "variables": ["tas"],
            "variables_lr": ["tas"],
            "data_dir": "/tmp/data",
            "pattern": {
                "lr_pattern": "{root}/{folder}/{var}_lr{file_suffix}.nc",
                "hr_pattern": "{root}/{folder}/{var}_hr{file_suffix}.nc",
            },
            "train": {"folder": "train", "file_suffix": ""},
            "preprocessing": {
                "norm_method_input": "none",
                "norm_method_output": "none",
            },
        },
        "inference": {
            "checkpoint_source": "pretrained",
            "pretrained_checkpoints": {"stage1": "/tmp/model.pt"},
            "split": "inference",
            "submode": "custom_a",
            "data_modes": {
                "inference": {
                    "submodes": {
                        "custom_a": {"folder": "infer/custom_a", "file_suffix": ""}
                    }
                }
            },
        },
    }

    infer_doc = {
        "inherits_from": str(train),
        "inference": {
            "checkpoint_source": "train_output",
            "train_run_dir": "/tmp/run",
            "split": "inference",
            "submode": "custom_a",
        },
    }

    train.write_text(yaml.safe_dump(train_doc))
    infer.write_text(yaml.safe_dump(infer_doc))

    cfg = load_train_and_inference_config_v2(str(train), str(infer))
    assert cfg.inference.checkpoint_source == "train_output"
    assert cfg.inference.train_run_dir == "/tmp/run"
    assert cfg.inference.submode == "custom_a"
