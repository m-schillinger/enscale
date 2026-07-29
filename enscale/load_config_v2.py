import os
from copy import deepcopy
from typing import Dict, Any

import yaml

from config_v2 import ConfigV2, DataPreprocessingV2


class ConfigInheritanceError(Exception):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _load_yaml_with_inheritance(path: str, visited=None) -> Dict[str, Any]:
    if visited is None:
        visited = set()

    abs_path = os.path.abspath(path)
    if abs_path in visited:
        raise ConfigInheritanceError(f"Cycle detected in config inheritance at: {abs_path}")
    visited.add(abs_path)

    with open(abs_path, "r") as f:
        doc = yaml.safe_load(f) or {}

    parent = doc.pop("inherits_from", None)
    if not parent:
        return doc

    if not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(abs_path), parent)

    parent_doc = _load_yaml_with_inheritance(parent, visited=visited)
    return _deep_merge(parent_doc, doc)


def _load_v2_doc(path: str) -> Dict[str, Any]:
    return _load_yaml_with_inheritance(path)


def load_config_v2(path: str) -> ConfigV2:
    d = _load_v2_doc(path)

    cfg = ConfigV2()

    for section_name, section_values in d.items():
        if not hasattr(cfg, section_name):
            raise TypeError(f"Unknown v2 config section: {section_name}")
        section = getattr(cfg, section_name)
        section_cls = section.__class__
        setattr(cfg, section_name, section_cls(**(section_values or {})))

    if "preprocessing" in d:
        legacy = d.get("preprocessing") or {}
        if isinstance(legacy, DataPreprocessingV2):
            cfg.data.preprocessing = legacy
        else:
            cfg.data.preprocessing = DataPreprocessingV2(**legacy)

    cfg.preprocessing = cfg.data.preprocessing

    return cfg


def load_train_and_inference_config_v2(train_path: str, inference_path: str) -> ConfigV2:
    train_doc = _load_v2_doc(train_path)
    inference_doc = _load_v2_doc(inference_path)

    merged = _deep_merge(train_doc, inference_doc)

    cfg = ConfigV2()
    for section_name, section_values in merged.items():
        if not hasattr(cfg, section_name):
            raise TypeError(f"Unknown v2 config section: {section_name}")
        section = getattr(cfg, section_name)
        section_cls = section.__class__
        setattr(cfg, section_name, section_cls(**(section_values or {})))

    cfg.preprocessing = cfg.data.preprocessing
    return cfg
