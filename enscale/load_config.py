# load_config.py
import yaml
from config import Config

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    cfg = Config()

    for section_name, section_values in d.items():
        section = getattr(cfg, section_name)
        section_cls = section.__class__
        setattr(cfg, section_name, section_cls(**section_values))

    return cfg
