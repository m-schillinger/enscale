import yaml
from enscale.archive.config import Config, DataPreprocessing

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    cfg = Config()

    for section_name, section_values in d.items():
        section = getattr(cfg, section_name)
        section_cls = section.__class__
        setattr(cfg, section_name, section_cls(**section_values))

    # Only use legacy top-level preprocessing if user actually provided it in YAML
    if "preprocessing" in d:
        legacy = d.get("preprocessing") or {}
        if isinstance(legacy, DataPreprocessing):
            cfg.data.preprocessing = legacy
        else:
            cfg.data.preprocessing = DataPreprocessing(**legacy)

    # Keep alias in sync
    cfg.preprocessing = cfg.data.preprocessing
    return cfg
