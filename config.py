import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")

def _load() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

_cfg = _load()

DEFAULT_MODEL_PATH: str = _cfg.get("model", {}).get("path", "static/ppo_sailing_marl")