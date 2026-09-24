import json
from pathlib import Path

from dproblem.common.hashing import stable_json_hash


def load_model_config(project_root):
    path = Path(project_root) / "config" / "model.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def energy_config_hash(model_config):
    return stable_json_hash(model_config["energy_model"])
