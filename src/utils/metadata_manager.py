import json
import os


def load_metadata(path):

    if not os.path.exists(path):
        return {"files": {}}

    with open(path, "r") as f:
        return json.load(f)


def save_metadata(path, metadata):

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)