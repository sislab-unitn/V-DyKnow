import os
import json


MODELS = ["qwen2-vl-7b", "llava-1-5-7b"]

def load_json(path: str) -> dict:
    """
    Load a JSON file.

    Parameters
    ----------
    path: str
        Path to the JSON file.
    Returns
    -------
    data: dict
        Loaded JSON data.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(path: str, data: dict):
    """
    Save data to a JSON file.

    Parameters
    ---------- 
    path: str
        Path to the JSON file.
    data: dict
        Data to be saved.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)