import re
import copy
from datetime import datetime
import random
from utils import load_json
from typing import List, Dict, Tuple

def unify_dicts(*dicts):
    unified = {}

    for d in dicts:
        for key, value in d.items():
            if key in unified:
                assert unified[key] == value, (
                    f"Conflict for key '{key}': "
                    f"{unified[key]} != {value}"
                )
            else:
                unified[key] = value

    return unified


def remove_data(indexs: List[str], dataset: Dict) -> Dict:
    # Work on a deep copy to avoid mutating the original dataset
    clean_dataset = copy.deepcopy(dataset)

    for index in indexs:
        try:
            category, entity, property_name, _ = index.split("|")
        except ValueError:
            raise ValueError(f"Invalid index format: {index}")

        # Check existence
        if (
            category in clean_dataset
            and entity in clean_dataset[category]
            and property_name in clean_dataset[category][entity]
        ):
            # Remove the property
            del clean_dataset[category][entity][property_name]

            # If entity has no properties left except optional "images"
            remaining_keys = [
                k for k in clean_dataset[category][entity].keys()
                if k != "images"
            ]

            if not remaining_keys:
                del clean_dataset[category][entity]

            # If category has no entities left
            if not clean_dataset[category]:
                del clean_dataset[category]

    return clean_dataset

            
def extract_indexes(dataset: Dict) -> List[str]:
    indexes = []

    for category, entities in dataset.items():
        for entity, entity_data in entities.items():
            images = entity_data.get("images", {})

            for property_name in entity_data:
                if property_name == "images":
                    continue

                for picture in images.keys():
                    indexes.append(
                        f"{category}|{entity}|{property_name}|{picture}"
                    )

    return indexes

def sample_from_dataset(
    dataset: Dict,
    k: int,
    seed: int
) -> List[str]:
    indexes = extract_indexes(dataset)

    if k > len(indexes):
        raise ValueError(
            f"Requested k={k}, but only {len(indexes)} indexes available"
        )

    rng = random.Random(seed)
    return rng.sample(indexes, k)


def get_gold_answer(answers: list) -> str:
    """
    Get the gold answer from the list of answers.

    Parameters
    ----------
    answers: list
        List of answers.

    Returns
    -------
    gold_answer: str
        The gold answer.
    """
    candidates = []

    for answer in answers:
        parts = [p.strip() for p in answer.split("|")]
        name = parts[0]

        start = next(p for p in parts if p.startswith("S:"))[2:].strip()
        end = next(p for p in parts if p.startswith("E:"))[2:].strip()

        # Normalize start date
        start = re.sub("-00", "-01", start)
        start = start.replace("+", "").replace("Z", "")

        if end == "":
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
            candidates.append((start_dt, name))

    assert candidates, "No answer without end date found"

    # Pick newest start date
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

if __name__ == "__main__":

    input_json_path_llava = "models_editing/editing_datasets/llava-1-5-7b/editing_dataset.json"
    input_json_path_qwen = "models_editing/editing_datasets/qwen2-vl-7b/editing_dataset.json"

    input_data_qwen = load_json(input_json_path_qwen)
    input_data_llava = load_json(input_json_path_llava)

    unified_data = unify_dicts(input_data_llava, input_data_qwen)
    
    wd = load_json("data/annotations/wikidata_combined.json")

    clean_dataset = remove_data(
        indexs=unified_data.keys(),
        dataset=wd
    )

    sampled_indexes = sample_from_dataset(
        dataset=clean_dataset,
        k=2,
        seed=42
    )

    print("Sampled Indexes:", sampled_indexes)

    for index in sampled_indexes:
        print(f"\nIndex: {index}")
        category, entity, property_name, picture = index.split("|")
        question = wd[category][entity][property_name]["visual_questions"]["generic"]
        answer = get_gold_answer(wd[category][entity][property_name]["answers"])
        img = wd[category][entity]["images"][picture]
        print(question, answer, img)
