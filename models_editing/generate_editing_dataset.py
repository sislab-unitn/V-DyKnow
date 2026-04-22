import os
import re
import argparse
from datetime import datetime
from argparse import Namespace
from utils import load_json, save_json, MODELS

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m generate_editing_dataset",
        description="Generate the dataset for editing a specific model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # arguments of the parser
    parser.add_argument(
        "--model-name",
        metavar="STR",
        default=MODELS[0],
        choices=MODELS,
        type=str,
        help="Name of the model for which the editing dataset will be generated.",
    )

    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        default="editing_datasets",
        type=str,
        help="Path to the output directory where the editing dataset will be saved.",
    )

    # parse arguments
    parsed_args = parser.parse_args()
   

    return parsed_args
    

def get_data_outdated(model_name: str) -> list:
    """
    Get the list of (type_category, category, entity, propriety) tuples for which the model output is outdated.

    Returns
    -------
    outdated_data: list
        List of (type_category, category, entity, propriety) tuples for which the model output is outdated.
    """

    data_outdated = []

    for type_category in ["countries", "athletes", "organizations"]:

        path_to_data = os.path.join(
            "models_output",
            "results",
            f"{model_name}",
            "visual",
            f"{type_category}_answer_sheet.json"
        )

        data = load_json(path_to_data)

        if not data:
            raise ValueError(f"No data found at path: {path_to_data}")

        for category, entities in data.items():
            for entity, proprieties in entities.items():
                for propriety, values in proprieties.items():
                    responses = set(values.values())
                    if "outdated" in responses and "correct" not in responses:
                        # The model output is outdated for this (category, entity, propriety)
                        data_outdated.append((type_category, category, entity, propriety))
                    
    return data_outdated

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
    
    
def get_image_path(img_url: str) -> str:
    """
    Get the local image path from the URL.

    Parameters
    ----------
    url: str
        URL of the image.

    Returns
    -------
    image_path: str
        Local path of the image.
    """
    base_dir = os.path.join("data", "imgs", "resized")

    if img_url is not None:
        img_file = img_url.split("/")[-1]
        img_name = ".".join(img_file.split(".")[:-1])
        img_path = os.path.join(base_dir, img_file)
        if img_file.endswith(".svg"):
            img_path = os.path.join(base_dir, f"{img_name}.png")
        try:
            assert os.path.exists(img_path)
        except:
            print(f"Path '{img_path}' does not exist for {img_url}'")

    return img_path

def create_editing_dataset(data_outdated: list) -> dict:
    """
    Create the dataset for editing the model.
    Parameters
    ----------
    data_outdated: list
        List of (type_category, category, entity, propriety) tuples for which the model output is outdated.
    Returns
    -------
    editing_dataset: dict
        The dataset for editing the model.
    """

    path_to_dataset = os.path.join(
        "data",
        "annotations",
        "wikidata_combined.json"
    )

    dataset = load_json(path_to_dataset)

    ret = {}

    for type_category, category, entity, propriety in data_outdated:
        
        # Check if the data exists in the dataset
        if type_category in dataset and category in dataset[type_category] and entity in dataset[type_category][category]:

            entry = {}
            # Get the element from the dataset
            elem = dataset[type_category][category][entity]
            
            for q_type, q in elem["visual_questions"].items():
                entry[q_type] = q
            
            entry["target"] = get_gold_answer(elem["answers"])
            entry["image"] = get_image_path(dataset[type_category][category]["images"][propriety])
            
            elem_id = f"{type_category}|{category}|{entity}|{propriety}"
            ret[elem_id] = entry
            
        else:
            print(f"Data for ({type_category}, {category}, {entity}) not found in dataset. Skipping.")
            continue

    return ret
    

def main():
    args = get_args()

    data_outdated = get_data_outdated(args.model_name)
    editing_dataset = create_editing_dataset(data_outdated)
    if not os.path.exists(f"models_editing/{args.out_dir}/{args.model_name}"):
        os.makedirs(f"models_editing/{args.out_dir}/{args.model_name}")
    save_json(f"models_editing/{args.out_dir}/{args.model_name}/editing_dataset.json", editing_dataset)
    
    print(f"Generated editing dataset with {len(editing_dataset)} entries for model '{args.model_name}'.")

if __name__ == "__main__":
    main()