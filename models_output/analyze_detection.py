import re
import json
import csv
import spacy
import argparse
from argparse import Namespace
from pathlib import Path
from pathlib import Path
from spacy.tokens.doc import Doc
# Import the MAP_MODELS dictionary to get model names
from generate_answers import MAP_MODELS

NLP = spacy.load("en_core_web_trf")

def load_model_answers(name_model: str) -> dict:
    """
    Function to load the model answers from a given file.

    Args:
        name_model (str): The name of the model whose answers are to be loaded.
    
    Returns:
        dict: A dictionary containing the model answers for all the categories.

    Raises:
        FileNotFoundError: If the model folder or any required JSON file is not found.
    """
    base_path = Path("models_output") / "results" / name_model / "detection"

    # Check if model folder exists
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {base_path}")

    required_files = {
        "athletes": "athletes_answers.json",
        "countries": "countries_answers.json",
        "organizations": "organizations_answers.json",
    }

    model_answers = {}

    # Check and load each required JSON file
    for key, filename in required_files.items():
        file_path = base_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            model_answers[key] = json.load(f)

    return model_answers

def find_main_chunk(doc: Doc):
    '''
    Function to find the main noun chunk in a spacy Doc object.
    Args:
        doc (Doc): A spacy Doc object.
    Returns:
        The main noun chunk in the Doc object.
    '''
    ancestor = None
    for chunk in doc.noun_chunks:
        if ancestor is None:
            ancestor = chunk
        elif chunk.root.is_ancestor(ancestor.root):
            ancestor = chunk.root
    return ancestor

def matcher(answer: str, pred: str):
    '''
    Function to match answer and prediction strings.
    It first tries to find an exact match (case insensitive).
    If no exact match is found and the answer consists of multiple words, it extracts the main noun chunk from the answer and tries to match that against the prediction.

    Args:
        answer (str): The ground truth answer string.
        pred (str): The predicted answer string.
    Returns:
        bool: True if a match is found, False otherwise.
    '''
    match = False
    res = re.search(rf"(^|[^\w]{{1}}){answer}($|[^\w]{{1}})", pred, flags=re.IGNORECASE)
    if bool(res):
        match = True
    elif len(answer.split()) > 1:
        doc = NLP(answer)
        main_chunk = find_main_chunk(doc)

        if main_chunk is not None:
            head_chunk = main_chunk.root.text
        else:
            head_chunk = answer

        res = re.search(
            rf"(^|[^\w]{{1}}){head_chunk}($|[^\w]{{1}})",
            pred,
            flags=re.IGNORECASE,
        )
        if bool(res):
            match = True
        
    return match


def analyze_detection(answers: dict, image_type: str) -> dict:
    '''
    Function to analyze detection by checking if the prediction matches the answer.

    Args:
        answers: input data containing the answers of the model for a given category.
        image_type: the type of the image to analyze.

        Possible values are:
        ['flag', 'coat_of_arms' for countries,
        'picture' for athletes,
        'logo' for organizations.]

    Returns:
        dict: A dictionary with the total and correct counts for each question type.
    '''

    results = {type: {"total": 0, "correct": 0} for type in ["generic", "contextualized", "rephrased", "upper_bound"]}


    for entity, props in answers.items():
        if image_type not in props["no_rel"]:
            print(f"Image type {image_type} is not present for this category.")
            return False
        
        questions = props["no_rel"][image_type]["answers"]

        corr_upper_bound = False
        for q_type, pred in questions.items():
            if q_type != "upper_bound":
                results[q_type]["total"] += 1
                if matcher(entity, pred):
                    results[q_type]["correct"] += 1
                    corr_upper_bound = True

        # Upper bound analysis
        if corr_upper_bound:
            results["upper_bound"]["correct"] += 1

        corr_upper_bound = False
        results["upper_bound"]["total"] += 1
                
    return results

def save_results(model_name: str, results: dict, name_file: str):
    """
    Saves accuracy results to a CSV file for a given model and analysis type.
    The results are appended to the file and saved in the "analysis/detection" directory.
    Args:
        model_name (str): The name of the model.
        results (dict): A dictionary containing the results of the analysis.
        name_file (str): The name of the file to save the results to (without extension).
    
    Raises:
        ValueError: If the results dictionary is empty.
    """

    base_path = Path("models_output") / "analysis" / "detection"
    base_path.mkdir(parents=True, exist_ok=True)

    file_path = base_path / f"{name_file}.csv"

    row = {"model_name": model_name}

    if not results:
        raise ValueError("No results to save. The results dictionary is empty.")

    for category, values in results.items():
        total = values.get("total", 0)
        correct = values.get("correct", 0)
        row[f"{category}_accuracy"] = correct / total if total > 0 else 0.0

    fieldnames = list(row.keys())
    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def merge_detection_results(*results_list: dict) -> dict:
    """
    Merge multiple detection result dictionaries by summing totals and correct counts.

    Args:
        results_list: any number of detection result dicts returned by analyze_detection

    Returns:
        dict: merged results
    """

    merged = {
        q_type: {"total": 0, "correct": 0}
        for q_type in ["generic", "contextualized", "rephrased", "upper_bound"]
    }

    for results in results_list:
        for q_type, values in results.items():
            merged[q_type]["total"] += values.get("total", 0)
            merged[q_type]["correct"] += values.get("correct", 0)

    return merged

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m analyze_detection",
        description="Analyze detection results for a given model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        type=str,
        choices=[model_name for model_name in MAP_MODELS.keys()],
        help="Model used for generation.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args

def main():
    args = get_args()

    model_name = args.model_name
    model_answers = load_model_answers(model_name)

    countries_flag = analyze_detection(model_answers["countries"], "flag")
    countries_coat = analyze_detection(model_answers["countries"], "coat_of_arms")
    athletes_picture = analyze_detection(model_answers["athletes"], "picture")
    organizations_logo = analyze_detection(model_answers["organizations"], "logo")

    save_results(model_name, countries_flag, "countries_flag")
    save_results(model_name, countries_coat, "countries_coat_of_arms")
    save_results(model_name, athletes_picture, "athletes_picture")
    save_results(model_name, organizations_logo, "organizations_logo")

    total_results = merge_detection_results(
        countries_flag,
        countries_coat,
        athletes_picture,
        organizations_logo,
    )

    save_results(
        model_name,
        total_results,
        "total_detection_all_categories"
    )





if __name__ == "__main__":

    main()