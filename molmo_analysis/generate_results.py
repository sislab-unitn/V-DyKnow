from email.mime import text
import os
from pydoc import text
import argparse
from urllib import response
from utils import load_json, save_json
from argparse import Namespace
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python molmo_analysis/generate_results.py",
        description="Generate results for the molmo analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--path-analysis-responses",
        metavar="PATH_ANALYSIS_RESPONSES",
        type=str,
        default="molmo_analysis/data/analysis/visual/passages_with_responses.json",
        help="Path to the analysis responses file",
    )
    parser.add_argument(
        "--results-folder",
        metavar="RESULTS_FOLDER",
        type=str,
        default="molmo_analysis/results/visual/",
        help="Path to the folder where the generated results will be saved.",
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.path_analysis_responses):
        raise ValueError(f"No analysis responses found at {parsed_args.path_analysis_responses}")

    if not os.path.exists(parsed_args.results_folder):
        os.makedirs(parsed_args.results_folder)
    
    return parsed_args

def refine_answers(answers: Dict[str, int]) -> Dict[str, int]:
    # Clean the answer from dates and times
    ret = {}
    for answer, count in answers.items():
        if "|" in answer:
            ret[answer.split("|")[0].strip()] = count
        else:
            raise ValueError(f"Answer {answer} does not contain a date or time to be removed.")
    return ret

def most_common_answer(answers: Dict[str, int]) -> str:

    refined_answers = refine_answers(answers)

    # Get the most common answers 
    # if there is a tie, return all the answers with the highest count
    if len(refined_answers.keys()) == 0:
        return []
    max_count = max(refined_answers.values())
    most_common_answers = [answer for answer, count in refined_answers.items() if count == max_count]

    return most_common_answers

def frequency_of_answers(answers: Dict[str, int]) -> Dict[str, List[str]]:
    """
    Get the most frequent answer(s) and the non-majority answers from a dictionary of answers and their counts.
    
    Parameters    ----------
    answers: Dict[str, int]
        A dictionary where the keys are the answers and the values are the counts of how many times
        each answer was given.
    Returns
    -------
    Dict[str, List[str]]
        A dictionary with two keys: "most_frequent" and "non_majority". 
        The value of "most_frequent" is a list of the answer(s) that were given the most times, and the value of "non_majority" is a list of the answers that were given less times than the most frequent answer(s).
    """

    ret = {"most_frequent": [], "non_majority": []}

    refined_answers = refine_answers(answers)

    if len(refined_answers.keys()) == 0:
        return ret
    
    max_count = max(refined_answers.values())

    ret["most_frequent"] = [answer for answer, count in refined_answers.items() if count == max_count]

    ret["non_majority"] = [answer for answer, count in refined_answers.items() if count != max_count]

    return ret


def main(args: Namespace):
    responses = load_json(args.path_analysis_responses)

    labels = ["correct", "outdated", "irrelevant"]

    results = {label: {} for label in labels}

    for catergory, entities in responses.items():
        for entity, properties in entities.items():
            for property, data in properties.items():
                dolma_answers = frequency_of_answers(data["dolma"])
                model_answers = frequency_of_answers(data.get("model_responses", {}))

                mf_model = model_answers["most_frequent"]
                mf_dolma = dolma_answers["most_frequent"]
                nm_model = model_answers["non_majority"]
                nm_dolma = dolma_answers["non_majority"]
                correct_answer = data["correct_answer"]
                response_validity = data["model_response_validity"]

                if correct_answer in mf_dolma and correct_answer in mf_model:
                    results[response_validity].setdefault("model_majority_dolma_majority", []).append((catergory, entity, property))
                elif correct_answer in mf_dolma and correct_answer in nm_model:
                    results[response_validity].setdefault("model_non_majority_dolma_majority", []).append((catergory, entity, property))
                elif correct_answer in mf_dolma and correct_answer not in nm_model and correct_answer not in mf_model:
                    results[response_validity].setdefault("model_absence_dolma_majority", []).append((catergory, entity, property))
                elif correct_answer in nm_dolma and correct_answer in mf_model:
                    results[response_validity].setdefault("model_majority_dolma_non_majority", []).append((catergory, entity, property))
                elif correct_answer in nm_dolma and correct_answer in nm_model:
                    results[response_validity].setdefault("model_non_majority_dolma_non_majority", []).append((catergory, entity, property))
                elif correct_answer in nm_dolma and correct_answer not in nm_model and correct_answer not in mf_model:
                    results[response_validity].setdefault("model_absence_dolma_non_majority", []).append((catergory, entity, property))
                elif correct_answer not in mf_dolma and correct_answer in mf_model:
                    results[response_validity].setdefault("model_majority_dolma_absence", []).append((catergory, entity, property))
                elif correct_answer not in mf_dolma and correct_answer in nm_model:
                    results[response_validity].setdefault("model_non_majority_dolma_absence", []).append((catergory, entity, property))
                elif correct_answer not in mf_dolma and correct_answer not in nm_dolma and correct_answer not in nm_model and correct_answer not in mf_model:
                    results[response_validity].setdefault("model_absence_dolma_absence", []).append((catergory, entity, property))
                else:
                    raise ValueError(f"Unexpected combination of answers for {catergory} - {entity} - {property}: correct answer {correct_answer}, model majority {mf_model}, model non-majority {nm_model}, dolma majority {mf_dolma}, dolma non-majority {nm_dolma}")
    
    for label, result_categories in results.items():
        for result_category in result_categories.keys():
            print(f"{label} - {result_category}: {len(result_categories[result_category])} samples")
    
    save_json(results, os.path.join(args.results_folder, "results.json"))

if __name__ == "__main__":
    args = get_args()
    main(args)