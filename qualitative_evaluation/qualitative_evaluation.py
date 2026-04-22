import os
import re
import csv
import spacy
import argparse
from tqdm import tqdm
from typing import List
from models_editing.utils import load_json
from argparse import Namespace
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from models_output.analyze_replies import remove_additional_bits, find_main_chunk, is_monarch, MONARCH_NUMS
from models_output.utils import ADDITIONAL_BITS

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m qualitative_evaluation.qualitative_evaluation",
        description="Select the answers of a model after editing it in order to analyze them qualitatively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "path_to_results",
        metavar="PATH_TO_RESULTS",
        type=str,
        help="Path to the results directory.",
    )

    parser.add_argument(
        "--path-to-wikidata-questions",
        metavar="PATH_TO_WIKIDATA_QUESTIONS",
        type=str,
        default="data/annotations/wikidata_combined.json",
        help="Path to the file containing the Wikidata questions and gold answers.",
    )

    parser.add_argument(
        "--results-folder",
        metavar="RESULTS_FOLDER",
        type=str,
        default="qualitative_evaluation/to_evaluate/",
        help="Path to the folder where the selected questions will be saved.",
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.path_to_results):
        raise ValueError(f"No results found at {parsed_args.path_to_results}")
    
    if not os.path.exists(parsed_args.path_to_wikidata_questions):
        raise ValueError(f"No Wikidata questions found at {parsed_args.path_to_wikidata_questions}")

    return parsed_args

def matcher(
    gold: str,
    pred: str,
    nlp: Language,
    additional_bits: List[str],
    monarch_nums: List[str] = MONARCH_NUMS
) -> bool:
    """
    Returns True if pred matches the gold target, otherwise False.
    Matching strategy:
      1. Exact match
      2. Simplified match (remove additional bits)
      3. Head-word match (spaCy-based)
    Parameters
    ----------
    gold: str
        The gold target string.
    pred: str
        The predicted string.
    nlp: Language
        The spaCy language model.
    additional_bits: List[str]
        List of additional bits to remove for simplified matching.
    """

    # --- 1. Exact match ---
    pattern = rf"(^|[^\w]){re.escape(gold)}($|[^\w])"
    if re.search(pattern, pred, flags=re.IGNORECASE):
        return True

    # --- 2. Simplified match ---
    simplified = remove_additional_bits(gold, additional_bits)
    if simplified != gold:
        pattern = rf"(^|[^\w]){re.escape(simplified)}($|[^\w])"
        if re.search(pattern, pred, flags=re.IGNORECASE):
            return True

    # --- 3. Head match ---
    if len(simplified.split()) > 1:
        doc = nlp(simplified)
        main_chunk = find_main_chunk(doc)

        if main_chunk is not None:
            if is_monarch(main_chunk, monarch_nums):
                head = main_chunk.text
            else:
                head = main_chunk.root.text
        else:
            head = simplified

        pattern = rf"(^|[^\w]){re.escape(head)}($|[^\w])"
        if re.search(pattern, pred, flags=re.IGNORECASE):
            return True

    return False

def get_list_answers(answers: List[str]) -> List[str]:
    """
    Extract the list of answers from the dataset format.
    Parameters
    ----------
    answers: List[str]
        List of answers in the format: ["answer1 |S: +year-month-dayTHH:MM:SS |E: +year-month-dayTHH:MM:SS", ...]
    Returns
    -------
    List[str]
        List of answers in the format: ["answer1", ...]
    """

    return [a.split(" |S:", 1)[0].strip() for a in answers]



def analyze_results(results: List[dict], wikidata_questions: str) -> None:
    """
    Analyze the results of the edited model and categoriize in correct, outdated, and irrelevant. 
    The results will be saved in list for qualitative analysis.
    Each element in the list will be a dictionary with the following keys:
    - Category
    - Element
    - Attribute
    - Answer type (generic, contextualized, rephrased)
    - Question
    - Target
    - Answer
    - Correct
    - Outdated
    - Irrelevant

    Parameters
    ----------
    results: List[dict]
        List of results to analyze.
    wikidata_questions: List[dict]
        List of Wikidata questions and gold answers.
    
    Returns
    -------
    List[dict]
        List of analyzed results with the above-mentioned keys.
    """

    analyzed = []

    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Whitespace tokenization

    type_questions = results[next(iter(results))]["answers"].keys()

    
    for key, res in tqdm(results.items(), desc="Preprocessing results"):
        category, element, attribute, _ = key.split("|")

        target = res["targets"]

        for answer_type in type_questions:
            question = res["questions"][answer_type]
            answer = res["answers"][answer_type]

            is_correct = 0
            is_outdated = 0
            is_irrelevant = 0

            # Check if the answer is correct first
            if matcher(
                gold=target,
                pred=answer,
                nlp=nlp,
                additional_bits=ADDITIONAL_BITS.get(category, []),
                monarch_nums=MONARCH_NUMS,
            ):
                is_correct = 1
            
            # if it is not correct, check if it is outdated
            if is_correct == 0:
                if(
                    category not in wikidata_questions 
                    or element not in wikidata_questions[category] 
                    or attribute not in wikidata_questions[category][element]
                ):
                    raise ValueError(f"Missing question for {key} in Wikidata questions file.")

                gold_answers = get_list_answers(wikidata_questions[category][element][attribute]["answers"])

                init_len = len(gold_answers)
                gold_answers = [g for g in gold_answers if g != target]
                if len(gold_answers) == init_len:
                    raise ValueError(f"Target answer '{target}' not found in gold answers for {key} in Wikidata questions file.")

                for gold in gold_answers:
                    if matcher(
                        gold=gold,
                        pred=answer,
                        nlp=nlp,
                        additional_bits=ADDITIONAL_BITS.get(category, []),
                        monarch_nums=MONARCH_NUMS,
                    ):
                        is_outdated = 1
                        break
            
                if is_outdated == 0:
                    is_irrelevant = 1
            
            analyzed.append(
                {
                    "Category": category,
                    "Element": element,
                    "Attribute": attribute,
                    "Answer type": answer_type,
                    "Question": question,
                    "Target": target,
                    "Answer": answer,
                    "Correct": is_correct,
                    "Outdated": is_outdated,
                    "Irrelevant": is_irrelevant,
                }
            )

    return analyzed

def save_csv(analyzed: List[dict], path: str) -> None:
    """
    Save the analyzed results in a CSV file.
    Parameters
    ----------
    analyzed: List[dict]
        List of analyzed results with the above-mentioned keys.
    path: str
        Path to the CSV file where the analyzed results will be saved.
    """
    keys = analyzed[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(analyzed)



def main(args: Namespace):

    results = load_json(args.path_to_results)
    
    wikidata_questions = load_json(args.path_to_wikidata_questions)

    results = analyze_results(results, wikidata_questions)

    path_parts = args.path_to_results.split(os.sep)

    edit_model = path_parts[-4]
    model = path_parts[-3]

    filename = os.path.basename(args.path_to_results)
    info = filename.replace("results_", "").replace(".json", "")

    folder_path = os.path.join(args.results_folder, edit_model, model)
    os.makedirs(folder_path, exist_ok=True)

    save_csv(results, os.path.join(folder_path, f"analyzed_results_{info}.csv"))


if __name__ == "__main__":
    args = get_args()
    main(args)
