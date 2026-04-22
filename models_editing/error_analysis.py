import os
import re
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
        prog="python -m models_editing.error_analysis",
        description="Analyze the errors of a model after editing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "path_to_results",
        metavar="PATH_TO_RESULTS",
        type=str,
        help="Path to the results directory.",
    )

    parsed_args = parser.parse_args()

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

def analyze_results(results: List[dict]):
    """
    Analyze model editing results.
    Parameters
    ----------
    results: List[dict]
        List of results from model editing.
    Returns
    -------
    
    """
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Whitespace tokenization

    total = 0

    correct = {key: 0 for key in next(iter(results.values()))["answers"].keys()}
    for key, res in tqdm(results.items(), desc="Analyzing results"):
        gold = res["targets"]
        pred = res["answers"]
        category, _ , _, _ = key.split("|")
        for answer_type in pred.keys():
            if matcher(
                gold=gold,
                pred=pred[answer_type],
                nlp=nlp,
                additional_bits=ADDITIONAL_BITS.get(category, []),
                monarch_nums=MONARCH_NUMS,
            ):
                correct[answer_type] += 1
        total += 1
    
    accuracy = {key: correct[key] / total if total > 0 else 0.0 for key in correct.keys()}

    acc_mean_contextualized_rephrased = (accuracy["contextualized"] + accuracy["rephrased"]) / 2
    harm_mean = 2*accuracy["generic"]*acc_mean_contextualized_rephrased / (accuracy["generic"] + acc_mean_contextualized_rephrased) if (accuracy["generic"] + acc_mean_contextualized_rephrased) > 0 else 0.0
    print("Accuracy results:")
    for key in accuracy.keys():
        print(f"{key}: {accuracy[key]:.4f}")
    print(f"Harmonic Mean: {harm_mean:.4f}")

    return harm_mean, accuracy


def main(args: Namespace):

    if not os.path.exists(args.path_to_results):
        raise ValueError(f"No results found at {args.path_to_results}")

    results = load_json(args.path_to_results)
    analyze_results(results)

if __name__ == "__main__":
    args = get_args()
    main(args)
