import argparse
import json
import os
import re
import copy
from argparse import Namespace
from typing import Dict, List, Optional, Set
from datetime import datetime


import spacy
from spacy.tokenizer import Tokenizer
from spacy.tokens.doc import Doc
from spacy.tokens import Span
from spacy.language import Language
from tqdm import tqdm

from models_output.utils import write_roman, EXCEPTIONS, ADDITIONAL_BITS, load_json, dump_json


MONARCH_NUMS = {write_roman(i) for i in range(1, 100, 1)}


def extract_category(file_to_analyze: str):
    return "_".join(file_to_analyze.split("_")[:-1])


def is_exception(
    answer_name: str,
    category: str,
    subject: str,
    relation: Optional[str],
    exceptions: dict,
) -> bool:
    if category in exceptions:
        if subject in exceptions[category]:
            return answer_name in exceptions[category][subject]

    return False


def extract_answer(
    answers: List[str],
    exceptions: dict,
    category: str,
    subject: str,
    relation: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    to_assign = {}

    no_end_entries = []
    # Sort the list so that we consider only the latest entry for a given candidate
    for answer in sorted(answers):
        # Skip the answer if it is an exception

        split_answer = answer.split("|")
        name, span = split_answer[0], split_answer[1:]
        name = name.strip()  # remove first and last spaces

        if is_exception(name, category, subject, relation, exceptions):
            if "national" not in name:
                print(f"Exception skipping: {name} {category} {subject} {relation}")
            continue

        assert len(span) == 2, f"There have to be start and end for {category} {subject} {relation if relation else ''}: {span}"

        start, end = span
        start, end = re.sub("-00", "-01", start[3:].strip()), re.sub(
            "-00", "-01", end[3:].strip()
        )
        if end == "":
            no_end_entries.append(answer)
            end = None
        if start == "":
            start = None

        # assert start != end, f"Start '{start}' and end '{end}' cannot be the same. Error for {category} {subject} {relation if relation else ''}"

        if name in to_assign and end is not None:
            end_date = datetime.strptime(end, "+%Y-%m-%dT%H:%M:%SZ")
            prev_end = datetime.strptime(to_assign[name]["end"], "+%Y-%m-%dT%H:%M:%SZ")
            if end_date < prev_end:
                end = to_assign[name]["end"]

        to_assign[name] = {"start": start, "end": end}

    # assert (
    #     len(no_end_entries) == 1
    # ), f"There are {len(no_end_entries)} entries with no end for {category} {subject} {relation if relation else ''}: {no_end_entries}"

    return to_assign


def prepare_answers(category: str, original: dict, exceptions: dict) -> dict:
    answers = {}

    for subject, relations in original[category].items():
        if subject not in answers:
            answers[subject] = {}
        for relation, grc_elem in relations.items():
            if relation == "images":
                continue
            try:
                to_assign = extract_answer(
                    grc_elem["answers"], exceptions, category, subject, relation
                )
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            answers[subject][relation] = to_assign

    return answers


def prepare_predictions(generated: dict, category: str, use_rfind: bool) -> dict:
    predictions = {}

    for subject, relations in generated.items():
        if subject not in predictions:
            predictions[subject] = {}

        for relation, by_img in relations.items():
            for img_type, gen_elem in by_img.items():
                for q, answer in gen_elem["answers"].items():
                    if relation not in predictions[subject]:
                        predictions[subject][relation] = {}
                    if img_type not in predictions[subject][relation]:
                        predictions[subject][relation][img_type] = {}

                    if use_rfind:
                        idx = answer.rfind(gen_elem["questions"][q])
                    else:
                        idx = answer.find(gen_elem["questions"][q])

                    assert idx == -1, "Matched question in the answer"

                    predictions[subject][relation][img_type][q] = answer

    return predictions


def find_main_chunk(doc: Doc):
    ancestor = None
    for chunk in doc.noun_chunks:
        if ancestor is None:
            ancestor = chunk
        elif chunk.root.is_ancestor(ancestor.root):
            ancestor = chunk.root
    return ancestor


def is_monarch(span: Span, monarch_nums: Set[str]):
    for name_chunk in span.text.split():
        if name_chunk in monarch_nums:
            return True
    return False


def remove_additional_bits(string: str, additional_bits: List[str]):
    for bit in additional_bits:
        string = re.sub(bit, "", string)
    return " ".join(string.split())  # remove additional whitespaces



def check_match_position(
    match_span_idx: int,
    start_idx: int,
    to_append: dict,
    match_type: str,
    matches: int,
    is_up_to_date: bool,
    answer: str,
):
    # If the current match occurs earlier than the other matches, discard
    # the previous matches (i.e. reset everything)
    if start_idx < match_span_idx:
        match_span_idx = start_idx
        to_append.update(
            {
                "matched_answers": [],
                "match_type": [],
            }
        )
        matches = 0
        is_up_to_date = False

    match = False
    matched_ans = None
    # If the current match occurs at the same position of the other macthes,
    # add it to the pool of correct matches
    if start_idx == match_span_idx:
        match = True
        matched_ans = answer
        to_append["match_type"].append(match_type)

    return match, matches, match_span_idx, is_up_to_date, matched_ans


def assign_question_to_group_based_on_answer(
    stats: dict,
    img_type: str,
    question_type: str,
    pred: str,
    ans: Dict[str, Dict[str, str]],
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: List[str],
    subject: str,
    relation: Optional[str] = None,
):

    if img_type not in stats:
        stats[img_type] = {}
    if question_type not in stats[img_type]:
        stats[img_type][question_type] = {
            "correct": [],
            "outdated": [],
            "irrelevant": [],
        }

    to_append = {
        "prediction": pred,
        "matched_answers": [],
        "match_type": [],
        "correct_answer": None,
        "subject": subject,
        "relation": relation,
    }

    # Consider only the most recent answer with no end date (remove others)
    # if sum([1 for answer, answ_prop in ans.items() if answ_prop["end"] is None]) > 1:
    #     no_end = {answer: answ_prop for answer, answ_prop in ans.items() if answ_prop["end"] is None}
    #     with_end = {answer: answ_prop for answer, answ_prop in ans.items() if answ_prop["end"] is not None}
    #     no_end_answ, no_end_answ_prop = sorted(no_end.items(), key=lambda x: (x[1]["start"]))[-1]
    #
    #     ans = {}
    #     ans.update(copy.deepcopy(with_end))
    #     ans.update(copy.deepcopy({no_end_answ: no_end_answ_prop}))

    matches = 0
    is_up_to_date = False
    match_span_idx = float("Inf")
    corr_answ_start = None
    # get correct answer
    for answer, ans_prop in ans.items():
        if ans_prop["end"] is None:
            # if there are multiple answers with no end date, keep the one with the most recent start date as correct
            if corr_answ_start is None or ans_prop["start"] > corr_answ_start:
                to_append["correct_answer"] = answer
                corr_answ_start = ans_prop["start"]

    # get matches
    for answer, ans_prop in ans.items():
        match = False

        res = re.search(rf"(^|[^\w]{{1}}){answer}($|[^\w]{{1}})", pred, flags=re.IGNORECASE)
        if bool(res):
            start_idx = res.span()[0]
            # match earliest in pred as answer (if two matches at same position both saved, ambiguous)
            match, matches, match_span_idx, is_up_to_date, matched_ans = check_match_position(
                match_span_idx,
                start_idx,
                to_append,
                "em",
                matches,
                is_up_to_date,
                answer,
            )
        else:
            answer = remove_additional_bits(answer, additional_bits)

            # try to see if you can match the simplified version
            res = re.search(
                rf"(^|[^\w]{{1}}){answer}($|[^\w]{{1}})", pred, flags=re.IGNORECASE
            )
            if bool(res):
                start_idx = res.span()[0]
                match, matches, match_span_idx, is_up_to_date, matched_ans = check_match_position(
                    match_span_idx,
                    start_idx,
                    to_append,
                    "simplified",
                    matches,
                    is_up_to_date,
                    answer,
                )

            # check if we are considering a single token
            elif len(answer.split()) > 1:
                doc = nlp(answer)
                main_chunk = find_main_chunk(doc)

                if main_chunk is not None:
                    if is_monarch(main_chunk, monarch_nums):
                        head_chunk = main_chunk.text
                    else:
                        head_chunk = main_chunk.root.text
                else:
                    head_chunk = answer

                res = re.search(
                    rf"(^|[^\w]{{1}}){head_chunk}($|[^\w]{{1}})",
                    pred,
                    flags=re.IGNORECASE,
                )
                if bool(res):
                    start_idx = res.span()[0]
                    match, matches, match_span_idx, is_up_to_date, matched_ans = check_match_position(
                        match_span_idx,
                        start_idx,
                        to_append,
                        "head",
                        matches,
                        is_up_to_date,
                        head_chunk,
                    )

        if match:
            matches += 1
            to_append["matched_answers"].append((matched_ans, ans_prop["start"]))
            # up to date only if most recent (correct answer)
            if ans_prop["end"] is None and ans_prop["start"] == corr_answ_start:
                is_up_to_date = True

    if matches == 1 and is_up_to_date:
        # if we match only one subject and is the correct one we can remove the question
        stats[img_type][question_type]["correct"].append(to_append)
    elif matches == 1 and not is_up_to_date:
        # if we matched more than one subject, but none of them is up to date
        stats[img_type][question_type]["outdated"].append(to_append)
    elif matches == 0:
        # we did not match anything at all
        stats[img_type][question_type]["irrelevant"].append(to_append)
    elif matches > 1:
        # we matched more than one

        # take most recent
        most_recent_start = None
        most_recent_answ = None
        old_matches = to_append["matched_answers"]
        for answer, start in to_append["matched_answers"]:
            # if there are multiple answers with no end date, keep the one with the most recent start date as correct
            if most_recent_start is None or start > most_recent_start:
                most_recent_answ = answer
                most_recent_start = start
        to_append["matched_answers"] = [(most_recent_answ, most_recent_start)]

        # upper bound:
        if is_up_to_date:
            # at least one match is up to date

            assert most_recent_start == corr_answ_start
            stats[img_type][question_type]["correct"].append(to_append)
        else:
            # all matches are outdated

            stats[img_type][question_type]["outdated"].append(to_append)

    else:
        raise "I forgot to consider some cases"


def compute_stats_for_qa(
    predictions: dict,
    answers: dict,
    category: str,
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: Dict[str, List[str]],
) -> Dict[str, Dict[str, dict]]:
    stats = {}

    # pass empty list if we are not considering athletes to avoid unwanted substitutions
    for (p_subject, p_relations), (a_relations) in tqdm(list(zip(predictions.items(), answers.values())), desc=category):
        for (p_relation, questions_by_img_type), (ans) in zip(p_relations.items(), a_relations.values()):
            for img_type, questions in questions_by_img_type.items():
                for question_type, pred in questions.items():
                    assign_question_to_group_based_on_answer(
                        stats,
                        img_type,
                        question_type,
                        pred,
                        ans,
                        nlp,
                        monarch_nums,
                        additional_bits.get(category, []),
                        p_subject,
                        p_relation,
                    )
    return stats


def save_stats(stats: dict, category: str, results_folder: str, indent: int = 4):
    path = os.path.join(results_folder, f"{category}_analysis.json")
    dump_json(path, stats, indent)


def analyze_model_replies(
    results_folder: str,
    questions_path: str,
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: Dict[str, List[str]],
    exceptions: dict,
    use_rfind: bool,
):
    for file_to_analyze in os.listdir(results_folder):
        if file_to_analyze.endswith("_answers.json"):
            stats_path = os.path.join(
                results_folder,
                "".join(
                    file_to_analyze.split("_answers.json")[:-1] + ["_analysis.json"]
                ),
            )
            if not os.path.isfile(stats_path):
                generated = load_json(os.path.join(results_folder, file_to_analyze))
                original = load_json(questions_path)
                category = extract_category(file_to_analyze)

                answers = prepare_answers(category, original, exceptions)
                predictions = prepare_predictions(generated, category, use_rfind)

                stats = compute_stats_for_qa(
                    predictions, answers, category, nlp, monarch_nums, additional_bits
                )
                save_stats(stats, category, results_folder)
            else:
               print(f"File {stats_path} already exists: SKIPPING")


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m analyze_replies",
        description="Analyze the generated answers of a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "results_dir",
        metavar="DIR_NAME",
        type=str,
        help="Folder containing the generated answers from a model.",
    )
    parser.add_argument(
        "--question-path",
        metavar="FILE_PATH",
        type=str,
        default="../grc_generated.json",
        help="Path to the file containing Q&A.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def analyze_replies(results_folder: str, questions_path: str):
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Whitespace tokenization

    analyze_model_replies(
        results_folder,
        questions_path,
        nlp,
        MONARCH_NUMS,
        ADDITIONAL_BITS,
        EXCEPTIONS,
        use_rfind=True,
    )


if __name__ == "__main__":
    args = get_args()
    analyze_replies(args.results_dir, args.question_path)
