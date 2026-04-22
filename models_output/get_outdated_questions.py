import argparse
import os
from argparse import Namespace
from pathlib import Path


from utils import load_json, dump_json
from analyze_replies import analyze_replies


def save_answer_sheet(results_folder: str, question_path: str):
    """
    Save the answer sheet based on the analysis of replies.

    Args:
        results_folder (str): The path to the folder containing the generated answers.
        question_path (str): The path to the question file.

    Returns:
        None
    """
    # make sure the _analysis.json files are there
    analyze_replies(results_folder, question_path)

    for file_to_analyze in os.listdir(results_folder):
        if file_to_analyze.endswith("_analysis.json"):
            analysis = load_json(os.path.join(results_folder, file_to_analyze))

            answer_sheet = {}
            for img_type in analysis:
                for question_type in analysis[img_type]:
                    for answer_type, answers in analysis[img_type][question_type].items():
                        for ans in answers:
                            category = "_".join(file_to_analyze.split("_")[:-1])
                            subject = ans["subject"]
                            relation = ans["relation"]

                            if subject not in answer_sheet:
                                answer_sheet[subject] = {}

                            # if relation is None:
                            #     answer_sheet[subject][question_type] = answer_type
                            # else:
                            assert relation is not None
                            if relation not in answer_sheet[subject]:
                                answer_sheet[subject][relation] = {}
                            if img_type not in answer_sheet[subject][relation]:
                                answer_sheet[subject][relation][img_type] = {}
                            answer_sheet[subject][relation][img_type][question_type] = answer_type

            dump_json(
                os.path.join(results_folder, f"{category}_answer_sheet.json"),
                answer_sheet,
            )


def save_questions_to_update(results_folder: str, questions_path: str, experiment):
    """
    Save questions to update based on the answer sheets.

    Args:
        results_folder (str): The path to the folder containing the answer sheets.
        questions_path (str): The path to the questions file.

    Returns:
        None
    """
    save_answer_sheet(results_folder, questions_path)

    questions_to_update = {}
    n_questions = 0
    for answer_sheet_file in os.listdir(results_folder):
        if answer_sheet_file.endswith("_answer_sheet.json"):
            questions = load_json(questions_path)
            answer_sheet = load_json(os.path.join(results_folder, answer_sheet_file))
            category = "_".join(answer_sheet_file.split("_")[:-2])

            if category not in questions_to_update:
                questions_to_update[category] = {}

            for subject in answer_sheet:
                for relation in answer_sheet[subject]:
                    for img_type in answer_sheet[subject][relation]:
                        answer_types = [ans for qt, ans in answer_sheet[subject][relation][img_type].items()]
                        assert len(answer_types) == 3, "I am not removing stuff"

                        if "correct" not in answer_types:
                            if all([x == "irrelevant" for x in answer_types]):
                                continue
                            elif all([x == "outdated" or x == "irrelevant" for x in answer_types]):
                                if subject not in questions_to_update[category]:
                                    questions_to_update[category][subject] = {}
                                if relation not in questions_to_update[category][subject]:
                                    questions_to_update[category][subject][relation] = {}
                                if relation not in questions_to_update[category][subject][relation]:
                                    questions_to_update[category][subject][relation][img_type] = {}
                                questions_to_update[category][subject][relation][img_type] = {
                                    "questions": questions[category][subject][relation][f"{experiment}_questions"],
                                    "answers": questions[category][subject][relation]["answers"]
                                }
                                n_questions += 1
                                assert "outdated" in answer_types, "No outdated in question types"
                            else:
                                raise AssertionError(f"Not covering case for {subject} -- {relation}: '{answer_types}'")

    print("Questions to update: ", n_questions)
    dump_json(os.path.join(results_folder, f"qa_to_update.json"), questions_to_update)


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m get_outdated_questions",
        description="Create the answer sheet and extract the outdated questions for a given model.",
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
        default="./data/annotations/wikidata_combined.json",
        help="Path to the file containing Q&A.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    args = get_args()
    experiment = Path(args.results_dir).name
    save_questions_to_update(args.results_dir, args.question_path, experiment)
