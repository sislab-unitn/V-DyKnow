import argparse
from argparse import Namespace
import os
from pathlib import Path
from typing import Dict, List

from analyze_replies import analyze_replies
from generate_answers import MAP_MODELS


from collections import Counter as count
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, List


from get_outdated_questions import save_questions_to_update
import pandas as pd
from save_dates import save_dates
import plotly.express as px
import plotly.graph_objects as go
import spacy
from IPython.display import display


from utils import dump_json, load_json, get_correct_year
from analyze_replies import EXCEPTIONS, prepare_answers



def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python models_output/analysis.py",
        description="Compute results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        metavar="MODEL_NAME",
        type=str,
        nargs="+",
        default=[model for model in MAP_MODELS],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="./models_output/results",
        help="Destination folder to save the generation results.",
    )

    parsed_args = parser.parse_args()

    return parsed_args

### % correct, outdated, irrelevant by domain
def load_stats(results_folder: str, show_percentage: bool = False, remove: List[str] = []) -> pd.DataFrame:
    """
    Load and analyze statistics from the given results folder.

    Parameters:
    - results_folder (str): The path to the folder containing the results files.
    - show_percentage (bool): Flag indicating whether to show the statistics as percentages.
    - remove (List[str]): List of strings to remove from the statistics.

    Returns:
    None
    """
    tables = {}
    for file_to_analyze in os.listdir(results_folder):
        if file_to_analyze.endswith("_analysis.json"):
            domain = file_to_analyze.split("_")[0]
            print(file_to_analyze)
            stats = load_json(os.path.join(results_folder, file_to_analyze))
            stats_summary = create_stats_summary(stats)
            tables[domain] = show_stats_summary(stats_summary, show_percentage, remove)
    return tables


def create_stats_summary(stats: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, int]]:
    """
    Create a summary of statistics based on the given stats dictionary.

    Args:
        stats (Dict[str, Dict[str, dict]]): A dictionary containing statistics for different question types and answer types.

    Returns:
        Dict[str, Dict[str, int]]: A summary of statistics, where the keys are question types and the values are dictionaries
        containing answer types and their corresponding counts.
    """
    stats_summary = {}

    stats_summary = {
        img_type: {
            question_type: {
                answ_type: 0 for answ_type in stats[img_type][question_type]
            } for question_type in stats[img_type]
        } for img_type in stats
    }

    for img_type in stats:
        for question_type in stats[img_type]:
            for anws_type in stats[img_type][question_type]:
                stats_summary[img_type][question_type][anws_type] = len(
                    stats[img_type][question_type][anws_type]
                )

    return stats_summary


def show_stats_summary(stats_summary: Dict[str, Dict[str, int]], show_percentage: bool, remove: List[str]):
    """
    Display a summary of statistics based on the provided stats_summary dictionary.

    Args:
        stats_summary (Dict[str, Dict[str, int]]): A dictionary containing the statistics summary.
        show_percentage (bool): Flag indicating whether to display the statistics as percentages.
        remove (List[str]): A list of keys to remove from the statistics summary.

    Returns:
        None
    """
    tables = {}
    for img_type, stats_summary_by_img in stats_summary.items():
        print(img_type)
        table_stats = pd.DataFrame.from_dict(stats_summary_by_img)
        total = table_stats.sum(0)

        if show_percentage:
            table_stats = table_stats / total * 100
            table_stats = table_stats.round(2)

        table_stats = table_stats.T

        for r in remove:
            del table_stats[r]

        table_stats['total #'] = total
        display(table_stats)
        tables[img_type] = table_stats
    return tables



### % correct, outupdated, irrelevant
def compute_stats(results_folder: str, show_percentage: bool = False, round_digits: int = 0):
    """
    Compute statistics for answer sheets in the given results folder.

    Parameters:
    results_folder (str): The path to the folder containing the answer sheets.
    show_percentage (bool, optional): Whether to show the statistics as percentages. Default is False.

    Returns:
    None
    """
    stats = {}
    for answer_sheet_file in os.listdir(results_folder):
        if answer_sheet_file.endswith("_answer_sheet.json"):
            answer_sheet = load_json(os.path.join(results_folder, answer_sheet_file))
            category = "_".join(answer_sheet_file.split("_")[:-2])

            if category not in stats:
                stats[category] = {"correct": 0, "outdated": 0, "irrelevant": 0}

            for subject in answer_sheet:
                for relation in answer_sheet[subject]:
                    for img_type in answer_sheet[subject][relation]:
                        # TODO do across image for countries?
                        answer_types = [ans for qt, ans in answer_sheet[subject][relation][img_type].items()]
                        assert "ambiguous" not in answer_types, f"Forgot to remove 'ambiguous' for {subject} -- {relation}: '{answer_types}'"
                        if "correct" in answer_types:
                            stats[category]["correct"] += 1
                        elif "outdated" in answer_types:
                            stats[category]["outdated"] += 1
                        elif "irrelevant" in answer_types:
                            stats[category]["irrelevant"] += 1
                        else:
                            raise AssertionError(
                                f"Not covering case for {subject} -- {relation}: '{answer_types}'"
                            )

    table = pd.DataFrame.from_dict(stats)
    table["Answers"] = 0
    for category in ["countries", "athletes", "organizations"]:
        table["Answers"] += table.get(category, 0)
        if category in table:
            del table[category]

    total = table.sum(0)
    if show_percentage:
        table = table / total * 100
        table = table.round(round_digits)
    table.loc['total #'] = total
    display(table.T)
    return table


### dates

### prompt agreement
def contradiction_analysis(results_folders: List[str]):
    results = {}
    total_questions = 0
    total_contradictions = 0
    for results_folder in results_folders:
        experiment = Path(results_folder).name
        model_name = Path(results_folder).parent.name
        n_questions = 0
        n_contradictions = 0
        for file_name in os.listdir(results_folder):
            if not file_name.endswith("_dates.json"):
                continue

            domain = file_name.split("_dates.json")[0]
            dates = load_json(os.path.join(results_folder, file_name))
            answer_sheet = load_json(os.path.join(results_folder, f"{domain}_answer_sheet.json"))

            for subject in dates:
                for relation in dates[subject]:
                    for img_type in dates[subject][relation]:
                        n_questions += 1
                        possible_dates = set()
                        set_answers = set([answer_type for qt, answer_type in answer_sheet[subject][relation][img_type].items()])
                        assert len(set_answers) > 0
                        if len(set_answers) > 1:
                            n_contradictions += 1
                        elif len(set_answers) == 1 and "outdated" in set_answers:
                            for question_type, date in dates[subject][relation][img_type].items():
                                possible_dates.add(date)
                            if len(possible_dates) <= 1:
                                continue
                            elif len(possible_dates) > 1:
                                n_contradictions += 1
                            else:
                                raise Exception("Something went wrong")

        if model_name not in results:
            results[model_name] = {}
        # results[model_name][experiment] = {
        #     "% contradictions": round(n_contradictions/n_questions*100, 2)
        # }
        if n_questions > 0:
            results[model_name][experiment] = round(n_contradictions/n_questions*100, 2)
        else:
            # if all responses were irrelevant, will have no entries in *_dates.json
            results[model_name][experiment] = 0.0
        total_questions += n_questions
        total_contradictions += n_contradictions
    table = pd.DataFrame.from_dict(results).T
    display(table)
    total_contradictions = round(total_contradictions/total_questions*100, 2)
    print("Total % contradictions: ", total_contradictions)
    return table, total_contradictions


### Plot Years
def plot_years(grc_path: str, results_folders: List[str], exp: str, min_year = 0, max_year = datetime.now().year):
    assert exp in EXPERIMENTS

    grc_file = load_json(grc_path)
    years = []

    for results_folder in results_folders:
        folder = os.path.join(results_folder, exp)
        if not os.path.exists(folder):
            continue
        for file_name in os.listdir(folder):
            if not file_name.endswith("_analysis.json"):
                continue

            domain = file_name.split("_analysis.json")[0]
            analysis = load_json(os.path.join(folder, file_name))
            dates = {}
            for img_type, questions_by_img in analysis.items():
                for question_type, answers_type in questions_by_img.items():
                    for answer_type, answers in answers_type.items():
                        if answer_type in ["correct", "outdated"]:
                            for ans in answers:
                                assert len(ans["matched_answers"]) == 1, f"More predictions for {folder} about {ans['subject']} --- {ans['relation']}: {ans['matched_answers']}"
                                year = ans["matched_answers"][0][-1]
                                subject = ans["subject"]
                                relation = ans["relation"]
                                matched_ans = ans["matched_answers"][0][0]
                                if subject not in dates:
                                    dates[subject] = {}
                                assert relation is not None
                                if relation not in dates[subject]:
                                    dates[subject][relation] = {}
                                if img_type not in dates[subject][relation]:
                                    dates[subject][relation][img_type] = {}
                                n_valid_answers = sum([matched_ans in a for a in grc_file[domain][subject][relation]["answers"]])
                                assert n_valid_answers > 0, f"'{matched_ans}' for {subject} --- {relation}: has no valid answers!"
                                if n_valid_answers == 1:
                                    dates[subject][relation][img_type][question_type] = year

                model = results_folder.split("/").pop()

                for subject in dates:
                    for relation in dates[subject]:
                        for img_type in dates[subject][relation]:
                            question_domain_years = count()
                            for question_type, date in dates[subject][relation][img_type].items():
                                year = datetime.strptime(date, "+%Y-%m-%dT%H:%M:%SZ").year
                                question_domain_years.update([year])
                            if len(question_domain_years) > 0:
                                correct_year = get_correct_year(question_domain_years)
                                if correct_year >= min_year and correct_year <= max_year:
                                    years.append({
                                        "year": correct_year,
                                        "model": model
                                    })
    df = pd.DataFrame().from_records(years)
    fig = px.box(df, x="model", y="year", color="model")
    fig.update_yaxes(dtick=2)
    os.makedirs("./models_output/analysis/plots/",exist_ok=True)
    for file_type in ["png", "pdf"]:
        fig.write_image(os.path.join("./models_output/analysis/plots", f"{exp}.{file_type}"))


EXPERIMENTS = ["visual", "text_only", "llm-text_only"]

def main():
    args = get_args()

    results = "./models_output/results/"
    annotations = "./data/annotations/wikidata_combined.json"

    stats = {}
    stats_by_domain = {}
    for model in args.models:
        assert model in MAP_MODELS
        model_res = os.path.join(results, model)
        if not os.path.exists(model_res):
            continue
        for exp in EXPERIMENTS:
            print("-"*100)
            print(f"Model: {model}| Exp: {exp}")
            print("-"*100)
            exp_res = os.path.join(model_res, exp)

            # skip gpt-4 and gpt-5 sice the results are not avaialable
            if model.startswith("gpt") and exp == "llm-text_only":
                continue

            # run analyze_replies.py: python models_output/analyze_replies.py ./models_output/results/qwen2-vl-7b/visual/ --question-path ./data/annotations/wikidata_combined.json
            analyze_replies(exp_res, annotations)

            print("Stats by domain:")
            if model not in stats_by_domain:
                stats_by_domain[model] = {}
            stats_by_domain[model][exp] = load_stats(exp_res, show_percentage=True)

            # obtain all files
            # python ./models_output/get_outdated_questions.py ./models_output/results/qwen2-vl-7b/visual/
            # answer sheets
            exp_name = exp
            if "llm" in exp:
                exp_name = exp.split("-")[1]
            save_questions_to_update(exp_res, annotations, exp_name)

            print("Stats:")
            if model not in stats:
                stats[model] = {}
            stats[model][exp] = compute_stats(exp_res, show_percentage=True, round_digits=0)

            # python ./models_output/save_dates.py ./models_output/results/qwen2-vl-7b/visual/
            save_dates(exp_res, annotations)
            # analysis.ipynb


    # as results put together, models / experiments (+llm-only) / img_type
    combined_tables_by_domain = []
    # merge and save stats_by_domain
    for model, model_data in stats_by_domain.items():
        for experiment, experiment_data in model_data.items():
            for domain, domain_data in experiment_data.items():
                for image_type, result_df in domain_data.items():
                    result_df_copy = result_df.copy()

                    result_df_copy['Model'] = model
                    result_df_copy['Experiment'] = experiment
                    result_df_copy['Domain'] = domain
                    result_df_copy['ImageType'] = image_type

                    result_df_copy = result_df_copy[['Model', 'Experiment', 'Domain', 'ImageType'] + result_df_copy.columns.tolist()[:-4]]

                    combined_tables_by_domain.append(result_df_copy)

    combined_tables_by_domain = pd.concat(combined_tables_by_domain, ignore_index=True)
    os.makedirs("./models_output/analysis/", exist_ok=True)
    combined_tables_by_domain.to_csv("./models_output/analysis/by_domain.csv")


    # merge and save stats
    combined_tables = []
    for model, model_data in stats.items():
        for experiment, result_df in model_data.items():
            result_df_copy = result_df.T.copy()

            result_df_copy['Model'] = model
            result_df_copy['Experiment'] = experiment

            result_df_copy = result_df_copy[['Model', 'Experiment'] + result_df_copy.columns.tolist()[:-2]]

            combined_tables.append(result_df_copy)

    combined_tables = pd.concat(combined_tables, ignore_index=True)
    print(combined_tables)
    combined_tables.to_csv("./models_output/analysis/upper_bound.csv")



    # prompt agreement (needs save_dates)
    exp_folders = []
    for model in MAP_MODELS:
        for exp in EXPERIMENTS:
            # GPT models do not have llm-text_only
            if model.startswith("gpt") and exp == "llm-text_only":
                continue

            folder = os.path.join("./models_output/results/", model, exp)
            if os.path.exists(folder):
                exp_folders.append(folder)

    contr_table, total_contr = contradiction_analysis(exp_folders)
    contr_table.to_csv("./models_output/analysis/contradictions.csv")

    # year plots
    model_folders = [os.path.join("./models_output/results/", model) for model in MAP_MODELS]
    for exp in EXPERIMENTS:
        plot_years(
            "./data/annotations/wikidata_combined.json",
            [f for f in model_folders if os.path.exists(f)],
            exp,
            min_year=2006
        )


if __name__ == "__main__":
    main()
