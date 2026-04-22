import os
import random
import argparse
from utils import load_json, save_json
from argparse import Namespace


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python molmo_analysis/sample_data.py",
        description="Sample 30 (category, entity, propriety) combinations from the MolMO data (10 correct, 10 outdated and 10 irrelevant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--path_to_molmo_results",
        metavar="PATH_TO_MOLMO_RESULTS",
        type=str,
        default="models_output/results/molmo-o-7b/visual",
        help="Path to the MolMO results directory.",
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
        default="molmo_analysis/data/sampled_entities/visual",
        help="Path to the folder where the selected questions will be saved.",
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.path_to_molmo_results):
        raise ValueError(f"No MolMO results found at {parsed_args.path_to_molmo_results}")
    
    if not os.path.exists(parsed_args.path_to_wikidata_questions):
        raise ValueError(f"No Wikidata questions found at {parsed_args.path_to_wikidata_questions}")
    
    if not os.path.exists(os.path.join(parsed_args.results_folder, "correct")):
        os.makedirs(os.path.join(parsed_args.results_folder, "correct"))
        os.makedirs(os.path.join(parsed_args.results_folder, "outdated"))
        os.makedirs(os.path.join(parsed_args.results_folder, "irrelevant"))
    
    return parsed_args


def main(args: Namespace):

    random.seed(42)
    molmo_data = {}
    
    for category in ["countries", "athletes", "organizations"]:
        molmo_data[category] = load_json(f"{args.path_to_molmo_results}/{category}_answer_sheet.json")

    stratified_samples = {"correct": [], "outdated": [], "irrelevant": []}

    for category, entities in molmo_data.items():
        for entity, proprieties in entities.items():
            for propriety, images in proprieties.items():
                
                # Get the responses for the images and determine the overall response for the (category, entity, propriety) combination
                res_images = {}
                for image, values in images.items():
                    responses = set(values.values())
                    if "correct" in responses:
                        res_images[image] = "correct"
                    elif "outdated" in responses:
                        res_images[image] = "outdated"
                    elif "irrelevant" in responses:
                        res_images[image] = "irrelevant"

                # Determine the overall response for the (category, entity, propriety) combination based on the responses of the images
                res_values = set(res_images.values())
                if len(res_values) == 1:
                    stratified_samples[res_values.pop()].append((category, entity, propriety))
                elif len(res_values) == 2:
                    if "correct" in res_values:
                        stratified_samples["correct"].append((category, entity, propriety))
                    elif "outdated" in res_values:
                        stratified_samples["outdated"].append((category, entity, propriety))
                    elif "irrelevant" in res_values:
                        stratified_samples["irrelevant"].append((category, entity, propriety))
                elif len(res_values) == 0:
                    print(f"No responses for {category} - {entity} - {propriety}")
                else:
                    print(f"Unexpected combination of responses for {category} - {entity} - {propriety}: {res_values}")
    
    for key in stratified_samples:
        # Randomly sample 10 entries from each category
        stratified_samples[key] = random.sample(stratified_samples[key], min(10, len(stratified_samples[key])))

    print("Stratified samples:")
    for key, samples in stratified_samples.items():
        print(f"\n{key.capitalize()}:")
        for sample in samples:
            print(sample)

    
    
    wikidata_combined = load_json(args.path_to_wikidata_questions)

    to_save = {"correct": {}, "outdated": {}, "irrelevant": {}}

    for key, samples in stratified_samples.items():
        for sample in samples:
            category, entity, propriety = sample
            if category in wikidata_combined and entity in wikidata_combined[category]:
                if propriety in wikidata_combined[category][entity]:
                    if category not in to_save[key]:
                        to_save[key][category] = {}
                    if entity not in to_save[key][category]:
                        to_save[key][category][entity] = {}
                    to_save[key][category][entity][propriety] = wikidata_combined[category][entity][propriety].copy()
            else:
                print(f"Sample not found in Wikidata: {sample}")

    for key, samples in to_save.items():
        save_json(samples, os.path.join(args.results_folder, f"{key}", "wikidata.json"))



if __name__ == "__main__":
    args = get_args()
    main(args)