from email.mime import text
import os
from pydoc import text
import argparse
from urllib import response
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
        prog="python molmo_analysis/analyze_molmo.py",
        description="Analyze the molmo dataset and molmo responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--path-passages",
        metavar="PATH_PASSAGES",
        type=str,
        default="molmo_analysis/data/passages/visual/",
        help="Path to the folder containing the passages retrieved for the sampled Wikidata questions.",
    )
    parser.add_argument(
        "--path-molmo-responses",
        metavar="PATH_MOLMO_RESPONSES",
        type=str,
        default="models_output/results/molmo-o-7b/visual/",
        help="Path to the folder containing the MolMO responses for the sampled Wikidata questions.",
    )
    parser.add_argument(
        "--path-wikidata-combined",
        metavar="PATH_WIKIDATA_COMBINED",
        type=str,
        default="data/annotations/wikidata_combined.json",
        help="Path to the complete dataset of Wikidata questions and answers",
    )
    parser.add_argument(
        "--results-folder",
        metavar="RESULTS_FOLDER",
        type=str,
        default="molmo_analysis/data/analysis/visual/",
        help="Path to the folder where the results of the analysis will be saved.",
    )
    parsed_args = parser.parse_args()
    
    for category in ["athletes", "countries", "organizations"]:
        if not os.path.exists(os.path.join(parsed_args.path_molmo_responses, f"{category}_analysis.json")):
            raise ValueError(f"No MolMO responses found for category {category} at {os.path.join(parsed_args.path_molmo_responses, category)}")
    
    for elem in ["correct", "outdated", "irrelevant"]:
        if not os.path.exists(os.path.join(parsed_args.path_passages, elem, "passages.json")):
            raise ValueError(f"No passages found for category {elem} at {os.path.join(parsed_args.path_passages, elem)}")
    
    if not os.path.exists(parsed_args.path_wikidata_combined):
        raise ValueError(f"No combined Wikidata dataset found at {parsed_args.path_wikidata_combined}")

    if not os.path.exists(parsed_args.results_folder):
        os.makedirs(parsed_args.results_folder)
    
    return parsed_args

def get_passages(passages: dict) -> dict:
    """
    Get the passages retrieved for the sampled Wikidata questions, organized by category, entity and relation.
    Parameters
    ----------
    passages: dict
        Dictionary containing the passages retrieved for the sampled Wikidata questions, organized by category, entity and relation.
    Returns
    -------
    dict
        Dictionary containing the passages retrieved for the sampled Wikidata questions, organized by category, entity and relation, and the number of times each passage was retrieved for each question.  
    """
    ret = {}

    for pred, categories in passages.items():
        for category, entities in categories.items():
            ret.setdefault(category, {})

            for entity, relations in entities.items():
                ret[category].setdefault(entity, {})

                # ATHLETES
                if category == "athletes":
                    relation = "Sports Team"
                    ret[category][entity].setdefault(
                        relation,
                        {"dolma": {}, "model_response_validity": pred}
                    )
                    #! FULL, EM AND SIMPLIFIED MATCHES FOR ATHLETES
                    for match_type in ["full", "em", "simplified"]:
                        for full_match in relations.get("matches", []).get(match_type, []):
                            answer = full_match["answer"]
                            dolma = ret[category][entity][relation]["dolma"]
                            dolma[answer] = dolma.get(answer, 0) + 1

                # COUNTRIES & ORGANIZATIONS
                elif category in {"countries", "organizations"}:
                    for rel, matches in relations.items():
                        ret[category][entity].setdefault(
                            rel,
                            {"dolma": {}, "model_response_validity": pred}
                        )
                        #! ONLY FULL MATCHES FOR COUNTRIES AND ORGANIZATIONS
                        for full_match in matches.get("matches", []).get("full", []):
                            answer = full_match["answer"]
                            dolma = ret[category][entity][rel]["dolma"]
                            dolma[answer] = dolma.get(answer, 0) + 1

    return ret
                    
       
def extract_responses(passages: dict, responses: dict, wikidata_combined: dict) -> None:
    for category, pictures in responses.items():
        for picture, prompt_types in pictures.items():
            for prompt_type, responses_validity in prompt_types.items():
                for res, elements in responses_validity.items():
                    for elem in elements:

                        subject = elem["subject"]
                        relation = elem["relation"]

                        if category not in passages or subject not in passages[category] or relation not in passages[category][subject]:
                            continue

                        if "correct_answer" not in passages[category][subject][relation]:
                            passages[category][subject][relation]["correct_answer"] = elem["correct_answer"]

                        matched = elem.get("matched_answers", [])

                        if len(matched) > 1:
                            raise ValueError(
                                f"Expected 1 matched answer for each response, "
                                f"but got {len(matched)} for category {category}, "
                                f"picture {picture}, prompt type {prompt_type}, "
                                f"response validity {res}, element {elem}"
                            )
                        
                        if len(matched) == 0:
                            continue

                        if len(matched[0]) != 2:
                            raise ValueError(
                                f"Expected matched answer to be a tuple of (entity, relation), "
                                f"but got {matched[0]} for category {category}, "
                                f"picture {picture}, prompt type {prompt_type}, "
                                f"response validity {res}, element {elem}"
                            )
                        matched_entity, start_date = matched[0]


                        if category not in wikidata_combined or subject not in wikidata_combined[category] or relation not in wikidata_combined[category][subject]:
                            raise ValueError(
                                f"Expected to find the relation {relation} for subject {subject} in category {category} in the combined Wikidata dataset, but it was not found. "
                            )
                        
                        entity = None

                        for answer in wikidata_combined[category][subject][relation]["answers"]:
                            answer_entity = answer.split(" |")[0]
                            answer_start_time = answer.split("|S: ")[1].split(" |")[0]

                            if "-00-00T00:00:00Z" in answer_start_time:
                                answer_start_time = answer_start_time.replace("-00-00T00:00:00Z", "-01-01T00:00:00Z")
                            if "-00T00:00:00Z" in answer_start_time:
                                answer_start_time = answer_start_time.replace("-00T00:00:00Z", "-01T00:00:00Z")
                            if matched_entity in answer_entity and start_date in answer_start_time:
                                entity = answer
                                break

                        if entity is None:
                            raise ValueError(
                                f"Expected to find an answer containing the matched entity {matched_entity} and start date {start_date} for relation {relation} and subject {subject} in category {category} in the combined Wikidata dataset, but it was not found. "
                            )


                        passages[category][subject][relation].setdefault(
                            "model_responses", {}
                        )
                        # Increment count
                        model_responses = passages[category][subject][relation]["model_responses"]
                        model_responses[entity] = model_responses.get(entity, 0) + 1

                    

def main(args: Namespace):
    passages = {elem: {} for elem in ["correct", "outdated", "irrelevant"]}

    for response_validity in ["correct", "outdated", "irrelevant"]:
         passages[response_validity] = load_json(os.path.join(args.path_passages, response_validity, "passages.json"))
    
    passages = get_passages(passages)
    

    responses = {}
    for category in ["athletes", "countries", "organizations"]:
        responses[category] = load_json(os.path.join(args.path_molmo_responses, f"{category}_analysis.json"))

    wikidata_combined = load_json(args.path_wikidata_combined)

    extract_responses(passages, responses, wikidata_combined)
    save_json(passages, os.path.join(args.results_folder, "passages_with_responses.json"))

if __name__ == "__main__":
    args = get_args()
    main(args)