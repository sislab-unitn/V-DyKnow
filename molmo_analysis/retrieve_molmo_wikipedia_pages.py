import os
import gzip
import json
import tqdm
import requests
import argparse
from argparse import Namespace
from typing import List, Dict
from utils import load_json, save_json

ENTITIES_EXCEPTIONS = {
    "George Russell": "George Russell (racing driver)",
    "Amazon": "Amazon (company)",
    "Shell": "Shell plc",
} 

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python molmo_analysis/retrieve_molmo_wikipedia_pages.py",
        description="Retrieve the Wikipedia pages corresponding to the entities in the sampled Wikidata questions and save them for further analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--dolma-data-folder",
        metavar="PATH_DOLMA_DATA",
        type=str,
        default="molmo_analysis/data/wikidata_dolma/",
        help="Path to the folder containing the compressed DOLMA data files (.json.gz).",
    )
    parser.add_argument(
        "--path-sampled-entities",
        metavar="PATH_SAMPLED_ENTITIES",
        type=str,
        default="molmo_analysis/data/sampled_entities/visual/",
        help="Path to the folder containing the sampled Wikidata entities and their corresponding questions.",
    )
    parser.add_argument(
        "--results-folder",
        metavar="RESULTS_FOLDER",
        type=str,
        default="molmo_analysis/data/dolma_selected_documents/visual",
        help="Path to the folder where the selected questions will be saved.",
    )
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.dolma_data_folder):
        raise ValueError(f"No DOLMA data found at {parsed_args.dolma_data_folder}")
    
    if not os.path.exists(parsed_args.path_sampled_entities):
        raise ValueError(f"No sampled entities found at {parsed_args.path_sampled_entities}")
    
    return parsed_args


def get_wikipedia_ids(entities_list: List[str], exceptions: Dict[str, str] = ENTITIES_EXCEPTIONS) -> Dict[str, str]:
    """
    Get Wikipedia page IDs for a list of entities.
    Args:
        entities_list (list[str]): List of entity names to query.
        exceptions (dict[str, str]): Optional mapping of entity names to their correct Wikipedia titles for disambiguation.
    Returns:
        dict[str, str]: Mapping of entity names to their corresponding Wikipedia page IDs.
    """

    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "MyDataScript/1.0 (your-email@example.com)"
    }
    entities_list_filtered = []
    for entity in entities_list:
        if entity in exceptions:
            print(f"Using exception for entity '{entity}': '{exceptions[entity]}'")
            entities_list_filtered.append(exceptions[entity])
        else:
            entities_list_filtered.append(entity)

    entities_string = "|".join(entities_list_filtered)

    params = {
        "action": "query",
        "titles": entities_string,
        "format": "json",
        "redirects": 1,
        "prop": "pageprops"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    mapping = {}

    for pid, info in pages.items():
        if int(pid) < 0:
            raise ValueError(f"Page not found for title {info.get('title', 'Unknown')}")

        if "pageprops" in info and "disambiguation" in info["pageprops"]:
            raise ValueError(f"Disambiguation page found for title {info['title']} with pid {pid}. Please provide a specific title in the exceptions mapping.")

        mapping[info["title"]] = pid

    return mapping


def main(args: Namespace) -> None:
    labels = ["correct", "outdated", "irrelevant"]

    data_samples = {labels[i]: load_json(f"{args.path_sampled_entities}{labels[i]}/wikidata.json") for i in range(len(labels))}
    

    list_entities = set()
    for label, samples in data_samples.items():
        for _, entities in samples.items():
            list_entities.update(entities.keys())

    # Get Wikipedia page IDs for all entities
    ids = get_wikipedia_ids(list_entities)
    print(ids)

    urls = {entity: f"https://en.wikipedia.org/wiki?curid={id}" for entity, id in ids.items()}

    to_save = {}
    dolma_files = [f for f in os.listdir(args.dolma_data_folder) if f.endswith(".json.gz")]
    
    for file in dolma_files:
        with gzip.open(os.path.join(args.dolma_data_folder, file), "r") as f:
            #data = []
            for line in tqdm.tqdm(f, desc=f"Loading data from {file}"):
                if line.strip():
                    #data.append(json.loads(line))
                    item = json.loads(line)
                    for entity, url in urls.items():
                        if url == item['metadata']['url']:
                            
                            entity_name = entity

                            for inital_name, exception_name in ENTITIES_EXCEPTIONS.items():
                                if exception_name == entity:
                                    entity_name = inital_name
                                    break

                            print(f"Found matching item for ID {entity_name}: {item['metadata']['url']}")
                            item_copy = item.copy()
                            to_save[entity_name] = item_copy
                            break
                            
    if len(to_save) != len(urls):
        print(f"Warning: Found {len(to_save)} matching items, but expected {len(urls)}. Missing entities: {set(urls.keys()) - set(to_save.keys())}")

    os.makedirs(args.results_folder, exist_ok=True)
    save_json(to_save, f"{args.results_folder}/data.json")

if __name__ == "__main__":
    args = get_args()
    main(args)



