import re
import os
import csv
import json
import requests
import tqdm
from urllib.parse import quote
from bs4 import BeautifulSoup

EXCEPTIONS = {
    "organizations|Amazon|chief executive officer": "https://en.wikipedia.org/wiki/Amazon_(company)",
    "organizations|Shell|chief executive officer": "https://en.wikipedia.org/wiki/Shell_plc",
    "countries|Russia|Prime Minister of Russia": "https://en.wikipedia.org/wiki/Mikhail_Mishustin",
    "countries|Canada|Prime Minister of Canada": "https://en.wikipedia.org/wiki/Mark_Carney",
    "countries|Canada|monarch of the United Kingdom": "https://en.wikipedia.org/wiki/Charles_III",
    "countries|Singapore|President of Singapore": "https://en.wikipedia.org/wiki/Tharman_Shanmugaratnam",
    "countries|Indonesia|President of Indonesia": "https://en.wikipedia.org/wiki/Prabowo_Subianto",
    "countries|Israel|President of Israel": "https://en.wikipedia.org/wiki/Isaac_Herzog",
    "countries|India|President of India": "https://en.wikipedia.org/wiki/Droupadi_Murmu",
    "organizations|Toyota|chief executive officer": "https://en.wikipedia.org/wiki/Koji_Sato_(engineer)",
    "countries|Netherlands|Prime Minister of the Netherlands": "https://en.wikipedia.org/wiki/Dick_Schoof",
    "countries|United States of America|President of the United States": "https://en.wikipedia.org/wiki/Donald_Trump",
}


def load_json(path: str) -> dict:
    """
    Load a JSON file.

    Parameters
    ----------
    path: str
        Path to the JSON file.
    Returns
    -------
    data: dict
        Loaded JSON data.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def remove_content_between_chars(string: str, start="[", end="]"):
    edited = []
    skip = False
    for char in string:
        if char == start:
            skip = True

        if not skip:
            edited.append(char)

        if char == end:
            assert skip, f"Closing character found without opening in {''.join(edited)}"
            skip = False
    return "".join(edited)

def resolve_wikipedia_links(
    input_dict: dict,
    output_csv_path: str,
    execeptions: dict = EXCEPTIONS,
    timeout: int = 10,
    min_paragraphs: int = 5
):
    output = []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Wikipedia-Link-Resolver)"
    })

    for key, value in tqdm.tqdm(input_dict.items(), desc="Resolving Wikipedia links"):
        attempted_url = None

        category, element, attribute = key.split("|")

        if key in execeptions:
            attempted_url = execeptions[key]
        else:
            encoded_element = quote(element.replace(" ", "_"))
            attempted_url = f"https://en.wikipedia.org/wiki/{encoded_element}"

        response = session.get(
            attempted_url,
            timeout=timeout,
            allow_redirects=True
        )

        if not (200 <= response.status_code < 300):
            raise ValueError(
                f"Failed to retrieve Wikipedia page for {element} "
                f"(status code: {response.status_code})"
            )

        page_text = response.text.lower()

        if "wikipedia does not have an article with this exact name" in page_text:
            raise ValueError("Wikipedia page does not exist")

        soup = BeautifulSoup(response.text, "html.parser")
        res = [soup.find("h1")]
        div = soup.find("div", class_="mw-content-ltr mw-parser-output")
        res += [i for i in div.find_all(re.compile("(h.)|p"))]

        paragraphs = 0
        for tag in res:
            if tag.name.startswith("h"):
                current_header = tag
            elif tag.name == "p" and tag.get("style") is None:
                # remove multiple whitespaces
                p = " ".join(tag.text.split())
                # remove additional whitespaces
                p = p.strip()
                # remove content between '[' and ']'
                p = remove_content_between_chars(p)
                if len(p) > 0:
                    paragraphs += 1

        if paragraphs <= min_paragraphs:
            raise ValueError(f"Insufficient content in Wikipedia page: {attempted_url}, only {paragraphs} paragraphs found")

        if not isinstance(value, dict) or "target" not in value:
            raise ValueError("Missing target")

        output.append({
            "category": category,
            "item": element,
            "attribute": attribute,
            "answer": value["target"],
            "page_url": response.url  
        })
    
    with open(output_csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

def combine_datasets(*datasets: dict) -> dict:
    """
    Combine multiple datasets into one, ensuring no conflicting targets.
    Parameters
    ----------
    *datasets: dicts
        Multiple datasets to combine with keys in the format "category|element|attribute|picture" 
        and as values dictionaries containing at least a "target" key.
    Returns
    -------
    combined: dict
        Combined dataset with unique keys in the format "category|element|attribute" and their corresponding targets.
    Raises
    -------
    ValueError
        If there are conflicting targets for the same key.
    """
    combined = {}
    for dataset in datasets:
        for key, value in dataset.items():
            new_key = key.rsplit("|", 1)
            target = value.get("target") if isinstance(value, dict) else None
            if new_key[0] not in combined:
                combined[new_key[0]] = {"target": target}
            else:
                if combined[new_key[0]]["target"] != target:
                    raise ValueError(f"Conflicting targets for key {new_key[0]}")
    return combined




if __name__ == "__main__":
    input_json_path_llava = "models_editing/editing_datasets/llava-1-5-7b/editing_dataset.json"
    input_json_path_qwen = "models_editing/editing_datasets/qwen2-vl-7b/editing_dataset.json"

    output_csv_path = "RAG/data/annotations/wikipedia_pages.csv"

    input_data_qwen = load_json(input_json_path_qwen)
    input_data_llava = load_json(input_json_path_llava)

    input_data_combined = combine_datasets(input_data_llava, input_data_qwen)

    resolved_links = resolve_wikipedia_links(
        input_dict=input_data_combined,
        output_csv_path=output_csv_path,
        timeout=10,
        min_paragraphs=2
    )