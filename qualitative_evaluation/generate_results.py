import os
import csv
from typing import List


def load_csv(file_path: str) -> List[dict]:
    """
    Load a CSV file and return a list of rows as dictionaries.
    Parameters
    ----------
    file_path: str
        The path to the CSV file to be loaded.
    Returns
    -------
    List[dict]
        A list of dictionaries, where each dictionary represents a row in the CSV file with keys corresponding to the column names.
    """
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_csv(rows: List[dict], file_path: str) -> None:
    """
    Save a list of dictionaries to a CSV file.
    Parameters
    ----------
    rows: List[dict]
        A list of dictionaries to be saved, where each dictionary represents a row in the CSV file with keys corresponding to the column names.
    file_path: str
        The path to the CSV file where the data will be saved.
    """
    if not rows:
        raise ValueError("No data to save")

    keys = rows[0].keys()
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(rows)

def compute_label_ratios(rows: List[dict]) -> dict:
    """
    Compute ratios of Correct, Outdated, Generic, Hallucination, Model Collapse.
    Parameters
    ----------
    rows: List[dict]
        List of rows from the CSV file, where each row is a dictionary with keys corresponding to the column names.
    Returns
    -------
    dict
        A dictionary containing the distribution of each category (Correct, Outdated, Generic, Hallucination, Model Collapse) normalized
    """

    labels = [
        "Correct",
        "Outdated",
        "Generic",
        "Hallucination",
        "Model Collapse"
    ]

    total_rows = len(rows)
    if total_rows == 0:
        raise ValueError("CSV file is empty")

    counts = {label: 0 for label in labels}
    count_irrelevant = 0
    for row in rows:
        for label in labels:
            if label not in row:
                raise ValueError(f"Missing expected column '{label}' in CSV file.")
            value = row[label].strip()
            if value == "1":
                counts[label] += 1
        if "Irrelevant" in row and row["Irrelevant"].strip() == "1":
            count_irrelevant += 1

    assert sum([counts["Generic"], counts["Hallucination"], counts["Model Collapse"]]) == count_irrelevant, "Irrelevant count does not match the sum of Generic, Hallucination, and Model Collapse counts"
    # Normalize so the sum is 1
    ratios = {
        label: counts[label] / total_rows
        for label in labels
    }

    return ratios

def compute_folder_results(folder_path: str) -> List[dict]:
    """
    Load all CSV files in a folder and compute qualitative analysis evaluation results for each file
    Parameters
    ----------
    folder_path: str
        The path to the folder containing the CSV files to be analyzed.
    Returns
    -------
    List[dict]
        A list of dictionaries, where each dictionary contains the model name, editing model name, and the results of the qualitative analysis evaluation (Correct, Outdated, Generic, Hallucination, Model Collapse) for each CSV file in the folder.
    """
    results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".csv"):
            continue

        name_part = filename.replace(".csv", "")
        name_part = name_part.replace("QA - ", "")
        name_part = name_part.replace(" copy", "")

        try:
            editing_model, model = name_part.split("_", 1)
        except ValueError:
            raise ValueError(f"Filename '{filename}' does not match the expected format 'QA - <EditingModel>_<Model>.csv'")

        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")
        data = load_csv(file_path)
        distribution = compute_label_ratios(data)

        row = {
            "Model": model,
            "Editing Model": editing_model,
            "Correct": distribution["Correct"],
            "Outdated": distribution["Outdated"],
            "Generic": distribution["Generic"],
            "Hallucination": distribution["Hallucination"],
            "Model Collapse": distribution["Model Collapse"]
        }

        results.append(row)

    return results



def main():
    results = compute_folder_results("qualitative_evaluation/evaluated")
    save_csv(results, "qualitative_evaluation/summary_results.csv")

if __name__ == "__main__":
    main()