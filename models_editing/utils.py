import os
import json
import random
import torch
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoTokenizer, PreTrainedModel
from qwen_vl_utils import process_vision_info
from PIL import Image


MODELS = ["qwen2-vl-7b", "llava-1-5-7b"]

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

def save_json(path: str, data: dict):
    """
    Save data to a JSON file.

    Parameters
    ---------- 
    path: str
        Path to the JSON file.
    data: dict
        Data to be saved.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_image(path: str, image: Image.Image):
    """
    Save an image in PNG format.

    Parameters
    ----------
    path: str
        Path to save the image.
    image: Image.Image
        Image to be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="PNG")

def get_indexes_dataset(path_to_dataset: str) -> List[str]:
    """
    Get the indexes dataset from the given path.

    Parameters
    ----------
    path_to_dataset: str
        Path to the dataset file.

    Returns
    -------
    indexes_dataset: List[str]
        The indexes dataset in format category|entity|propriety|image_type.
    """

    dataset = load_json(path_to_dataset)

    ret = []

    for category in dataset:
        for entity in dataset[category]:
            proprieties = sorted(dataset[category][entity].keys() - set(['images']))
            for propriety in proprieties:
                for image_type in dataset[category][entity]['images']:
                    ret.append(f"{category}|{entity}|{propriety}|{image_type}")
    return ret

def encode_inputs(
        model_name: str,
        tok: AutoTokenizer,
        ids: List[str],
        prompts: Dict[str, List[str]],
        targets: List[str],
        images: List[str],
        demonstrations: Dict[str, List[str]] = None,
) -> Dict[str, Dict[str, List]]:
    """
    Encode inputs for model editing.
    Parameters
    ----------
    model_name: str
        Name of the model.
    tok: AutoTokenizer
        Tokenizer for the model.
    ids: List[str]
        IDs corresponding to each prompt/target/image.
    prompts: Dict[str, List[str]]
        Prompts for testing the model: generic, contextualized, rephrased.
    targets: List[str]
        Target outputs for the prompts.
    images: List[str]
        Paths to the images corresponding to the prompts.
    demonstrations: Dict[str, List[str]], optional
        Demonstrations only for IKE editing algorithm.
    Returns
    -------
    Dict[str, Dict[str, List]]
        Encoded inputs for each type of prompt.
    """

    ret = {key: [] for key in prompts.keys()}

    for key in prompts.keys():
        for idx, prompt in enumerate(prompts[key]):

            demonstrations_content = []
            if demonstrations is not None:
                for demo_prompt, demo_image in zip(demonstrations['prompts'], demonstrations['images']):
                    demonstrations_content.extend([{"type": "image", "image": Image.open(demo_image)}, {"type": "text", "text": demo_prompt}])
                demonstrations_content.extend([{"type": "image", "image": Image.open(images[idx])}, {"type": "text", "text": f"New Fact: {prompts['generic'][idx]} {targets[idx]}"}])

            content = [
                {
                    "type": "image",
                    "image": Image.open(images[idx]),
                },
                {"type": "text", "text": f"Prompt: {prompt}" if demonstrations is not None else prompt},
            ]

            demonstrations_content.extend(content)

            messages = [
                    {
                        "role": "user",
                        "content": demonstrations_content,
                    }
                ]
            
            if model_name == "qwen2-vl-7b":

                texts =tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                input_ids = tok(
                    text=[texts],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                entry = {"input_ids": input_ids, "prompt": prompt, "target": targets[idx], "id": ids[idx]}
                ret[key].append(entry)

            elif model_name == "llava-1-5-7b":
                input_ids = tok.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    padding=True,
                    return_tensors="pt",
                )
                entry = {"input_ids": input_ids, "prompt": prompt, "target": targets[idx], "id": ids[idx]}
                ret[key].append(entry)
            else:
                raise NotImplementedError(f"Model {model_name} not supported for encoding the inputs.")
            
    return ret

def test_model_editing(
    model_name: str,
    model: PreTrainedModel,
    tok: AutoTokenizer,
    ids: List[str],
    prompts: Dict[str, List[str]],
    targets: List[str],
    image: List[str],
    demonstrations: Dict[str, List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, str]]]:   
    
    encoded_inputs = encode_inputs(model_name, tok, ids, prompts, targets, image, demonstrations)
    
    results = {}
    with torch.no_grad():
        for key in encoded_inputs.keys():
            for entry in tqdm(encoded_inputs[key], desc=f"Testing model editing - {key}"):

                input_ids = entry["input_ids"].to(model.device)
                input_ids.to(model.device)

                output = model.generate(
                    **input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tok.tokenizer.eos_token_id,
                )

                output = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(input_ids.input_ids, output)
                ]

                generated_answers = tok.batch_decode(  # type: ignore
                    output, skip_special_tokens=True
                )

                if entry["id"] not in results:
                    results[entry["id"]] = {"answers": {}, "targets": {}, "questions": {}}
                
                results[entry["id"]]["answers"][key] = generated_answers[0]
                results[entry["id"]]["targets"] = entry["target"]
                results[entry["id"]]["questions"][key] = entry["prompt"]

    return results

def save_locality_inputs(number_of_examples: int, path: str):
    """
    Save locality datasets for MME and NQ.
    Parameters
    ----------
    number_of_examples: int
        Number of examples to save for each dataset.
    path: str
        Path to save the datasets.
    """
    from datasets import load_dataset
    mme_data_path = os.path.join(path, "MME", "data.json")
    mme_image_path = os.path.join(path, "MME", "images")
    nq_path = os.path.join(path, "NQ", "data.json")

    if not os.path.exists(os.path.join(path, "MME")):
        os.makedirs(f"{path}/MME")
    if not os.path.exists(os.path.join(path, "NQ")):
        os.makedirs(f"{path}/NQ")
    
    mme_data = load_dataset("lmms-lab/MME", split="test")
    mme_data.shuffle(seed=42)

    mme_to_save = []
    for i in tqdm(range(number_of_examples), desc="Saving MME locality dataset"):
        image_save_path = os.path.join(mme_image_path, f"{mme_data[i]['question_id']}".split(".")[0] + ".png")
        save_image(image_save_path, mme_data[i]["image"])
        mme_to_save.append({
            "question_id": mme_data[i]["question_id"],
            "image_path": image_save_path,
            "question": mme_data[i]["question"],
            "answer": mme_data[i]["answer"],
            "category": mme_data[i]["category"],
        })
        
    save_json(mme_data_path, mme_to_save)

    nq_data = load_dataset("sentence-transformers/natural-questions", split="train")
    nq_data.shuffle(seed=42)

    nq_to_save = []
    for i in tqdm(range(number_of_examples), desc="Saving NQ locality dataset"):
        nq_to_save.append(nq_data[i])

    save_json(nq_path, nq_to_save)


def get_locality_inputs(num_locality_examples: int, path: str) -> Dict:
    """
    Get locality inputs from saved datasets for MME and NQ.
    Parameters
    ----------
    num_locality_examples: int
        Number of locality examples to retrieve.
    path: str
        Path where the datasets are saved.
    Returns
    -------
    ret: Dict
        Locality inputs containing text and vision data.
    """
    mme_path = os.path.join(path, "MME", "data.json")
    nq_path = os.path.join(path, "NQ", "data.json")

    if not os.path.exists(mme_path) or not os.path.exists(nq_path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    
    mme_dataset = load_json(mme_path)[:num_locality_examples]
    nq_dataset = load_json(nq_path)[:num_locality_examples]
    
    ret = {
        "text": {
            "prompt": [], 
            "ground_truth": [],
        }, 
        "vision": {
            "prompt": [], 
            "ground_truth": [], 
            "image": [],
    }
    }

    for mme_elem, nq_elem in zip(mme_dataset, nq_dataset):
        #textual data from NQ
        ret["text"]["prompt"].append(nq_elem["query"])
        ret["text"]["ground_truth"].append(nq_elem["answer"])

        #vision data from MME
        ret["vision"]["prompt"].append(mme_elem["question"])
        ret["vision"]["ground_truth"].append(mme_elem["answer"])
        ret["vision"]["image"].append(Image.open(mme_elem["image_path"]))
    
    return ret



if __name__ == "__main__":
    #get_locality_inputs(5)
    save_locality_inputs(100, "models_editing/locality_datasets")