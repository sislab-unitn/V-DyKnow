import os
import yaml
import torch
import random
import argparse
from typing import List
from argparse import Namespace
from utils import test_model_editing, MODELS, get_locality_inputs, get_indexes_dataset
from utils import load_json, save_json
from EasyEdit.easyeditor import MultimodalEditor, WISEMultimodalHyperParams, GraceHyperParams
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaImageProcessor,
    Qwen2VLForConditionalGeneration,
)
def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m edit_model",
        description="Edit a model using a specific method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "edit_alg",
        metavar="EDIT_ALG_NAME",
        type=str,
        default="GRACE",
        choices={"WISE", "GRACE", "IKE"},
        help="Editing algorithm to update a model.",
    )
    parser.add_argument(
        "hparams_path",
        metavar="HPARAMS_PATH",
        type=str,
        help="Path to the hparams path with the configuration.",
    )
    parser.add_argument(
        "edit_dataset_path",
        metavar="DATASET_PATH",
        type=str,
        help="Path to the editing dataset.",
    )
    parser.add_argument(
        "--locality-inputs-path",
        metavar="LOCALITY_INPUTS_PATH",
        type=str,
        default="models_editing/locality_datasets/",
        help="Path to the locality inputs dataset.",
    )
    parser.add_argument(
        "--editing-ratio",
        metavar="FLOAT",
        type=float,
        default=0.01,
        help="Ratio of entries from the editing dataset to be used for editing.",
    )
    parser.add_argument(
        "--complete-dataset_path",
        metavar="DATASET_PATH",
        type=str,
        default="data/annotations/wikidata_combined.json",
        help="Path to the complete dataset.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, saves the edited model.",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only.",
        help="Prompt for Q&A generation.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    if parsed_args.edit_alg == "GRACE":
        parsed_args.edit_alg = {"name": parsed_args.edit_alg, "params": GraceHyperParams}
    elif parsed_args.edit_alg == "WISE":
        parsed_args.edit_alg = {"name": parsed_args.edit_alg, "params": WISEMultimodalHyperParams}
    elif parsed_args.edit_alg == "IKE":
        parsed_args.edit_alg = {"name": parsed_args.edit_alg, "params": None} 
    else:
        raise NotImplementedError
    
    if not os.path.exists(parsed_args.hparams_path):
        raise FileNotFoundError(f"Hparams path {parsed_args.hparams_path} does not exist.")
    
    if not os.path.exists(parsed_args.edit_dataset_path):
        raise FileNotFoundError(f"Editing dataset path {parsed_args.edit_dataset_path} does not exist.")
    
    if not os.path.exists(parsed_args.complete_dataset_path):
        raise FileNotFoundError(f"Complete dataset path {parsed_args.complete_dataset_path} does not exist.")
    
    if not (0.0 <= parsed_args.editing_ratio <= 1.0):
        raise ValueError("Editing ratio must be between 0.0 and 1.0.")
    
    return parsed_args

def main(args: Namespace):

    seed = 42
    random.seed(seed)

    #get the indexes dataset from the complete dataset
    indexes_dataset = get_indexes_dataset(args.complete_dataset_path)
    
    print(f"Total number of entries in the complete dataset: {len(indexes_dataset)}")

    random.shuffle(indexes_dataset)
    
    num_edits = int(len(indexes_dataset) * args.editing_ratio)
    indexes_dataset_edit = indexes_dataset[:num_edits]

    editing_dataset = load_json(args.edit_dataset_path)

    print(f"Total number of entries in the editing dataset: {len(editing_dataset)}")

    editing_dataset_filtered = {elem_id: editing_dataset[elem_id] for elem_id in indexes_dataset_edit if elem_id in editing_dataset}

    ids = []
    prompts = {"generic": [], "contextualized": [], "rephrased": []}
    targets = []
    image = []

    for entry_id, entry in editing_dataset_filtered.items():
        prompts["generic"].append(f"{args.prompt} {entry['generic']}")# if args.edit_alg["name"] != "IKE" else entry['generic']) 
        prompts["contextualized"].append(f"{args.prompt} {entry['contextualized']}")# if args.edit_alg["name"] != "IKE" else entry['contextualized'])
        prompts["rephrased"].append(f"{args.prompt} {entry['rephrased']}")# if args.edit_alg["name"] != "IKE" else entry['rephrased'])
        targets.append(entry["target"])
        image.append(entry["image"])
        ids.append(entry_id)

    print(f"Number of entries in the filtered editing dataset: {len(editing_dataset_filtered)}")

    metrics, edited_model, processor = None, None, None
    editor = None
    model_name = None
    demonstrations = None
    hparams = None

    if args.edit_alg["name"] == "WISE" or args.edit_alg["name"] == "GRACE":
        hparams = args.edit_alg["params"].from_hparams(args.hparams_path) 

        editor = MultimodalEditor.from_hparams(hparams)
        normalized_name = editor.model_name.replace(".", "-")

        model_name = next(
            (model for model in MODELS if model.startswith(normalized_name)),
            None
        )
        if not model_name:
            raise ValueError(f"Model name '{editor.model_name}' not recognized. Available models: {MODELS}")
    

    if args.edit_alg["name"]  == "WISE":

        locality_inputs = get_locality_inputs(
            num_locality_examples=len(prompts["generic"]),
            path=args.locality_inputs_path,
        )
        metrics, edited_model, _ = editor.edit(
            prompts=prompts["generic"],
            targets=targets,
            image=image,
            locality_inputs=locality_inputs,
            sequential_edit=False,
            keep_original_weight=False,
            eval_metric='token em',
            file_type=["image" for _ in range(len(prompts["generic"]))],      
        )

        processor = editor.tok

    elif args.edit_alg["name"]  == "GRACE":

        metrics, edited_model, _ = editor.edit(
            prompts=prompts["generic"],
            targets=targets,
            image=image,
            locality_inputs={},
            sequential_edit=False,
            keep_original_weight=False,
            eval_metric='token em',
            file_type=["image" for _ in range(len(prompts["generic"]))],      
        )

        processor = editor.tok

    elif args.edit_alg["name"]  == "IKE":

        with open(args.hparams_path, "r") as f:
            hparams = yaml.safe_load(f)

        model_name = hparams["model_name"]

        demonstrations = {
            "images": [
                "data/imgs/resized/Exxon_Mobil_Logo.png", 
                "data/imgs/resized/Exxon_Mobil_Logo.png", 
                "data/imgs/resized/Exxon_Mobil_Logo.png",
                "data/imgs/resized/Exxon_Mobil_Logo.png",
                "data/imgs/resized/Exxon_Mobil_Logo.png", 
                "data/imgs/resized/Flag_of_Turkey.png",
                ], 
            "prompts": [
                "New Fact: Who is the current chief executive officer of this company? Darren Woods", 
                "Prompt: Who is the current chief executive officer of this company? Darren Woods", 
                "New Fact: Who is the current chief executive officer of this company? Darren Woods", 
                "Prompt:  Who currently serves as CEO of this company? Darren Woods",
                "New Fact: Who is the current chief executive officer of this company? Darren Woods", 
                "Prompt: Who is the current head of government of this country? Recep Tayyip Erdoğan"
                ],
            }
        demonstrations["images"] = demonstrations["images"][: hparams["k"] * 2]
        demonstrations["prompts"] = demonstrations["prompts"][: hparams["k"] * 2]

        if model_name == "qwen2-vl-7b":
            edited_model = Qwen2VLForConditionalGeneration.from_pretrained(
                hparams["hf_model_name"],
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{hparams['device']}",
                low_cpu_mem_usage=True,
            )

            processor = AutoProcessor.from_pretrained(
                hparams["hf_model_name"],
                min_pixels=100 * 28 * 28,
                max_pixels=16384 * 28 * 28,
            )

        elif model_name == "llava-1-5-7b":
            edited_model = LlavaForConditionalGeneration.from_pretrained(
                hparams["hf_model_name"],
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{hparams['device']}",
                low_cpu_mem_usage=True,
            )
            image_processor = LlavaImageProcessor.from_pretrained(
                hparams["hf_model_name"], do_pad=True
            )
            processor = AutoProcessor.from_pretrained(
                hparams["hf_model_name"], image_processor=image_processor
            )
        else:
            raise NotImplementedError(f"Model {model_name} not supported for IKE editing.")
    else:
        raise NotImplementedError
    

    if args.save_model:
        if not os.path.exists(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/saved_models/edited_model_ratio_{args.editing_ratio}"):
            os.makedirs(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/saved_models/edited_model_ratio_{args.editing_ratio}")
            
        edited_model.save(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/saved_models/edited_model_ratio_{args.editing_ratio}/checkpoint.pt")

        #processor.tokenizer.save_pretrained(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/saved_models/edited_tokenizer_ratio_{args.editing_ratio}")
        #processor.image_processor.save_pretrained(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/model/edited_image_processor_ratio_{args.editing_ratio}")

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(metrics)

    result = test_model_editing(
        model_name=model_name,
        model=edited_model,
        tok=processor,
        ids=ids,
        prompts=prompts,
        targets=targets,
        image=image,
        demonstrations=demonstrations,
    )

    if not os.path.exists(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results"):
        os.makedirs(f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results")
    
    params_to_save = {
        "edit_alg": args.edit_alg["name"],
        "model_name": model_name,
        "editing_ratio": args.editing_ratio,
        "prompt": args.prompt,
        "number_entries_complete_dataset": len(indexes_dataset),
        "number_entries_total_editing_dataset": len(editing_dataset),
        "number_entries_filtered_editing_dataset": len(editing_dataset_filtered),
        "seed": seed,
        "hparams": hparams if hparams != None else None,
        "demonstrations": demonstrations if demonstrations else None,
    }

    if args.edit_alg["name"] == "IKE":
        save_json(
            f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results/results_ratio_{args.editing_ratio}_k_{hparams['k']}.json",
            result
        )
        save_json(
            f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results/params_ratio_{args.editing_ratio}_k_{hparams['k']}.json",
            params_to_save
        )
    else:
        save_json(
            f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results/results_ratio_{args.editing_ratio}.json",
            result
        )
        save_json(
            f"models_editing/{args.out_dir}/{args.edit_alg['name']}/{model_name}/generation_results/params_ratio_{args.editing_ratio}.json",
            params_to_save
        )

if __name__ == "__main__":
    args = get_args()
    main(args)