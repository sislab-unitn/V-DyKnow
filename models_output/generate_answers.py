import argparse
from argparse import Namespace
import os
from functools import partial

import torch
import random
import numpy as np
import json
import gc

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2ForCausalLM,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoModelForImageTextToText,
    LlavaImageProcessor,
    LlamaForCausalLM,
    LlamaTokenizer,
    PaliGemmaForConditionalGeneration
)

from dataset import SAMPLE_IDS_SEP, DyKnowDataset
from models import qwen2_vl, qwen2_5_vl, llava_onevision, internVL3_5_8B, molmo, llava_1_5_7b, paligemma2_10b, gpt


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python models_output/generate_answers.py",
        description="Generate answers to questions using a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        type=str,
        choices=[model_name for model_name in MAP_MODELS.keys()],
        help="Model used for generation.",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only evaluate the LLM of the model (only for text_only questions).",
    )
    parser.add_argument(
        "--experiment",
        metavar="TYPE",
        type=str,
        choices=["text_only", "visual", "detection"],
        default="visual",
        help="Type of experiment.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--max-length",
        metavar="LENGTH",
        type=int,
        default=20,
        help="Max number of tokens that can be generated.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="./models_output/results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--data",
        metavar="FOLDER_PATH",
        type=str,
        default="./data",
        help="Path to the data folder containing Q&A and images.",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only.",
        help="Prompt for Q&A generation.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiple GPUs (device='auto').",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute results.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args

MAP_MODELS = {
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        qwen2_vl.GenerationCollator,
    ),
    "qwen2-5-vl-7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        partial(
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        qwen2_5_vl.GenerationCollator,
    ),
    "llava-onevision-7b": (
        "llava-hf/llava-onevision-qwen2-7b-si-hf",
        partial(
            LlavaOnevisionForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        llava_onevision.GenerationCollator,
    ),
    "internVL3-5-8b": (
        "OpenGVLab/InternVL3_5-8B-HF",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            #use_flash_attn=True,
            trust_remote_code=True
        ),
        internVL3_5_8B.GenerationCollator,
    ),
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # correct torch_dtype is torch.float32, but performance is negligible
        ),
        molmo.GenerationCollator,
    ),
    "llava-1-5-7b": (
        "llava-hf/llava-1.5-7b-hf",
        partial(
            LlavaForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        llava_1_5_7b.GenerationCollator,
    ),
    "paligemma2-10b": (
        "google/paligemma2-10b-mix-448",
        partial(
            PaliGemmaForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        paligemma2_10b.GenerationCollator,
    ),
    "gpt-4": (
        "gpt-4.1",
        partial(
            gpt.GPTModel,
            temperature=0.0,
        ),
        gpt.GenerationCollator,
    ),
    "gpt-5": (
        "gpt-5.1",
        partial(
            gpt.GPTModel,
            temperature=0.0,
        ),
        gpt.GenerationCollator,
    )
}

MAP_MODELS_LLM_ONLY = {
    "qwen2-vl-7b": (
        "Qwen/Qwen2-7B-Instruct", # unclear if instruct used for VL model (or only pre-trained was used)
        partial(
            Qwen2ForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        qwen2_vl.LLMGenerationCollator,
    ),
    "qwen2-5-vl-7b": (
        "Qwen/Qwen2.5-7B-Instruct", # unclear if instruct used for VL model (or only pre-trained was used)
        partial(
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        qwen2_5_vl.LLMGenerationCollator,
    ),
    "llava-onevision-7b": (
        "Qwen/Qwen2-7B-Instruct", # unclear if instruct used for VL model (or only pre-trained was used)
        partial(
            Qwen2ForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        llava_onevision.LLMGenerationCollator,
    ),
    "internVL3-5-8b": (
        "Qwen/Qwen3-8B",
        partial(
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        internVL3_5_8B.LLMGenerationCollator,
    ),
    "molmo-o-7b": (
        "allenai/OLMo-7B-Instruct-hf",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # correct torch_dtype is torch.float32, but performance is negligible
        ),
        molmo.LLMGenerationCollator,
    ),
    "llava-1-5-7b": (
        "lmsys/vicuna-7b-v1.5", # Selected based on the following paper: https://arxiv.org/pdf/2310.03744
        partial(
            LlamaForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        llava_1_5_7b.LLMGenerationCollator,
    ),
    "paligemma2-10b": (
        "google/gemma-2-9b-it", # Selected gemma2 based on the following article: https://arxiv.org/pdf/2412.03555
        partial(
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        paligemma2_10b.LLMGenerationCollator,
    ),
}

LIST_API_MODELS = ["gpt-4", "gpt-5"]

def generate(
    model_name: str,
    data_folder: str,
    instruction: str = "",
    batch_size: int = 1,
    out_dir: str = "output",
    max_new_tokens: int = 15,
    parallel: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    experiment: str = "visual",
    llm_only: bool = False,
    skip_existing: bool = False,
):
    res_folder = experiment
    if llm_only:
        assert experiment == "text_only"
        res_folder = f"llm-{experiment}"
    out_dir = os.path.join(out_dir, model_name, res_folder)

    if skip_existing and os.path.exists(os.path.join(out_dir, "organizations_answers.json")):
        print(f"Results for {model_name} {res_folder} exist. Skipping ... ")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # get the model and processor based on the model name
    map_models = MAP_MODELS
    if llm_only:
        map_models = MAP_MODELS_LLM_ONLY
    hf_model_name, Model, Collator = map_models[model_name]

    model = Model(
        hf_model_name,
        low_cpu_mem_usage=True,
        device_map="auto" if parallel else device,
    )

    # model config
    model.eval()

    Processor = partial(AutoProcessor.from_pretrained, hf_model_name)
    if model_name == "qwen2-vl-7b":
        # as declared in the paper
        processor = Processor(
            min_pixels=100 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )
    elif model_name == "molmo-o-7b":
        processor = Processor(
            trust_remote_code=True,
        )
    elif model_name == "llava-1-5-7b" and not llm_only:
        # use LlavaImageProcessor to ensure reproducing original implementation
        image_processor = LlavaImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", do_pad=True)
        processor = Processor(image_processor=image_processor)
    # elif model_name == "llava-1-5-7b" and llm_only:
    #     processor = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    else:
        processor = Processor()

    # Tok config
    tokenizer = processor
    if not llm_only:
        tokenizer = processor.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore
    tokenizer.padding_side = "left"  # type: ignore

    # set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load test data
    test_ds = DyKnowDataset(data_folder, experiment)
    test_loader = DataLoader(
        test_ds,
        batch_size=1 if model_name == "molmo-o-7b" else batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(
            processor=processor,  # type: ignore
            instruction=instruction,
        ),
    )


    responses = {}

    # generate
    with torch.no_grad():
        for input_ids, questions, sample_ids in tqdm(
            test_loader, desc=f"Generating responses"
        ):
            if model_name != "molmo-o-7b" or llm_only:
                input_ids.to(model.device)
                output = model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                if model_name in ["qwen2-vl-7b", "qwen2-5-vl-7b", "llava-onevision-7b", "internVL3-5-8b", "molmo-o-7b", "llava-1-5-7b", "paligemma2-10b"]:
                    output = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(input_ids.input_ids, output)
                    ]

                generated_answers = processor.batch_decode(  # type: ignore
                    output, skip_special_tokens=True
                )
            else:
                # molmo works only with transformers <= 4.48 (but qwen2.5-vl requires more recent version)

                # put the inputs on the correct device and make a batch of size 1
                input_ids = {
                    k: v.to(model.device).unsqueeze(0) for k, v in input_ids.items()
                }
                # cast the inputs to the right torch_dtype
                if "images" in input_ids:
                    input_ids["images"] = input_ids["images"].to(torch.bfloat16)

                # model.generate_from_batch( input_ids, GenerationConfig( max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>",   do_sample=False,), tokenizer=processor.tokenizer,)
                output = model.generate_from_batch(
                    input_ids,
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        stop_strings="<|endoftext|>",  # suggested on MOLMo huggingface page
                        do_sample=False,
                    ),
                    tokenizer=processor.tokenizer,
                )
                # discard the input part
                generated_tokens = output[0, input_ids["input_ids"].size(1) :]
                generated_answers = [
                    processor.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                ]

            for answ, q, s_id in zip(generated_answers, questions, sample_ids):
                category, subject, relation, q_type, img_type = s_id.split(SAMPLE_IDS_SEP)

                if category not in responses:
                    responses[category] = {}
                if subject not in responses[category]:
                    responses[category][subject] = {}

                if relation not in responses[category][subject]:
                    responses[category][subject][relation] = {}

                if img_type not in responses[category][subject][relation]:
                    responses[category][subject][relation][img_type] = {"answers": {}, "questions": {}}

                responses[category][subject][relation][img_type]["answers"][q_type] = answ
                responses[category][subject][relation][img_type]["questions"][q_type] = q


            # save outputs
            for cat, cat_resps in responses.items():
                with open(os.path.join(out_dir, f"{cat}_answers.json"), "w") as f:
                    json.dump(cat_resps, f, indent=4)


def generate_api(
    model_name: str,
    data_folder: str,
    instruction: str = "",
    out_dir: str = "output",
    max_new_tokens: int = 15,
    seed: int = 42,
    experiment: str = "visual",
    #llm_only: bool = False,
    skip_existing: bool = False,
):
    res_folder = experiment
    # if llm_only:
    #     assert experiment == "text_only"
    #     res_folder = f"llm-{experiment}"
    out_dir = os.path.join(out_dir, model_name, res_folder)

    if skip_existing and os.path.exists(os.path.join(out_dir, "organizations_answers.json")):
        print(f"Results for {model_name} {res_folder} exist. Skipping ... ")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # get the model and processor based on the model name
    map_models = MAP_MODELS

    # if llm_only:
    #     map_models = MAP_MODELS_LLM_ONLY

    model_name, Model, Collator = map_models[model_name]

    model = None

    if model_name == "gemini-2-5-flash":
        model = Model(
            model_name,
            seed=seed,
            )
    else:
        model = Model(
            model_name,
        )

    # set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_ds = None

    # load test data; use b64 encoding for gpt models
    if model_name.startswith("gpt"):
        test_ds = DyKnowDataset(data_folder, experiment, b64_encode=True)
    else:
        test_ds = DyKnowDataset(data_folder, experiment)

    collator = Collator(
        instruction=instruction,
    )

    responses = {}

    #generate
    for sample in tqdm(
        test_ds, desc=f"Generating responses"
    ):
        input_id , question, sample_id = collator(*sample)
        generated_answer = model.generate(
            input_id,
            max_new_tokens=max_new_tokens,
        )

        category, subject, relation, q_type, img_type = sample_id.split(SAMPLE_IDS_SEP)

        if category not in responses:
            responses[category] = {}
        if subject not in responses[category]:
            responses[category][subject] = {}

        if relation not in responses[category][subject]:
            responses[category][subject][relation] = {}

        if img_type not in responses[category][subject][relation]:
            responses[category][subject][relation][img_type] = {"answers": {}, "questions": {}}

        responses[category][subject][relation][img_type]["answers"][q_type] = generated_answer
        responses[category][subject][relation][img_type]["questions"][q_type] = question


        # save outputs
        for cat, cat_resps in responses.items():
            with open(os.path.join(out_dir, f"{cat}_answers.json"), "w") as f:
                json.dump(cat_resps, f, indent=4)


def main():
    args = get_args()

    if args.model_name in LIST_API_MODELS:

        generate_api(
            model_name=args.model_name,
            data_folder=args.data,
            instruction=args.prompt,
            # batch_size=args.batch_size,
            out_dir=args.out_dir,
            max_new_tokens=args.max_length,
            # seed: int = 42,
            experiment=args.experiment,
            #llm_only=args.llm_only,
            skip_existing=args.skip_existing,
        )

    else:
        generate(
            model_name=args.model_name,
            data_folder=args.data,
            instruction=args.prompt,
            # batch_size=args.batch_size,
            out_dir=args.out_dir,
            max_new_tokens=args.max_length,
            parallel=args.parallel,
            device=args.device,
            # seed: int = 42,
            experiment=args.experiment,
            llm_only=args.llm_only,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
