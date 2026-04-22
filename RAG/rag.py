import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from urllib.parse import unquote
from argparse import Namespace
from typing import List, Dict, Tuple
from qwen_vl_utils import process_vision_info
from utils import load_json, save_json, MODELS
from qwen3_vl_scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from qwen3_vl_scripts.qwen3_vl_reranker import Qwen3VLReranker
from transformers import (
    AutoProcessor,
    PreTrainedModel,
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
        prog="python -m rag",
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        type=str,
        choices=[model for model in MODELS],
        help="Name of the model to be used.",
    )
    parser.add_argument(
        "query_dataset_path",
        metavar="QUERY_DATASET_PATH",
        type=str,
        help="Dataset path to be used for querying the VLM model.",
    )
    parser.add_argument(
        "--retrieve-k",
        metavar="K1",
        type=int,
        default=3,
        help="Number of documents to retrieve for each query. (retrieval)",
    )
    parser.add_argument(
        "--context-k",
        metavar="K2",
        type=int,
        default=1,
        help="Number of documents to use as context for generation. (reranking)",
    )
    parser.add_argument(
        "--rag-annotations-path",
        metavar="RAG_ANNOTATIONS_PATH",
        type=str,
        default="RAG/data/annotations/final_refined_passages_with_images.json",
        help="Path to the RAG dataset annotations.",
    )
    parser.add_argument(
        "--rag-images-path",
        metavar="RAG_IMAGES_PATH",
        type=str,
        default="RAG/data/imgs/resized/",
        help="Path to the RAG dataset images.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only.",
        help="Prompt for Q&A generation.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use multiple GPUs (device='auto').",
    )
    parser.add_argument(
        "--gold-documents",
        action="store_true",
        default=False,
        help="Use gold documents as retrieved documents.",
    )
    # parse arguments
    parsed_args = parser.parse_args()

    if not os.path.exists(parsed_args.query_dataset_path):
        raise FileNotFoundError(f"Query dataset path {parsed_args.query_dataset_path} does not exist.")
    
    if not os.path.exists(parsed_args.rag_annotations_path):
        raise FileNotFoundError(f"RAG annotations path {parsed_args.rag_annotations_path} does not exist.")

    if not os.path.exists(parsed_args.rag_images_path):
        raise FileNotFoundError(f"RAG images path {parsed_args.rag_images_path} does not exist.")
    
    if parsed_args.model_name == "qwen2-vl-7b":
        hf_model_name = "Qwen/Qwen2-VL-7B-Instruct"

        parsed_args.model_class = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if parsed_args.parallel else parsed_args.device,
            low_cpu_mem_usage=True,
        )

        parsed_args.processor_class = AutoProcessor.from_pretrained(
            hf_model_name,
            min_pixels=100 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )

    elif parsed_args.model_name == "llava-1-5-7b":
        hf_model_name = "llava-hf/llava-1.5-7b-hf"

        parsed_args.model_class = LlavaForConditionalGeneration.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if parsed_args.parallel else parsed_args.device,
            low_cpu_mem_usage=True,
        )
        image_processor = LlavaImageProcessor.from_pretrained(
            hf_model_name, do_pad=True
        )
        parsed_args.processor_class = AutoProcessor.from_pretrained(
            hf_model_name, image_processor=image_processor
        )
    else:
        raise NotImplementedError(f"Model {parsed_args.model_name} not supported.")
   
    if not parsed_args.gold_documents:
        parsed_args.embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")
        parsed_args.reranker = Qwen3VLReranker("Qwen/Qwen3-VL-Reranker-2B", torch_dtype=torch.bfloat16)

    return parsed_args


def process_dataset(entry: Dict) -> List[Dict]:
    """
    Process the dataset in the format required for RAG.
    Parameters
    ----------
    entry: Dict
        Dataset entries.
    Returns
    -------
    List[Dict]
        Dataset processed.
    """
    
    query_dataset = []
    for entry_id, entry in tqdm(entry.items(), desc="Processing query dataset"):
        elem_query = {}

        elem_query["generic"] = entry["generic"]
        elem_query["contextualized"] = entry["contextualized"]
        elem_query["rephrased"] = entry["rephrased"]
        elem_query["target"] = entry["target"]
        elem_query["image"] = entry["image"]
        elem_query["id"] = entry_id

        query_dataset.append(elem_query)

    return query_dataset

def top_k_embeddings(retrieve_data: List[Dict], query, k: int) -> List[Dict]:
    documents = torch.cat([elem["embedding"] for elem in retrieve_data], dim=0)

    query = query.squeeze(0)

    if torch.is_tensor(query):
        query = query.cpu().numpy()
    if torch.is_tensor(documents):
        documents = documents.cpu().numpy()
    similarities = query @ documents.T
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_documents = [retrieve_data[i] for i in top_k_indices]
    return top_k_documents

def top_k_reranking(documents: List[Dict], query: Dict, k: int, reranker: Qwen3VLReranker) -> List[Dict]:

    docs = []
    for doc in documents:
        if doc["image"] is not None:
            docs.append({"text": doc["text"], "image": doc["image"]})
        else:
            docs.append({"text": doc["text"]})

    reranker_inputs = {
        "instruction": "Rerank the following documents based on their relevance to the query.",
        "query": query,
        "documents": docs,
        "fps": 1.0,
    }
    ranks_scores = reranker.process(reranker_inputs)
    indexes = np.argpartition(ranks_scores, -k)[-k:]
    top_k_documents = [documents[i] for i in indexes]
    return top_k_documents

def encode_input(
        model_name: str,
        tok: AutoProcessor,
        prompt: str,
        image: str,
        retrieved_docs: List[Dict],
) -> Dict[str, Dict[str, List]]:
    """
    Encode inputs for model editing.
    Parameters
    ----------
    model_name: str
        Name of the model.
    tok: AutoProcessor
        Processor for the model.
    id: str
        ID corresponding to the prompt/target/image.
    prompt: str
        Prompt for testing the model.
    target: str
        Target outputs for the prompts.
    images: List[str]
        Paths to the images corresponding to the prompts.
    Returns
    -------
    Dict[str, Dict[str, List]]
        Encoded inputs for each type of prompt.
    """
    input_ids = None
    context = []

    context.append({"type": "text", "text": "Retrieved documents:"})
    for doc in retrieved_docs:
        if doc["image"] is not None:
            context.append({"type": "image", "image": Image.open(doc["image"])})
        context.append({"type": "text", "text": f"{doc['text']}"})

    context.append({"type": "image", "image": Image.open(image)})
    context.append({"type": "text", "text": f"Question: {prompt}"})


    messages = [
            {
                "role": "user",
                "content": context,
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

    elif model_name == "llava-1-5-7b":
        input_ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported for encoding the inputs.")
            
    return input_ids

def model_generate(
    model_name: str,
    model: PreTrainedModel,
    tok: AutoProcessor,
    prompt: str,
    image: str,
    retrieved_docs: List[Dict],
) -> str: 
    """
    Generate the answer using the documents retrieved.
    Parameters
    model_name: str
        Name of the model.
    model: PreTrainedModel
        Model for generation.
    tok: AutoProcessor
        Processor for the model.
    prompt: str
        Prompt for testing the model.
    target: str
        Target outputs for the prompts.
    image: str
        Path to the image corresponding to the prompt.
    Returns
    -------
    str
        Generated answer.
    """
    encoded_inputs = encode_input(model_name, tok, prompt, image, retrieved_docs)

    with torch.no_grad():
        input_ids = encoded_inputs.to(model.device)
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
        
    return generated_answers[0]

def embedd_dataset(entry: Dict, path_images: str, embedder: Qwen3VLEmbedder) -> List[Dict]:
    """
    Embedd the dataset in the format required for RAG.
    Parameters
    entry: Dict
        Dataset entries.
    path_images: str
        Path to the images folder.
    embedder: Qwen3VLEmbedder
        Embedder model.
    Returns
    -------
    List[Dict]
        Dataset processed with embeddings.
    """
    retrieve_dataset = []
    for entry_id, entry in tqdm(entry.items(), desc="Processing RAG dataset"):
        elem_retrieve = {}
        img_filename = ""

        elem_retrieve["image_url"] = unquote(entry["image_url"]) if entry["image_url"] else None
        
        if elem_retrieve["image_url"] is not None:
            img_filename = elem_retrieve["image_url"].split("/")[-1].rsplit(".", 1)[0]

            for ext in [".png", ".jpg", ".JPG"]:
                if os.path.exists(os.path.join(path_images, img_filename + ext)):
                    img_filename = img_filename + ext
                    break
            
            if not os.path.exists(os.path.join(path_images, img_filename)):
                raise FileNotFoundError(f"Image {img_filename} not found in {path_images}.")

        elem_retrieve["text"] = entry["text"]
        elem_retrieve["image"] = os.path.join(path_images, img_filename) if elem_retrieve["image_url"] is not None else None
        elem_retrieve["id"] = entry_id

        # If the embedder is None, it means we are using gold documents, so we skip the embedding step
        if embedder is not None:
            elem_retrieve["embedding"] = embedder.process([{"text": elem_retrieve["text"], "image": elem_retrieve["image"]}]) if elem_retrieve["image"] is not None else embedder.process([{"text": elem_retrieve["text"]}])

        retrieve_dataset.append(elem_retrieve)

    return retrieve_dataset

def select_dataset(entry: Dict, keys: List[str]) -> Dict:
    """
    Select the dataset based on the model name.
    Parameters
    ----------
    entry: Dict
        Dataset entries.
    keys: List[str]
        Dataset indexes to select from the RAG Dataset.
    Returns
    -------
    Dict
        Selected dataset.
    """

    ret_dataset = {}

    print("Complete RAG dataset size:", len(entry.keys()))
    print("Number of entries in the query dataset:", len(keys))

    keys = [key.rsplit("|", 1)[0] for key in keys]
    keys = list(set(keys))

    for key in keys:
        if key in entry:
            ret_dataset[key] = entry[key]
        else:
            # If the key is not found in the RAG dataset we select the entry with the same category and element
            category, element, _ = key.split("|")
            matching_keys = [k for k in entry if k.startswith(f"{category}|{element}")]

            assert len(matching_keys) <= 1, f"Expected at most 1 matching key for {key}, but got {len(matching_keys)}. Matching keys: {matching_keys}"
            
            ret_dataset[key] = entry[matching_keys[0]]

            print(f"Key {key} not found in RAG dataset. Selected key {matching_keys[0]} with category {category} and element {element} instead.")
    
    print(f"Selected dataset size: {len(ret_dataset.keys())}")
    return ret_dataset


def main(args: Namespace):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    query_dataset = load_json(args.query_dataset_path)

    query_data = process_dataset(query_dataset)

    rag_dataset = load_json(args.rag_annotations_path)

    indexs_query_dataset = list(query_dataset.keys())
    rag_dataset = select_dataset(rag_dataset, indexs_query_dataset)

    rag_data = None

    if args.gold_documents:
        rag_data = embedd_dataset(rag_dataset, args.rag_images_path, None)
    else:
        rag_data = embedd_dataset(rag_dataset, args.rag_images_path, args.embedder)

    results = {}
    for elem in tqdm(query_data, desc="Processing queries"):
        for key in ["generic", "contextualized", "rephrased"]:
            query = {"text": elem[key], "image": elem["image"]}
            
            reranked_documents = None

            if not args.gold_documents:
                query_embedding = args.embedder.process([query])

                top_k_documents = top_k_embeddings(
                    retrieve_data=rag_data, 
                    query=query_embedding, 
                    k=args.retrieve_k,
                )

                reranked_documents = top_k_reranking(
                    documents=top_k_documents, 
                    query=query, 
                    k=args.context_k, 
                    reranker=args.reranker
                )
            else:
                reranked_documents = [data for data in rag_data if elem["id"].startswith(data["id"])]
                assert len(reranked_documents) == 1, f"Expected 1 document for query {elem['id']} with prompt type {key}, but got {len(reranked_documents)}. Reranked documents: {[doc['id'] for doc in reranked_documents]}"

            prompt = f"{args.prompt} {elem[key]}"

            output = model_generate(
                model_name=args.model_name,
                model=args.model_class,
                tok=args.processor_class,
                prompt=prompt,
                image=elem["image"],
                retrieved_docs=reranked_documents,
            )
            if elem["id"] not in results:
                results[elem["id"]] = {"answers": {}, "targets": {}, "questions": {}}

            results[elem["id"]]["answers"][key] = output
            results[elem["id"]]["questions"][key] = prompt
        results[elem["id"]]["targets"] = elem["target"]
    
    if not os.path.exists(f"RAG/{args.out_dir}/RAG/{args.model_name}/generation_results"):
        os.makedirs(f"RAG/{args.out_dir}/RAG/{args.model_name}/generation_results")
    
    save_json(
        f"RAG/{args.out_dir}/RAG/{args.model_name}/generation_results/results_retrieve_{args.retrieve_k}_context_{args.context_k}_gold_{args.gold_documents}.json",
        results,
    )

    params_to_save = {
        "model_name": args.model_name,
        "seed": seed,
        "prompt": args.prompt,
        "retrieve_k": args.retrieve_k,
        "context_k": args.context_k,
        "gold_documents": args.gold_documents,
        "number_entries_complete_rag_dataset": len(rag_dataset),
        "number_entries_selected_rag_dataset": len(rag_data),
    }
    
    save_json(
        f"RAG/{args.out_dir}/RAG/{args.model_name}/generation_results/params_retrieve_{args.retrieve_k}_context_{args.context_k}_gold_{args.gold_documents}.json",
        params_to_save,
    )
    
if __name__ == "__main__":
    args = get_args()
    main(args)
