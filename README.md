# V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models

Official repository for the paper [V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models](https://arxiv.org/abs/2603.16581)

- [Introduction](#introduction)
- [Usage](#usage)
  - [Installation](#installation)
  - [Structure](#structure)
  - [Dataset Construction](#dataset-construction)
  - [Benchmarking a Model with Visual DyKnow](#benchmarking-a-model-with-visual-dyknow)
    - [Can I Add Another Model?](#can-i-add-another-model)
  - [Analyzing Answers](#analyzing-answers)
  - [Editing a Model](#editing-a-model)
    - [Computing the Performance of the Edited Model](#computing-the-performance-of-the-edited-model)
  - [Testing RAG](#testing-rag)
  - [Analyzing the MolMo Training Dataset](#analyzing-the-molmo-training-dataset)
- [License](#license)
- [How to Cite](#how-to-cite)

---

## Introduction

Vision-Language Models (VLMs) are trained on data snapshots of documents including images and texts. Their training data and evaluation benchmarks are typically static, implicitly treating factual knowledge as time-invariant. However, real-world facts are intrinsically time-sensitive and subject to erratic and periodic changes, causing model predictions to become outdated. We present V-DyKnow, a Visual Dynamic Knowledge benchmark for evaluating time-sensitive factual knowledge in VLMs. Using V-DyKnow, we benchmark closed- and open-source VLMs and analyze *a)* the reliability (correctness and consistency) of model responses across modalities and input perturbations; *b)* the efficacy of knowledge editing and multi-modal RAG methods for knowledge updates across modalities; and *c)* the sources of outdated predictions, through data and mechanistic analysis. Our results show that VLMs frequently output outdated facts, reflecting outdated snapshots used in the (pre-)training phase. Factual reliability degrades from textual to visual stimuli, even when entities are correctly recognized. Besides, existing alignment approaches fail to consistently update the models' knowledge across modalities. Together, these findings highlight fundamental limitations in how current VLMs acquire and update time-sensitive knowledge across modalities. We release the benchmark, code, and evaluation data.

---

## Usage

This section describes how to install the environment and use the code to reproduce the results obtained in the paper.

### Installation

Use the following command to clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/sislab-unitn/V-DyKnow.git
```

Run the following command to create the conda environment:

```bash
conda create -n vdyknow python=3.10.13
```

Install the requirements using the following command:

```bash
pip install -r requirements.txt
```

---

### Structure

```shell
.
├── data                      # V-DyKnow questions + answers + Python script to download the images
├── models_editing            # Model editing utilities and results
│   ├── EasyEdit
│   ├── editing_datasets      # Outdated questions for each model
│   ├── edit_model.py
│   ├── error_analysis.py
│   ├── generate_editing_dataset.py
│   ├── hparams               # Hyper-parameters for editing
│   ├── results               # Editing results
│   └── utils.py
├── models_output
│   ├── analysis/             # Analysis of the model outputs
│   ├── analysis.py
│   ├── analyze_detection.py
│   ├── analyze_replies.py
│   ├── dataset.py
│   ├── generate_answers.py
│   ├── get_outdated_questions.py
│   ├── models/               # Code for each model
│   ├── results/              # Model outputs to DyKnow
│   └── utils.py
├── molmo_analysis
│   ├── analyze_molmo.py
│   ├── data/
│   ├── dolma/
│   ├── download_wikidata_dolma.sh
│   ├── generate_results.py
│   ├── get_wikipedia_passages.ipynb
│   ├── results/
│   ├── retrieve_molmo_wikipedia_pages.py
│   ├── sample_data.py
│   └── utils.py
├── RAG
│   ├── data/
│   ├── qwen3_vl_scripts
│   ├── rag.py                # Execute RAG
│   ├── results/              # Results
│   └── utils.py
├── README.md
└── requirements.txt
```

---

### Dataset Construction

The benchmark is built on Wikidata and evolves over time. To ensure reproducibility of our results, we rely on the November 2025 snapshot, provided at `data/annotations/wikidata_combined.json`.

To download and format the images linked to the annotations:

**1. Navigate to the image scripts folder:**

```bash
cd data/imgs/script/
```

**2. Download the images:**

```bash
python download_wikimedia_imgs.py
```

**3. Resize the images to the fixed dimension of 672×672:**

```bash
python resize_imgs.py
```

---

### Benchmarking a Model with Visual DyKnow

To evaluate a model, navigate to the root folder and run the `generate_answers` script:

```bash
python models_output/generate_answers.py MODEL --experiment EXPERIMENT_TYPE --out-dir OUTPUT_DIR
```

- `MODEL`: The name of the target model (e.g., `qwen2-vl-7b`).
- `EXPERIMENT_TYPE`: Choose from `text_only`, `visual`, or `detection`.

**Example:** To query `qwen2-vl-7b` using the visual experiment and save the results in a `results` folder, run:

```bash
python models_output/generate_answers.py qwen2-vl-7b --experiment visual --out-dir results
```

This will generate the following directory structure:

```text
results
└── qwen2-vl-7b
    └── visual
          ├── athletes_answers.json
          ├── countries_answers.json
          └── organizations_answers.json
```

> **Tip:** To see a list of all available models or optional parameters, run:
> ```bash
> python models_output/generate_answers.py --help
> ```

---

#### Can I Add Another Model?

Yes! You can add a custom model collator by creating or duplicating a Python file in the `models_output/models` directory (e.g., refer to `models_output/models/qwen2_vl.py`).

Once you have created the collator function, you can:

**1. Import your model collator module into `generate_answers.py`.**

**2. Add the model definition to the `MAP_MODELS` dictionary inside `generate_answers.py`:**

```python
"qwen2-vl-7b": (
    "Qwen/Qwen2-VL-7B-Instruct",              # Identifier of the VLM model
    partial(                                    # PyTorch class used to load the model and all its parameters
        Qwen2VLForConditionalGeneration.from_pretrained,
        torch_dtype=torch.bfloat16,
    ),
    qwen2_vl.GenerationCollator,               # The collator you created in models_output/models
),
```

If you also want to test the LLM backbone of the VLM, you can create a custom function for the LLM and add the same information to `MAP_MODELS_LLM_ONLY`. In this way it is possible to test and compare `text_only` question performance between the VLM and the LLM to see the differences.

---

### Analyzing Answers

To understand if a model's answers are correct (up-to-date), outdated, or irrelevant, you must first generate the answers as described in the section above.

Once generated, run the analysis script:

```bash
python models_output/analyze_replies.py MODEL_RESULTS --question-path QUESTION_DIR
```

**Example:**

```bash
python models_output/analyze_replies.py ./models_output/results/qwen2-vl-7b/visual/ --question-path ./data/annotations/wikidata_combined.json
```

This will generate three new analysis files (`_analysis.json`, `_answers_sheet.json`, `_answers_dates.json`) for every category alongside your original output files:

```text
results
└── qwen2-vl-7b
    └── visual
          ├── athletes_answers.json
          ├── athletes_analysis.json        (NEW)
          ├── athletes_answers_sheet.json   (NEW)
          ├── athletes_answers_dates.json   (NEW)
          ...
```

If you want to analyze all the results with a single script, run:

```bash
python models_output/analysis.py
```

This script will not only generate the results but also an `analysis/` folder containing aggregated outputs.

---

### Editing a Model

**1. Environment Setup:** Navigate to the `models_editing/EasyEdit` folder and install the requirements:

```bash
cd models_editing/EasyEdit
conda create -n EasyEdit python=3.10 -y
conda activate EasyEdit
pip install -r requirements.txt
```

**2. Prerequisite — Benchmark the Model:** Before applying any edits, you should benchmark the model using Visual DyKnow (see the [Benchmarking a Model with Visual DyKnow](#benchmarking-a-model-with-visual-dyknow) section).

**3. Generate the Editing Dataset:** Navigate to the `V-DyKnow` root folder and generate the dataset for your specific model:

```bash
python models_editing/generate_editing_dataset.py --model-name MODEL
```

**Example:**

```bash
python models_editing/generate_editing_dataset.py --model-name qwen2-vl-7b
```

With this command, the file `editing_datasets/qwen2-vl-7b/editing_dataset.json` will be created inside the `models_editing` folder.

**4. Extract the Locality Dataset:** This is needed for the various editing algorithms. Run using the `vdyknow` environment:

```bash
python models_editing/utils.py
```

**5. Apply the Edit:** Edit the model using your generated dataset and an `EDITING_METHOD` (e.g., WISE):

```bash
python models_editing/edit_model.py EDITING_METHOD HPARAM_FILE EDITING_DATASET_PATH
```

**Example (editing Qwen2-VL with WISE):**

```bash
python models_editing/edit_model.py WISE models_editing/hparams/WISE/qwen2vl-7b.yaml models_editing/editing_datasets/qwen2-vl-7b/editing_dataset.json
```

---

#### Computing the Performance of the Edited Model

After performing the edits, the model will generate new answers to the outdated questions in your output directory. Calculate the harmonic mean and overall performance of the edited model by running (using the `vdyknow` environment):

```bash
python -m models_editing.error_analysis MY_RESULTS_PATH
```

---

### Testing RAG

To test Retrieval-Augmented Generation (RAG) approaches, you first need to prepare the retrieval data.

**1. Retrieve Wikipedia Data:** Run `RAG/data/annotations/get_wikipedia_pages.py` to fetch web pages related to the subject entities:

```bash
python RAG/data/annotations/get_wikipedia_pages.py
```

**2. Extract Passages:** Run the `get_wikipedia_passages.ipynb` notebook to extract the specific text passages that the RAG system will use.

**3. Prepare Images:** Just like the main dataset, navigate to `RAG/data/imgs/script/` and run `download_wikimedia_imgs.py` followed by `resize_imgs.py` to ensure all retrieved images are the correct dimensions:

```bash
cd RAG/data/imgs/script/
python download_wikimedia_imgs.py
python resize_imgs.py
```

**4. Run the RAG Benchmark:** Where `QUERY_DATASET_PATH` is the path to the dataset used to query the VLM:

```bash
python RAG/rag.py MODEL QUERY_DATASET_PATH
```

**Example (RAG with qwen2-vl-7b):**

```bash
python RAG/rag.py qwen2-vl-7b models_editing/editing_datasets/qwen2-vl-7b/editing_dataset.json
```

For additional RAG parameters, run:

```bash
python RAG/rag.py --help
```

---

### Analyzing the MolMo Training Dataset

**1. Download the Wikidata section of Dolma:**

```bash
bash molmo_analysis/download_wikidata_dolma.sh
```

**2. Sample 30 (category, entity, property) combinations from the MolMo data** (10 correct, 10 outdated, and 10 irrelevant):

```bash
python molmo_analysis/sample_data.py
```

**3. Retrieve the Wikipedia pages** corresponding to the entities in the sampled Wikidata questions and save them for further analysis:

```bash
python molmo_analysis/retrieve_molmo_wikipedia_pages.py
```

**4. Extract the passages** from the Wikipedia pages by running the `get_wikipedia_passages.ipynb` notebook.

**5. Get the results of this analysis:**

```bash
python molmo_analysis/generate_results.py
```

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## How to Cite

```bibtex
@article{mousavi-etal-2026-v-dyknow,
  title={V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models},
  author={Mousavi, Seyed Mahed and Moiola, Christian and Rizzoli, Massimo and Alghisi, Simone and Riccardi, Giuseppe},
  journal={arXiv preprint arXiv:2603.16581},
  year={2026}
}
```
