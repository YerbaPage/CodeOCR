# CodeOCR: On the Effectiveness of Vision Language Models in Code Understanding

This repository contains the official implementation for the paper **"Seeing is Coding: On the Effectiveness of Vision Language Models in Code Understanding"**.

## Introduction

Large Language Models (LLMs) have achieved remarkable success in source code understanding, yet as software systems grow in scale, computational efficiency has become a critical bottleneck. This paper explores the feasibility of representing source code as rendered images to optimization efficiency through "optical compression". We evaluate state-of-the-art MLLMs across multiple downstream tasks to demonstrate the effectiveness of this paradigm.

## Project Structure

```text
CodeOCR/
├── downstream/             # Code for downstream tasks (RQ1-RQ4)
│   ├── standalone          # Code Clone Detection dataset
│   ├── tasks/              # Task-specific implementations
│   ├── run_pipeline.py     # Main entry point for experiments
│   ├── text_to_image.py    # Code rendering utility
│   ├── llm_utils.py        # LLM API client and utilities
│   └──qa_dataset_test_no_comments.json # QA dataset
├── reconstruction/         # Code for reconstruction task (RQ5)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Hardware Requirements

*   **GPU**: A GPU (NVIDIA with CUDA) is recommended for running local embedding models (`microsoft/unixcoder-base`) and Qwen tokenizer/processor.
*   **VRAM**: At least 4GB VRAM is suggested for the embedding model.
*   **Disk**: Sufficient space for storing rendered images and datasets.

## Installation

Please ensure you have Python >= 3.10 installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. **Code Completion**: Download [microsoft/LCC_python](https://huggingface.co/datasets/microsoft/LCC_python) (Python) and [microsoft/LCC_Java](https://huggingface.co/datasets/microsoft/LCC_Java) (Java) from Hugging Face.
2. **Code QA**: The dataset is already included (`qa_dataset_test_no_comments.json`).
3. **Code Clone Detection**: Follow [GPTCloneBench](https://github.com/srlabUsask/GPTCloneBench?tab=readme-ov-file) instructions and place the `standalone` folder under the `downstream` directory.
4. **Code Summarization**: Download [JetBrains-Research/lca-module-summarization](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization) from Hugging Face.

## Configuration

To run the experiments using LLM/VLM APIs, you need to configure the API credentials.

1.  Create a `config.json` file in the `downstream` directory (or set environment variables).
2.  Example `config.json`:
    ```json
    {
        "api_key": "your_api_key",
        "base_url": "https://api.openai.com/v1",
        "azure_endpoint": "",
        "azure_api_version": ""
    }
    ```
    *   Alternatively, set `OPENAI_API_KEY` environment variable.
    *   Default `base_url` in code points to a proxy; change it if using official OpenAI/Azure endpoints.

## Usage

### Code Transformation Tool

To render a text instruction as images and send it to OpenAI API with a specific compression ratio (e.g., 4x). Instruction is passed in, and code context can be passed via flag or file:

```bash
# Run from the project root
cat CodeOCR/sample.txt | python CodeOCR/render_instruction_to_openai.py --resize-ratios 4 --code-context-file CodeOCR/sample.txt
```

### RQ1-RQ4: Downstream Tasks

Navigate to the `downstream` directory to run experiments for Code Completion, Code QA, Clone Detection, and Summarization.

```bash
cd downstream
```

**Code Completion (RAG)**
> python dataset
```bash
python -u run_pipeline.py --run-tasks --task code_completion_rag --models glm-4.6v --resize-mode --preserve-newlines
```
> java dataset
```bash
python -u run_pipeline.py --run-tasks --task code_completion_rag --models glm-4.6v --resize-mode --preserve-newlines --language java --dataset_path microsoft/LCC_Java
```

**Code Question Answering (QA)**
```bash
python -u ./run_pipeline.py --run-tasks --task code_qa --models glm-4.6v --resize-mode --preserve-newlines
```

**Code Clone Detection**
> Append `--language java` (or another language) to command below to switch target language.
```bash
python -u ./run_pipeline.py --run-tasks --task code_clone_detection --models glm-4.6v --resize-mode --preserve-newlines --code-clone-detection-separate-mode
```

**Code Summarization**
```bash
python -u run_pipeline.py --run-tasks --task code_summarization --models glm-4.6v --resize-mode --preserve-newlines
```

### RQ5: Code Reconstruction

Navigate to the `reconstruction` directory and configure the environment variables to run the reconstruction experiments.

```bash
cd reconstruction
```

**Configuration & Execution**
```bash
export USE_EXISTING_IMAGES="1"
export EXISTING_IMAGES_DIR="./dataset/images"
export DATASET_FILENAME="dataset.json"
export RUN_MODULE_3="1"
export RUN_MODULE_4="1"
export OCR_CONCURRENCY="4"
export OCR_PARALLEL_MIN_INTERVAL_SECONDS="0"
export GEMINI_ENABLE_SAFETY_SETTINGS="1"
export OCR_PROMPT_PERSONAL_OFFLINE="1"

python run_gemini.py
```

