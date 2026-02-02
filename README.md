# CodeOCR: On the Effectiveness of Vision Language Models in Code Understanding

This repository contains the official implementation for the paper **"Seeing is Coding: On the Effectiveness of Vision Language Models in Code Understanding"**.

## Introduction

Large Language Models (LLMs) have achieved remarkable success in source code understanding, yet as software systems grow in scale, computational efficiency has become a critical bottleneck. This paper explores the feasibility of representing source code as rendered images to optimize efficiency through "optical compression". We evaluate state-of-the-art MLLMs across multiple downstream tasks to demonstrate the effectiveness of this paradigm.

## Project Structure

```text
CodeOCR/
├── CodeOCR/                  # Code rendering tool (modular structure)
│   ├── __init__.py           # Main entry, exports public API
│   ├── api.py                # Simplified API (render_code_to_images, render_and_query)
│   ├── client.py             # LLM client (OpenAI/Azure)
│   ├── demo.py               # CLI demo tool
│   ├── render.py             # Full CLI tool
│   ├── core/                 # Core modules
│   │   ├── constants.py      # Constants
│   │   ├── fonts.py          # Font handling
│   │   ├── text_processing.py # Text preprocessing
│   │   ├── syntax.py         # Syntax highlighting
│   │   ├── rendering.py      # Image rendering
│   │   ├── layout.py         # Layout optimization
│   │   ├── compression.py    # Image compression
│   │   └── tokens.py         # Token calculation
│   └── sample.txt            # Sample code file
├── downstream/               # Downstream tasks (RQ1-RQ4)
│   ├── tasks/                # Task implementations
│   ├── dataset/              # Datasets
│   ├── run_pipeline.py       # Main entry
│   └── llm_utils.py          # LLM utilities
├── reconstruction/           # Code reconstruction task (RQ5)
│   ├── run.py                # Main entry
│   ├── pipeline/             # OCR and evaluation modules
│   └── dataset/              # Datasets
├── README.md
└── requirements.txt
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

**Option 1: Environment Variables (Recommended)**

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"  # Optional, defaults to OpenRouter
```

**Option 2: Config File**

Create a `config.json` file in the `downstream` directory:

```json
{
    "api_key": "your_api_key",
    "base_url": "https://openrouter.ai/api/v1"
}
```

For Azure OpenAI, add:

```json
{
    "api_key": "your_azure_key",
    "azure_endpoint": "https://your-resource.openai.azure.com",
    "azure_api_version": "2024-03-01-preview"
}
```

## Usage

### Quick Start (Demo)

```bash
# Render code to image
python -m CodeOCR.demo render --file example.py -o output.png

# Query LLM with code image
python -m CodeOCR.demo query --file example.py -i "Explain this code"

# End-to-end: Code -> Image -> OCR -> Evaluate
python -m CodeOCR.demo ocr --file example.py

# E2E with custom ratios (render only, no API call)
python -m CodeOCR.demo ocr --file example.py --ratios "1,2,4" --render-only
```

### Python API

```python
from CodeOCR import render_code_to_images, call_llm_with_images, create_client

code = "def hello():\n    print('world')"

# Step 1: Render code to images
images = render_code_to_images(code, language="python", theme="modern")
images[0].save("output.png")

# Step 2: Send images to LLM
client = create_client()
response, token_info = call_llm_with_images(
    client,
    model_name="gpt-5-mini",
    images=images,
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain this code in the image.",
)
print(response)
```

### Advanced CLI

```bash
# Use full CLI tool
python -m CodeOCR.render --code-context-file example.py --instruction "Explain code" --model gpt-4o --enable-syntax-highlight
```

**Parameters:**
- `--code-context-file`: Code file path
- `--instruction`: Instruction
- `--model`: Model name (default: gpt-4o)
- `--enable-syntax-highlight`: Enable syntax highlighting
- `--theme`: Theme (light/modern)
- `--language`: Code language (default: python)

### RQ1-RQ4: Downstream Tasks

Navigate to the `downstream` directory to run experiments for Code Completion, Code QA, Clone Detection, and Summarization.

```bash
cd downstream
```

**Code Completion (RAG)**
> python dataset
```bash
python -u run_pipeline.py --run-tasks --task code_completion_rag --models gpt-5-mini --resize-mode --preserve-newlines
```
> java dataset
```bash
python -u run_pipeline.py --run-tasks --task code_completion_rag --models gpt-5-mini --resize-mode --preserve-newlines --language java --dataset_path microsoft/LCC_Java
```

**Code Question Answering (QA)**
```bash
python -u ./run_pipeline.py --run-tasks --task code_qa --models gpt-5-mini --resize-mode --preserve-newlines
```

**Code Clone Detection**
> Append `--language java` (or another language) to command below to switch target language.
```bash
python -u ./run_pipeline.py --run-tasks --task code_clone_detection --models gpt-5-mini --resize-mode --preserve-newlines --code-clone-detection-separate-mode
```

**Code Summarization**
```bash
python -u run_pipeline.py --run-tasks --task code_summarization --models gpt-5-mini --resize-mode --preserve-newlines
```

### RQ5: Code Reconstruction

Run all reconstruction experiments:

```bash
cd reconstruction
python run.py
```

This runs the full pipeline: fetch code from dataset → render images → OCR → evaluate.

