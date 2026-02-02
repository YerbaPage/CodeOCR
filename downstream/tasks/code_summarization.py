import os
import json
import torch
import logging
import re
import random
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from openai import OpenAI
from PIL import Image as PIL_Image
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_utils import (
    create_client,
    build_folder,
    call_llm_with_images,
    call_llm_with_text_only,
    get_text_tokens,
    EVALUATOR_MODELS,
    get_config,
    call_llm_with_logit_bias,
)
from tasks.code_summarization.data import (
    load_code_summarization_data,
    prepare_code_context,
)

# Import image generation and compression tools
from text_to_image import (
    text_to_image,
    generate_compressed_images_dynamic,
    analyze_text_structure,
    calculate_image_tokens_from_paths,
    calculate_image_tokens_with_processor,
    COMPRESSION_RATIOS,
    get_all_modes,
    text_to_image_stream,
    optimize_layout_config_dry,
    find_closest_resolution_prefer_larger,
    get_expanded_resolution_list,
    calculate_image_tokens_qwen3,
    calculate_fill_rate,
)

def generate_one_text(row, code_context, client, model_name):
    """Generate single document summary using text mode."""
    # if not CODE_SUMMARIZATION_AVAILABLE:
    #     return ""

    intent = row["intent"]
    filename = row["docfile_name"]

    prompt = "I have code collected from one or more files joined into one string. "
    prompt += f"Using the code generate text for {filename} file with documentation about {intent}.\n\n"
    prompt += f"My code:\n\n{code_context}"
    prompt += f"\n\n\n\nAs answer return text for {filename} file about {intent}. Do not return the instruction how to make documentation, return only documentation itself."

    max_tokens = get_config()["summarization"].get("max_output_tokens", 4096)
    try:
        # Use calling method from llm_utils
        system_prompt = "You are a helpful assistant."
        generated_text, token_info = call_llm_with_text_only(
            client, model_name, system_prompt, prompt, max_tokens=max_tokens
        )
        return generated_text, token_info
    except Exception as e:
        print(f"Error generating documentation: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def generate_one_image(row, image_paths, client, model_name):
    """Generate summary using images."""
    intent = row["intent"]
    filename = row["docfile_name"]

    user_prompt = (
        "I have code collected from one or more files joined into one string. "
    )
    user_prompt += f"Using the code generate text for {filename} file with documentation about {intent}.\n\n"
    user_prompt += f"My codes are in the images."
    user_prompt += f"\n\n\n\nAs answer return text for {filename} file about {intent}. Do not return the instruction how to make documentation, return only documentation itself. Attention: If you are unsure or the code is unclear, do not make things up. Only describe what can be clearly summarized."
    system_prompt = (
        "You are a helpful assistant that generates documentation from code images."
    )

    max_tokens = get_config()["summarization"].get("max_output_tokens", 4096)
    try:
        generated_text, token_info = call_llm_with_images(
            client,
            model_name,
            image_paths,
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
        )
        return generated_text, token_info
    except Exception as e:
        print(f"Generation error: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_kwargs": {}}


def process_example_with_text(
    example_id: int,
    row,
    tokenizer,
    max_context_toks,
    client,
    model_name,
    output_dir: str,
    no_context_mode: bool = False,
) -> Dict:
    """Process a single example (text mode)."""
    code_context, text_tks = prepare_code_context(row, max_context_toks, tokenizer)
    if no_context_mode:
        code_context = ""
    generated_text, token_info = generate_one_text(
        row, code_context, client, model_name
    )

    # Get gold_doc (ground truth document)
    gold_doc = row.get("target_text", "")

    # Save results
    output_file = f"{output_dir}/example_{example_id:05d}_text_only.jsonl"

    result = {
        "id": example_id,
        "intent": row["intent"],
        "docfile_name": row["docfile_name"],
        "gold_doc": gold_doc,  # Add gold_doc to results
        "output": generated_text,
        "api_kwargs": token_info.get("api_kwargs", {}),
        "prompt": code_context,
        "tokens": token_info,
        "mode": "text_only",
        "compression_ratio": None,
        "resolution": "text_only",
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result, code_context, text_tks


def process_example_with_images(
    example_id: int,
    row,
    image_paths: List[str],
    client,
    model_name,
    output_dir: str,
    compression_ratio: Optional[float] = None,
    resolution: str = "2240x2240",
) -> Dict:
    """Process a single example (image mode)."""
    generated_text, token_info = generate_one_image(
        row, image_paths, client, model_name
    )

    # Get gold_doc (ground truth document)
    gold_doc = row.get("target_text", "")

    # Save results
    mode_suffix = (
        "image"
        if compression_ratio is None
        else f"image_ratio{compression_ratio}"  # same as get_all_modes() in text_to_image.py
    )
    output_file = f"{output_dir}/example_{example_id:05d}_{mode_suffix}.jsonl"

    # Calculate image token count
    image_tokens = calculate_image_tokens_from_paths(image_paths) if image_paths else 0

    result = {
        "id": example_id,
        "intent": row["intent"],
        "docfile_name": row["docfile_name"],
        "gold_doc": gold_doc,  # Add gold_doc to results
        "output": generated_text,
        "api_kwargs": token_info.get("api_kwargs", {}),
        "image_paths": image_paths,
        "tokens": token_info,
        "mode": "image",
        "compression_ratio": compression_ratio,
        "resolution": resolution,
        "image_tokens": image_tokens,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        # Clean up base64 image data in api_kwargs
        api_kwargs = token_info.get("api_kwargs", {})
        if "messages" in api_kwargs:
            new_messages = []
            for msg in api_kwargs["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        new_content = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                # Replace base64 data with placeholder
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": "base64_image_truncated"}
                                })
                            else:
                                new_content.append(item)
                        new_messages.append({**msg, "content": new_content})
                    else:
                        new_messages.append(msg)
                else:
                    new_messages.append(msg)
            api_kwargs["messages"] = new_messages

        result["api_kwargs"] = api_kwargs
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def run_code_summarization_task(
    model_name: str,
    output_dir: str,
    width: int = 2240,
    height: int = 2240,
    font_size: int = 40,
    line_height: float = 1.2,
    dpi: int = 300,
    font_path: str = None,
    preserve_newlines: bool = True,
    enable_syntax_highlight: bool = False,
    language: str = "python",
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    resize_mode: bool = False,
    processor=None,
    qwen_tokenizer=None,
    client_type: str = "OpenAI",
    num_examples: int = 139,
    test_single: bool = False,
    no_context_mode: bool = False,
    enable_bold: bool = False,
    extreme_mode: bool = False,
    enable_text_only_test: bool = True,
) -> List[Dict]:
    """Run code summarization task (supports both text and image modes)."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    lh_str = str(line_height).replace('.', '_')
    folder_parts = [
        f"code_completion_rag_{model_name.replace('/', '_slash_')}",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}"
    ]
    folder_kwargs = {
        "enable_syntax_highlight": enable_syntax_highlight,
        "preserve_newlines": preserve_newlines,
        "enable_bold": enable_bold,
    }
    output_dir = build_folder(
        output_dir,
        folder_parts,
        **folder_kwargs
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    dataset, tokenizer, config = load_code_summarization_data()
    if dataset is None:
        return []

    # Create client
    client = create_client(client_type)

    # Extract parameters
    max_context_toks = config.get("max_context_toks", None)

    # Filter dataset by relevant_code_context length
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    def count_tokens(text):
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Filter by code context field
    dataset = dataset.filter(
        lambda ex: count_tokens(ex.get("relevant_code_context", "")) <= max_context_toks,
        num_proc=4,
    )
    if 0 < num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))

    print(f"Code Summarization: Processing {len(dataset)} examples (max tokens: {max_context_toks})")
    all_ratios = get_all_modes()

    all_results = {
        "total": 0,
        "empty": 0,
        "prompt_tokens": {r: 0 for r in all_ratios},
        "completion_tokens": {r: 0 for r in all_ratios},
        "total_tokens": {r: 0 for r in all_ratios},
    }
    results_lock = Lock()

    lh_str = str(line_height).replace(".", "_")

    # Create image output directory
    images_base_dir = "./generated_images"
    images_folder_parts = [
        f"code_summarization",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
    ]
    images_dir = build_folder(
        images_base_dir,
        images_folder_parts,
        **folder_kwargs
    )
    os.makedirs(images_dir, exist_ok=True)

    def process_example(example_id, row):
        """Process all modes for a single example."""
        example_results = {
            "total": 0,
            "empty": 0,
            "prompt_tokens": {r: 0 for r in all_ratios},
            "completion_tokens": {r: 0 for r in all_ratios},
            "total_tokens": {r: 0 for r in all_ratios},
        }

        def summ_res(t, r):
            example_results["total"] += 1
            example_results["empty"] += (
                1 if len(r["output"]) == 0 or not r["output"] else 0
            )
            tks = r.get("tokens", {})
            example_results["prompt_tokens"][t] += tks.get("prompt_tokens", 0)
            example_results["completion_tokens"][t] += tks.get("completion_tokens", 0)
            example_results["total_tokens"][t] += tks.get("total_tokens", 0)

        # Initialize to avoid unassigned errors
        code_context = ""
        text_tks = 0

        # 1. Text mode
        if enable_text_only_test:
            try:
                text_result, code_context, text_tks = process_example_with_text(
                    example_id,
                    row,
                    tokenizer,
                    max_context_toks,
                    client,
                    model_name,
                    output_dir,
                    no_context_mode=no_context_mode,
                )
                summ_res("text_only", text_result)
            except Exception as e:
                print(f"  Example {example_id}: Text mode processing failed: {e}")
                if not no_context_mode:
                    try:
                        code_context, text_tks = prepare_code_context(
                            row, max_context_toks, tokenizer
                        )
                    except Exception as e2:
                        print(f"  Example {example_id}: Code context preparation failed: {e2}")
        elif not no_context_mode:
            try:
                code_context, text_tks = prepare_code_context(
                    row, max_context_toks, tokenizer
                )
            except Exception as e2:
                print(f"  Example {example_id}: Code context preparation failed: {e2}")

        # 2. Image mode
        if not no_context_mode:
            try:
                if resize_mode:
                    if extreme_mode:
                        text_structure = analyze_text_structure(code_context)

                        def renderer_func(w, h, fs):
                            margin = int(w * 0.01)
                            return text_to_image(
                                code_context,
                                width=w,
                                height=h,
                                font_size=fs,
                                line_height=line_height,
                                margin_px=margin,
                                dpi=dpi,
                                font_path=font_path,
                                preserve_newlines=preserve_newlines,
                                enable_syntax_highlight=enable_syntax_highlight,
                                filename=None,
                                language=language or "python",
                                should_crop_whitespace=should_crop_whitespace,
                                enable_two_column=enable_two_column,
                                enable_bold=enable_bold,
                            )

                        resized_results = generate_compressed_images_dynamic(
                            text_tks,
                            renderer_func,
                            compression_ratios=COMPRESSION_RATIOS,
                            text_structure=text_structure,
                            data_id=f"summary_{example_id}",
                        )

                        for ratio in COMPRESSION_RATIOS:
                            try:
                                resized_images, target_resolution, target_font_size = resized_results.get(
                                    ratio, ([], None, None)
                                )
                                if not resized_images:
                                    continue

                                resized_image_paths = []
                                for page_num, resized_img in enumerate(resized_images, 1):
                                    image_filename = f"summary_{example_id:05d}_ratio{ratio}_{target_resolution}x{target_resolution}_fs{target_font_size}_page_{page_num:03d}.png"
                                    image_path = os.path.join(images_dir, image_filename)
                                    resized_img.save(image_path)
                                    resized_image_paths.append(os.path.abspath(image_path))

                                compressed_result = process_example_with_images(
                                    example_id,
                                    row,
                                    resized_image_paths,
                                    client,
                                    model_name,
                                    output_dir,
                                    compression_ratio=ratio,
                                    resolution=f"{target_resolution}x{target_resolution}",
                                )
                                summ_res(f"image_ratio{ratio}", compressed_result)
                            except Exception as e:
                                print(
                                    f"  Example {example_id}: Compression ratio {ratio} processing failed: {e}"
                                )
                    else:
                        text_structure = analyze_text_structure(code_context)
                        res_1x, fs_1x, _ = optimize_layout_config_dry(
                            target_tokens=text_tks,
                            previous_configs=[],
                            text_tokens=text_tks,
                            line_height=line_height,
                            text_structure=text_structure,
                            compression_ratio=1.0,
                            page_limit=100,
                            text=code_context,
                            enable_syntax_highlight=enable_syntax_highlight,
                            language=language or "python",
                            preserve_newlines=preserve_newlines,
                            font_path=font_path,
                        )
                        margin_1x = int(res_1x * 0.01)
                        base_images = []
                        base_paths = []
                        start_time = time.time()
                        page_idx = 0
                        for img in text_to_image_stream(
                            code_context,
                            width=res_1x,
                            height=res_1x,
                            font_size=fs_1x,
                            line_height=line_height,
                            margin_px=margin_1x,
                            dpi=dpi,
                            font_path=font_path,
                            preserve_newlines=preserve_newlines,
                            enable_syntax_highlight=enable_syntax_highlight,
                            filename=None,
                            language=language or "python",
                            should_crop_whitespace=should_crop_whitespace,
                            enable_two_column=enable_two_column,
                            enable_bold=enable_bold,
                        ):
                            page_idx += 1
                            bp = os.path.join(
                                images_dir,
                                f"summary_{example_id:05d}_ratio{1.0}_{res_1x}x{res_1x}_fs{fs_1x}_page_{page_idx:03d}.png",
                            )
                            img.save(bp)
                            base_images.append(img)
                            base_paths.append(bp)
                            fr = calculate_fill_rate(fs_1x, res_1x, 1, text_structure["num_lines"], int(text_structure["avg_line_chars"]), line_height)
                            tk = calculate_image_tokens_qwen3(res_1x, res_1x)
                            tt = text_tks / 1.0
                            et = time.time() - start_time
                            print(f"[summary_{example_id:05d}_1x] Ratio 1.0: Res {res_1x}x{res_1x}, Count {page_idx}, Font {fs_1x}, Fill {int(fr*100)}%, Tokens {tk} (Target {tt:.1f}) [Time: {et:.3f}s]")
                        resolution_list = get_expanded_resolution_list()
                        for ratio in COMPRESSION_RATIOS:
                            try:
                                image_token_limit = text_tks / ratio
                                num_images = len(base_images)
                                per_image_tokens = image_token_limit / num_images if num_images > 0 else image_token_limit
                                target_res = find_closest_resolution_prefer_larger(per_image_tokens, resolution_list, tolerance_ratio=1.4)
                                fs_scaled = int(fs_1x * (target_res / res_1x)) if res_1x > 0 else fs_1x
                                resized_image_paths = []
                                for page_num, bp in enumerate(base_paths, 1):
                                    try:
                                        with PIL_Image.open(bp) as im:
                                            resized_img = im.resize((target_res, target_res), PIL_Image.Resampling.LANCZOS)
                                        image_filename = f"summary_{example_id:05d}_ratio{ratio}_{target_res}x{target_res}_fs{fs_scaled}_page_{page_num:03d}.png"
                                        image_path = os.path.join(images_dir, image_filename)
                                        resized_img.save(image_path)
                                        resized_image_paths.append(os.path.abspath(image_path))
                                    except Exception:
                                        continue
                                compressed_result = process_example_with_images(
                                    example_id,
                                    row,
                                    resized_image_paths,
                                    client,
                                    model_name,
                                    output_dir,
                                    compression_ratio=ratio,
                                    resolution=f"{target_res}x{target_res}",
                                )
                                summ_res(f"image_ratio{ratio}", compressed_result)
                            except Exception as e:
                                print(
                                    f"  Example {example_id}: Compression ratio {ratio} processing failed: {e}"
                                )
            except Exception as e:
                print(f"  Example {example_id}: Image mode processing failed: {e}")

        with results_lock:
            all_results["total"] += example_results["total"]
            all_results["empty"] += example_results["empty"]
            for t in all_ratios:
                all_results["prompt_tokens"][t] += example_results["prompt_tokens"][t]
                all_results["completion_tokens"][t] += example_results[
                    "completion_tokens"
                ][t]
                all_results["total_tokens"][t] += example_results["total_tokens"][t]
        return example_results

    # Process examples concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(process_example, i, row) for i, row in enumerate(dataset)
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing example: {e}")

    print(f"Generation complete")
    return all_results, output_dir


def get_metric(client, eval_model, intent, code_context, gold_doc, pred_doc):

    tk_infos = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def update_tks(_t_i):
        tk_infos["prompt_tokens"] += _t_i["prompt_tokens"]
        tk_infos["completion_tokens"] += _t_i["completion_tokens"]
        tk_infos["total_tokens"] += _t_i["total_tokens"]

    prompt = f"I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n"
    prompt += f"My code:\n\n{code_context}\n\n\n\n"
    prompt += f"Documentation A:\n\n{gold_doc}\n\n\n\n"
    prompt += f"Documentation B:\n\n{pred_doc}\n\n\n\n"
    prompt += "Better documentation is documentation "

    options = ["A", "B"]
    unnorm_logprobs, _tk_infos = call_llm_with_logit_bias(
        client, eval_model, prompt, options
    )
    norm_probs1 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    update_tks(_tk_infos)

    prompt = f"I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n"
    prompt += f"My code:\n\n{code_context}\n\n\n\n"
    prompt += f"Documentation A:\n\n{pred_doc}\n\n\n\n"
    prompt += f"Documentation B:\n\n{gold_doc}\n\n\n\n"
    prompt += "Better documentation is documentation "
    unnorm_logprobs, _tk_infos = call_llm_with_logit_bias(
        client, eval_model, prompt, options
    )
    update_tks(_tk_infos)
    norm_probs2 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))

    p_better1 = (norm_probs1[1] + norm_probs2[0]) / 2
    return float(p_better1), tk_infos


def get_metric_with_llm(client, eval_model, intent, code_context, gold_doc, pred_doc):
    """Evaluate documentation quality using LLM."""
    # Try scorer method first
    try:
        score, tk_infos = get_metric(
            client, eval_model, intent, code_context, gold_doc, pred_doc
        )
        if score is not None:
            return score, tk_infos
    except Exception as e:
        print(f"Scorer method failed, falling back to text method: {e}")
    # Fallback to text method
    tk_infos = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def update_tks(_t_i):
        tk_infos["prompt_tokens"] += _t_i["prompt_tokens"]
        tk_infos["completion_tokens"] += _t_i["completion_tokens"]
        tk_infos["total_tokens"] += _t_i["total_tokens"]

    prompt = f"I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n"
    prompt += f"My code:\n\n{code_context}\n\n\n\n"
    prompt += f"Documentation A:\n\n{gold_doc}\n\n\n\n"
    prompt += f"Documentation B:\n\n{pred_doc}\n\n\n\n"
    prompt += "Better documentation is documentation "

    system_prompt = (
        "You are a code quality assessment engine. Respond with only 'A' or 'B'."
    )

    try:
        # First comparison: A vs B
        response1, _ = call_llm_with_text_only(
            client, eval_model, system_prompt, prompt
        )
        update_tks(_)

        # Second comparison: B vs A
        prompt2 = f"I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n"
        prompt2 += f"My code:\n\n{code_context}\n\n\n\n"
        prompt2 += f"Documentation A:\n\n{pred_doc}\n\n\n\n"
        prompt2 += f"Documentation B:\n\n{gold_doc}\n\n\n\n"
        prompt2 += "Better documentation is documentation "

        response2, _ = call_llm_with_text_only(
            client, eval_model, system_prompt, prompt2
        )
        update_tks(_)

        # Calculate probability that B is better
        prob1 = 1.0 if response1.strip().upper() == "B" else 0.0
        prob2 = 1.0 if response2.strip().upper() == "A" else 0.0

        p_better = (prob1 + prob2) / 2
        return float(p_better), tk_infos
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.5, {}  # Return neutral score by default


def evaluate_model_results(
    model_name: str, output_dir: str, use_pbar: bool = True
) -> Dict:
    """Evaluate results for a single model."""
    # Load dataset and configuration
    dataset, tokenizer, config = load_code_summarization_data()
    if dataset is None:
        return {}

    # Get evaluation model
    eval_model = config.get("eval_model", "gpt-4")
    print(f"Using evaluation model: {eval_model} to evaluate model: {model_name}")

    # Create evaluation client
    eval_client = create_client(config.get("client", "OpenAI"), **config)

    # Extract parameters
    max_context_toks = config.get("max_context_toks", None)

    # Group results by mode
    results_by_mode = {}

    print(f"Loading result files: {output_dir}")
    # Recursively find all result files
    jsonl_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))

    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {output_dir} and subdirectories")
        return {}

    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        match = re.search(r"example_\d+_(.+)\.jsonl", file_name)
        if not match:
            continue
        mode = match.group(1)

        if mode not in results_by_mode:
            results_by_mode[mode] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results_by_mode[mode].append(json.loads(line))

    # Evaluate each mode
    evaluation_results = {}

    for mode, results in results_by_mode.items():
        print(f"Evaluating mode: {mode}, Examples: {len(results)}")
        eva_tks = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if not results:
            continue

        # Prepare evaluation data
        golds, preds, intents, codes, empty_docs = [], [], [], [], 0

        for result in results:
            # Read required fields directly from result file
            gold_doc = result.get("gold_doc", "")
            pred_doc = result.get("output", "")
            intent = result.get("intent", "")
            code_context = result.get("prompt", "")  # Code context already saved in results

            if gold_doc and pred_doc:
                golds.append(gold_doc)
                preds.append(pred_doc)
                intents.append(intent)
                codes.append(code_context)
            if not pred_doc:
                empty_docs += 1
        print(f"{empty_docs} empty generation results")

        # Calculate evaluation metrics
        metrics = []
        pbar = range(len(golds))
        if use_pbar:
            from tqdm.auto import tqdm

            pbar = tqdm(
                pbar, total=len(pbar), position=0, leave=True, desc=f"Evaluating {mode}"
            )

        for idx in pbar:
            metric, i_tks = get_metric_with_llm(
                eval_client,
                eval_model,
                intents[idx],
                codes[idx],
                golds[idx],
                preds[idx],
            )
            metrics.append(metric)
            eva_tks["prompt_tokens"] += i_tks["prompt_tokens"]
            eva_tks["completion_tokens"] += i_tks["completion_tokens"]
            eva_tks["total_tokens"] += i_tks["total_tokens"]

        # Calculate statistics
        if metrics:
            evaluation_results[mode] = {
                "count": len(metrics),
                "mean_score": np.mean(metrics),
                "std_score": np.std(metrics),
                "min_score": np.min(metrics),
                "max_score": np.max(metrics),
                "median_score": np.median(metrics),
                "scores": metrics,
                "avg_tokens_per_result": sum(
                    r.get("tokens", {}).get("total_tokens", 0) for r in results
                )
                / max(len(results), 1),
                "total_tokens": sum(
                    r.get("tokens", {}).get("total_tokens", 0) for r in results
                ),
                "eval_tokens": eva_tks,
            }
        else:
            evaluation_results[mode] = {
                "count": 0,
                "mean_score": 0.0,
                "error": "No valid results for evaluation",
            }

    return {
        "model_name": model_name,
        "eval_model": eval_model,
        "output_dir": output_dir,
        "results_by_mode": results_by_mode,
        "evaluation_results": evaluation_results,
        "status": "completed",
    }


def evaluate_code_summarization(
    model_name: str, output_dir: str, client_type: str = "OpenAI"
) -> Dict:
    """Evaluate code summarization results (simplified version for backward compatibility)."""
    # Group results by mode
    results_by_mode = {}

    # Find all result files
    for file in os.listdir(output_dir):
        if file.endswith(".jsonl"):
            mode = (
                "text_only"
                if "text_only" in file
                else (
                    "original"
                    if "original" in file
                    else "compressed" if "ratio" in file else "unknown"
                )
            )

            if mode not in results_by_mode:
                results_by_mode[mode] = []

            with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results_by_mode[mode].append(json.loads(line))

    # Calculate basic statistics
    evaluation = {}
    for mode, results in results_by_mode.items():
        evaluation[mode] = {
            "count": len(results),
            "avg_tokens_per_result": sum(
                r.get("tokens", {}).get("total_tokens", 0) for r in results
            )
            / max(len(results), 1),
            "total_tokens": sum(
                r.get("tokens", {}).get("total_tokens", 0) for r in results
            ),
        }

    return {
        "model_name": model_name,
        "output_dir": output_dir,
        "results_by_mode": results_by_mode,
        "evaluation": evaluation,
        "status": "completed",
    }


def run_all_evaluations(models_to_test: List[str], base_output_dir: str, model_output_dir: str) -> Dict:
    """Run evaluations for all models and aggregate results."""
    print("\n=== Starting evaluation for all models ===")

    all_evaluations = {}
    summary_stats = {}

    for model_name in models_to_test:
        print(f"\n--- Evaluating model: {model_name} ---")

        if not os.path.exists(model_output_dir):
            print(f"Warning: Output directory for model {model_name} does not exist: {model_output_dir}")
            continue

        try:
            # Evaluate single model
            eval_result = evaluate_model_results(
                model_name=model_name,
                output_dir=model_output_dir,
                use_pbar=True,
            )

            all_evaluations[model_name] = eval_result

            # Aggregate statistics
            for mode, stats in eval_result.get("evaluation_results", {}).items():
                if mode not in summary_stats:
                    summary_stats[mode] = []

                if stats.get("count", 0) > 0:
                    summary_stats[mode].append(
                        {
                            "model": model_name,
                            "mean_score": stats["mean_score"],
                            "count": stats["count"],
                        }
                    )

            # Save evaluation results for single model
            eval_result_file = os.path.join(model_output_dir, "evaluation_results.json")
            with open(eval_result_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=2)

            print(f"Model {model_name} evaluation complete, results saved to: {eval_result_file}")

        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            import traceback

            traceback.print_exc()

    # Generate summary report
    summary_report = {
        "timestamp": json.dumps({"__time__": int(os.path.getmtime(__file__))}),
        "models_evaluated": list(all_evaluations.keys()),
        "summary_stats": summary_stats,
        "all_evaluations": all_evaluations,
    }

    # Save summary report
    summary_file = os.path.join(base_output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print(f"\n=== All evaluations complete ===")
    print(f"Summary report saved to: {summary_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for mode, model_stats in summary_stats.items():
        print(f"\nMode: {mode}")
        for stat in model_stats:
            print(
                f"  {stat['model']}: Mean score = {stat['mean_score']:.3f}, Examples = {stat['count']}"
            )

    return summary_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate module summarization results.")
    parser.add_argument("output_dir", type=str, help="Directory containing the results to evaluate")
    parser.add_argument("--model_name", type=str, default="unknown_model", help="Name of the model being evaluated")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist.")
        exit(1)

    print(f"Evaluating results in: {args.output_dir}")
    # Note: This assumes config.json is in the current working directory or handled by get_config
    results = evaluate_model_results(args.model_name, args.output_dir)

    # Print summary
    if "evaluation_results" in results:
        print("\n=== Evaluation Summary ===")
        for mode, stats in results["evaluation_results"].items():
            if stats.get("count", 0) > 0:
                print(f"Mode: {mode}")
                print(f"  Count: {stats['count']}")
                print(f"  Mean Score: {stats['mean_score']:.4f}")
                print(f"  Std Dev: {stats.get('std_score', 0.0):.4f}")
                print(f"  Avg Tokens: {stats.get('avg_tokens_per_result', 0):.1f}")
            else:
                print(f"Mode: {mode} - {stats.get('error', 'No valid results')}")

        # Generate font size summary (similar to run_pipeline.py)
        try:
            from report_utils import generate_font_size_summary
            import re

            # Extract average scores
            evaluation_results = results["evaluation_results"]
            average_scores = {
                mode: stats.get("mean_score", 0.0)
                for mode, stats in evaluation_results.items()
                if stats.get("count", 0) > 0
            }

            if average_scores:
                # Try to extract font size from output_dir path
                font_size = 40  # Default
                font_match = re.search(r"font(\d+)", args.output_dir)
                if font_match:
                    font_size = int(font_match.group(1))

                processed_results = [
                    {
                        "font_size": font_size,
                        "config_name": "module_summarization",
                        "filename": "module_summarization",
                        "compression_ratio": None,
                        "resolution": "N/A",
                        "evaluation_results": {
                            "module_summarization": average_scores
                        },
                    }
                ]

                generate_font_size_summary(
                    processed_results, args.model_name, "module_summarization"
                )
        except ImportError:
            print("Warning: report_utils not found, skipping generate_font_size_summary")
        except Exception as e:
            print(f"Error generating font size summary: {e}")
    else:
        print("No evaluation results returned.")
