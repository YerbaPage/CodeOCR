import os
import argparse
import sys
import json
import time
from typing import List, Tuple, Dict, Optional
from PIL import Image as PIL_Image
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_utils import (
    create_client,
    build_folder,
    call_llm_with_images,
    call_llm_with_text_only,
    get_text_tokens,
    get_text_tokens_qwen,
)
from task_utils import extract_json_from_response as extract_json_from_response_shared
from tasks.code_clone_detection.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_LANG,
    DEFAULT_MAX_WORKERS,
    DEFAULT_NUM_EXAMPLES,
    DEFAULT_TIER,
)
from tasks.code_clone_detection.data import (
    build_balanced_dataset,
    load_code_clone_detection_pairs,
)
from tasks.code_clone_detection.rendering import (
    _find_existing_images,
    render_images_for_file,
    render_pair_images,
    render_pair_images_stream,
)

from text_to_image import (
    get_font,
    prepare_text_for_rendering,
    text_to_image,
    text_to_image_stream,
    resize_images_for_compression,
    generate_compressed_images_dynamic,
    calculate_image_tokens_from_paths,
    calculate_image_tokens_with_processor,
    calculate_image_tokens_qwen3,
    COMPRESSION_RATIOS,
    DEFAULT_MARGIN_RATIO,
    find_closest_resolution_prefer_larger,
    get_expanded_resolution_list,
    optimize_layout_config_dry,
    calculate_fill_rate,
)

def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """
    Extract JSON object from model response.
    
    Args:
        response_text: Model response text
        
    Returns:
        Parsed JSON object or None
    """
    return extract_json_from_response_shared(response_text)

def process_single_code_clone_detection_item(
    idx: int,
    item: Dict,
    client,
    model_name: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: Optional[str],
    preserve_newlines: bool,
    resize_mode: bool,
    lang: str,
    difficulty: str,
    tier: str,
    processor,
    qwen_tokenizer,
    enable_text_only_test: bool,
    enable_syntax_highlight: bool,
    enable_bold: bool,
    separate_mode: bool,
    images_dir: str,
    output_dir: str,
    config_name: str,
    existing_results_dir: Optional[str] = None,
    target_ratios: Optional[List[float]] = None,
    theme: str = "light",
    extreme_mode: bool = False,
) -> Tuple[Dict, Dict]:
    """Process a single code clone detection task item."""
    left = item["left"]
    right = item["right"]
    label = item["label"]
    file_path = item["file"]
    unique_id = f"{lang}_{difficulty}_{tier}_{idx:04d}"
    
    # Function to save image immediately to save memory
    def save_img(img, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
    
    
    # 2. Use in-memory images for text_only test
    results_item = {}
    token_stats_item = []
    
    system_prompt = "You judge whether two code snippets implement the same logic. Return strict JSON only."
    # user_prompt will be set dynamically based on actual image count in resize flow
    user_prompt = ""  # Placeholder, will be set in resize processing
    
    # Initialize prompt_text_tokens (for resize mode token statistics)
    prompt_text_tokens = get_text_tokens(system_prompt + "\n" + user_prompt)
    
    if enable_text_only_test:
        tp = f"A:\n```{lang}\n{left}\n```\n\nB:\n```{lang}\n{right}\n```"
        system_prompt_text = "You judge whether two code snippets implement the same logic. Return strict JSON only."
        user_prompt_text = f"The following are two code snippets.\n{tp}\nReturn JSON: {{\"is_same\": true|false}} without extra text."
        
        response_text = None
        tk_info_t = {}
        flag_suffix = "_nl"
        result_filename = f"{model_name}_{config_name}_{width}x{height}_lh{str(line_height).replace('.','_')}_{unique_id}_text_only{flag_suffix}.txt"
        
        if existing_results_dir:
            existing_file = os.path.join(existing_results_dir, result_filename)
            if os.path.exists(existing_file):
                try:
                    with open(existing_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    response_text = existing_data.get("output", "")
                    tk_info_t = existing_data.get("tokens", {})
                    existing_clean = existing_data.get("clean_output")
                    if (not response_text or (isinstance(response_text, str) and response_text.strip() == "")) or (not existing_clean):
                        response_text = None
                        tk_info_t = {}
                except Exception as e:
                    print(f"Error reading existing file {existing_file}: {e}")

        if response_text is None:
            response_text, tk_info_t = call_llm_with_text_only(client, model_name, system_prompt_text, user_prompt_text)
            
        pred = None
        obj = extract_json_from_response(response_text)
        if obj:
            pred = bool(obj.get("is_same"))
        
        prompt_tokens_text = get_text_tokens(system_prompt_text + "\n" + user_prompt_text)
        source_tokens_left = get_text_tokens(left)
        source_tokens_right = get_text_tokens(right)
        source_tokens_qwen_left = get_text_tokens_qwen(left, qwen_tokenizer)
        source_tokens_qwen_right = get_text_tokens_qwen(right, qwen_tokenizer)
        
        result_file = os.path.join(output_dir, result_filename)
        result_payload_text = {
            "id": unique_id,
            "file": file_path,
            "label": label,
            "dataset_type": item.get("dataset_type", ""),
            "lang": lang,
            "difficulty": difficulty,
            "tier": tier,
            "model": model_name,
            "mode": "text_only",
            "left": left,
            "right": right,
            "system_prompt": system_prompt_text,
            "user_prompt": user_prompt_text,
            "output": response_text or "",
            "clean_output": obj,
            "tokens": tk_info_t,
            "config": {
                "width": width,
                "height": height,
                "font_size": font_size,
                "line_height": line_height,
                "dpi": dpi,
                "preserve_newlines": preserve_newlines,
                "enable_syntax_highlight": enable_syntax_highlight,
                "language": lang,
                "enable_bold": enable_bold,
                "separate_mode": separate_mode,
            },
        }
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result_payload_text, ensure_ascii=False, indent=2))
        token_stats_item.append({
            "model": model_name,
            "config": config_name,
            "unique_id": unique_id,
            "mode": "text_only",
            "image_tokens": 0,
            "image_tokens_processor": None,
            "text_tokens_left": source_tokens_left,
            "text_tokens_right": source_tokens_right,
            "text_tokens_qwen_left": source_tokens_qwen_left,
            "text_tokens_qwen_right": source_tokens_qwen_right,
            "prompt_text_tokens": prompt_tokens_text,
            "completion_tokens": get_text_tokens(response_text or ""),
            "api_prompt_tokens": tk_info_t.get("prompt_tokens", 0),
            "api_image_tokens_estimate": 0,
            "total_tokens": tk_info_t.get("total_tokens", 0),
            "num_images": 0,
        })
        results_item["text_only"] = {"pred": pred, "label": label}
    
    
    # 5. Process resize mode
    if resize_mode:
        ratios_to_use = target_ratios if target_ratios is not None else COMPRESSION_RATIOS
        # Ensure 1.0 is always included (treat it as "original")
        if 1.0 not in ratios_to_use:
            ratios_to_use = sorted(list(set([1.0] + list(ratios_to_use))))
        if separate_mode:
            # Separate mode: generate compressed images for left and right separately
            left_tks = get_text_tokens(left)
            right_tks = get_text_tokens(right)
            
            # Initialize variables to avoid UnboundLocalError
            resize_image_tokens = 0
            resize_image_tokens_processor = None
            resize_num_images = 0
            
            if extreme_mode:
                # Extreme mode: dynamically re-render at different resolutions
                def make_renderer(content):
                    def _render(w, h, fs):
                        # Dynamically adjust margin (1% of width)
                        margin = int(w * 0.01)
                        return text_to_image(
                            content,
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
                            language=lang,
                            should_crop_whitespace=False,
                            enable_two_column=False,
                            enable_bold=enable_bold,
                            theme=theme
                        )
                    return _render
                
                # Calculate text structure information to help optimize layout
                left_lines = left.count('\n') + 1
                left_chars = [len(line) for line in left.split('\n')]
                left_structure = {
                    'num_lines': left_lines,
                    'max_line_chars': max(left_chars) if left_chars else 0,
                    'avg_line_chars': sum(left_chars) / len(left_chars) if left_chars else 0
                }
                
                right_lines = right.count('\n') + 1
                right_chars = [len(line) for line in right.split('\n')]
                right_structure = {
                    'num_lines': right_lines,
                    'max_line_chars': max(right_chars) if right_chars else 0,
                    'avg_line_chars': sum(right_chars) / len(right_chars) if right_chars else 0
                }
                
                # Check existing results to avoid duplicate generation
                results_map = {}
                missing_ratios = []
                
                for ratio in COMPRESSION_RATIOS:
                    found_existing = False
                    if existing_results_dir and os.path.exists(existing_results_dir):
                        # Match filename: ignore resolution part
                        prefix = f"{model_name}_{config_name}_resize_ratio{ratio}_"
                        suffix = f"_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                        try:
                            for fname in os.listdir(existing_results_dir):
                                if fname.startswith(prefix) and fname.endswith(suffix):
                                    full_path = os.path.join(existing_results_dir, fname)
                                    try:
                                        with open(full_path, "r", encoding="utf-8") as f:
                                            results_map[ratio] = (json.load(f), fname)
                                        found_existing = True
                                    except:
                                        pass
                                    break
                        except Exception as e:
                            print(f"Error searching existing results: {e}")
                    
                    if not found_existing:
                        missing_ratios.append(ratio)
                
                resized_left = {}
                resized_right = {}
                if missing_ratios:
                    resized_left = generate_compressed_images_dynamic(left_tks, make_renderer(left), missing_ratios, text_structure=left_structure, data_id=f"{unique_id}_A")
                    resized_right = generate_compressed_images_dynamic(right_tks, make_renderer(right), missing_ratios, text_structure=right_structure, data_id=f"{unique_id}_B")
            else:
                # Standard mode: use dynamic optimization for 1x, then resize other ratios from 1x base
                # No longer generate fixed-resolution images first
                
                # Now create renderer functions for dynamic optimization
                # Create renderer functions for dynamic optimization
                def make_renderer(content):
                    def _render(w, h, fs):
                        margin = int(w * 0.01)
                        return text_to_image(
                            content,
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
                            language=lang,
                            should_crop_whitespace=False,
                            enable_two_column=False,
                            enable_bold=enable_bold,
                            theme=theme
                        )
                    return _render
                
                # Calculate text structure for optimization
                left_lines = left.count('\n') + 1
                left_chars = [len(line) for line in left.split('\n')]
                left_structure = {
                    'num_lines': left_lines,
                    'max_line_chars': max(left_chars) if left_chars else 0,
                    'avg_line_chars': sum(left_chars) / len(left_chars) if left_chars else 0
                }
                
                right_lines = right.count('\n') + 1
                right_chars = [len(line) for line in right.split('\n')]
                right_structure = {
                    'num_lines': right_lines,
                    'max_line_chars': max(right_chars) if right_chars else 0,
                    'avg_line_chars': sum(right_chars) / len(right_chars) if right_chars else 0
                }
                
                left_res_1x, left_fs_1x, _ = optimize_layout_config_dry(
                    target_tokens=left_tks,
                    previous_configs=[],
                    text_tokens=left_tks,
                    line_height=line_height,
                    text_structure=left_structure,
                    compression_ratio=1.0,
                    page_limit=100,
                    text=left,
                    enable_syntax_highlight=enable_syntax_highlight,
                    language=lang,
                    preserve_newlines=preserve_newlines,
                    font_path=font_path,
                    theme=theme,
                )
                right_res_1x, right_fs_1x, _ = optimize_layout_config_dry(
                    target_tokens=right_tks,
                    previous_configs=[],
                    text_tokens=right_tks,
                    line_height=line_height,
                    text_structure=right_structure,
                    compression_ratio=1.0,
                    page_limit=100,
                    text=right,
                    enable_syntax_highlight=enable_syntax_highlight,
                    language=lang,
                    preserve_newlines=preserve_newlines,
                    font_path=font_path,
                    theme=theme,
                )
                resized_left = {}
                resized_right = {}
                resized_left = {}
                resized_right = {}
                results_map = {}
                base_A_paths = []
                margin_a = int(left_res_1x * 0.01)
                page_idx = 0
                start_time_a = time.time()
                for img in text_to_image_stream(
                    left,
                    width=left_res_1x,
                    height=left_res_1x,
                    font_size=left_fs_1x,
                    line_height=line_height,
                    margin_px=margin_a,
                    dpi=dpi,
                    font_path=font_path,
                    bg_color="white",
                    text_color="black",
                    preserve_newlines=preserve_newlines,
                    enable_syntax_highlight=enable_syntax_highlight,
                    filename=None,
                    language=lang,
                    should_crop_whitespace=False,
                    enable_two_column=False,
                    enable_bold=enable_bold,
                    theme=theme,
                ):
                    page_idx += 1
                    bp = os.path.join(
                        images_dir,
                        f"{unique_id}_sep_A_{'extreme' if extreme_mode else 'standard'}_ratio{1.0}_{left_res_1x}x{left_res_1x}_fs{left_fs_1x}_page_{page_idx:03d}.png",
                    )
                    save_img(img, bp)
                    base_A_paths.append(bp)
                    fr = calculate_fill_rate(left_fs_1x, left_res_1x, 1, left_structure["num_lines"], int(left_structure["avg_line_chars"]), line_height)
                    tk = calculate_image_tokens_qwen3(left_res_1x, left_res_1x)
                    tt = left_tks / 1.0
                    et = time.time() - start_time_a
                    print(f"[{unique_id}_A_1x] Ratio 1.0: Res {left_res_1x}x{left_res_1x}, Count {page_idx}, Font {left_fs_1x}, Fill {int(fr*100)}%, Tokens {tk} (Target {tt:.1f}) [Time: {et:.3f}s]")
                base_B_paths = []
                margin_b = int(right_res_1x * 0.01)
                page_idx_b = 0
                start_time_b = time.time()
                for img in text_to_image_stream(
                    right,
                    width=right_res_1x,
                    height=right_res_1x,
                    font_size=right_fs_1x,
                    line_height=line_height,
                    margin_px=margin_b,
                    dpi=dpi,
                    font_path=font_path,
                    bg_color="white",
                    text_color="black",
                    preserve_newlines=preserve_newlines,
                    enable_syntax_highlight=enable_syntax_highlight,
                    filename=None,
                    language=lang,
                    should_crop_whitespace=False,
                    enable_two_column=False,
                    enable_bold=enable_bold,
                    theme=theme,
                ):
                    page_idx_b += 1
                    bp = os.path.join(
                        images_dir,
                        f"{unique_id}_sep_B_{'extreme' if extreme_mode else 'standard'}_ratio{1.0}_{right_res_1x}x{right_res_1x}_fs{right_fs_1x}_page_{page_idx_b:03d}.png",
                    )
                    save_img(img, bp)
                    base_B_paths.append(bp)
                    fr_b = calculate_fill_rate(right_fs_1x, right_res_1x, 1, right_structure["num_lines"], int(right_structure["avg_line_chars"]), line_height)
                    tk_b = calculate_image_tokens_qwen3(right_res_1x, right_res_1x)
                    tt_b = right_tks / 1.0
                    et_b = time.time() - start_time_b
                    print(f"[{unique_id}_B_1x] Ratio 1.0: Res {right_res_1x}x{right_res_1x}, Count {page_idx_b}, Font {right_fs_1x}, Fill {int(fr_b*100)}%, Tokens {tk_b} (Target {tt_b:.1f}) [Time: {et_b:.3f}s]")
            
            for ratio in ratios_to_use:
                rl_imgs, rl_res, rl_fs = [], None, None
                rr_imgs, rr_res, rr_fs = [], None, None
                resp_r, tk_info_r = None, {}
                current_image_paths = []
                if float(ratio) == 0.0:
                    os.makedirs("./generated_images", exist_ok=True)
                    blank_path = os.path.join("./generated_images", "blank_14x14.png")
                    if not os.path.exists(blank_path):
                        PIL_Image.new("RGB", (14, 14), color="white").save(blank_path)
                    user_prompt_r = "Two code snippets are shown in the images. Decide if they are logically equivalent and return: {\"is_same\": true|false}."
                    rl_res = 14
                    rr_res = 14
                    rl_fs = 0
                    rr_fs = 0
                    current_image_paths = [os.path.abspath(blank_path)]
                    result_filename = f"{model_name}_{config_name}_resize_ratio{ratio}_14x14_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                    result_file_r = os.path.join(output_dir, result_filename)
                    resp_r, tk_info_r = call_llm_with_images(client, model_name, current_image_paths, system_prompt, user_prompt_r)
                    obj = extract_json_from_response(resp_r or "")
                    result_payload_r = {
                        "id": unique_id,
                        "file": file_path,
                        "label": label,
                        "dataset_type": item.get("dataset_type", ""),
                        "lang": lang,
                        "difficulty": difficulty,
                        "tier": tier,
                        "model": model_name,
                        "mode": f"image_ratio{ratio}",
                        "left": left,
                        "right": right,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt_r,
                        "output": resp_r or "",
                        "clean_output": obj,
                        "tokens": tk_info_r,
                        "image_paths": current_image_paths,
                        "config": {
                            "width": width,
                            "height": height,
                            "resize_ratio": ratio,
                            "target_res_left": rl_res,
                            "target_res_right": rr_res,
                            "font_size_left": rl_fs,
                            "font_size_right": rr_fs,
                            "line_height": line_height,
                            "dpi": dpi,
                            "preserve_newlines": preserve_newlines,
                            "enable_syntax_highlight": enable_syntax_highlight,
                            "language": lang,
                            "enable_bold": enable_bold,
                            "separate_mode": separate_mode,
                        },
                    }
                    with open(result_file_r, "w", encoding="utf-8") as f:
                        f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                    pred_r = None
                    if obj:
                        pred_r = bool(obj.get("is_same"))
                    resize_image_tokens = calculate_image_tokens_from_paths(current_image_paths)
                    resize_image_tokens_processor = calculate_image_tokens_with_processor(current_image_paths, processor)
                    resize_num_images = len(current_image_paths)
                    token_stats_item.append({
                        "model": model_name,
                        "config": f"{config_name}_resize_ratio{ratio}",
                        "unique_id": unique_id,
                        "mode": f"image_ratio{ratio}",
                        "image_tokens": resize_image_tokens,
                        "image_tokens_processor": resize_image_tokens_processor,
                        "text_tokens_left": get_text_tokens(left),
                        "text_tokens_right": get_text_tokens(right),
                        "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                        "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                        "prompt_text_tokens": prompt_text_tokens,
                        "completion_tokens": get_text_tokens(resp_r or ""),
                        "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                        "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                        "total_tokens": tk_info_r.get("total_tokens", 0),
                        "num_images": resize_num_images,
                    })
                    results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
                    continue
                
                existing = _find_existing_images(images_dir, unique_id, ratio, True)
                if existing:
                    try:
                        for p in existing.get("paths", []):
                            print(f"use_cache: {os.path.basename(p)}")
                    except Exception:
                        pass
                    a_count = len([p for p in existing["paths"] if "_sep_A_" in os.path.basename(p)])
                    b_count = len([p for p in existing["paths"] if "_sep_B_" in os.path.basename(p)])
                    user_prompt_r = f"There are two code snippets shown in the images. The first {a_count} image(s) correspond to A, and the next {b_count} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                    rl_res = existing["res_left"]
                    rr_res = existing["res_right"]
                    rl_fs = existing["fs_left"]
                    rr_fs = existing["fs_right"]
                    result_filename = f"{model_name}_{config_name}_resize_ratio{ratio}_{rl_res}x{rl_res}_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                    result_file_r = os.path.join(output_dir, result_filename)
                    if existing_results_dir:
                        existing_file = os.path.join(existing_results_dir, result_filename)
                        if os.path.exists(existing_file):
                            try:
                                with open(existing_file, "r", encoding="utf-8") as f:
                                    result_payload_r = json.load(f)
                                resp_r = result_payload_r.get("output", "")
                                tk_info_r = result_payload_r.get("tokens", {})
                                obj = result_payload_r.get("clean_output")
                                if not obj:
                                    obj = extract_json_from_response(resp_r or "")
                                    result_payload_r["clean_output"] = obj
                            except Exception:
                                resp_r = None
                                tk_info_r = {}
                    if resp_r is None:
                        resp_r, tk_info_r = call_llm_with_images(client, model_name, existing["paths"], system_prompt, user_prompt_r)
                        obj = extract_json_from_response(resp_r or "")
                        result_payload_r = {
                            "id": unique_id,
                            "file": file_path,
                            "label": label,
                            "dataset_type": item.get("dataset_type", ""),
                            "lang": lang,
                            "difficulty": difficulty,
                            "tier": tier,
                            "model": model_name,
                            "mode": f"image_ratio{ratio}",
                            "left": left,
                            "right": right,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt_r,
                            "output": resp_r or "",
                            "clean_output": obj,
                            "tokens": tk_info_r,
                            "image_paths": existing["paths"],
                            "config": {
                                "width": width,
                                "height": height,
                                "resize_ratio": ratio,
                                "target_res_left": rl_res,
                                "target_res_right": rr_res,
                                "font_size_left": rl_fs,
                                "font_size_right": rr_fs,
                                "line_height": line_height,
                                "dpi": dpi,
                                "preserve_newlines": preserve_newlines,
                                "enable_syntax_highlight": enable_syntax_highlight,
                                "language": lang,
                                "enable_bold": enable_bold,
                                "separate_mode": separate_mode,
                            },
                        }
                    with open(result_file_r, "w", encoding="utf-8") as f:
                        f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                    pred_r = None
                    if result_payload_r.get("clean_output"):
                        pred_r = bool(result_payload_r["clean_output"].get("is_same"))
                    token_stats_item.append({
                        "model": model_name,
                        "config": f"{config_name}_resize_ratio{ratio}",
                        "unique_id": unique_id,
                        "mode": f"image_ratio{ratio}",
                        "image_tokens": calculate_image_tokens_from_paths(existing["paths"]),
                        "image_tokens_processor": calculate_image_tokens_with_processor(existing["paths"], processor),
                        "text_tokens_left": get_text_tokens(left),
                        "text_tokens_right": get_text_tokens(right),
                        "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                        "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                        "prompt_text_tokens": prompt_text_tokens,
                        "completion_tokens": get_text_tokens(resp_r or ""),
                        "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                        "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                        "total_tokens": tk_info_r.get("total_tokens", 0),
                        "num_images": len(existing["paths"]),
                    })
                    results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
                    continue

                if ratio in results_map:
                    result_payload_r, result_filename = results_map[ratio]
                    resp_r = result_payload_r.get("output", "")
                    tk_info_r = result_payload_r.get("tokens", {})
                    current_image_paths = result_payload_r.get("image_paths", [])
                    existing_clean = result_payload_r.get("clean_output")
                    obj = extract_json_from_response(resp_r)
                    need_refetch = (not resp_r or (isinstance(resp_r, str) and resp_r.strip() == "")) or (not existing_clean and not obj)
                    if need_refetch:
                        a_paths = [p for p in current_image_paths if "_sep_A_" in p]
                        b_paths = [p for p in current_image_paths if "_sep_B_" in p]
                        ordered_paths = a_paths + b_paths if a_paths or b_paths else current_image_paths
                        ordered_paths = ordered_paths if ordered_paths else current_image_paths
                        if not ordered_paths:
                            gen_left = generate_compressed_images_dynamic(left_tks, make_renderer(left), [ratio], text_structure=left_structure, data_id=f"{unique_id}_A")
                            rl_imgs, rl_res, rl_fs = gen_left.get(ratio, ([], None, None))
                            gen_right = generate_compressed_images_dynamic(right_tks, make_renderer(right), [ratio], text_structure=right_structure, data_id=f"{unique_id}_B")
                            rr_imgs, rr_res, rr_fs = gen_right.get(ratio, ([], None, None))
                            current_image_paths = []
                            for i, img in enumerate(rl_imgs, 1):
                                rp = os.path.join(images_dir, f"{unique_id}_sep_A_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rl_res}x{rl_res}_fs{rl_fs}_page_{i:03d}.png")
                                save_img(img, rp)
                                current_image_paths.append(rp)
                            for i, img in enumerate(rr_imgs, 1):
                                rp = os.path.join(images_dir, f"{unique_id}_sep_B_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rr_res}x{rr_res}_fs{rr_fs}_page_{i:03d}.png")
                                save_img(img, rp)
                                current_image_paths.append(rp)
                            user_prompt_r = f"There are two code snippets shown in the images. The first {len(rl_imgs)} image(s) correspond to A, and the next {len(rr_imgs)} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                        else:
                            a_count = len(a_paths) if a_paths else 0
                            b_count = len(b_paths) if b_paths else 0
                            user_prompt_r = f"There are two code snippets shown in the images. The first {a_count} image(s) correspond to A, and the next {b_count} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                        
                        # Calculate image metrics BEFORE calling LLM
                        resize_image_tokens = calculate_image_tokens_from_paths(current_image_paths) if current_image_paths else 0
                        resize_image_tokens_processor = calculate_image_tokens_with_processor(current_image_paths, processor) if current_image_paths else None
                        resize_num_images = len(current_image_paths)
                        resp_r, tk_info_r = call_llm_with_images(client, model_name, current_image_paths, system_prompt, user_prompt_r)
                        
                        obj = extract_json_from_response(resp_r)
                        result_payload_r["output"] = resp_r or ""
                        result_payload_r["clean_output"] = obj
                        result_payload_r["tokens"] = tk_info_r
                        result_payload_r["image_paths"] = current_image_paths
                    else:
                        if "clean_output" not in result_payload_r:
                            result_payload_r["clean_output"] = obj
                else:
                    # Generate new results
                    rl_imgs, rl_res, rl_fs = resized_left.get(ratio, ([], None, None))
                    rr_imgs, rr_res, rr_fs = resized_right.get(ratio, ([], None, None))
                    
                    current_image_paths = []
                    if ratio == 1.0:
                        current_image_paths.extend(base_A_paths)
                        current_image_paths.extend(base_B_paths)
                        rl_res = left_res_1x
                        rr_res = right_res_1x
                        rl_fs = left_fs_1x
                        rr_fs = right_fs_1x
                        a_count = len(base_A_paths)
                        b_count = len(base_B_paths)
                        user_prompt_r = f"There are two code snippets shown in the images. The first {a_count} image(s) correspond to A, and the next {b_count} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                    elif not extreme_mode:
                        image_token_limit_left = left_tks / ratio
                        image_token_limit_right = right_tks / ratio
                        num_images_left = len(base_A_paths)
                        num_images_right = len(base_B_paths)
                        per_image_tokens_left = image_token_limit_left / num_images_left if num_images_left > 0 else image_token_limit_left
                        per_image_tokens_right = image_token_limit_right / num_images_right if num_images_right > 0 else image_token_limit_right
                        rl_res = find_closest_resolution_prefer_larger(
                            per_image_tokens_left, get_expanded_resolution_list(), tolerance_ratio=1.4
                        )
                        rr_res = find_closest_resolution_prefer_larger(
                            per_image_tokens_right, get_expanded_resolution_list(), tolerance_ratio=1.4
                        )
                        rl_fs = int(left_fs_1x * (rl_res / left_res_1x)) if left_res_1x > 0 else 0
                        rr_fs = int(right_fs_1x * (rr_res / right_res_1x)) if right_res_1x > 0 else 0
                        for i, bp in enumerate(base_A_paths, 1):
                            try:
                                with PIL_Image.open(bp) as im:
                                    resized_img = im.resize((rl_res, rl_res), PIL_Image.Resampling.LANCZOS)
                                rp = os.path.join(images_dir, f"{unique_id}_sep_A_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rl_res}x{rl_res}_fs{rl_fs}_page_{i:03d}.png")
                                resized_img.save(rp)
                                current_image_paths.append(rp)
                            except Exception:
                                continue
                        for i, bp in enumerate(base_B_paths, 1):
                            try:
                                with PIL_Image.open(bp) as im:
                                    resized_img = im.resize((rr_res, rr_res), PIL_Image.Resampling.LANCZOS)
                                rp = os.path.join(images_dir, f"{unique_id}_sep_B_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rr_res}x{rr_res}_fs{rr_fs}_page_{i:03d}.png")
                                resized_img.save(rp)
                                current_image_paths.append(rp)
                            except Exception:
                                continue
                        a_count = len(base_A_paths)
                        b_count = len(base_B_paths)
                        user_prompt_r = f"There are two code snippets shown in the images. The first {a_count} image(s) correspond to A, and the next {b_count} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                    else:
                        user_prompt_r = f"There are two code snippets shown in the images. The first {len(rl_imgs)} image(s) correspond to A, and the next {len(rr_imgs)} image(s) correspond to B. Decide if they are logically equivalent and return: {{\"is_same\": true|false}}."
                        obj = extract_json_from_response(resp_r)
                        for i, img in enumerate(rl_imgs, 1):
                            rp = os.path.join(images_dir, f"{unique_id}_sep_A_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rl_res}x{rl_res}_fs{rl_fs}_page_{i:03d}.png")
                            save_img(img, rp)
                            current_image_paths.append(rp)
                        for i, img in enumerate(rr_imgs, 1):
                            rp = os.path.join(images_dir, f"{unique_id}_sep_B_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{rr_res}x{rr_res}_fs{rr_fs}_page_{i:03d}.png")
                            save_img(img, rp)
                            current_image_paths.append(rp)
                        try:
                            import gc
                            rl_imgs = []
                            rr_imgs = []
                            gc.collect()
                        except Exception:
                            pass
                    
                    resize_image_tokens = calculate_image_tokens_from_paths(current_image_paths) if current_image_paths else 0
                    resize_image_tokens_processor = calculate_image_tokens_with_processor(current_image_paths, processor) if current_image_paths else None
                    resize_num_images = len(current_image_paths)
                    resp_r, tk_info_r = call_llm_with_images(client, model_name, current_image_paths, system_prompt, user_prompt_r)

                    # Construct result and its filename
                    result_filename = f"{model_name}_{config_name}_resize_ratio{ratio}_{rl_res}x{rl_res}_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                    result_payload_r = {
                        "id": unique_id,
                        "file": file_path,
                        "label": label,
                        "dataset_type": item.get("dataset_type", ""),
                        "lang": lang,
                        "difficulty": difficulty,
                        "tier": tier,
                        "model": model_name,
                        "mode": f"image_ratio{ratio}",
                        "left": left,
                        "right": right,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt_r,
                        "output": resp_r or "",
                        "clean_output": obj,
                        "tokens": tk_info_r,
                        "image_paths": current_image_paths,
                        "config": {
                            "width": width,
                            "height": height,
                            "resize_ratio": ratio,
                            "target_res_left": rl_res,
                            "target_res_right": rr_res,
                            "font_size_left": rl_fs,
                            "font_size_right": rr_fs,
                            "line_height": line_height,
                            "dpi": dpi,
                            "preserve_newlines": preserve_newlines,
                            "enable_syntax_highlight": enable_syntax_highlight,
                            "language": lang,
                            "enable_bold": enable_bold,
                            "separate_mode": separate_mode,
                        },
                    }

                pred_r = None
                if obj:
                    pred_r = bool(obj.get("is_same"))

                # Save LLM results (ensure clean_output is saved)
                result_file_r = os.path.join(output_dir, result_filename)
                with open(result_file_r, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                
                # Collect token statistics
                token_stats_item.append({
                    "model": model_name,
                    "config": f"{config_name}_resize_ratio{ratio}",
                    "unique_id": unique_id,
                    "mode": f"image_ratio{ratio}",
                    "image_tokens": resize_image_tokens,
                    "image_tokens_processor": resize_image_tokens_processor,
                    "text_tokens_left": get_text_tokens(left),
                    "text_tokens_right": get_text_tokens(right),
                    "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                    "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                    "prompt_text_tokens": prompt_text_tokens,
                    "completion_tokens": get_text_tokens(resp_r or ""),
                    "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                    "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                    "total_tokens": tk_info_r.get("total_tokens", 0),
                    "num_images": resize_num_images,
                })
                results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
        else:
            # Merged mode: generate compressed images for merged images
            combined_text_tokens = get_text_tokens(left) + get_text_tokens(right)
            
            # Common setup for both extreme and standard mode
            combined_text = f"{left}\n\n{right}"
            combined_lines = combined_text.count('\n') + 1
            combined_chars = [len(line) for line in combined_text.split('\n')]
            combined_structure = {
                'num_lines': combined_lines,
                'max_line_chars': max(combined_chars) if combined_chars else 0,
                'avg_line_chars': sum(combined_chars) / len(combined_chars) if combined_chars else 0
            }
            
            def renderer_combined(w, h, fs):
                # render_pair_images already has dynamic margin logic internally (1% of width)
                return render_pair_images(
                    left,
                    right,
                    width=w,
                    height=h,
                    font_size=fs,
                    line_height=line_height,
                    dpi=dpi,
                    font_path=font_path,
                    preserve_newlines=preserve_newlines,
                    enable_bold=enable_bold,
                    bg_color="white",
                    text_color="black",
                    enable_syntax_highlight=enable_syntax_highlight,
                    language=lang,
                )
            
            if extreme_mode:
                # Extreme mode: dynamically re-render at different resolutions
                resized = generate_compressed_images_dynamic(combined_text_tokens, renderer_combined, ratios_to_use, text_structure=combined_structure, data_id=unique_id)
            else:
                # Standard mode: resize from dynamically generated 1x base
                # Stream-generate 1x base merged images and save to disk to minimize memory
                resized = {}
                base_1x_paths = []
                res_1x, fs_1x, _ = optimize_layout_config_dry(
                    target_tokens=combined_text_tokens,
                    previous_configs=[],
                    text_tokens=combined_text_tokens,
                    line_height=line_height,
                    text_structure=combined_structure,
                    compression_ratio=1.0,
                    page_limit=100,
                    text=combined_text,
                    enable_syntax_highlight=enable_syntax_highlight,
                    language=lang,
                    preserve_newlines=preserve_newlines,
                    font_path=font_path,
                    theme=theme,
                )
                page_idx = 0
                for img in render_pair_images_stream(
                    left_text=left,
                    right_text=right,
                    width=res_1x,
                    height=res_1x,
                    font_size=fs_1x,
                    line_height=line_height,
                    dpi=dpi,
                    font_path=font_path,
                    preserve_newlines=preserve_newlines,
                    enable_bold=enable_bold,
                    bg_color="white",
                    text_color="black",
                    enable_syntax_highlight=enable_syntax_highlight,
                    language=lang,
                    filename=None,
                    theme=theme,
                ):
                    page_idx += 1
                    bp = os.path.join(
                        images_dir,
                        f"{unique_id}_{'extreme' if extreme_mode else 'standard'}_ratio{1.0}_{res_1x}x{res_1x}_fs{fs_1x}_page_{page_idx:03d}.png",
                    )
                    save_img(img, bp)
                    base_1x_paths.append(bp)
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
            
            for ratio in ratios_to_use:
                res_imgs, target_res, fs = [], None, None
                existing = _find_existing_images(images_dir, unique_id, ratio, False)
                if float(ratio) == 0.0:
                    os.makedirs("./generated_images", exist_ok=True)
                    blank_path = os.path.join("./generated_images", "blank_14x14.png")
                    if not os.path.exists(blank_path):
                        PIL_Image.new("RGB", (14, 14), color="white").save(blank_path)
                    current_image_paths = [os.path.abspath(blank_path)]
                    target_res = 14
                    fs = 0
                    result_filename_r = f"{model_name}_{config_name}_resize_ratio{ratio}_{target_res}x{target_res}_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                    resp_r, tk_info_r = call_llm_with_images(client, model_name, current_image_paths, system_prompt, user_prompt)
                    obj = extract_json_from_response(resp_r or "")
                    pred_r = None
                    if obj:
                        pred_r = bool(obj.get("is_same"))
                    result_file_r = os.path.join(output_dir, result_filename_r)
                    result_payload_r = {
                        "id": unique_id,
                        "file": file_path,
                        "label": label,
                        "dataset_type": item.get("dataset_type", ""),
                        "lang": lang,
                        "difficulty": difficulty,
                        "tier": tier,
                        "model": model_name,
                        "mode": f"image_ratio{ratio}",
                        "left": left,
                        "right": right,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "output": resp_r or "",
                        "clean_output": obj,
                        "tokens": tk_info_r,
                        "image_paths": current_image_paths,
                        "config": {
                            "width": width,
                            "height": height,
                            "resize_ratio": ratio,
                            "target_res": target_res,
                            "font_size": fs,
                            "line_height": line_height,
                            "dpi": dpi,
                            "preserve_newlines": preserve_newlines,
                            "enable_syntax_highlight": enable_syntax_highlight,
                            "language": lang,
                            "enable_bold": enable_bold,
                            "separate_mode": separate_mode,
                        },
                    }
                    with open(result_file_r, "w", encoding="utf-8") as f:
                        f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                    token_stats_item.append({
                        "model": model_name,
                        "config": f"{config_name}_resize_ratio{ratio}",
                        "unique_id": unique_id,
                        "mode": f"image_ratio{ratio}",
                        "image_tokens": calculate_image_tokens_from_paths(current_image_paths),
                        "image_tokens_processor": calculate_image_tokens_with_processor(current_image_paths, processor),
                        "text_tokens_left": get_text_tokens(left),
                        "text_tokens_right": get_text_tokens(right),
                        "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                        "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                        "prompt_text_tokens": prompt_text_tokens,
                        "completion_tokens": get_text_tokens(resp_r or ""),
                        "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                        "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                        "total_tokens": tk_info_r.get("total_tokens", 0),
                        "num_images": len(current_image_paths),
                    })
                    results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
                    continue
                if existing:
                    try:
                        for p in existing.get("paths", []):
                            print(f"use_cache: {os.path.basename(p)}")
                    except Exception:
                        pass
                    user_prompt_m = user_prompt
                    result_filename_r = f"{model_name}_{config_name}_resize_ratio{ratio}_{existing['res']}x{existing['res']}_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                    result_file_r = os.path.join(output_dir, result_filename_r)
                    resp_r = None
                    tk_info_r = {}
                    if existing_results_dir:
                        existing_file = os.path.join(existing_results_dir, result_filename_r)
                        if os.path.exists(existing_file):
                            try:
                                with open(existing_file, "r", encoding="utf-8") as f:
                                    result_payload_r = json.load(f)
                                resp_r = result_payload_r.get("output", "")
                                tk_info_r = result_payload_r.get("tokens", {})
                                obj = result_payload_r.get("clean_output")
                                if not obj:
                                    obj = extract_json_from_response(resp_r or "")
                                    result_payload_r["clean_output"] = obj
                            except Exception:
                                resp_r = None
                                tk_info_r = {}
                    if resp_r is None:
                        resp_r, tk_info_r = call_llm_with_images(client, model_name, existing["paths"], system_prompt, user_prompt_m)
                        obj = extract_json_from_response(resp_r or "")
                        result_payload_r = {
                            "id": unique_id,
                            "file": file_path,
                            "label": label,
                            "dataset_type": item.get("dataset_type", ""),
                            "lang": lang,
                            "difficulty": difficulty,
                            "tier": tier,
                            "model": model_name,
                            "mode": f"image_ratio{ratio}",
                            "left": left,
                            "right": right,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt_m,
                            "output": resp_r or "",
                            "clean_output": obj,
                            "tokens": tk_info_r,
                            "image_paths": existing["paths"],
                            "config": {
                                "width": width,
                                "height": height,
                                "resize_ratio": ratio,
                                "target_res": existing["res"],
                                "font_size": existing["fs"],
                                "line_height": line_height,
                                "dpi": dpi,
                                "preserve_newlines": preserve_newlines,
                                "enable_syntax_highlight": enable_syntax_highlight,
                                "language": lang,
                                "enable_bold": enable_bold,
                                "separate_mode": separate_mode,
                            },
                        }
                    with open(result_file_r, "w", encoding="utf-8") as f:
                        f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                    pred_r = None
                    if result_payload_r.get("clean_output"):
                        pred_r = bool(result_payload_r["clean_output"].get("is_same"))
                    token_stats_item.append({
                        "model": model_name,
                        "config": f"{config_name}_resize_ratio{ratio}",
                        "unique_id": unique_id,
                        "mode": f"image_ratio{ratio}",
                        "image_tokens": calculate_image_tokens_from_paths(existing["paths"]),
                        "image_tokens_processor": calculate_image_tokens_with_processor(existing["paths"], processor),
                        "text_tokens_left": get_text_tokens(left),
                        "text_tokens_right": get_text_tokens(right),
                        "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                        "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                        "prompt_text_tokens": prompt_text_tokens,
                        "completion_tokens": get_text_tokens(resp_r or ""),
                        "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                        "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                        "total_tokens": tk_info_r.get("total_tokens", 0),
                        "num_images": len(existing["paths"]),
                    })
                    results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
                    continue
                if ratio in resized:
                    res_imgs, target_res, fs = resized[ratio]
                    current_image_paths = []
                    for i, img in enumerate(res_imgs):
                        rp = os.path.join(images_dir, f"{unique_id}_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{target_res}x{target_res}_fs{fs}_page_{i+1:03d}.png")
                        save_img(img, rp)
                        current_image_paths.append(rp)
                    try:
                        import gc
                        res_imgs = []
                        gc.collect()
                    except Exception:
                        pass
                elif not extreme_mode:
                    if ratio == 1.0:
                        current_image_paths = list(base_1x_paths)
                        target_res = res_1x
                        fs = fs_1x
                    else:
                        image_token_limit = combined_text_tokens / ratio
                        num_images = len(base_1x_paths)
                        per_image_tokens = image_token_limit / num_images if num_images > 0 else image_token_limit
                        target_res = find_closest_resolution_prefer_larger(
                            per_image_tokens, get_expanded_resolution_list(), tolerance_ratio=1.4
                        )
                        fs = int(fs_1x * (target_res / res_1x)) if res_1x > 0 else 0
                        current_image_paths = []
                        for i, bp in enumerate(base_1x_paths, 1):
                            try:
                                with PIL_Image.open(bp) as im:
                                    resized_img = im.resize((target_res, target_res), PIL_Image.Resampling.LANCZOS)
                                rp = os.path.join(images_dir, f"{unique_id}_{'extreme' if extreme_mode else 'standard'}_ratio{ratio}_{target_res}x{target_res}_fs{fs}_page_{i:03d}.png")
                                resized_img.save(rp)
                                current_image_paths.append(rp)
                            except Exception:
                                continue
                else:
                    continue
                result_filename_r = f"{model_name}_{config_name}_resize_ratio{ratio}_{target_res}x{target_res}_lh{str(line_height).replace('.','_')}_{unique_id}_nl.txt"
                resp_r = None
                tk_info_r = {}
                if existing_results_dir:
                    existing_file = os.path.join(existing_results_dir, result_filename_r)
                    if os.path.exists(existing_file):
                        try:
                            with open(existing_file, "r", encoding="utf-8") as f:
                                existing_data = json.load(f)
                            resp_r = existing_data.get("output", "")
                            tk_info_r = existing_data.get("tokens", {})
                            existing_clean = existing_data.get("clean_output")
                            if (not resp_r or (isinstance(resp_r, str) and resp_r.strip() == "")) or (not existing_clean):
                                resp_r = None
                                tk_info_r = {}
                        except Exception as e:
                            print(f"Error reading existing file {existing_file}: {e}")
                if resp_r is None:
                    resp_r, tk_info_r = call_llm_with_images(client, model_name, current_image_paths, system_prompt, user_prompt)
                obj = extract_json_from_response(resp_r or "")
                pred_r = None
                if obj:
                    pred_r = bool(obj.get("is_same"))
                result_file_r = os.path.join(output_dir, result_filename_r)
                result_payload_r = {
                    "id": unique_id,
                    "file": file_path,
                    "label": label,
                    "dataset_type": item.get("dataset_type", ""),
                    "lang": lang,
                    "difficulty": difficulty,
                    "tier": tier,
                    "model": model_name,
                    "mode": f"image_ratio{ratio}",
                    "left": left,
                    "right": right,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "output": resp_r or "",
                    "clean_output": obj,
                    "tokens": tk_info_r,
                    "image_paths": current_image_paths,
                    "config": {
                        "width": width,
                        "height": height,
                        "resize_ratio": ratio,
                        "target_res": target_res,
                        "font_size": fs,
                        "line_height": line_height,
                        "dpi": dpi,
                        "preserve_newlines": preserve_newlines,
                        "enable_syntax_highlight": enable_syntax_highlight,
                        "language": lang,
                        "enable_bold": enable_bold,
                        "separate_mode": separate_mode,
                    },
                }
                with open(result_file_r, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result_payload_r, ensure_ascii=False, indent=2))
                token_stats_item.append({
                    "model": model_name,
                    "config": f"{config_name}_resize_ratio{ratio}",
                    "unique_id": unique_id,
                    "mode": f"image_ratio{ratio}",
                    "image_tokens": calculate_image_tokens_from_paths(current_image_paths),
                    "image_tokens_processor": calculate_image_tokens_with_processor(current_image_paths, processor),
                    "text_tokens_left": get_text_tokens(left),
                    "text_tokens_right": get_text_tokens(right),
                    "text_tokens_qwen_left": get_text_tokens_qwen(left, qwen_tokenizer),
                    "text_tokens_qwen_right": get_text_tokens_qwen(right, qwen_tokenizer),
                    "prompt_text_tokens": prompt_text_tokens,
                    "completion_tokens": get_text_tokens(resp_r or ""),
                    "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                    "api_image_tokens_estimate": max(0, tk_info_r.get("prompt_tokens", 0) - prompt_text_tokens),
                    "total_tokens": tk_info_r.get("total_tokens", 0),
                    "num_images": len(current_image_paths),
                })
                results_item[f"image_ratio{ratio}"] = {"pred": pred_r, "label": label}
    
    # All images already saved to disk during processing
    # Final cleanup
    import gc
    gc.collect()
    
    return results_item, token_stats_item


def run_code_clone_detection_task(
    model_name: str,
    output_dir: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: Optional[str],
    preserve_newlines: bool,
    resize_mode: bool,
    base_dir: str,
    dataset_type: str = "true",
    lang: str = "py",
    difficulty: str = "prompt_2",
    tier: str = "T4",
    num_examples: int = 200,
    processor=None,
    qwen_tokenizer=None,
    enable_text_only_test: bool = True,
    client_type: str = "OpenAI",
    enable_syntax_highlight: bool = False,
    enable_bold: bool = False,
    separate_mode: bool = False,
    max_workers: int = 20,
    evaluation_only_file: Optional[str] = None,    existing_results_dir: Optional[str] = None,
    target_ratios: Optional[List[float]] = None,
    theme: str = "",
    extreme_mode: bool = False,
) -> Tuple[Dict, List[Dict], str]:
    os.makedirs(output_dir, exist_ok=True)

    if evaluation_only_file:
        if not os.path.exists(evaluation_only_file):
            print(f"Error: Evaluation file not found: {evaluation_only_file}")
            return {}, [], output_dir
            
        print(f"Loading results from: {evaluation_only_file}")
        try:
            with open(evaluation_only_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if not isinstance(results, dict):
                print("Warning: Loaded data is not a dictionary. Expected results dictionary.")
                return {}, [], output_dir
                
            return results, [], output_dir
        except Exception as e:
            print(f"Error loading evaluation file: {e}")
            return {}, [], output_dir

    client = create_client(client_type)
    config_name = f"COMPACT_font{font_size}"
    lh_str = str(line_height).replace(".", "_")
    folder_parts_res = [
        f"code_clone_detection_{model_name.replace('/', '_slash_')}",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
    ]
    folder_kwargs = {
        "enable_syntax_highlight": enable_syntax_highlight,
        "preserve_newlines": preserve_newlines,
        "enable_bold": enable_bold,
    }
    if separate_mode:
        folder_kwargs["separate_mode"] = True

    output_dir = build_folder(output_dir, folder_parts_res, **folder_kwargs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    if dataset_type == "true":
        dataset = build_balanced_dataset(base_dir, lang, difficulty, tier, num_total=num_examples)
    else:
        dataset = load_code_clone_detection_pairs(base_dir, "false", lang, difficulty, tier, num_examples=num_examples)
    results = {"text_only": [], "image": []}
    token_stats = []
    images_base_dir = "./generated_images"
    images_folder_parts = [
        "code_clone_detection",
        "java" if lang == "java" else "python",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
    ]
    if theme:
        images_folder_parts.append(theme)
    images_dir = build_folder(images_base_dir, images_folder_parts, **folder_kwargs)
    os.makedirs(images_dir, exist_ok=True)
    
    # Process dataset in parallel
    print(f"Starting parallel processing of {len(dataset)} samples, workers: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(dataset):
            future = executor.submit(
                process_single_code_clone_detection_item,
                idx=idx,
                item=item,
                client=client,
                model_name=model_name,
                width=width,
                height=height,
                font_size=font_size,
                line_height=line_height,
                dpi=dpi,
                font_path=font_path,
                preserve_newlines=preserve_newlines,
                resize_mode=resize_mode,
                lang=lang,
                difficulty=difficulty,
                tier=tier,
                processor=processor,
                qwen_tokenizer=qwen_tokenizer,
                enable_text_only_test=enable_text_only_test,
                enable_syntax_highlight=enable_syntax_highlight,
                enable_bold=enable_bold,
                separate_mode=separate_mode,
                images_dir=images_dir,
                output_dir=output_dir,
                config_name=config_name,                existing_results_dir=existing_results_dir,
                theme=theme,
                extreme_mode=extreme_mode,
            )
            futures.append(future)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Semantic Clones ({model_name})"):
            try:
                results_item, token_stats_item = future.result()
                # Merge results
                for mode, value in results_item.items():
                    if mode not in results:
                        results[mode] = []
                    results[mode].append(value)
                token_stats.extend(token_stats_item)
            except Exception as e:
                print(f"Error processing task: {e}")
                import traceback
                traceback.print_exc()
    
    result_dir = output_dir
    return results, token_stats, result_dir


def evaluate_code_clone_detection_results(results: Dict) -> Dict:
    """Evaluate code clone detection results."""
    eval_res = {}
    for mode, items in results.items():
        if not items:
            continue
        correct = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        valid_total = 0
        for it in items:
            pred = it.get("pred")
            label = it.get("label")
            if pred is None:
                total += 1
                continue
            if pred == label:
                correct += 1
            total += 1
            valid_total += 1
            if label:
                if pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred:
                    fp += 1
                else:
                    tn += 1
        acc = (correct / total) if total > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        eval_res[mode] = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": total,
            "valid_total": valid_total,
        }
    return {"code_clone_detection": eval_res}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Clones image rendering test entry point")
    parser.add_argument("--render-file", type=str, required=True, help="Dataset file path")
    parser.add_argument("--output-dir", type=str, default="./generated_images", help="Output directory")
    parser.add_argument("--width", type=int, default=2240, help="Width")
    parser.add_argument("--height", type=int, default=2240, help="Height")
    parser.add_argument("--font-size", type=int, default=40, help="Font size")
    parser.add_argument("--line-height", type=float, default=1.0, help="Line height")
    parser.add_argument("--dpi", type=int, default=300, help="DPI")
    parser.add_argument("--font-path", type=str, default=None, help="Font path")
    parser.add_argument("--preserve-newlines", action="store_true", default=True, help="Preserve newlines")
    parser.add_argument("--enable-bold", action="store_true", help="Enable bold")
    parser.add_argument("--resize-mode", action="store_true", default=False, help="Enable resize")
    parser.add_argument("--enable-syntax-highlight", action="store_true", help="Enable syntax highlighting")
    parser.add_argument("--language", type=str, default="py", help="Programming language name")
    parser.add_argument("--separate-mode", action="store_true", default=False, help="Separate mode: render A/B segments separately")
    parser.add_argument("--skip-original-image", action="store_true", default=True, help="Skip original image generation (enabled by default)")
    args = parser.parse_args()
    paths = render_images_for_file(
        args.render_file,
        args.output_dir,
        args.width,
        args.height,
        args.font_size,
        args.line_height,
        args.dpi,
        args.font_path,
        args.preserve_newlines,
        args.enable_bold,
        args.resize_mode,
        args.enable_syntax_highlight,
        args.language,
        args.separate_mode,
    )
    for p in paths:
        print(p)
