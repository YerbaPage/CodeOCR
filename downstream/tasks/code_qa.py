"""
Code QA (Long Context Benchmark) task module.
Load LQA dataset and test text version, image version, resize version, and no-context version.
Supports Function RAG mode: retrieve most relevant code blocks from repo_text using function RAG.
"""
from text_to_image import (
    text_to_image,
    generate_compressed_images_dynamic,
    analyze_text_structure,
    COMPRESSION_RATIOS,
    find_closest_resolution_prefer_larger,
    get_expanded_resolution_list,
    optimize_layout_config_dry,
    calculate_image_tokens_qwen3,
    calculate_fill_rate,
    estimate_fill_rate,
)
from llm_utils import (
    create_client,
    build_folder,
    call_llm_with_images,
    call_llm_with_text_only,
    get_text_tokens,
    get_text_tokens_qwen,
    get_appropriate_device
)
from task_utils import function_rag_retrieve
from tasks.code_qa.constants import (
    DEFAULT_LQA_FILES,
    DEFAULT_NUM_EXAMPLES_PER_FILE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RAG_TOP_K,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
)
from tasks.code_qa.data import load_code_qa_data, extract_answer_letter
import os
import sys
import json
import re
import glob
from typing import List, Tuple, Dict, Optional
from PIL import Image as PIL_Image
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))



RAG_GPU_DEVICE = None


def process_single_code_qa_item(
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
    no_context_mode: bool,
    processor,
    qwen_tokenizer,
    enable_text_only_test: bool,
    enable_image_test: bool,
    enable_syntax_highlight: bool,
    language: Optional[str],
    enable_bold: bool,
    theme: str,
    images_dir: str,
    output_dir: str,
    config_name: str,
    extreme_mode: bool = False,
    # Function RAG related parameters
    enable_function_rag: bool = False,
    embed_model=None,
    embed_tokenizer=None,
    rag_device=None,
    rag_top_k: int = 3,
) -> Tuple[Dict, List[Dict]]:
    """Process a single code_qa task item."""

    # Extract data fields
    prompt = item.get("prompt", "")
    repo = item.get("repo", "")
    question = item.get("question", "")
    correct_letter = item.get("correct_letter", "")
    repo_text = item.get("repo_text", "")
    prompt_goal = item.get("prompt_goal", "")
    is_hard = item.get("is_hard", False)
    source_file = item.get("source_file", "unknown")

    unique_id = f"{source_file}_{idx:04d}"

    results_item = {}
    token_stats_item = []

    # System prompt (following original prompt field format)
    system_prompt = "You are going to be provided the content of a repository and a question about it. Provide the answer to the question by stating ONLY the letter associated to the question."

    # If Function RAG is enabled, use retrieved context
    retrieved_context = ""
    if enable_function_rag and embed_model is not None and embed_tokenizer is not None and repo_text:
        # Use question as query, retrieve most relevant code blocks from repo_text
        retrieved_context = function_rag_retrieve(
            repo_text,
            question,
            embed_model,
            embed_tokenizer,
            rag_device,
            language or "python",
            rag_top_k,
        )

    # Construct repo_text for testing (use retrieved results if RAG enabled, otherwise use original)
    effective_repo_text = retrieved_context if (
        enable_function_rag and retrieved_context) else repo_text

    # 1. Text-only test: construct prompt using effective_repo_text (mimicking original prompt field format)
    if enable_text_only_test:
        # Construct prompt following original data format: Repository: + repo_text + question
        user_prompt_text = f"Repository: {effective_repo_text}\n\n{question}"
        response_text, tk_info_t = call_llm_with_text_only(
            client, model_name, system_prompt, user_prompt_text, data_id=unique_id)

        pred_letter = extract_answer_letter(response_text)
        is_correct = pred_letter == correct_letter.upper(
        ) if pred_letter and correct_letter else False

        prompt_tokens_text = get_text_tokens(
            system_prompt + "\n" + user_prompt_text)

        result_payload_text = {
            "id": unique_id,
            "repo": repo,
            "question": question,
            "source_file": source_file,
            "correct_letter": correct_letter,
            "output": response_text or "",
            "pred_letter": pred_letter,
            "is_correct": is_correct,
            "mode": "text_only",
            "prompt_goal": prompt_goal,
            "is_hard": is_hard,
            "tokens": tk_info_t,
        }

        result_file = os.path.join(
            output_dir, f"{model_name}_{config_name}_{unique_id}_text_only.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_payload_text, f, ensure_ascii=False, indent=2)

        token_stats_item.append({
            "model": model_name,
            "config": config_name,
            "unique_id": unique_id,
            "mode": "text_only",
            "source_file": source_file,
            "prompt_text_tokens": prompt_tokens_text,
            "completion_tokens": get_text_tokens(response_text or ""),
            "api_prompt_tokens": tk_info_t.get("prompt_tokens", 0),
            "total_tokens": tk_info_t.get("total_tokens", 0),
            "is_correct": is_correct,
        })

        results_item["text_only"] = {
            "pred": pred_letter,
            "label": correct_letter,
            "is_correct": is_correct,
            "repo": repo,
            "question": question,
            "source_file": source_file,
            "output": response_text or "",
        }

    # 2. No-context test: use only question, without repo_text
    if no_context_mode:
        user_prompt_no_ctx = f"Please answer the following question:\n\n{question}\n\nReturn your answer as a single letter."
        response_no_ctx, tk_info_nc = call_llm_with_text_only(
            client, model_name, system_prompt, user_prompt_no_ctx, data_id=unique_id)

        pred_letter_nc = extract_answer_letter(response_no_ctx)
        is_correct_nc = pred_letter_nc == correct_letter.upper(
        ) if pred_letter_nc and correct_letter else False

        result_payload_nc = {
            "id": unique_id,
            "repo": repo,
            "question": question,
            "source_file": source_file,
            "correct_letter": correct_letter,
            "output": response_no_ctx or "",
            "pred_letter": pred_letter_nc,
            "is_correct": is_correct_nc,
            "mode": "no_context",
            "prompt_goal": prompt_goal,
            "is_hard": is_hard,
            "tokens": tk_info_nc,
        }

        result_file_nc = os.path.join(
            output_dir, f"{model_name}_{config_name}_{unique_id}_no_context.json")
        with open(result_file_nc, "w", encoding="utf-8") as f:
            json.dump(result_payload_nc, f, ensure_ascii=False, indent=2)

        token_stats_item.append({
            "model": model_name,
            "config": config_name,
            "unique_id": unique_id,
            "mode": "no_context",
            "source_file": source_file,
            "prompt_text_tokens": get_text_tokens(system_prompt + "\n" + user_prompt_no_ctx),
            "completion_tokens": get_text_tokens(response_no_ctx or ""),
            "api_prompt_tokens": tk_info_nc.get("prompt_tokens", 0),
            "total_tokens": tk_info_nc.get("total_tokens", 0),
            "is_correct": is_correct_nc,
        })

        results_item["no_context"] = {
            "pred": pred_letter_nc,
            "label": correct_letter,
            "is_correct": is_correct_nc,
            "repo": repo,
            "question": question,
            "source_file": source_file,
            "output": response_no_ctx or "",
        }

    # 3. Image test: render effective_repo_text as images, let LLM read and answer question
    if enable_image_test and effective_repo_text:
        # User prompt for image version (shared by original and resize)
        user_prompt_img = f"The images show some code/context. Please read them carefully and answer the following question:\n\n{question}\n\nReturn your answer as a single letter (A, B, C, D, etc.)."

        # Original image test: only in extreme_mode
        if extreme_mode:
            page_limit = 50 if ("glm" in model_name.lower()) else 100
            existing_original_paths = sorted(
                glob.glob(os.path.join(images_dir, f"{unique_id}_original_page_*.png")))
            if existing_original_paths:
                images = []
                image_paths = []
                for p in existing_original_paths[:page_limit]:
                    try:
                        img = PIL_Image.open(p)
                        images.append(img)
                        image_paths.append(os.path.abspath(p))
                    except Exception:
                        continue
            else:
                images = text_to_image(
                    effective_repo_text,
                    width=width,
                    height=height,
                    font_size=font_size,
                    line_height=line_height,
                    dpi=dpi,
                    font_path=font_path,
                    preserve_newlines=preserve_newlines,
                    enable_syntax_highlight=enable_syntax_highlight,
                    filename=None,
                    language=language or "python",
                    should_crop_whitespace=False,
                    enable_two_column=False,
                    enable_bold=enable_bold,
                    theme=theme,
                )
                images = images[:page_limit]
                image_paths = []
                for i, img in enumerate(images, 1):
                    img_path = os.path.join(
                        images_dir, f"{unique_id}_original_page_{i:03d}.png")
                    img.save(img_path)
                    image_paths.append(os.path.abspath(img_path))

            response_img, tk_info_img = call_llm_with_images(
                client, model_name, images, system_prompt, user_prompt_img, data_id=unique_id)

            pred_letter_img = extract_answer_letter(response_img)
            is_correct_img = pred_letter_img == correct_letter.upper(
            ) if pred_letter_img and correct_letter else False

            result_payload_img = {
                "id": unique_id,
                "repo": repo,
                "question": question,
                "source_file": source_file,
                "correct_letter": correct_letter,
                "output": response_img or "",
                "pred_letter": pred_letter_img,
                "is_correct": is_correct_img,
                "mode": "image",
                "prompt_goal": prompt_goal,
                "is_hard": is_hard,
                "tokens": tk_info_img,
                "image_paths": image_paths,
                "num_pages": len(images),
            }

            result_file_img = os.path.join(
                output_dir, f"{model_name}_{config_name}_{unique_id}_image.json")
            with open(result_file_img, "w", encoding="utf-8") as f:
                json.dump(result_payload_img, f, ensure_ascii=False, indent=2)

            token_stats_item.append({
                "model": model_name,
                "config": config_name,
                "unique_id": unique_id,
                "mode": "image",
                "source_file": source_file,
                "prompt_text_tokens": get_text_tokens(system_prompt + "\n" + user_prompt_img),
                "completion_tokens": get_text_tokens(response_img or ""),
                "api_prompt_tokens": tk_info_img.get("prompt_tokens", 0),
                "total_tokens": tk_info_img.get("total_tokens", 0),
                "is_correct": is_correct_img,
                "num_images": len(images),
            })

            results_item["image"] = {
                "pred": pred_letter_img,
                "label": correct_letter,
                "is_correct": is_correct_img,
                "repo": repo,
                "question": question,
                "source_file": source_file,
                "output": response_img or "",
            }

        # 4. Resize mode: choose compression method based on extreme_mode
        if resize_mode:
            page_limit = 50 if ("glm" in model_name.lower()) else 100
            
            if extreme_mode:
                # Extreme mode: dynamically re-render at different resolutions
                ratio_existing = {}
                ratios_to_generate = []
                for ratio in COMPRESSION_RATIOS:
                    pattern = os.path.join(
                        images_dir, f"{unique_id}_ratio{ratio}_*x*_fs*_page_*.png")
                    paths = glob.glob(pattern)
                    if not paths:
                        ratios_to_generate.append(ratio)
                        continue
                    cfg_map: Dict[Tuple[int, int], List[str]] = {}
                    for p in paths:
                        m = re.search(
                            rf"{re.escape(unique_id)}_ratio{ratio}_(\d+)x\1_fs(\d+)_page_(\d+)\.png$", os.path.basename(p))
                        if not m:
                            continue
                        res = int(m.group(1))
                        fs = int(m.group(2))
                        cfg_map.setdefault((res, fs), []).append(p)
                    if len(cfg_map) == 0:
                        ratios_to_generate.append(ratio)
                        continue
                    if len(cfg_map) > 1:
                        for p in paths:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                        ratios_to_generate.append(ratio)
                        continue
                    (res, fs), files = next(iter(cfg_map.items()))
                    files_sorted = sorted(files, key=lambda x: int(re.search(r"_page_(\d+)\.png$", os.path.basename(
                        x)).group(1)) if re.search(r"_page_(\d+)\.png$", os.path.basename(x)) else 0)
                    ratio_existing[ratio] = (res, fs, files_sorted)
                if ratios_to_generate:
                    text_tokens = get_text_tokens(effective_repo_text)
                    text_structure = analyze_text_structure(effective_repo_text)

                    def renderer(w, h, fs):
                        margin = int(w * 0.01)
                        return text_to_image(
                            effective_repo_text,
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
                            should_crop_whitespace=False,
                            enable_two_column=False,
                            enable_bold=enable_bold,
                            theme=theme,
                        )
                    resized_results = generate_compressed_images_dynamic(
                        text_tokens, renderer, ratios_to_generate, text_structure=text_structure, data_id=unique_id, page_limit=page_limit
                    )
                else:
                    resized_results = {}
            else:
                text_tokens = get_text_tokens(effective_repo_text)
                ratio_existing = {}
                for ratio in COMPRESSION_RATIOS:
                    pattern = os.path.join(
                        images_dir, f"{unique_id}_ratio{ratio}_*x*_fs*_page_*.png")
                    paths = glob.glob(pattern)
                    if not paths:
                        continue
                    cfg_map: Dict[Tuple[int, int], List[str]] = {}
                    for p in paths:
                        m = re.search(
                            rf"{re.escape(unique_id)}_ratio{ratio}_(\d+)x\1_fs(\d+)_page_(\d+)\.png$", os.path.basename(p))
                        if not m:
                            continue
                        res = int(m.group(1))
                        fs = int(m.group(2))
                        cfg_map.setdefault((res, fs), []).append(p)
                    if len(cfg_map) == 0:
                        continue
                    if len(cfg_map) > 1:
                        for p in paths:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                        continue
                    (res, fs), files = next(iter(cfg_map.items()))
                    files_sorted = sorted(files, key=lambda x: int(re.search(r"_page_(\d+)\.png$", os.path.basename(
                        x)).group(1)) if re.search(r"_page_(\d+)\.png$", os.path.basename(x)) else 0)
                    ratio_existing[ratio] = (res, fs, files_sorted)
                text_structure = analyze_text_structure(effective_repo_text)
                base_ratio = None
                for r in COMPRESSION_RATIOS:
                    if abs(float(r) - 1.0) < 1e-9:
                        base_ratio = r
                        break
                if base_ratio is None:
                    base_ratio = 1.0
                res_1x = None
                fs_1x = None
                base_1x_paths: List[str] = []
                if base_ratio in ratio_existing:
                    res_1x, fs_1x, files_sorted = ratio_existing[base_ratio]
                    base_1x_paths = files_sorted[:page_limit]
                    base_start = time.time()
                    tk_1x = len(base_1x_paths) * calculate_image_tokens_qwen3(res_1x, res_1x)
                    tgt_1x = text_tokens / float(base_ratio)
                    if text_structure:
                        fr_1x = calculate_fill_rate(
                            fs_1x, res_1x, len(base_1x_paths),
                            text_structure.get("num_lines", 0),
                            text_structure.get("avg_line_chars", 0),
                            line_height
                        )
                    else:
                        fr_1x = estimate_fill_rate(text_tokens, res_1x, fs_1x, line_height)
                    elapsed_1x = time.time() - base_start
                    print(f"[{unique_id}_1x] Ratio {base_ratio}: Res {res_1x}x{res_1x}, Count {len(base_1x_paths)}, Font {fs_1x}, Fill {int(fr_1x*100)}%, Tokens {tk_1x} (Target {tgt_1x:.1f}) [Time: {elapsed_1x:.3f}s]")
                    import sys as _sys
                    _sys.stdout.flush()
                else:
                    text_structure = analyze_text_structure(effective_repo_text)
                    base_start = time.time()
                    res_1x, fs_1x, _ = optimize_layout_config_dry(
                        target_tokens=text_tokens,
                        previous_configs=[],
                        text_tokens=text_tokens,
                        line_height=line_height,
                        text_structure=text_structure,
                        compression_ratio=1.0,
                        page_limit=page_limit,
                        text=effective_repo_text,
                        enable_syntax_highlight=enable_syntax_highlight,
                        language=language or "python",
                        preserve_newlines=preserve_newlines,
                        font_path=font_path,
                        theme=theme,
                    )
                    base_imgs = text_to_image(
                        effective_repo_text,
                        width=res_1x,
                        height=res_1x,
                        font_size=fs_1x,
                        line_height=line_height,
                        dpi=dpi,
                        font_path=font_path,
                        preserve_newlines=preserve_newlines,
                        enable_syntax_highlight=enable_syntax_highlight,
                        filename=None,
                        language=language or "python",
                        should_crop_whitespace=False,
                        enable_two_column=False,
                        enable_bold=enable_bold,
                        theme=theme,
                    )
                    base_imgs = base_imgs[:page_limit]
                    for i, img in enumerate(base_imgs, 1):
                        bp = os.path.join(
                            images_dir, f"{unique_id}_ratio{base_ratio}_{res_1x}x{res_1x}_fs{fs_1x}_page_{i:03d}.png")
                        img.save(bp)
                        base_1x_paths.append(bp)
                    tk_1x = len(base_1x_paths) * calculate_image_tokens_qwen3(res_1x, res_1x)
                    tgt_1x = text_tokens / float(base_ratio)
                    if text_structure:
                        fr_1x = calculate_fill_rate(
                            fs_1x, res_1x, len(base_1x_paths),
                            text_structure.get("num_lines", 0),
                            text_structure.get("avg_line_chars", 0),
                            line_height
                        )
                    else:
                        fr_1x = estimate_fill_rate(text_tokens, res_1x, fs_1x, line_height)
                    elapsed_1x = time.time() - base_start
                    print(f"[{unique_id}_1x] Ratio {base_ratio}: Res {res_1x}x{res_1x}, Count {len(base_1x_paths)}, Font {fs_1x}, Fill {int(fr_1x*100)}%, Tokens {tk_1x} (Target {tgt_1x:.1f}) [Time: {elapsed_1x:.3f}s]")
                    import sys as _sys
                    _sys.stdout.flush()
                resized_results = {}
            
            for ratio in COMPRESSION_RATIOS:
                ratio_start = time.time()
                if abs(float(ratio)) < 1e-9:
                    os.makedirs("./generated_images", exist_ok=True)
                    blank_path = os.path.join("./generated_images", "blank_14x14.png")
                    if not os.path.exists(blank_path):
                        PIL_Image.new("RGB", (14, 14), color="white").save(blank_path)
                    target_res = 14
                    fs = 0
                    res_imgs = [blank_path]
                    resize_image_paths = [os.path.abspath(blank_path)]
                elif ratio in ratio_existing:
                    target_res, fs, files_sorted = ratio_existing[ratio]
                    res_imgs = []
                    resize_image_paths = []
                    for p in files_sorted[:page_limit]:
                        try:
                            img = PIL_Image.open(p)
                            res_imgs.append(img)
                            resize_image_paths.append(os.path.abspath(p))
                        except Exception:
                            continue
                else:
                    image_token_limit = text_tokens / float(ratio)
                    base_count = len(base_1x_paths)
                    candidates = []
                    if base_count >= 2:
                        candidates.append(2)
                    candidates.append(1)
                    if base_count not in candidates:
                        candidates.append(base_count)
                    desired_count = None
                    chosen_res = None
                    for c in candidates:
                        per_image_tokens_c = image_token_limit / c if c > 0 else image_token_limit
                        cand_res = find_closest_resolution_prefer_larger(
                            per_image_tokens_c, get_expanded_resolution_list(), tolerance_ratio=1.4
                        )
                        if cand_res > 980:
                            desired_count = c
                            chosen_res = cand_res
                            break
                    if desired_count is None:
                        desired_count = 1 if base_count >= 1 else base_count
                        per_image_tokens_default = image_token_limit / desired_count if desired_count > 0 else image_token_limit
                        chosen_res = find_closest_resolution_prefer_larger(
                            per_image_tokens_default, get_expanded_resolution_list(), tolerance_ratio=1.4
                        )
                    selected_paths = base_1x_paths[:desired_count]
                    if desired_count == 2 and base_count == 1:
                        selected_paths = base_1x_paths * 2
                    if abs(float(ratio) - float(base_ratio)) < 1e-9:
                        target_res = res_1x
                        fs = fs_1x
                        resize_image_paths = [os.path.abspath(p) for p in selected_paths]
                        res_imgs = []
                        for p in resize_image_paths:
                            try:
                                img = PIL_Image.open(p)
                                res_imgs.append(img)
                            except Exception:
                                continue
                    else:
                        target_res = chosen_res
                        fs = int(fs_1x * (target_res / res_1x)) if res_1x and res_1x > 0 else 0
                        resize_image_paths = []
                        res_imgs = []
                        for i, bp in enumerate(selected_paths, 1):
                            try:
                                with PIL_Image.open(bp) as im:
                                    resized_img = im.resize((target_res, target_res), PIL_Image.Resampling.LANCZOS)
                                rp = os.path.join(
                                    images_dir, f"{unique_id}_ratio{ratio}_{target_res}x{target_res}_fs{fs}_page_{i:03d}.png")
                                resized_img.save(rp)
                                resize_image_paths.append(os.path.abspath(rp))
                                res_imgs.append(PIL_Image.open(rp))
                            except Exception:
                                continue
                ratio_elapsed = time.time() - ratio_start
                actual_tokens = len(res_imgs) * calculate_image_tokens_qwen3(target_res, target_res)
                target_tokens = 0.0 if abs(float(ratio)) < 1e-9 else (text_tokens / float(ratio))
                if text_structure:
                    fill_rate = calculate_fill_rate(
                        fs, target_res, len(res_imgs),
                        text_structure.get("num_lines", 0),
                        text_structure.get("avg_line_chars", 0),
                        line_height
                    )
                else:
                    fill_rate = 0.0 if abs(float(ratio)) < 1e-9 else estimate_fill_rate(text_tokens, target_res, fs, line_height)
                print(f"[{unique_id}] Ratio {ratio}: Res {target_res}x{target_res}, Count {len(res_imgs)}, Font {fs}, Fill {int(fill_rate*100)}%, Tokens {actual_tokens} (Target {target_tokens:.1f}) [Time: {ratio_elapsed:.3f}s]")
                import sys as _sys
                _sys.stdout.flush()
                response_r, tk_info_r = call_llm_with_images(
                    client, model_name, res_imgs, system_prompt, user_prompt_img, data_id=unique_id)
                pred_letter_r = extract_answer_letter(response_r)
                is_correct_r = pred_letter_r == correct_letter.upper(
                ) if pred_letter_r and correct_letter else False
                mode_key = f"image_ratio{ratio}"
                result_payload_r = {
                    "id": unique_id,
                    "repo": repo,
                    "question": question,
                    "source_file": source_file,
                    "correct_letter": correct_letter,
                    "output": response_r or "",
                    "pred_letter": pred_letter_r,
                    "is_correct": is_correct_r,
                    "mode": mode_key,
                    "compression_ratio": ratio,
                    "resolution": f"{target_res}x{target_res}",
                    "font_size": fs,
                    "prompt_goal": prompt_goal,
                    "is_hard": is_hard,
                    "tokens": tk_info_r,
                    "image_paths": resize_image_paths,
                    "num_pages": len(res_imgs),
                }
                result_file_r = os.path.join(
                    output_dir, f"{model_name}_{config_name}_{unique_id}_{mode_key}.json")
                with open(result_file_r, "w", encoding="utf-8") as f:
                    json.dump(result_payload_r, f,
                              ensure_ascii=False, indent=2)
                token_stats_item.append({
                    "model": model_name,
                    "config": config_name,
                    "unique_id": unique_id,
                    "mode": mode_key,
                    "source_file": source_file,
                    "compression_ratio": ratio,
                    "resolution": f"{target_res}x{target_res}",
                    "prompt_text_tokens": get_text_tokens(system_prompt + "\n" + user_prompt_img),
                    "completion_tokens": get_text_tokens(response_r or ""),
                    "api_prompt_tokens": tk_info_r.get("prompt_tokens", 0),
                    "total_tokens": tk_info_r.get("total_tokens", 0),
                    "is_correct": is_correct_r,
                    "num_images": len(res_imgs),
                })
                results_item[mode_key] = {
                    "pred": pred_letter_r,
                    "label": correct_letter,
                    "is_correct": is_correct_r,
                    "repo": repo,
                    "question": question,
                    "source_file": source_file,
                    "output": response_r or "",
                }

    return results_item, token_stats_item


def run_code_qa_task(
    model_name: str,
    output_dir: str,
    width: int = 2240,
    height: int = 2240,
    font_size: int = 40,
    line_height: float = 1.0,
    dpi: int = 300,
    font_path: Optional[str] = None,
    preserve_newlines: bool = True,
    resize_mode: bool = True,
    no_context_mode: bool = True,
    qa_dir: str = "./dataset/code_qa",
    processor=None,
    qwen_tokenizer=None,
    enable_text_only_test: bool = True,
    enable_image_test: bool = True,
    client_type: str = "OpenAI",
    enable_syntax_highlight: bool = False,
    language: Optional[str] = None,
    enable_bold: bool = False,
    theme: str = "light",
    max_workers: int = 20,
    test_single: bool = False,
    # Function RAG related parameters
    enable_function_rag: bool = False,
    embed_model=None,
    embed_tokenizer=None,
    rag_top_k: int = 3,
    indices_to_run: Optional[List[int]] = None,
    prev_result_dir: Optional[str] = None,
    extreme_mode: bool = False,
) -> Tuple[Dict, List[Dict], str]:
    """
    Run Code QA task.

    Args:
        model_name: Model name
        output_dir: Output directory
        width: Image width
        height: Image height
        font_size: Font size
        line_height: Line height
        dpi: DPI
        font_path: Font path
        preserve_newlines: Preserve newlines
        resize_mode: Enable dynamic compression mode
        no_context_mode: Enable no-context test mode
        lqa_dir: LQA data directory
        file_list: List of files to load
        num_examples_per_file: Number of examples to read per file
        processor: Qwen processor
        qwen_tokenizer: Qwen tokenizer
        enable_text_only_test: Enable text test
        enable_image_test: Enable image test
        client_type: API client type
        enable_syntax_highlight: Enable syntax highlighting
        language: Programming language
        enable_bold: Enable bold text
        max_workers: Number of concurrent workers
        test_single: Test mode, process only first entry
        enable_function_rag: Enable Function RAG mode
        embed_model: Embedding model (for Function RAG)
        embed_tokenizer: Embedding model tokenizer (for Function RAG)
        rag_top_k: Number of top-k code blocks to retrieve in RAG
        indices_to_run: Specify list of data indices to run (0-based). If provided, only rerun these indices, others try loading from disk.
        prev_result_dir: Specify directory with existing results. If indices_to_run is specified, will load skipped data results from this directory.

    Returns:
        Tuple of (results, token_stats, result_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    client = create_client(client_type)

    # Initialize RAG device
    global RAG_GPU_DEVICE
    rag_device = None
    if enable_function_rag:
        if RAG_GPU_DEVICE is None:
            RAG_GPU_DEVICE = get_appropriate_device()
        rag_device = RAG_GPU_DEVICE
        print(f"✓ Function RAG enabled, using device: {rag_device}, top_k={rag_top_k}")

    config_name = f"COMPACT_font{font_size}"
    lh_str = str(line_height).replace(".", "_")

    folder_parts = [
        f"code_qa_{model_name.replace('/', '_slash_')}",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
    ]
    if theme == "modern" or theme == "morden":
        folder_parts.append("mor")

    folder_kwargs = {
        "enable_syntax_highlight": enable_syntax_highlight,
        "preserve_newlines": preserve_newlines,
        "enable_bold": enable_bold,
    }

    result_dir = build_folder(output_dir, folder_parts, **folder_kwargs)
    os.makedirs(result_dir, exist_ok=True)

    # Load data
    dataset = load_code_qa_data(qa_dir)

    if test_single:
        dataset = dataset[:1]
        print(f"✓ Test mode: processing only 1 entry")

    if not dataset:
        print("Warning: No data loaded")
        return {}, [], result_dir

    # Image output directory
    images_base_dir = "./generated_images"
    images_folder_parts = [
        "code_qa",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}",
    ]
    if theme == "modern" or theme == "morden":
        images_folder_parts.append("mor")
    
    images_dir = build_folder(
        images_base_dir, images_folder_parts, **folder_kwargs)
    os.makedirs(images_dir, exist_ok=True)

    results = {}
    token_stats = []
    results_lock = Lock()

    if indices_to_run is not None:
        print(f"Note: Specified to run only indices {indices_to_run}")
        if prev_result_dir:
            print(f"  - Loading existing results from specified directory: {prev_result_dir}")
        else:
            print(f"  - Loading existing results from current output directory: {result_dir}")

    print(f"Starting parallel processing of {len(dataset)} samples, workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(dataset):
            # Check if should skip (load existing results)
            if indices_to_run is not None and idx not in indices_to_run:
                # Try loading results from disk
                source_file = item.get("source_file", "unknown")
                unique_id = f"{source_file}_{idx:04d}"

                # Determine search directory
                search_dir = prev_result_dir if prev_result_dir else result_dir

                # All modes to check
                modes_to_check = []
                if enable_text_only_test:
                    modes_to_check.append("text_only")
                if no_context_mode:
                    modes_to_check.append("no_context")
                if enable_image_test:
                    if extreme_mode:
                        modes_to_check.append("image")
                    if resize_mode:
                        for ratio in COMPRESSION_RATIOS:
                            modes_to_check.append(f"image_ratio{ratio}")

                for mode in modes_to_check:
                    result_file = os.path.join(
                        search_dir, f"{model_name}_{config_name}_{unique_id}_{mode}.json")
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                loaded_result = json.load(f)

                            # Construct simplified result item for evaluation
                            # Note: reconstructing fields needed by evaluate_code_qa_results
                            simplified_result = {
                                "pred": loaded_result.get("pred_letter"),
                                "label": loaded_result.get("correct_letter"),
                                "is_correct": loaded_result.get("is_correct"),
                                "repo": loaded_result.get("repo"),
                                "question": loaded_result.get("question"),
                                "source_file": loaded_result.get("source_file"),
                                "output": loaded_result.get("output"),
                            }

                            with results_lock:
                                if mode not in results:
                                    results[mode] = []
                                results[mode].append(simplified_result)

                            # Note: we don't load token_stats here since evaluation only needs results
                            # If token_stats are needed, they can also be recovered from loaded_result, but field structure differs

                        except Exception as e:
                            print(f"Warning: Failed to load existing result {result_file}: {e}")

                continue

            future = executor.submit(
                process_single_code_qa_item,
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
                no_context_mode=no_context_mode,
                processor=processor,
                qwen_tokenizer=qwen_tokenizer,
                enable_text_only_test=enable_text_only_test,
                enable_image_test=enable_image_test,
                enable_syntax_highlight=enable_syntax_highlight,
                language=language,
                enable_bold=enable_bold,
                theme=theme,
                images_dir=images_dir,
                output_dir=result_dir,
                config_name=config_name,
                extreme_mode=extreme_mode,
                # Function RAG related parameters
                enable_function_rag=enable_function_rag,
                embed_model=embed_model,
                embed_tokenizer=embed_tokenizer,
                rag_device=rag_device,
                rag_top_k=rag_top_k,
            )
            futures.append(future)

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Code QA ({model_name})"):
            try:
                results_item, token_stats_item = future.result()
                # Merge results
                with results_lock:
                    for mode, value in results_item.items():
                        if mode not in results:
                            results[mode] = []
                        results[mode].append(value)
                    token_stats.extend(token_stats_item)
            except Exception as e:
                print(f"Error processing task: {e}")
                import traceback
                traceback.print_exc()

    return results, token_stats, result_dir


def evaluate_code_qa_results(results: Dict) -> Dict:
    """
    Evaluate Code QA results and calculate accuracy.

    Args:
        results: Results grouped by mode

    Returns:
        Dictionary of evaluation results
    """
    eval_res = {}

    for mode, items in results.items():
        if not items:
            continue

        correct = 0
        total = 0
        by_source = {}  # Statistics by source file

        for it in items:
            is_correct = it.get("is_correct", False)
            source_file = it.get("source_file", "unknown")

            if is_correct:
                correct += 1
            total += 1

            # Statistics by source file
            if source_file not in by_source:
                by_source[source_file] = {"correct": 0, "total": 0}
            by_source[source_file]["total"] += 1
            if is_correct:
                by_source[source_file]["correct"] += 1

        accuracy = correct / total if total > 0 else 0.0

        # Calculate accuracy for each source
        source_accuracy = {}
        for source, stats in by_source.items():
            source_accuracy[source] = stats["correct"] / \
                stats["total"] if stats["total"] > 0 else 0.0

        eval_res[mode] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_source": source_accuracy,
        }

    return {"code_qa": eval_res}


def print_code_qa_results(eval_results: Dict, model_name: str):
    """
    Print Code QA results.

    Args:
        eval_results: Evaluation results
        model_name: Model name
    """
    print("\n" + "=" * 70)
    print(f"Code QA Results - Model: {model_name}")
    print("=" * 70)

    code_qa_res = eval_results.get("code_qa", {})

    if not code_qa_res:
        print("  No results")
        return

    # Sort by mode
    modes = sorted(code_qa_res.keys())

    for mode in modes:
        stats = code_qa_res[mode]
        accuracy = stats.get("accuracy", 0.0)
        correct = stats.get("correct", 0)
        total = stats.get("total", 0)
        by_source = stats.get("by_source", {})

        print(f"\n[{mode}]")
        print(f"  Overall accuracy: {accuracy:.4f} ({correct}/{total})")

        if by_source:
            print("  By source file:")
            for source, source_acc in sorted(by_source.items()):
                print(f"    {source}: {source_acc:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code QA task test entry point")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--output-dir", type=str,
                        default="./llm_outputs/code_qa", help="Output directory")
    parser.add_argument("--lqa-dir", type=str,
                        default="./LQA", help="LQA data directory")
    parser.add_argument("--file-list", type=str,
                        default="32K,64K", help="File list, comma-separated")
    parser.add_argument("--num-examples", type=int,
                        default=100, help="Number of examples per file")
    parser.add_argument("--width", type=int, default=2240, help="Image width")
    parser.add_argument("--height", type=int, default=2240, help="Image height")
    parser.add_argument("--font-size", type=int, default=40, help="Font size")
    parser.add_argument("--test-single", action="store_true", help="Test mode")
    parser.add_argument("--client-type", type=str,
                        default="OpenAI", help="API client type")
    # Function RAG related parameters
    parser.add_argument("--enable-function-rag",
                        action="store_true", help="Enable Function RAG mode")
    parser.add_argument("--rag-top-k", type=int, default=3,
                        help="Number of top-k code blocks to retrieve in RAG")
    parser.add_argument("--embed-model-name", type=str,
                        default="microsoft/codebert-base", help="Embedding model name")
    parser.add_argument("--indices", type=str, default=None,
                        help="Specify index list to run, comma-separated, e.g. '1,2,4'")
    parser.add_argument("--prev-result-dir", type=str,
                        default=None, help="Specify directory with existing results (when using --indices)")

    args = parser.parse_args()

    file_list = [f.strip() for f in args.file_list.split(",")]

    indices_to_run = None
    if args.indices:
        try:
            indices_to_run = [int(i.strip())
                              for i in args.indices.split(",") if i.strip()]
            print(f"✓ Will run only the following indices: {indices_to_run}")
        except ValueError:
            print("Error: --indices parameter format incorrect, should be comma-separated integers")
            sys.exit(1)

    # Initialize embedding model (if Function RAG enabled)
    embed_model = None
    embed_tokenizer = None
    if args.enable_function_rag:
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            print("Error: Function RAG requires torch and transformers libraries")
            sys.exit(1)
        print(f"Loading embedding model: {args.embed_model_name}...")
        embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model_name)
        embed_model = AutoModel.from_pretrained(args.embed_model_name)
        device = get_appropriate_device()
        if device:
            embed_model = embed_model.to(device)
        embed_model.eval()
        print(f"✓ Embedding model loaded")

    results, token_stats, result_dir = run_code_qa_task(
        model_name=args.model,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        font_size=args.font_size,
        lqa_dir=args.lqa_dir,
        file_list=file_list,
        num_examples_per_file=args.num_examples,
        test_single=args.test_single,
        client_type=args.client_type,
        enable_function_rag=args.enable_function_rag,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
        rag_top_k=args.rag_top_k,
        indices_to_run=indices_to_run,
        prev_result_dir=args.prev_result_dir,
    )

    eval_results = evaluate_code_qa_results(results)
    print_code_qa_results(eval_results, args.model)

    # Save evaluation results
    eval_file = os.path.join(result_dir, "evaluation_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to: {eval_file}")
