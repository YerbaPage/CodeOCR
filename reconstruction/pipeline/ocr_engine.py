import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .config import (
    GEMINI_MODEL_NAME, OCR_TEMPERATURE, OCR_MAX_TOKENS, 
    OCR_SYSTEM_PROMPT, OCR_MAX_RETRIES, OCR_CONCURRENCY,
    OCR_PARALLEL_MIN_INTERVAL_SECONDS, OCR_SLEEP_SECONDS,
    GEMINI_ENABLE_SAFETY_SETTINGS, GEMINI_SAFETY_SETTINGS
)
from .utils import (
    _create_openai_client, _try_load_api_key_from_env_files, _mask_api_key,
    _load_done_set, _iter_image_files, _parse_ratio_from_filename,
    _extract_page_num_from_filename, _get_ocr_user_prompt,
    _encode_image_to_data_url, _clean_ocr_text, _extract_response_diagnostics
)

def _retry_ocr_single_case(code_id: str, ratio, image_paths: list[str], client=None) -> dict:
    """
    Re-run OCR for a single test case.

    Args:
        code_id: code ID
        ratio: compression ratio
        image_paths: list of image paths
        client: OpenAI client (optional; created internally if omitted)

    Returns:
        dict: OCR record containing text or error
    """
    if client is None:
        client = _create_openai_client()
        if client is None:
            return {"code_id": code_id, "ratio": ratio, "error": "Failed to create OpenAI client"}
    
    # Build request content
    content = [{"type": "text", "text": _get_ocr_user_prompt()}]
    for p in image_paths:
        if os.path.exists(p):
            data_url = _encode_image_to_data_url(p)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    
    if len(content) == 1:  # Text only, no images
        return {"code_id": code_id, "ratio": ratio, "error": "No valid image files found"}
    
    last_err = None
    text = ""
    diagnostics = {}
    
    for attempt in range(1, OCR_MAX_RETRIES + 1):
        try:
            extra_body = {"safety_settings": GEMINI_SAFETY_SETTINGS} if GEMINI_ENABLE_SAFETY_SETTINGS else None
            resp = client.chat.completions.create(
                model=GEMINI_MODEL_NAME,
                temperature=OCR_TEMPERATURE,
                max_tokens=OCR_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                extra_body=extra_body,
            )
            text = _clean_ocr_text(resp.choices[0].message.content or "")
            diagnostics = _extract_response_diagnostics(resp)
            last_err = None
            break
        except Exception as e:
            last_err = str(e)
            backoff = min(30.0, float(2 ** (attempt - 1)))
            time.sleep(backoff)
    
    rec = {
        "code_id": code_id,
        "ratio": ratio,
        "num_pages": len(image_paths),
        "image_paths": image_paths,
        "image_path": image_paths[0] if image_paths else "",
        "model": GEMINI_MODEL_NAME,
    }
    
    if diagnostics:
        rec.update(diagnostics)
    
    if last_err is None:
        rec["text"] = text
        rec["text_len"] = len(text)
        if rec.get("finish_reason") in ("content_filter", "safety"):
            rec["blocked_by_safety"] = True
    else:
        rec["error"] = last_err
    
    return rec


def run_module_3_gemini(images_dir: str, output_dir: str, continue_from: str = None):
    """
    Args:
        images_dir: images directory
        output_dir: output directory
        continue_from: optional; resume from an OCR results file (checkpoint)
    """
    print("\n" + "=" * 40)
    print(f"🚀 Running Module 3: Inference Engine ({GEMINI_MODEL_NAME})")
    print("=" * 40)

    # Load API key for display and parallel mode
    api_key = os.getenv("AIHUBMIX_API_KEY") or _try_load_api_key_from_env_files()
    if not api_key:
        print("❌ Missing AIHUBMIX_API_KEY.")
        return

    print(f"🔑 API_KEY loaded: {_mask_api_key(api_key)}")

    os.makedirs(output_dir, exist_ok=True)
    
    # If continue_from is specified, append to that file
    timestamp = time.strftime('%m%d_%H%M%S')
    if continue_from:
        if os.path.isabs(continue_from):
            out_jsonl = continue_from
        else:
            out_jsonl = os.path.join(output_dir, continue_from)
        print(f"📂 Continuing from: {out_jsonl}")
    else:
        out_jsonl = os.path.join(output_dir, f"qwen_ocr_{timestamp}.jsonl")
    
    done = _load_done_set(out_jsonl)

    client = _create_openai_client(api_key)
    if client is None:
        return

    total = 0
    skipped = 0
    errors = 0

    # single-turn: group by (code_id, ratio) and send all pages in one request
    image_paths = list(_iter_image_files(images_dir))

    grouped_images = defaultdict(list)  # (code_id, ratio) -> [image_path...]
    for image_path in image_paths:
        parent_dir = os.path.dirname(image_path)
        code_id_dir = os.path.dirname(parent_dir)
        code_id = os.path.basename(code_id_dir)
        ratio = _parse_ratio_from_filename(image_path)
        grouped_images[(code_id, ratio)].append(image_path)

    cases = []  # [(code_id, ratio, [paths...])]
    for (code_id, ratio), paths in grouped_images.items():
        paths.sort(key=lambda p: (_extract_page_num_from_filename(p), os.path.basename(p)))
        cases.append((code_id, ratio, paths))
    cases.sort(key=lambda x: (x[0], x[1]))

    print(f"🧩 Total cases to OCR (single-turn): {len(cases)}")

    if OCR_CONCURRENCY <= 1:
        for i, (code_id, ratio, page_paths) in enumerate(cases, start=1):
            case_key = f"{code_id}|{ratio}"
            if case_key in done:
                skipped += 1
                continue
            print(
                f"[{i}/{len(cases)}] OCR(single-turn): {code_id} @ ratio {ratio}x ({len(page_paths)} pages)"
            )

            content = [{"type": "text", "text": _get_ocr_user_prompt()}]
            for p in page_paths:
                data_url = _encode_image_to_data_url(p)
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            last_err = None
            text = ""
            diagnostics = {}

            for attempt in range(1, OCR_MAX_RETRIES + 1):
                try:
                    extra_body = {"safety_settings": GEMINI_SAFETY_SETTINGS} if GEMINI_ENABLE_SAFETY_SETTINGS else None
                    resp = client.chat.completions.create(
                        model=GEMINI_MODEL_NAME,  # 🌟 Use the Gemini model
                        temperature=OCR_TEMPERATURE,
                        max_tokens=OCR_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": OCR_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        extra_body=extra_body,
                    )
                    text = _clean_ocr_text(resp.choices[0].message.content or "")
                    diagnostics = _extract_response_diagnostics(resp)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    # exponential backoff: 1,2,4,8,... capped at 30s
                    backoff = min(30.0, float(2 ** (attempt - 1)))
                    time.sleep(backoff)

            rec = {
                "code_id": code_id,
                "ratio": ratio,
                "num_pages": len(page_paths),
                "image_paths": page_paths,
                "image_path": page_paths[0] if page_paths else "",
                "model": GEMINI_MODEL_NAME,  # 🌟 Record model name
            }

            if diagnostics:
                rec.update(diagnostics)

            if last_err is None:
                rec["text"] = text
                rec["text_len"] = len(text)
                if rec.get("finish_reason") in ("content_filter", "safety"):
                    rec["blocked_by_safety"] = True
                total += 1
            else:
                rec["error"] = last_err
                errors += 1

            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(OCR_SLEEP_SECONDS)
    else:
        pending_cases = [(code_id, ratio, page_paths) for (code_id, ratio, page_paths) in cases if f"{code_id}|{ratio}" not in done]
        skipped = len(cases) - len(pending_cases)

        print(
            f"⚡ Parallel OCR enabled: workers={OCR_CONCURRENCY}, "
            f"global_min_interval={OCR_PARALLEL_MIN_INTERVAL_SECONDS}s"
        )

        client_local = threading.local()
        write_lock = threading.Lock()
        rate_lock = threading.Lock()
        next_allowed_time = 0.0

        def _get_client():
            c = getattr(client_local, "client", None)
            if c is None:
                c = _create_openai_client(api_key)
                client_local.client = c
            return c

        def _rate_limit_wait():
            nonlocal next_allowed_time
            interval = float(OCR_PARALLEL_MIN_INTERVAL_SECONDS)
            if interval <= 0:
                return
            with rate_lock:
                now = time.monotonic()
                if now < next_allowed_time:
                    wait_s = next_allowed_time - now
                    next_allowed_time = next_allowed_time + interval
                else:
                    wait_s = 0.0
                    next_allowed_time = now + interval
            if wait_s > 0:
                time.sleep(wait_s)

        def _ocr_one_case(code_id: str, ratio: int, page_paths: list[str]):
            content = [{"type": "text", "text": _get_ocr_user_prompt()}]
            for p in page_paths:
                data_url = _encode_image_to_data_url(p)
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            last_err = None
            text = ""
            diagnostics = {}

            for attempt in range(1, OCR_MAX_RETRIES + 1):
                try:
                    _rate_limit_wait()
                    extra_body = {"safety_settings": GEMINI_SAFETY_SETTINGS} if GEMINI_ENABLE_SAFETY_SETTINGS else None
                    resp = _get_client().chat.completions.create(
                        model=GEMINI_MODEL_NAME,
                        temperature=OCR_TEMPERATURE,
                        max_tokens=OCR_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": OCR_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        extra_body=extra_body,
                    )
                    text = _clean_ocr_text(resp.choices[0].message.content or "")
                    diagnostics = _extract_response_diagnostics(resp)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    backoff = min(30.0, float(2 ** (attempt - 1)))
                    time.sleep(backoff)

            rec = {
                "code_id": code_id,
                "ratio": ratio,
                "num_pages": len(page_paths),
                "image_paths": page_paths,
                "image_path": page_paths[0] if page_paths else "",
                "model": GEMINI_MODEL_NAME,
            }
            if diagnostics:
                rec.update(diagnostics)
            if last_err is None:
                rec["text"] = text
                rec["text_len"] = len(text)
                if rec.get("finish_reason") in ("content_filter", "safety"):
                    rec["blocked_by_safety"] = True
                return rec, True
            rec["error"] = last_err
            return rec, False

        completed = 0
        total_jobs = len(pending_cases)

        with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as ex:
            futures = {ex.submit(_ocr_one_case, code_id, ratio, page_paths): (code_id, ratio, page_paths) for (code_id, ratio, page_paths) in pending_cases}
            for fut in as_completed(futures):
                code_id, ratio, page_paths = futures[fut]
                try:
                    rec, ok = fut.result()
                except Exception as e:
                    rec = {
                        "code_id": code_id,
                        "ratio": ratio,
                        "num_pages": len(page_paths),
                        "image_paths": page_paths,
                        "image_path": page_paths[0] if page_paths else "",
                        "model": GEMINI_MODEL_NAME,
                        "error": f"worker_exception: {e}",
                    }
                    ok = False

                with write_lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                completed += 1
                if ok:
                    total += 1
                else:
                    errors += 1
                print(
                    f"[{completed}/{total_jobs}] OCR done: {code_id} @ ratio {ratio}x "
                    f"({'ok' if ok else 'error'})"
                )

    print(f"✅ Module 3 finished. ok={total}, skipped={skipped}, error={errors}")
    print(f"📄 Output: {os.path.abspath(out_jsonl)}")
