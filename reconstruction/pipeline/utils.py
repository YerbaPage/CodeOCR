import os
import re
import json
import base64
import time
from .config import (
    CLIENT_TYPE, AZURE_ENDPOINT, AZURE_API_VERSION, 
    AIHUBMIX_BASE_URL, OCR_USER_PROMPT_OVERRIDE, 
    OCR_PROMPT_PERSONAL_OFFLINE, OCR_USER_PROMPT
)

try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    OpenAI = None
    AzureOpenAI = None

def _get_ocr_user_prompt() -> str:
    """Get the OCR user prompt.

    Priority:
    1) OCR_USER_PROMPT_OVERRIDE (full override)
    2) OCR_PROMPT_PERSONAL_OFFLINE=1 (add usage context without changing constraints)
    3) Default OCR_USER_PROMPT
    """
    if OCR_USER_PROMPT_OVERRIDE:
        return OCR_USER_PROMPT_OVERRIDE
    if OCR_PROMPT_PERSONAL_OFFLINE:
        return (
            "Transcribe the code in these images exactly as it appears. "
            "This is for a personal offline syntax check project.\n"
            "- These images are consecutive pages of the SAME code file, in order.\n"
            "- The page may start mid-block (e.g., indented lines without a visible 'def' header). Keep the indentation exactly as shown.\n"
            "- Do NOT invent missing context. Do NOT add wrapper code such as 'def', 'class', imports, or any extra lines.\n"
            "- Output plain text only (no Markdown, no code fences).\n"
            "- Preserve all whitespace, indentation, and newlines.\n"
            "- Do not add, remove, or rename anything.\n"
        )
    return OCR_USER_PROMPT

def _mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return key[:2] + "..." + key[-2:]
    return key[:6] + "..." + key[-6:]


def _try_load_api_key_from_env_files() -> str:
    """Try reading AIHUBMIX_API_KEY from .env files.

    Priority: environment variable (handled by caller) > repo root .env > ocr/.env
    """
    script_dir = os.path.dirname(__file__)
    # Assume we are in reconstruction/gemini_pipeline, so repo root is up 2 levels
    repo_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    candidates = [
        os.path.join(os.getcwd(), ".env"),   # depends on where you launch from
        os.path.join(repo_dir, ".env"),      # repo root (more reliable)
        os.path.join(script_dir, ".env"),    # local .env
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() != "AIHUBMIX_API_KEY":
                        continue
                    value = v.strip().strip('"').strip("'")
                    if value:
                        return value
        except Exception:
            continue

    return ""


def _create_openai_client(api_key: str = None):
    """
    Create an OpenAI client.

    The client type is selected via CLIENT_TYPE:
    - CLIENT_TYPE=Azure: use AzureOpenAI
    - otherwise: use the default OpenAI client (via aihubmix)

    Args:
        api_key: API key; if omitted, read from environment variables

    Returns:
        An OpenAI or AzureOpenAI client instance; returns None if creation fails
    """
    if api_key is None:
        api_key = os.getenv("AIHUBMIX_API_KEY") or _try_load_api_key_from_env_files()
    
    if not api_key:
        print("❌ Missing API key (AIHUBMIX_API_KEY)")
        return None
    
    if CLIENT_TYPE.lower() == "azure":
        if AzureOpenAI is None:
            print("❌ Missing dependency: openai (AzureOpenAI). Run: pip install openai")
            return None
        print(f"🔑 Using AzureOpenAI client (endpoint: {AZURE_ENDPOINT})")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
    else:
        if OpenAI is None:
            print("❌ Missing dependency: openai. Run: pip install openai")
            return None
        print(f"🔑 Using OpenAI client (base_url: {AIHUBMIX_BASE_URL})")
        return OpenAI(api_key=api_key, base_url=AIHUBMIX_BASE_URL)


def _safe_filename_component(text: str) -> str:
    """Convert a string (e.g., a model name) into a filename-safe component."""
    value = (text or "").strip()
    if not value:
        return "model"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value[:80]


def _remove_file_if_exists(path: str) -> bool:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False


def _dataset_filename_for_model(model_name: str) -> str:
    """Generate a per-model dataset filename to avoid cross-run overwrites."""
    model_tag = _safe_filename_component(model_name)
    return f"dataset_{model_tag}.json"


def _iter_image_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                yield os.path.join(dirpath, fn)


def _load_done_set(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("image_path"):
                        done.add(obj["image_path"])
                    if obj.get("code_id") and ("ratio" in obj):
                        done.add(f"{obj.get('code_id')}|{obj.get('ratio')}")
                except Exception:
                    continue
    except Exception:
        return done
    return done


def _encode_image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    # Upstream providers (incl. GLM / some proxies) may include wrapper markers
    cleaned = cleaned.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return cleaned.strip("\n")


def _extract_response_diagnostics(resp) -> dict:
    """Best-effort extraction of useful diagnostics from an OpenAI-compat response.

    Note: field shapes can vary across proxies/SDK versions; keep this resilient and non-blocking.
    """
    diag: dict = {}
    try:
        resp_id = getattr(resp, "id", None)
        if resp_id:
            diag["response_id"] = resp_id
    except Exception:
        pass

    try:
        model = getattr(resp, "model", None)
        if model:
            diag["response_model"] = model
    except Exception:
        pass

    finish_reason = None
    try:
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
    except Exception:
        finish_reason = None
    if finish_reason is not None:
        diag["finish_reason"] = finish_reason

    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # usage may be an object or a dict
            if hasattr(usage, "model_dump"):
                diag["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                diag["usage"] = usage
    except Exception:
        pass

    # Some implementations may expose refusal / safety info (best-effort; do not rely on it)
    try:
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            msg = getattr(resp.choices[0], "message", None)
            refusal = getattr(msg, "refusal", None) if msg is not None else None
            if refusal:
                diag["refusal"] = refusal
    except Exception:
        pass

    return diag


def _parse_ratio_from_filename(image_path: str):
    # e.g. page_001_ratio2.png -> 2 ; page_001_ratio1.5.png -> 1.5 ; page_001.png -> 1
    stem = os.path.splitext(os.path.basename(image_path))[0]
    marker = "_ratio"
    if marker in stem:
        try:
            tail = stem.split(marker, 1)[1].strip().replace("_", ".")
            if "." in tail:
                return float(tail)
            return int(tail)
        except Exception:
            return 1
    return 1


def _extract_page_num_from_filename(image_path: str) -> int:
    """page_001_ratio2.png -> 1; returns 0 if extraction fails."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    m = re.search(r"page_(\d+)", stem)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0
