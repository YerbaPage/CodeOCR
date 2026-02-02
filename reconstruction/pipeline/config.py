import os
from datetime import datetime

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")

# ================= Configuration =================
OUTPUT_DIR = "./experiment_output"
DATASET_DIR = "./dataset"
IMAGES_DIR_DEFAULT = os.path.join(OUTPUT_DIR, "images_gemini")  # 🌟 Gemini-specific directory
# Whether to OCR + judge directly from an existing image set (for fair cross-model comparison).
# - USE_EXISTING_IMAGES=1: skip modules 1/2, do not clean images; use images under EXISTING_IMAGES_DIR (or default IMAGES_DIR_DEFAULT).
# - DATASET_FILENAME: set the GT dataset filename (under OUTPUT_DIR) so different models can run on the same table for comparison.
USE_EXISTING_IMAGES = os.getenv("USE_EXISTING_IMAGES", "0").strip().lower() in ("1", "true", "yes", "y")
EXISTING_IMAGES_DIR = os.getenv("EXISTING_IMAGES_DIR", "").strip()
IMAGES_DIR = EXISTING_IMAGES_DIR or IMAGES_DIR_DEFAULT
DEFAULT_DATASET_FILENAME = "dataset_new.json"
DATASET_FILENAME = os.getenv("DATASET_FILENAME", DEFAULT_DATASET_FILENAME).strip() or DEFAULT_DATASET_FILENAME
TARGET_RATIOS = [1, 1.5, 2, 4, 6, 8]  # Target compression ratios

# ================= Module 3 Config (Inference Engine) =================
# Use Gemini (via the aihubmix OpenAI-compat endpoint)
RUN_MODULE_3 = _env_bool("RUN_MODULE_3", True)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").strip()
GEMINI_MODEL_NAME = "gpt-5.1-2025-11-13"  # 🌟 Set the Gemini model name

# ================= Client Config =================
# CLIENT_TYPE: "Azure" uses AzureOpenAI; other values use the default OpenAI client
CLIENT_TYPE = os.getenv("CLIENT_TYPE", "").strip()
# Azure settings
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-03-01-preview").strip()
OCR_SYSTEM_PROMPT = "You are an OCR engine for code images."
OCR_USER_PROMPT = (
    "Transcribe the code in these images exactly.\n"
    "- These images are consecutive pages of the SAME code file, in order.\n"
    "- The page may start mid-block (e.g., indented lines without a visible 'def' header). Keep the indentation exactly as shown.\n"
    "- Do NOT invent missing context. Do NOT add wrapper code such as 'def', 'class', imports, or any extra lines.\n"
    "- Output plain text only (no Markdown, no code fences).\n"
    "- Preserve all whitespace, indentation, and newlines.\n"
    "- Do not add, remove, or rename anything.\n"
)

# Gemini Safety Settings (disabled by default to preserve existing behavior; enable via env vars if needed)
# Note: support for this field varies across OpenAI-compat proxies; if enabling causes parameter errors, turn it off.
GEMINI_ENABLE_SAFETY_SETTINGS = _env_bool("GEMINI_ENABLE_SAFETY_SETTINGS", False)
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Optional prompt enhancement/override (disabled by default to preserve existing behavior)
OCR_PROMPT_PERSONAL_OFFLINE = _env_bool("OCR_PROMPT_PERSONAL_OFFLINE", False)
OCR_USER_PROMPT_OVERRIDE = os.getenv("OCR_USER_PROMPT_OVERRIDE", "").strip()
TIMESTAMP = datetime.now().strftime('%m%d_%H%M%S')
EVAL_ONLY = os.getenv("EVAL_ONLY", "0").strip().lower() in ("1", "true", "yes", "y", "on")
EVAL_OCR_JSONL_PATH = os.getenv("EVAL_OCR_JSONL_PATH", "").strip()
EVAL_DATASET_JSON_PATH = os.getenv("EVAL_DATASET_JSON_PATH", "").strip()
EVAL_MODEL_NAME = (os.getenv("EVAL_MODEL_NAME", GEMINI_MODEL_NAME) or GEMINI_MODEL_NAME).strip()

OCR_MAX_TOKENS = 16384  # Gemini supports larger context windows; keep this high
OCR_TEMPERATURE = 0.0
OCR_SLEEP_SECONDS = 0.2
OCR_MAX_RETRIES = 5
# OCR concurrency: set via env vars.
# OCR_CONCURRENCY=4 can speed things up; if you hit rate limits, set OCR_PARALLEL_MIN_INTERVAL_SECONDS for global throttling.
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))
OCR_PARALLEL_MIN_INTERVAL_SECONDS = float(os.getenv("OCR_PARALLEL_MIN_INTERVAL_SECONDS", "0"))

# ================= Module 4 Config (Auto-Judge) =================
RUN_MODULE_4 = _env_bool("RUN_MODULE_4", True)  # Whether to run the evaluation module
JUDGE_LLM_MODEL = "gpt-5-mini"  # Model used for soft taxonomy classification

# Error taxonomy (8 classes)
ERROR_TAXONOMY = [
    "Visual_Typo",          # Visually similar character substitutions (e.g., O/0, l/1)
    "Symbol_Loss",          # Punctuation/symbol loss (e.g., missing brackets, colons)
    "Indentation_Error",    # Indentation errors
    "Line_Skipped",         # Skipped lines / missed reads
    "Variable_Hallucination", # Variable-name hallucinations (e.g., 'data' read as 'date')
    "Code_Invention",       # Invented code that does not exist
    "Repetition",           # Repeated output of some lines
    "Comment_Loss"          # Missing or garbled comments
]
