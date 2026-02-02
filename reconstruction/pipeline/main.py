import os
import sys
import argparse

# Add the parent directory to sys.path to ensure we can import 'gemini_pipeline' if run as script
# But if we run from root as module, it's fine.
# Let's assume this is run as `python -m reconstruction.gemini_pipeline.main` or similar.

from .config import (
    OUTPUT_DIR, IMAGES_DIR, DATASET_FILENAME, 
    RUN_MODULE_3, RUN_MODULE_4, GEMINI_MODEL_NAME,
    EVAL_ONLY, EVAL_OCR_JSONL_PATH, EVAL_DATASET_JSON_PATH, EVAL_MODEL_NAME
)
from .utils import _dataset_filename_for_model
from .data_pipeline import run_module_1_and_2
from .ocr_engine import run_module_3_gemini
from .evaluator import run_module_4_judge

def run_pipeline(continue_from: str = None):
    """
    Orchestrate the full pipeline:
    1. Data Mining (if not existing)
    2. Visual Corruption (if not existing)
    3. OCR Inference (Module 3)
    4. Auto-Judge (Module 4)
    """
    
    # --- Module 1 & 2: Data Generation ---
    # Only run if we are not just evaluating results
    run_module_1_and_2()
    
    # Determine dataset filename
    # In original code, DATASET_FILENAME was fixed or env var.
    # Here we might want to ensure consistency.
    
    # --- Module 3: OCR Inference ---
    if RUN_MODULE_3:
        run_module_3_gemini(
            images_dir=IMAGES_DIR, 
            output_dir=OUTPUT_DIR, 
            continue_from=continue_from
        )
    else:
        print("⏭️  Skipping Module 3 (RUN_MODULE_3=False)")

    # --- Module 4: Auto-Judge ---
    if RUN_MODULE_4:
        # Determine which OCR result file to judge
        # If we just ran Module 3, it would be the latest one.
        # But run_module_3_gemini writes to a timestamped file or continue_from.
        # We need to know the filename. 
        # For simplicity, let's find the latest jsonl in output_dir matching pattern if not explicit.
        
        ocr_filename = continue_from
        if not ocr_filename:
            # Find latest qwen_ocr_*.jsonl or gemini_ocr?
            # The ocr_engine writes to `qwen_ocr_{timestamp}.jsonl` (I kept the name from original for compat, maybe should change to gemini)
            # Let's check the directory.
            files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jsonl") and "ocr" in f]
            if files:
                # Sort by modification time
                files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
                ocr_filename = files[0]
        
        if ocr_filename:
            run_module_4_judge(
                output_dir=OUTPUT_DIR,
                ocr_jsonl_filename=ocr_filename,
                ocr_model_name=GEMINI_MODEL_NAME,
                dataset_json_filename=DATASET_FILENAME
            )
        else:
            print("⚠️ No OCR results found to judge.")
    else:
        print("⏭️  Skipping Module 4 (RUN_MODULE_4=False)")

def run_eval_only():
    print("🔍 Running in EVAL_ONLY mode")
    if not EVAL_OCR_JSONL_PATH:
        print("❌ EVAL_OCR_JSONL_PATH is required for eval-only mode.")
        return

    run_module_4_judge(
        output_dir=os.path.dirname(EVAL_OCR_JSONL_PATH) or OUTPUT_DIR,
        ocr_jsonl_filename=os.path.basename(EVAL_OCR_JSONL_PATH),
        ocr_model_name=EVAL_MODEL_NAME,
        dataset_json_filename=EVAL_DATASET_JSON_PATH or DATASET_FILENAME
    )

def main():
    parser = argparse.ArgumentParser(description="CodeOCR Gemini Pipeline")
    parser.add_argument("--continue_from", type=str, help="Resume OCR from this JSONL file")
    parser.add_argument("--eval_only", action="store_true", help="Run only evaluation (Module 4)")
    args = parser.parse_args()

    # Prioritize CLI args over Env vars where appropriate, or mix.
    # EVAL_ONLY env var is checked in config, but CLI arg is explicit.
    
    if args.eval_only or EVAL_ONLY:
        run_eval_only()
    else:
        run_pipeline(continue_from=args.continue_from)

if __name__ == "__main__":
    main()
