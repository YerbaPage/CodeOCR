import os
import json
import re
import difflib
import math
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import (
    ERROR_TAXONOMY, JUDGE_LLM_MODEL, GEMINI_MODEL_NAME, 
    DEFAULT_DATASET_FILENAME, OUTPUT_DIR, DATASET_DIR
)
from .utils import _create_openai_client, _try_load_api_key_from_env_files, _mask_api_key

def normalize_code(code: str) -> str:
    if not code:
        return ""
    lines = code.split('\n')
    # rstrip each line
    lines = [line.rstrip() for line in lines]
    # filter empty lines at head/tail
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)

def _split_nonblank_lines_for_diff(code: str):
    """
    Split code into lines, keeping only non-blank lines (stripped).
    This mimics a 'ignore-all-space' approach for line-diff counting logic if desired,
    OR we can keep indentation.
    For standard diff metrics, we usually compare line-by-line exactly.
    But to be robust against empty lines, let's filter them out or keep them?
    The original logic seemed to just use splitlines().
    Let's stick to simple splitlines() but maybe rstrip().
    """
    return [line.rstrip() for line in code.splitlines()]

def _compute_codediff_metrics_no_blank(ref_code: str, hyp_code: str) -> dict:
    """
    Compute edit-distance based metrics (CER, WER) and line-based metrics,
    ignoring blank lines and trailing spaces for fair comparison.
    """
    import Levenshtein  # Ensure python-Levenshtein is installed

    # 1. Normalize: strip trailing spaces, remove empty lines
    ref_lines = [line.rstrip() for line in ref_code.splitlines() if line.strip()]
    hyp_lines = [line.rstrip() for line in hyp_code.splitlines() if line.strip()]
    
    ref_norm = "\n".join(ref_lines)
    hyp_norm = "\n".join(hyp_lines)
    
    # 2. CER (Character Error Rate) on the normalized string
    # dist / len(ref)
    if not ref_norm:
        cer = 1.0 if hyp_norm else 0.0
    else:
        dist = Levenshtein.distance(ref_norm, hyp_norm)
        cer = dist / len(ref_norm)
        
    # 3. WER (Word Error Rate) - naive splitting by whitespace
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if not ref_words:
        wer = 1.0 if hyp_words else 0.0
    else:
        # Levenshtein distance on list of words? python-Levenshtein usually works on strings.
        # We can use difflib or map words to chars.
        # For simplicity/speed, let's use a standard edit distance on lists.
        # (Levenshtein module works on strings mostly, or sequences of hashables?
        #  Levenshtein.distance() expects strings. Levenshtein.seqratio() etc. exist.)
        # Let's implement a simple DP for list-based edit distance or use editdistance package.
        # Fallback: map unique words to chars and use Levenshtein (tokenization trick).
        
        # Simple DP implementation for list distance is slow for large files.
        # Let's use difflib.SequenceMatcher for ratio, but we want edit distance.
        # optimization: map words to integers/chars
        vocab = {}
        next_id = 0
        def to_ids(words):
            nonlocal next_id
            ids = []
            for w in words:
                if w not in vocab:
                    vocab[w] = chr(next_id % 65535) # potential collision if >65k words? Unlikely for single file.
                    next_id += 1
                ids.append(vocab[w])
            return "".join(ids)
        
        r_str = to_ids(ref_words)
        h_str = to_ids(hyp_words)
        w_dist = Levenshtein.distance(r_str, h_str)
        wer = w_dist / len(ref_words)

    # 4. Line-level accuracy / similarity
    # We can use SequenceMatcher ratio
    sm = difflib.SequenceMatcher(None, ref_lines, hyp_lines)
    line_similarity = sm.ratio()  # 2*M / T
    
    return {
        "cer": cer,
        "wer": wer,
        "line_sim": line_similarity,
        "ref_len_char": len(ref_norm),
        "hyp_len_char": len(hyp_norm),
        "ref_lines": len(ref_lines),
        "hyp_lines": len(hyp_lines),
    }

def _compute_line_token_metrics(ref_code: str, hyp_code: str) -> dict:
    """
    Compute line-level token metrics (precision, recall, f1)
    Logic: For each corresponding line (or aligned lines), compute token overlap.
    This is complex if lines are not aligned.
    Simplified approach: Treat the whole file as a bag of lines, or just use CodeBLEU?
    Let's rely on standard CodeBLEU if available, or simple token overlap.
    
    Here, we implement a simple Jaccard similarity on the set of (line_content) or trigrams?
    Actually, let's stick to the previous _compute_codediff_metrics_no_blank for core stats.
    This function was just a placeholder in my thought process.
    Let's add CodeBLEU if the library is available.
    """
    metrics = {}
    try:
        from codebleu import calc_codebleu
        # CodeBLEU expects list of references and list of hypotheses
        # calc_codebleu([ref], [hyp], lang="python", ...)
        # It handles tokenization/AST if 'python' is specified.
        # We assume the input is Python code.
        result = calc_codebleu([ref_code], [hyp_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
        metrics["codebleu"] = result["codebleu"]
        metrics["ngram_match_score"] = result["ngram_match_score"]
        metrics["weighted_ngram_match_score"] = result["weighted_ngram_match_score"]
        metrics["syntax_match_score"] = result["syntax_match_score"]
        metrics["dataflow_match_score"] = result["dataflow_match_score"]
    except ImportError:
        metrics["codebleu"] = -1.0 # Not available
    except Exception:
        metrics["codebleu"] = -1.0
        
    return metrics

def _compute_cer(ref: str, hyp: str) -> float:
    import Levenshtein
    if not ref:
        return 1.0 if hyp else 0.0
    return Levenshtein.distance(ref, hyp) / len(ref)

# ----------------- Taxonomy Detection Rules (Heuristic) -----------------

def _detect_visual_typo(ref: str, hyp: str) -> bool:
    """
    Heuristic: if CER is small (< 0.05) but not 0, and length is very close,
    it might be a visual typo (e.g. 0 vs O).
    This is weak; we mainly rely on LLM for this.
    """
    return False

def _detect_symbol_loss(ref: str, hyp: str) -> bool:
    """
    Check if common symbols count dropped significantly.
    """
    symbols = set("()[]{},.:;=")
    def count_sym(s): return sum(1 for c in s if c in symbols)
    r_c = count_sym(ref)
    h_c = count_sym(hyp)
    if r_c > 0 and h_c < r_c * 0.8:
        return True
    return False

# ----------------- LLM-based Judge -----------------

def _call_llm_for_taxonomy(client, code_id: str, ratio: str, ref_code: str, hyp_code: str) -> dict:
    """
    Ask LLM (gpt-4o or similar) to classify the error into the taxonomy.
    """
    if not client:
        return {}
        
    prompt = f"""
You are an expert code reviewer. Compare the Ground Truth (GT) code with the OCR Output code.
Identify if any of the following error types exist in the OCR Output.
Taxonomy: {ERROR_TAXONOMY}

GT Code:
```python
{ref_code}
```

OCR Output:
```python
{hyp_code}
```

Instructions:
- Return a JSON object with boolean keys for each error type.
- Example: {{"Visual_Typo": true, "Symbol_Loss": false, ...}}
- If the OCR Output is identical or semantically equivalent (ignoring minor whitespace), all false.
- "Visual_Typo": e.g. '1' vs 'l', '0' vs 'O'.
- "Symbol_Loss": missing punctuation like ':', '(', '}}'.
- "Indentation_Error": wrong indentation levels.
- "Line_Skipped": missing entire lines of code.
- "Variable_Hallucination": variable names changed (e.g. 'cnt' -> 'cut').
- "Code_Invention": extra code not in GT.
- "Repetition": loops of repeated text.
- "Comment_Loss": missing comments.
"""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a code diff analyzer. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        print(f"      ⚠️ Judge LLM failed for {code_id}: {e}")
        return {}


def run_module_4_judge(output_dir: str, ocr_jsonl_filename: str, ocr_model_name: str, dataset_json_filename: str = None, print_space_diff_report: bool = False):
    """
    Compare OCR results (jsonl) against Ground Truth (json).
    Compute metrics: CER, WER, CodeBLEU (if installed).
    Perform taxonomy classification (Heuristic + LLM).
    Aggregate results by Ratio and Overall.
    """
    print("\n" + "="*40)
    print("⚖️  Running Module 4: Auto-Judge")
    print(f"   - Model being judged: {ocr_model_name}")
    print(f"   - Judge LLM: {JUDGE_LLM_MODEL}")
    print("="*40)

    # 1. Load GT Dataset
    if not dataset_json_filename:
        dataset_json_filename = DEFAULT_DATASET_FILENAME
    
    ds_path = os.path.join(DATASET_DIR, dataset_json_filename)
    if not os.path.exists(ds_path):
        # Fallback to output_dir
        ds_path = os.path.join(output_dir, dataset_json_filename)
        
    if not os.path.exists(ds_path):
        print(f"❌ Dataset not found: {ds_path}")
        return

    with open(ds_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    gt_map = {item['id']: item['code'] for item in dataset}
    print(f"📚 Loaded GT dataset: {len(gt_map)} items")

    # 2. Load OCR Results
    ocr_path = os.path.join(output_dir, ocr_jsonl_filename)
    if not os.path.exists(ocr_path):
        print(f"❌ OCR results not found: {ocr_path}")
        return
        
    ocr_results = []
    with open(ocr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                ocr_results.append(json.loads(line))
            except: pass
    
    print(f"📝 Loaded OCR results: {len(ocr_results)} records")
    
    # 3. Prepare Evaluation
    # Group by ratio to compute per-ratio stats
    # Also keep detailed records for taxonomy analysis
    
    # Check if we have API key for Judge LLM
    api_key = os.getenv("AIHUBMIX_API_KEY") or _try_load_api_key_from_env_files()
    judge_client = None
    if api_key:
        judge_client = _create_openai_client(api_key)
    else:
        print("⚠️ No API Key found. Skipping LLM-based taxonomy classification.")

    eval_results = []
    
    # We can parallelize the LLM judging if we want, but metrics are fast.
    # Let's do metrics locally, and LLM calls in parallel.
    
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for rec in ocr_results:
            code_id = rec.get('code_id')
            ratio = rec.get('ratio')
            hyp_text = rec.get('text', "")
            
            if code_id not in gt_map:
                continue
                
            ref_text = gt_map[code_id]
            
            # Compute deterministic metrics
            # Normalize for fair comparison
            ref_norm = normalize_code(ref_text)
            hyp_norm = normalize_code(hyp_text)
            
            metrics = _compute_codediff_metrics_no_blank(ref_norm, hyp_norm)
            
            # Add simple heuristics
            taxonomy_flags = {k: False for k in ERROR_TAXONOMY}
            if _detect_symbol_loss(ref_norm, hyp_norm):
                taxonomy_flags["Symbol_Loss"] = True
            
            # Prepare record
            eval_entry = {
                "code_id": code_id,
                "ratio": ratio,
                "model": ocr_model_name,
                "metrics": metrics,
                "taxonomy": taxonomy_flags, # initial heuristic
                "ref_len": len(ref_norm),
                "hyp_len": len(hyp_norm)
            }
            
            # Schedule LLM judge if needed (sample or all? Let's do all for now or skip if perfect match)
            if metrics['cer'] > 0.0 and judge_client:
                # Only call LLM if there is an error
                f = executor.submit(_call_llm_for_taxonomy, judge_client, code_id, ratio, ref_norm, hyp_norm)
                futures.append((f, eval_entry))
            else:
                # Perfect match -> all false
                pass
                
            eval_results.append(eval_entry)
        
        # Collect LLM results
        completed = 0
        for f, entry in futures:
            try:
                llm_flags = f.result()
                if llm_flags:
                    # Update flags (LLM overrides or merges? Let's merge True)
                    for k, v in llm_flags.items():
                        if k in entry["taxonomy"] and v is True:
                            entry["taxonomy"][k] = True
            except Exception as e:
                pass
            completed += 1
            if completed % 10 == 0:
                print(f"   ... Judged {completed}/{len(futures)} cases via LLM")

    # 4. Aggregate & Report
    # Group by ratio
    by_ratio = collections.defaultdict(list)
    for res in eval_results:
        by_ratio[res['ratio']].append(res)
        
    print("\n" + "-"*60)
    print(f"{'Ratio':<8} | {'Count':<6} | {'CER':<8} | {'WER':<8} | {'LineSim':<8} | {'Perfect':<8}")
    print("-"*60)
    
    overall_metrics = {"cer": [], "wer": [], "line_sim": [], "perfect": []}
    
    sorted_ratios = sorted(by_ratio.keys())
    for r in sorted_ratios:
        items = by_ratio[r]
        n = len(items)
        avg_cer = sum(i['metrics']['cer'] for i in items) / n
        avg_wer = sum(i['metrics']['wer'] for i in items) / n
        avg_sim = sum(i['metrics']['line_sim'] for i in items) / n
        n_perfect = sum(1 for i in items if i['metrics']['cer'] == 0.0)
        pct_perfect = (n_perfect / n) * 100
        
        print(f"{r:<8} | {n:<6} | {avg_cer:.4f}   | {avg_wer:.4f}   | {avg_sim:.4f}   | {pct_perfect:.1f}%")
        
        overall_metrics['cer'].extend([i['metrics']['cer'] for i in items])
        overall_metrics['wer'].extend([i['metrics']['wer'] for i in items])
        overall_metrics['line_sim'].extend([i['metrics']['line_sim'] for i in items])
        overall_metrics['perfect'].extend([1 if i['metrics']['cer'] == 0.0 else 0 for i in items])

    print("-"*60)
    
    if overall_metrics['cer']:
        n_total = len(overall_metrics['cer'])
        o_cer = sum(overall_metrics['cer']) / n_total
        o_wer = sum(overall_metrics['wer']) / n_total
        o_sim = sum(overall_metrics['line_sim']) / n_total
        o_perf = (sum(overall_metrics['perfect']) / n_total) * 100
        print(f"{'ALL':<8} | {n_total:<6} | {o_cer:.4f}   | {o_wer:.4f}   | {o_sim:.4f}   | {o_perf:.1f}%")
    print("-"*60)

    # Save detailed report
    report_path = os.path.join(output_dir, f"eval_report_{_safe_filename(ocr_model_name)}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"📄 Detailed evaluation report saved to: {report_path}")

def _safe_filename(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
