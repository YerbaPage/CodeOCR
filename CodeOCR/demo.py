#!/usr/bin/env python3
"""
CodeOCR Demo - Code image rendering and OCR examples

Usage:
    # Render code to image
    python -m CodeOCR.demo render --file example.py -o output.png
    
    # Query LLM with code image
    python -m CodeOCR.demo query --file example.py -i "Explain this code"
    
    # End-to-end: Code -> Image -> OCR -> Evaluate
    python -m CodeOCR.demo ocr --file example.py
"""

import argparse
import os
import json
import base64
from pathlib import Path

SAMPLE_CODE = '''
def quicksort(arr):
    """Quick sort implementation."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

if __name__ == "__main__":
    data = [3, 6, 8, 10, 1, 2, 1]
    print(quicksort(data))
'''.strip()


def _detect_language(filepath: str) -> str:
    """Detect language from file extension."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
        ".rs": "rust", ".rb": "ruby", ".php": "php",
    }
    return ext_map.get(Path(filepath).suffix.lower(), "python")


def cmd_render(args):
    """Render code to images."""
    from .api import render_code_to_images
    
    if args.file:
        code = Path(args.file).read_text(encoding="utf-8")
        language = _detect_language(args.file)
    elif args.code:
        code = args.code
        language = args.language
    else:
        print("Please provide --code or --file")
        return
    
    images = render_code_to_images(
        code, language=language,
        enable_syntax_highlight=args.highlight,
        theme=args.theme, auto_optimize=args.auto,
    )
    
    output = args.output or "output.png"
    if len(images) == 1:
        images[0].save(output)
        print(f"Saved to {output}")
    else:
        base, ext = Path(output).stem, Path(output).suffix or ".png"
        for i, img in enumerate(images, 1):
            path = f"{base}_{i:03d}{ext}"
            img.save(path)
            print(f"Saved to {path}")
    print(f"Total {len(images)} page(s)")


def cmd_query(args):
    """Render and query LLM."""
    from .api import render_and_query
    
    if args.file:
        code = Path(args.file).read_text(encoding="utf-8")
        language = _detect_language(args.file)
    elif args.code:
        code = args.code
        language = args.language
    else:
        print("Please provide --code or --file")
        return
    
    instruction = args.instruction or "Please explain this code"
    print(f"Calling {args.model}...")
    
    response, token_info = render_and_query(
        code, instruction=instruction, model=args.model,
        language=language, enable_syntax_highlight=args.highlight,
        theme=args.theme, max_tokens=args.max_tokens,
    )
    
    print("\n=== Response ===")
    print(response)
    print(f"\n=== Tokens: {token_info['prompt_tokens']} in, {token_info['completion_tokens']} out ===")


def cmd_ocr(args):
    """End-to-end: Code -> Image -> OCR -> Evaluate."""
    from .core import text_to_image, get_text_tokens
    from .client import create_client, OPENAI_AVAILABLE
    from PIL import Image
    
    # Get code
    if args.file:
        code = Path(args.file).read_text(encoding="utf-8")
        language = _detect_language(args.file)
        print(f"[Input] {args.file}")
    elif args.code:
        code = args.code
        language = args.language
        print("[Input] Code string")
    else:
        code = SAMPLE_CODE
        language = "python"
        print("[Input] Sample quicksort code")
    
    print(f"[Input] {len(code)} chars, {get_text_tokens(code)} tokens\n")
    
    # Calculate render size
    num_lines = code.count('\n') + 1
    max_line_len = max(len(line) for line in code.split('\n'))
    font_size, margin = 18, 20
    res = max(800, int(max_line_len * font_size * 0.6) + margin * 2,
              int(num_lines * font_size * 1.2) + margin * 2)
    
    # Parse ratios
    ratios = [int(r.strip()) for r in args.ratios.split(",")]
    
    # Render
    print("=" * 50)
    print("Step 1: Render")
    print("=" * 50)
    
    images = text_to_image(
        code, width=res, height=res, font_size=font_size,
        line_height=1.2, margin_px=margin, preserve_newlines=True,
        enable_syntax_highlight=True, language=language, theme="modern",
    )
    print(f"[Render] {res}x{res}, {len(images)} page(s)")
    
    # Generate compressed versions
    os.makedirs(args.output_dir, exist_ok=True)
    image_map = {}
    
    for ratio in ratios:
        ratio_dir = os.path.join(args.output_dir, f"ratio_{ratio}")
        os.makedirs(ratio_dir, exist_ok=True)
        paths = []
        
        for i, img in enumerate(images, 1):
            if ratio == 1:
                target = img
            else:
                w, h = img.size
                small = img.resize((max(1, w // ratio), max(1, h // ratio)), Image.Resampling.BILINEAR)
                target = small.resize((w, h), Image.Resampling.BILINEAR)
            
            path = os.path.join(ratio_dir, f"page_{i:03d}.png")
            target.save(path)
            paths.append(path)
        
        image_map[ratio] = paths
        print(f"[Render] ratio {ratio}x -> {ratio_dir}")
    
    if args.render_only:
        print("\n--render-only: skipping OCR")
        return
    
    # OCR
    print("\n" + "=" * 50)
    print("Step 2: OCR")
    print("=" * 50)
    
    if not OPENAI_AVAILABLE:
        print("[Error] pip install openai")
        return
    
    model = args.model or os.getenv("OCR_MODEL", "gpt-5-mini")
    client = create_client()
    results = []
    
    for ratio in ratios:
        paths = image_map[ratio]
        print(f"\n[OCR] ratio {ratio}x with {model}...")
        
        content = [{"type": "text", "text": (
            "Transcribe the code in these images exactly.\n"
            "Output plain text only (no Markdown). Preserve all whitespace."
        )}]
        for p in paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an OCR engine for code images."},
                {"role": "user", "content": content}
            ],
            temperature=0.0, max_tokens=8192,
        )
        
        ocr_text = resp.choices[0].message.content or ""
        # Strip markdown fences
        if ocr_text.startswith("```"):
            lines = ocr_text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            ocr_text = "\n".join(lines)
        
        # Save
        ocr_path = os.path.join(args.output_dir, f"ratio_{ratio}", "ocr.txt")
        Path(ocr_path).write_text(ocr_text, encoding="utf-8")
        
        # Evaluate
        metrics = _evaluate(code, ocr_text)
        results.append({"ratio": ratio, **metrics})
        print(f"[Eval] CER={metrics['cer']:.4f}, WER={metrics['wer']:.4f}, exact={metrics['exact']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Ratio':<8} | {'CER':<10} | {'WER':<10} | {'Exact':<6}")
    print("-" * 42)
    for r in results:
        print(f"{r['ratio']:<8} | {r['cer']:<10.4f} | {r['wer']:<10.4f} | {'Yes' if r['exact'] else 'No':<6}")
    
    # Save
    Path(os.path.join(args.output_dir, "results.json")).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {args.output_dir}/results.json")


def _evaluate(ref: str, hyp: str) -> dict:
    """Evaluate OCR result."""
    import Levenshtein
    
    def norm(s):
        lines = [l.rstrip() for l in s.strip().splitlines()]
        while lines and not lines[0].strip(): lines.pop(0)
        while lines and not lines[-1].strip(): lines.pop()
        return "\n".join(lines)
    
    r, h = norm(ref), norm(hyp)
    cer = Levenshtein.distance(r, h) / len(r) if r else (1.0 if h else 0.0)
    wer = Levenshtein.distance(" ".join(r.split()), " ".join(h.split())) / len(r.split()) if r.split() else (1.0 if h.split() else 0.0)
    return {"cer": cer, "wer": wer, "exact": r == h}


def main():
    parser = argparse.ArgumentParser(description="CodeOCR Demo")
    sub = parser.add_subparsers(dest="command")
    
    # render
    p = sub.add_parser("render", help="Render code to images")
    p.add_argument("--code", type=str)
    p.add_argument("--file", type=str)
    p.add_argument("--output", "-o", type=str)
    p.add_argument("--language", default="python")
    p.add_argument("--theme", default="modern", choices=["light", "modern"])
    p.add_argument("--no-highlight", dest="highlight", action="store_false")
    p.add_argument("--no-auto", dest="auto", action="store_false")
    p.set_defaults(highlight=True, auto=True)
    
    # query
    p = sub.add_parser("query", help="Render and query LLM")
    p.add_argument("--code", type=str)
    p.add_argument("--file", type=str)
    p.add_argument("--instruction", "-i", type=str)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--language", default="python")
    p.add_argument("--theme", default="modern", choices=["light", "modern"])
    p.add_argument("--no-highlight", dest="highlight", action="store_false")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.set_defaults(highlight=True)
    
    # ocr (e2e)
    p = sub.add_parser("ocr", help="End-to-end: Code -> Image -> OCR -> Eval")
    p.add_argument("--code", type=str)
    p.add_argument("--file", type=str)
    p.add_argument("--language", default="python")
    p.add_argument("--model", type=str, help="OCR model (default: gpt-5-mini)")
    p.add_argument("--ratios", default="1,2,4,8", help="Compression ratios")
    p.add_argument("--output-dir", default="./demo_output")
    p.add_argument("--render-only", action="store_true", help="Skip OCR")
    
    args = parser.parse_args()
    
    if args.command == "render":
        cmd_render(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "ocr":
        cmd_ocr(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
