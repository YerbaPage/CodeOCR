#!/usr/bin/env python3
"""
CodeOCR Render CLI Tool

Usage:
    python -m CodeOCR.render --code-context-file example.py --instruction "Explain the code"
"""

import argparse
import json
import sys
from pathlib import Path

from .api import render_code_to_images
from .client import create_client, call_llm_with_images
from .core import get_text_tokens, DEFAULT_LINE_HEIGHT


def main():
    parser = argparse.ArgumentParser(description="CodeOCR Render Tool")
    parser.add_argument("--instruction", type=str, help="Instruction")
    parser.add_argument("--code-context", type=str, help="Code string")
    parser.add_argument("--code-context-file", type=str, help="Code file path")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name")
    parser.add_argument("--client-type", type=str, default="OpenAI", choices=["OpenAI", "Azure"])
    parser.add_argument("--system-prompt", type=str, default="You are a helpful coding assistant.")
    parser.add_argument("--user-prompt", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--enable-syntax-highlight", action="store_true")
    parser.add_argument("--theme", type=str, default="light", choices=["light", "modern"])
    
    args = parser.parse_args()
    
    # Get instruction
    instruction = args.instruction
    if instruction is None:
        instruction = sys.stdin.read()
    if not instruction or not instruction.strip():
        print("Empty instruction")
        return
    
    # Get code
    if args.code_context:
        code = args.code_context
    elif args.code_context_file:
        path = Path(args.code_context_file)
        if not path.exists():
            print(f"File not found: {args.code_context_file}")
            return
        code = path.read_text(encoding="utf-8")
    else:
        print("Empty code context")
        return
    
    if not code.strip():
        print("Empty code context")
        return
    
    # Render images
    images = render_code_to_images(
        code,
        language=args.language,
        enable_syntax_highlight=args.enable_syntax_highlight,
        theme=args.theme,
        auto_optimize=True,
    )
    
    # Call LLM
    client = create_client(args.client_type)
    user_prompt = f"{instruction}"
    if args.user_prompt:
        user_prompt += f"\n\n{args.user_prompt}"
    user_prompt += "\n\nThe code context is in the image."
    
    response_text, token_info = call_llm_with_images(
        client, args.model, images, args.system_prompt, user_prompt, args.max_tokens
    )
    
    # Output result
    text_tokens = get_text_tokens(code)
    result = {
        "compression_ratio": 1.0,
        "resolution": f"{images[0].width}x{images[0].height}",
        "num_images": len(images),
        "response_text": response_text,
        "token_info": token_info,
        "original_text_tokens": text_tokens,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
