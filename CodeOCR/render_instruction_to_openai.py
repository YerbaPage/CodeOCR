import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import tiktoken
from openai import OpenAI, AzureOpenAI
from PIL import Image as PIL_Image


def _ensure_import_path():
    root = Path(__file__).resolve().parents[1]
    downstream_dir = root / "downstream"
    if str(downstream_dir) not in sys.path:
        sys.path.insert(0, str(downstream_dir))


_ensure_import_path()


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "downstream" / "config.json"
    if config_path.exists():
        try:
            return json.load(open(config_path, "r"))
        except Exception:
            return {}
    return {}


def create_client(client_type: str = "OpenAI"):
    config = load_config()
    if client_type == "Azure":
        api_key = config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY", "")
        api_version = config.get("azure_api_version") or os.environ.get(
            "AZURE_OPENAI_API_VERSION", ""
        )
        endpoint = config.get("azure_endpoint") or os.environ.get(
            "AZURE_OPENAI_ENDPOINT", ""
        )
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_text_tokens(text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _inject_llm_utils_stub():
    import types

    stub = types.ModuleType("llm_utils")
    stub.get_text_tokens = get_text_tokens
    sys.modules["llm_utils"] = stub


_inject_llm_utils_stub()

import builtins
if not hasattr(builtins, "AutoProcessor"):
    builtins.AutoProcessor = object

from text_to_image import (
    DEFAULT_DPI,
    DEFAULT_LINE_HEIGHT,
    DEFAULT_MARGIN_RATIO,
    analyze_text_structure,
    optimize_layout_config_dry,
    resize_images_for_compression,
    text_to_image_stream,
)


def encode_pil_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_api_kwargs(model_name: str, max_tokens: int):
    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": 0.0}


def call_llm_with_images(
    client,
    model_name: str,
    images,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
):
    base64_images = [encode_pil_image_to_base64(img) for img in images]
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": user_prompt}]
                + [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                    for image in base64_images
                ]
            ),
        },
    ]
    kwargs = {"model": model_name, "messages": messages, **prepare_api_kwargs(model_name, max_tokens)}
    response = client.chat.completions.create(**kwargs)
    generated_text = response.choices[0].message.content
    usage = response.usage
    return generated_text, {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "api_kwargs": {k: v for k, v in kwargs.items() if k != "messages"},
    }


def parse_ratios(value: str):
    if not value:
        return []
    ratios = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        ratios.append(float(item))
    return ratios


def read_instruction(args):
    if args.instruction is not None:
        return args.instruction
    return sys.stdin.read()


def read_code_context(args):
    if args.code_context is not None:
        return args.code_context
    if args.code_context_file:
        path = Path(args.code_context_file)
        if not path.exists():
            print(f"Code context file not found: {args.code_context_file}")
            return None
        return path.read_text(encoding="utf-8")
    return ""


def build_user_prompt(instruction: str, extra_prompt: str):
    parts = []
    if instruction and instruction.strip():
        parts.append(instruction.strip())
    if extra_prompt and extra_prompt.strip():
        parts.append(extra_prompt.strip())
    parts.append("The code context is in the image.")
    return "\n\n".join(parts)


def render_images(
    text: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    dpi: int,
    font_path: str,
    preserve_newlines: bool,
    enable_syntax_highlight: bool,
    language: str,
    should_crop_whitespace: bool,
    enable_two_column: bool,
    enable_bold: bool,
    theme: str,
):
    margin_px = int(width * DEFAULT_MARGIN_RATIO)
    images = []
    for img in text_to_image_stream(
        text,
        width=width,
        height=height,
        font_size=font_size,
        line_height=line_height,
        margin_px=margin_px,
        dpi=dpi,
        font_path=font_path,
        preserve_newlines=preserve_newlines,
        enable_syntax_highlight=enable_syntax_highlight,
        filename=None,
        language=language,
        should_crop_whitespace=should_crop_whitespace,
        enable_two_column=enable_two_column,
        enable_bold=enable_bold,
        theme=theme,
    ):
        images.append(img)
    return images


def call_and_print(
    client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    images,
    ratio,
    resolution,
    max_tokens: int,
    original_text_tokens: int,
):
    response_text, token_info = call_llm_with_images(
        client, model_name, images, system_prompt, user_prompt, max_tokens=max_tokens
    )
    payload = {
        "compression_ratio": ratio,
        "resolution": resolution,
        "num_images": len(images),
        "response_text": response_text,
        "token_info": token_info,
        "original_text_tokens": original_text_tokens,
    }
    print(json.dumps(payload, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--code-context", type=str, default=None)
    parser.add_argument("--code-context-file", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--client-type", type=str, default="OpenAI", choices=["OpenAI", "Azure"])
    parser.add_argument("--system-prompt", type=str, default="You are a helpful coding assistant.")
    parser.add_argument(
        "--user-prompt",
        type=str,
        default="The image contains a code instruction. Follow it and respond directly.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resize-mode", action="store_true", default=True)
    parser.add_argument("--no-resize-mode", action="store_false", dest="resize_mode")
    parser.add_argument("--resize-ratios", type=str, default="1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--line-height", type=float, default=DEFAULT_LINE_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--preserve-newlines", action="store_true", default=True)
    parser.add_argument("--enable-syntax-highlight", action="store_true", default=False)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--crop-whitespace", action="store_true", default=False)
    parser.add_argument("--enable-two-column", action="store_true", default=False)
    parser.add_argument("--enable-bold", action="store_true", default=False)
    parser.add_argument("--theme", type=str, default="light", choices=["light", "modern"])

    args = parser.parse_args()

    instruction = read_instruction(args)
    if instruction is None or not instruction.strip():
        print("Empty instruction")
        return
    code_context = read_code_context(args)
    if code_context is None:
        return
    if not code_context.strip():
        print("Empty code context")
        return

    client = create_client(args.client_type)
    user_prompt = build_user_prompt(instruction, args.user_prompt)
    ratios = parse_ratios(args.resize_ratios)
    if not ratios:
        ratios = [1.0]

    if args.resize_mode:
        try:
            text_tokens = get_text_tokens(code_context)
        except Exception:
            text_tokens = max(1, len(code_context) // 4)

        text_structure = analyze_text_structure(code_context)
        res_1x, fs_1x, _ = optimize_layout_config_dry(
            target_tokens=text_tokens,
            previous_configs=[],
            text_tokens=text_tokens,
            line_height=args.line_height,
            text_structure=text_structure,
            compression_ratio=1.0,
            page_limit=100,
            text=code_context,
            enable_syntax_highlight=args.enable_syntax_highlight,
            language=args.language,
            preserve_newlines=args.preserve_newlines,
            font_path=args.font_path,
            theme=args.theme,
        )

        base_images = render_images(
            code_context,
            width=res_1x,
            height=res_1x,
            font_size=fs_1x,
            line_height=args.line_height,
            dpi=args.dpi,
            font_path=args.font_path,
            preserve_newlines=args.preserve_newlines,
            enable_syntax_highlight=args.enable_syntax_highlight,
            language=args.language,
            should_crop_whitespace=args.crop_whitespace,
            enable_two_column=args.enable_two_column,
            enable_bold=args.enable_bold,
            theme=args.theme,
        )

        nonzero_ratios = [r for r in ratios if float(r) != 0.0]
        resized_map = {}
        if nonzero_ratios:
            resized_map = resize_images_for_compression(
                base_images,
                text_tokens,
                compression_ratios=nonzero_ratios,
            )

        for ratio in ratios:
            if float(ratio) == 0.0:
                blank = PIL_Image.new("RGB", (14, 14), color="white")
                call_and_print(
                    client,
                    args.model,
                    args.system_prompt,
                    user_prompt,
                    [blank],
                    ratio,
                    "14x14",
                    args.max_tokens,
                    text_tokens,
                )
                continue
            resized_images, target_resolution = resized_map.get(ratio, ([], None))
            if not resized_images or not target_resolution:
                continue
            call_and_print(
                client,
                args.model,
                args.system_prompt,
                user_prompt,
                resized_images,
                ratio,
                f"{target_resolution}x{target_resolution}",
                args.max_tokens,
                text_tokens,
            )
    else:
        images = render_images(
            code_context,
            width=args.width,
            height=args.height,
            font_size=args.font_size,
            line_height=args.line_height,
            dpi=args.dpi,
            font_path=args.font_path,
            preserve_newlines=args.preserve_newlines,
            enable_syntax_highlight=args.enable_syntax_highlight,
            language=args.language,
            should_crop_whitespace=args.crop_whitespace,
            enable_two_column=args.enable_two_column,
            enable_bold=args.enable_bold,
            theme=args.theme,
        )
        call_and_print(
            client,
            args.model,
            args.system_prompt,
            user_prompt,
            images,
            None,
            f"{args.width}x{args.height}",
            args.max_tokens,
        )


if __name__ == "__main__":
    main()
