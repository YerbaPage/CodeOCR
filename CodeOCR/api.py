# CodeOCR Simplified API

from typing import List, Tuple, Dict
from PIL import Image as PIL_Image

from .core import (
    text_to_image,
    text_to_image_stream,
    optimize_layout_config_dry,
    analyze_text_structure,
    resize_images_for_compression,
    get_text_tokens,
    DEFAULT_LINE_HEIGHT,
    DEFAULT_MARGIN_RATIO,
)
from .client import create_client, call_llm_with_images


def render_code_to_images(
    code: str,
    language: str = "python",
    enable_syntax_highlight: bool = True,
    theme: str = "modern",
    auto_optimize: bool = True,
    width: int = 1024,
    height: int = 1024,
    font_size: int = 32,
) -> List[PIL_Image.Image]:
    """
    Render code to images.
    
    Args:
        code: Source code text
        language: Programming language
        enable_syntax_highlight: Enable syntax highlighting
        theme: Theme ('light' or 'modern')
        auto_optimize: Auto-optimize layout
        width: Image width (when auto_optimize=False)
        height: Image height (when auto_optimize=False)
        font_size: Font size (when auto_optimize=False)
    
    Returns:
        List of PIL images
    """
    if auto_optimize:
        # Auto-optimize layout
        text_tokens = get_text_tokens(code)
        text_structure = analyze_text_structure(code)
        
        res, fs, _ = optimize_layout_config_dry(
            target_tokens=text_tokens,
            text_tokens=text_tokens,
            text_structure=text_structure,
            compression_ratio=1.0,
            text=code,
            enable_syntax_highlight=enable_syntax_highlight,
            language=language,
            preserve_newlines=True,
            theme=theme,
        )
        width = height = res
        font_size = fs
    
    margin_px = int(width * DEFAULT_MARGIN_RATIO)
    
    return text_to_image(
        code,
        width=width,
        height=height,
        font_size=font_size,
        margin_px=margin_px,
        preserve_newlines=True,
        enable_syntax_highlight=enable_syntax_highlight,
        language=language,
        theme=theme,
    )


def render_and_query(
    code: str,
    instruction: str,
    model: str = "gpt-5-mini",
    language: str = "python",
    enable_syntax_highlight: bool = True,
    theme: str = "modern",
    system_prompt: str = "You are a helpful coding assistant.",
    max_tokens: int = 2048,
    client_type: str = "OpenAI",
) -> Tuple[str, Dict]:
    """
    Render code to images and send to LLM.
    
    Args:
        code: Source code text
        instruction: User instruction
        model: Model name
        language: Programming language
        enable_syntax_highlight: Enable syntax highlighting
        theme: Theme
        system_prompt: System prompt
        max_tokens: Maximum output tokens
        client_type: Client type ("OpenAI" or "Azure")
    
    Returns:
        (response_text, token_info)
    """
    # Render images
    images = render_code_to_images(
        code,
        language=language,
        enable_syntax_highlight=enable_syntax_highlight,
        theme=theme,
        auto_optimize=True,
    )
    
    # Call LLM
    client = create_client(client_type)
    user_prompt = f"{instruction}\n\nThe code context is in the image."
    
    return call_llm_with_images(
        client, model, images, system_prompt, user_prompt, max_tokens
    )


def compress_images(
    images: List[PIL_Image.Image],
    text_tokens: int,
    ratios: List[float] = None,
) -> Dict[float, Tuple[List[PIL_Image.Image], int]]:
    """
    Resize images based on compression ratio.
    
    Args:
        images: Original image list
        text_tokens: Text token count
        ratios: Compression ratios, default [1, 2, 4, 8]
    
    Returns:
        {ratio: (resized_images, resolution)}
    """
    if ratios is None:
        ratios = [1.0, 2.0, 4.0, 8.0]
    return resize_images_for_compression(images, text_tokens, ratios)
