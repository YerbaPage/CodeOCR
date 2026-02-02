# CodeOCR - Code Image Rendering Tool
#
# Simple usage:
#   from CodeOCR import render_code_to_images, render_and_query
#
# CLI:
#   python -m CodeOCR.demo render --file example.py -o output.png
#   python -m CodeOCR.demo query --file example.py -i "Explain this code"

from .api import render_code_to_images, render_and_query, compress_images
from .client import create_client, call_llm_with_images
from .core import (
    text_to_image,
    text_to_image_stream,
    optimize_layout_config_dry,
    analyze_text_structure,
    resize_images_for_compression,
    get_text_tokens,
    calculate_image_tokens_qwen3,
    get_expanded_resolution_list,
    COMPRESSION_RATIOS,
    DEFAULT_DPI,
    DEFAULT_FONT_SIZE,
    DEFAULT_LINE_HEIGHT,
    DEFAULT_MARGIN_RATIO,
)

__version__ = "1.0.0"

__all__ = [
    # High-level API
    "render_code_to_images",
    "render_and_query",
    "compress_images",
    # Client
    "create_client",
    "call_llm_with_images",
    # Core functions
    "text_to_image",
    "text_to_image_stream",
    "optimize_layout_config_dry",
    "analyze_text_structure",
    "resize_images_for_compression",
    # Token calculation
    "get_text_tokens",
    "calculate_image_tokens_qwen3",
    "get_expanded_resolution_list",
    # Constants
    "COMPRESSION_RATIOS",
    "DEFAULT_DPI",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_LINE_HEIGHT",
    "DEFAULT_MARGIN_RATIO",
]
