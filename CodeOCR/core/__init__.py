# CodeOCR Core Module
# Core functionality for rendering text to images

from .constants import (
    COMPRESSION_RATIOS,
    DEFAULT_DPI,
    DEFAULT_FONT_SIZE,
    DEFAULT_LINE_HEIGHT,
    DEFAULT_MARGIN_RATIO,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
)
from .fonts import get_font
from .text_processing import prepare_text_for_rendering, crop_whitespace
from .syntax import parse_code_with_syntax_highlighting, PYGMENTS_AVAILABLE
from .tokens import (
    get_text_tokens,
    calculate_image_tokens_qwen3,
    get_expanded_resolution_list,
)
from .rendering import text_to_image, text_to_image_stream
from .layout import (
    optimize_layout_config_dry,
    analyze_text_structure,
    estimate_page_count,
)
from .compression import resize_images_for_compression

__all__ = [
    # Constants
    "COMPRESSION_RATIOS",
    "DEFAULT_DPI",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_LINE_HEIGHT",
    "DEFAULT_MARGIN_RATIO",
    "MIN_FONT_SIZE",
    "MAX_FONT_SIZE",
    # Fonts
    "get_font",
    # Text processing
    "prepare_text_for_rendering",
    "crop_whitespace",
    # Syntax highlighting
    "parse_code_with_syntax_highlighting",
    "PYGMENTS_AVAILABLE",
    # Token calculation
    "get_text_tokens",
    "calculate_image_tokens_qwen3",
    "get_expanded_resolution_list",
    # Rendering
    "text_to_image",
    "text_to_image_stream",
    # Layout
    "optimize_layout_config_dry",
    "analyze_text_structure",
    "estimate_page_count",
    # Compression
    "resize_images_for_compression",
]
