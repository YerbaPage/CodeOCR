# Layout optimization module

import math
from typing import List, Tuple, Dict

from .constants import (
    DEFAULT_MARGIN_RATIO, DEFAULT_LINE_HEIGHT, MIN_FONT_SIZE, MAX_FONT_SIZE,
    CHAR_WIDTH_RATIO, AVG_CHARS_PER_TOKEN
)
from .tokens import calculate_image_tokens_qwen3, get_expanded_resolution_list
from .fonts import get_font
from PIL import Image as PIL_Image, ImageDraw


def analyze_text_structure(text: str) -> Dict:
    """
    Analyze text structure.
    
    Returns:
        {'num_lines': int, 'max_line_chars': int, 'avg_line_chars': float}
    """
    lines = text.split("\n")
    num_lines = len(lines)
    line_lengths = [len(line) for line in lines]
    max_line_chars = max(line_lengths) if line_lengths else 0
    avg_line_chars = sum(line_lengths) / num_lines if num_lines > 0 else 0
    
    return {
        "num_lines": num_lines,
        "max_line_chars": max_line_chars,
        "avg_line_chars": avg_line_chars,
    }


def estimate_page_count(
    text_tokens: int,
    resolution: int,
    font_size: int,
    line_height: float = DEFAULT_LINE_HEIGHT,
) -> int:
    """
    Quickly estimate required page count for given configuration (without rendering).
    """
    margin = resolution * DEFAULT_MARGIN_RATIO
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    if font_size <= 0 or available_width <= 0 or available_height <= 0:
        return 999
    
    char_width = font_size * CHAR_WIDTH_RATIO
    line_height_px = font_size * line_height
    
    chars_per_line = max(1, int(available_width / char_width))
    lines_per_page = max(1, int(available_height / line_height_px))
    chars_per_page = chars_per_line * lines_per_page
    
    total_chars = text_tokens * AVG_CHARS_PER_TOKEN
    newline_overhead = 1.3
    effective_chars = total_chars * newline_overhead
    
    return max(1, math.ceil(effective_chars / chars_per_page))


def calculate_max_font_size_at_resolution(
    text: str,
    resolution: int,
    target_pages: int,
    line_height: float = DEFAULT_LINE_HEIGHT,
    font_path: str = None,
    enable_syntax_highlight: bool = False,
    preserve_newlines: bool = True,
    language: str = None,
    theme: str = "light",
) -> int:
    """
    Find maximum font size at given resolution and page count using binary search.
    """
    from .rendering import text_to_image
    
    margin_px = int(resolution * DEFAULT_MARGIN_RATIO)
    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_fs = low
    
    while low <= high:
        mid = (low + high) // 2
        images = text_to_image(
            text,
            width=resolution,
            height=resolution,
            font_size=mid,
            line_height=line_height,
            margin_px=margin_px,
            font_path=font_path,
            preserve_newlines=preserve_newlines,
            enable_syntax_highlight=enable_syntax_highlight,
            language=language,
            theme=theme,
        )
        
        if len(images) <= target_pages:
            best_fs = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_fs


def optimize_layout_config_dry(
    target_tokens: float,
    previous_configs: List[Tuple[int, int]] = None,
    text_tokens: int = None,
    line_height: float = 1.0,
    text_structure: dict = None,
    compression_ratio: float = None,
    page_limit: int = 100,
    text: str = None,
    enable_syntax_highlight: bool = False,
    language: str = None,
    preserve_newlines: bool = True,
    font_path: str = None,
    theme: str = "light",
) -> Tuple[int, int, int]:
    """
    Optimize layout configuration to find best (resolution, font_size, pages) combination.
    
    Args:
        target_tokens: Target token count
        previous_configs: Previously tried configurations
        text_tokens: Text token count
        line_height: Line height
        text_structure: Text structure info
        compression_ratio: Compression ratio
        page_limit: Maximum page limit
        text: Original text (for accurate calculation)
        enable_syntax_highlight: Enable syntax highlighting
        language: Language
        preserve_newlines: Preserve newlines
        font_path: Font path
        theme: Theme
    
    Returns:
        (resolution, font_size, pages)
    """
    if previous_configs is None:
        previous_configs = []
    
    estimated_chars = text_tokens * 3.5 if text_tokens else 10000
    resolutions = get_expanded_resolution_list()
    
    # Dynamic token tolerance range
    if target_tokens < 50:
        token_min_ratio, token_max_ratio = 0.5, 2.0
    elif target_tokens < 100:
        token_min_ratio, token_max_ratio = 0.7, 1.5
    elif target_tokens < 3000:
        token_min_ratio, token_max_ratio = 0.8, 1.25
    elif target_tokens < 5000:
        token_min_ratio, token_max_ratio = 0.9, 1.12
    else:
        token_min_ratio, token_max_ratio = 0.95, 1.05
    
    all_configs = []
    
    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        min_pages = max(1, math.ceil(target_tokens * token_min_ratio / per_image_tokens))
        max_pages = math.floor(target_tokens * token_max_ratio / per_image_tokens)
        
        if min_pages > max_pages:
            continue
        
        for pages in range(min_pages, min(max_pages + 1, page_limit + 1)):
            is_new = (res, pages) not in previous_configs
            
            # Area-based font size estimation
            margin = res * 0.01
            total_area = ((res - 2 * margin) ** 2) * pages
            
            if estimated_chars > 0:
                fs_squared = total_area / (estimated_chars * 0.6 * line_height)
                optimal_fs = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, int(math.sqrt(fs_squared))))
            else:
                optimal_fs = MAX_FONT_SIZE
            
            char_area = (optimal_fs * 0.6) * (optimal_fs * line_height)
            fill_rate = (estimated_chars * char_area) / total_area if total_area > 0 else 0
            
            actual_tokens = pages * per_image_tokens
            token_ratio = actual_tokens / target_tokens if target_tokens > 0 else 1.0
            
            all_configs.append({
                "resolution": res,
                "pages": pages,
                "font_size": optimal_fs,
                "fill_rate": fill_rate,
                "tokens": actual_tokens,
                "token_ratio": token_ratio,
                "is_new": is_new,
            })
    
    if not all_configs:
        return 112, MIN_FONT_SIZE, 1
    
    # Scoring function
    def _score(c):
        token_diff = abs(c["token_ratio"] - 1.0)
        token_score = 1.0 / (1.0 + token_diff * 2.0)
        
        fill_rate = c["fill_rate"]
        fill_score = 1.0 if fill_rate >= 0.8 else (0.2 + fill_rate) if fill_rate >= 0.2 else 0.1
        
        # Prefer larger resolution
        resolution_bonus = 1.0 + (c["resolution"] - 112) / (4480 - 112)
        
        score = token_score * fill_score * resolution_bonus
        score = score / (c["pages"] ** 0.3)  # Prefer fewer pages
        
        if not c["is_new"]:
            score *= 0.95
        
        return score
    
    for c in all_configs:
        c["score"] = _score(c)
    
    all_configs.sort(key=lambda x: -x["score"])
    best = all_configs[0]
    
    # If original text provided, calculate font size accurately
    if text:
        refined_fs = calculate_max_font_size_at_resolution(
            text, best["resolution"], best["pages"],
            line_height=line_height, font_path=font_path,
            enable_syntax_highlight=enable_syntax_highlight,
            preserve_newlines=preserve_newlines,
            language=language, theme=theme
        )
        return best["resolution"], refined_fs, best["pages"]
    
    return best["resolution"], best["font_size"], best["pages"]
