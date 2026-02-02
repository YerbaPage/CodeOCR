# Token calculation module

import math
from typing import List

# Try to import tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def get_text_tokens(text: str) -> int:
    """Calculate the number of tokens in text."""
    if TIKTOKEN_AVAILABLE:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    # Fallback: estimate ~4 chars/token
    return max(1, len(text) // 4)


def calculate_image_tokens_qwen3(width: int, height: int) -> int:
    """
    Calculate image tokens for Qwen3/GPT-4V models.
    Based on 112x112 tile calculation.
    """
    tile_size = 112
    tiles_w = math.ceil(width / tile_size)
    tiles_h = math.ceil(height / tile_size)
    total_tiles = tiles_w * tiles_h
    # ~170 tokens per tile
    return total_tiles * 170


def get_expanded_resolution_list() -> List[int]:
    """
    Generate expanded resolution list.
    Includes multiples of 112: 1/8, 1/4, 1/2, 1, 1.25, 1.5, ... up to 40.
    """
    resolutions = []
    
    # Fractional steps: 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    resolutions.extend([14, 28, 42, 56, 70, 84, 98])
    
    # Quarter-step multiples: 1, 1.25, 1.5, 1.75, 2, ..., 40
    for i in range(4, 161):
        res = int(112 * i / 4)
        if res not in resolutions:
            resolutions.append(res)
    
    return sorted(resolutions)
