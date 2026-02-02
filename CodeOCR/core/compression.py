# Image compression module

from typing import List, Dict, Tuple
from PIL import Image as PIL_Image

from .constants import COMPRESSION_RATIOS
from .tokens import calculate_image_tokens_qwen3, get_expanded_resolution_list


def find_closest_resolution(target_tokens: int, resolutions: List[int]) -> int:
    """Find resolution closest to target token count."""
    best_res = resolutions[0]
    best_diff = float("inf")
    
    for res in resolutions:
        tokens = calculate_image_tokens_qwen3(res, res)
        diff = abs(tokens - target_tokens)
        if diff < best_diff:
            best_diff = diff
            best_res = res
    
    return best_res


def resize_images_for_compression(
    images: List[PIL_Image.Image],
    text_tokens: int,
    compression_ratios: List[float] = None,
) -> Dict[float, Tuple[List[PIL_Image.Image], int]]:
    """
    Resize images based on compression ratio.
    
    Args:
        images: Original image list
        text_tokens: Text token count
        compression_ratios: Compression ratio list
    
    Returns:
        {ratio: (resized_images, target_resolution)}
    """
    if compression_ratios is None:
        compression_ratios = COMPRESSION_RATIOS
    
    if not images:
        return {}
    
    resolution_list = get_expanded_resolution_list()
    results = {}
    
    base_images = images
    base_resolution = images[0].width
    
    for ratio in sorted(compression_ratios):
        # ratio=0 returns blank image
        if float(ratio) == 0.0:
            blank = PIL_Image.new("RGB", (14, 14), color="white")
            results[ratio] = ([blank], 14)
            continue
        
        image_token_limit = text_tokens / ratio
        
        if ratio == 1.0:
            # 1x uses original images
            results[1.0] = (base_images, base_resolution)
        else:
            # Other ratios: resize
            num_images = len(base_images)
            per_image_tokens = image_token_limit / num_images if num_images > 0 else image_token_limit
            target_resolution = find_closest_resolution(int(per_image_tokens), resolution_list)
            
            resized_images = [
                img.resize((target_resolution, target_resolution), PIL_Image.Resampling.LANCZOS)
                for img in base_images
            ]
            results[ratio] = (resized_images, target_resolution)
    
    return results
