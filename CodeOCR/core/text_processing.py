# Text preprocessing module

from typing import Tuple
from PIL import Image as PIL_Image
from .constants import NEWLINE_MARKER, TYPOGRAPHIC_REPLACEMENTS, TAB_SPACES


def prepare_text_for_rendering(text: str, preserve_newlines: bool = False) -> str:
    """
    Prepare text for rendering.
    
    Args:
        text: Original text
        preserve_newlines: Whether to preserve newlines
    
    Returns:
        Processed text
    """
    # Replace tabs
    text = text.replace("\t", TAB_SPACES)
    
    # Handle newlines
    if not preserve_newlines:
        text = text.replace("\n", NEWLINE_MARKER)
    
    # Replace typographic characters
    for orig, repl in TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(orig, repl)
    
    return text


def crop_whitespace(
    img: PIL_Image.Image, 
    bg_color: str = "white", 
    keep_margin: Tuple[int, int] = (0, 0)
) -> PIL_Image.Image:
    """
    Crop whitespace from right and bottom of image.
    
    Args:
        img: PIL Image object
        bg_color: Background color
        keep_margin: Margins to keep (left, top)
    
    Returns:
        Cropped image
    """
    gray = img.convert("L")
    
    # Determine background threshold
    if bg_color == "white":
        bg_threshold = 240
    elif bg_color == "black":
        bg_threshold = 15
    else:
        from PIL import ImageColor
        rgb = ImageColor.getrgb(bg_color)
        bg_threshold = int(sum(rgb) / 3)
    
    # Create mask
    mask = gray.point(lambda p: 0 if p > bg_threshold else 255, mode="1")
    bbox = mask.getbbox()
    
    if bbox is None:
        return img
    
    left, top, right, bottom = bbox
    left = max(0, left - keep_margin[0])
    top = max(0, top - keep_margin[1])
    
    return img.crop((left, top, right, bottom))
