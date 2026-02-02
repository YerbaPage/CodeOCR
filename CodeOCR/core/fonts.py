# Font handling module

import os
from PIL import ImageFont


def get_font(font_size: int, font_path: str = None):
    """
    Get monospace font object.
    
    Args:
        font_size: Font size in pixels
        font_path: Optional font path, if None will search system fonts
    
    Returns:
        ImageFont object
    """
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)
    
    # Try system monospace fonts by priority
    monospace_fonts = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        # macOS
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
        # Windows
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    
    for path in monospace_fonts:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    
    return ImageFont.load_default()
