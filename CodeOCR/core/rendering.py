# Core rendering module

from typing import List, Iterator
from PIL import Image as PIL_Image, ImageDraw

from .constants import DEFAULT_MARGIN_RATIO, NEWLINE_MARKER, TYPOGRAPHIC_REPLACEMENTS
from .fonts import get_font
from .text_processing import prepare_text_for_rendering, crop_whitespace
from .syntax import parse_code_with_syntax_highlighting, PYGMENTS_AVAILABLE


def text_to_image(
    text: str,
    width: int = 800,
    height: int = 800,
    font_size: int = 40,
    line_height: float = 1.0,
    margin_px: int = None,
    dpi: int = 300,
    font_path: str = None,
    bg_color: str = "white",
    text_color: str = "black",
    preserve_newlines: bool = True,
    enable_syntax_highlight: bool = False,
    filename: str = None,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
    theme: str = "light",
) -> List[PIL_Image.Image]:
    """
    Render text as images (may produce multiple pages).
    
    Args:
        text: Text to render
        width: Image width in pixels
        height: Image height in pixels
        font_size: Font size
        line_height: Line height multiplier
        margin_px: Margin in pixels, None uses default ratio
        dpi: DPI setting
        font_path: Font file path
        bg_color: Background color
        text_color: Text color
        preserve_newlines: Whether to preserve newlines
        enable_syntax_highlight: Enable syntax highlighting
        filename: Filename for language detection
        language: Language name
        should_crop_whitespace: Whether to crop whitespace
        enable_two_column: Enable two-column layout
        enable_bold: Enable bold text
        theme: Theme ('light' or 'modern')
    
    Returns:
        List of PIL Images
    """
    if margin_px is None:
        margin_px = int(width * DEFAULT_MARGIN_RATIO)
    
    font = get_font(font_size, font_path)
    temp_img = PIL_Image.new("RGB", (width, height), color=bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    
    text_area_width = width - 2 * margin_px
    text_area_height = height - 2 * margin_px
    line_height_px = int(font_size * line_height)
    max_lines_per_page = max(1, int(text_area_height / line_height_px))
    
    # Determine column width
    if enable_two_column:
        column_width = (width - 2 * margin_px) // 2
        column_gap = 10
    else:
        column_width = text_area_width
        column_gap = 0
    
    # Syntax highlighting mode
    if enable_syntax_highlight and PYGMENTS_AVAILABLE:
        return _render_with_highlight(
            text, width, height, margin_px, font, temp_draw,
            line_height_px, max_lines_per_page, column_width, column_gap,
            bg_color, dpi, preserve_newlines, enable_two_column, enable_bold,
            filename, language, theme, should_crop_whitespace
        )
    
    # Plain mode
    return _render_plain(
        text, width, height, margin_px, font, temp_draw, text_area_width,
        line_height_px, max_lines_per_page, bg_color, text_color, dpi,
        preserve_newlines, enable_two_column, enable_bold, should_crop_whitespace
    )


def _render_with_highlight(
    text, width, height, margin_px, font, temp_draw,
    line_height_px, max_lines_per_page, column_width, column_gap,
    bg_color, dpi, preserve_newlines, enable_two_column, enable_bold,
    filename, language, theme, should_crop_whitespace
):
    """Render with syntax highlighting."""
    from PIL import ImageColor
    
    colored_tokens = parse_code_with_syntax_highlighting(
        text, filename=filename, language=language, theme=theme
    )
    
    # Preprocess tokens
    processed_tokens = []
    for token_text, token_color in colored_tokens:
        processed = token_text.replace("\t", "    ")
        for orig, repl in TYPOGRAPHIC_REPLACEMENTS.items():
            processed = processed.replace(orig, repl)
        if not preserve_newlines:
            processed = processed.replace("\n", NEWLINE_MARKER)
        processed_tokens.append((processed, token_color))
    
    pages = []
    current_page_lines = 0
    current_x = margin_px
    current_y = margin_px
    current_column = 0
    column_start_x = margin_px
    
    img = PIL_Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    def _new_page():
        nonlocal img, draw, current_page_lines, current_y, current_column, column_start_x, current_x
        if should_crop_whitespace:
            img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        pages.append(img)
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        current_page_lines = 0
        current_y = margin_px
        current_column = 0
        column_start_x = margin_px
        current_x = margin_px
    
    def _check_new_page():
        nonlocal current_column, column_start_x, current_x, current_y, current_page_lines
        if current_page_lines >= max_lines_per_page:
            if enable_two_column and current_column == 0:
                current_column = 1
                column_start_x = width // 2 + column_gap // 2
                current_x = column_start_x
                current_y = margin_px
                current_page_lines = 0
            else:
                _new_page()
    
    for token_text, token_color in processed_tokens:
        rgb_color = ImageColor.getrgb(token_color)
        
        for char in token_text:
            if preserve_newlines and char == "\n":
                current_y += line_height_px
                current_x = column_start_x
                current_page_lines += 1
                _check_new_page()
                continue
            
            char_w = temp_draw.textlength(char, font=font)
            column_right = column_start_x + column_width if current_column == 0 else width - margin_px
            
            if current_x + char_w > column_right and current_x > column_start_x:
                current_y += line_height_px
                current_x = column_start_x
                current_page_lines += 1
                _check_new_page()
            
            if enable_bold:
                for dx in [0, 1]:
                    draw.text((current_x + dx, current_y), char, font=font, fill=rgb_color)
            else:
                draw.text((current_x, current_y), char, font=font, fill=rgb_color)
            current_x += char_w
    
    # Save last page
    if current_page_lines > 0 or current_x > margin_px:
        if should_crop_whitespace:
            img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        pages.append(img)
    
    return pages if pages else [img]


def _render_plain(
    text, width, height, margin_px, font, temp_draw, text_area_width,
    line_height_px, max_lines_per_page, bg_color, text_color, dpi,
    preserve_newlines, enable_two_column, enable_bold, should_crop_whitespace
):
    """Plain text rendering."""
    processed_text = prepare_text_for_rendering(text, preserve_newlines=preserve_newlines)
    
    # Split into lines
    lines = []
    if preserve_newlines:
        for original_line in processed_text.split("\n"):
            current_line = ""
            current_width = 0
            for char in original_line:
                char_w = temp_draw.textlength(char, font=font)
                if current_width + char_w > text_area_width and current_line:
                    lines.append(current_line)
                    current_line = char
                    current_width = char_w
                else:
                    current_line += char
                    current_width += char_w
            if current_line:
                lines.append(current_line)
    else:
        current_line = ""
        current_width = 0
        for char in processed_text:
            char_w = temp_draw.textlength(char, font=font)
            if current_width + char_w > text_area_width and current_line:
                lines.append(current_line)
                current_line = char
                current_width = char_w
            else:
                current_line += char
                current_width += char_w
        if current_line:
            lines.append(current_line)
    
    # Render pages
    pages = []
    page_start = 0
    
    while page_start < len(lines):
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        y = margin_px
        page_lines = 0
        
        while page_start < len(lines) and page_lines < max_lines_per_page:
            line = lines[page_start]
            x = margin_px
            for char in line:
                char_w = temp_draw.textlength(char, font=font)
                if enable_bold:
                    for dx in [0, 1]:
                        draw.text((x + dx, y), char, font=font, fill=text_color)
                else:
                    draw.text((x, y), char, font=font, fill=text_color)
                x += char_w
            y += line_height_px
            page_lines += 1
            page_start += 1
        
        if should_crop_whitespace:
            img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        pages.append(img)
    
    return pages


def text_to_image_stream(
    text: str,
    width: int = 800,
    height: int = 800,
    font_size: int = 40,
    line_height: float = 1.0,
    margin_px: int = None,
    dpi: int = 300,
    font_path: str = None,
    bg_color: str = "white",
    text_color: str = "black",
    preserve_newlines: bool = True,
    enable_syntax_highlight: bool = False,
    filename: str = None,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
    theme: str = "light",
) -> Iterator[PIL_Image.Image]:
    """
    Stream render text to images (generator version).
    Parameters same as text_to_image.
    """
    images = text_to_image(
        text, width, height, font_size, line_height, margin_px, dpi,
        font_path, bg_color, text_color, preserve_newlines,
        enable_syntax_highlight, filename, language, should_crop_whitespace,
        enable_two_column, enable_bold, theme
    )
    for img in images:
        yield img
