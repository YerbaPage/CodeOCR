import os
import math
import sys
import argparse
import time
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image as PIL_Image, ImageDraw, ImageFont
import tiktoken
from llm_utils import get_text_tokens

# Try to import pygments (for code highlighting)
try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
    from pygments.token import Token

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("Warning: pygments not installed, syntax highlighting will be unavailable")

# Try to import transformers (for AutoProcessor)
try:
    from transformers import AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed, AutoProcessor token calculation will be unavailable")


# Constants
NEWLINE_MARKER = "⏎"  # Marker for preserved newlines in compact mode
COMPRESSION_RATIOS = [1, 2, 4, 8]  # Standard compression ratios
DEFAULT_MARGIN_RATIO = 0.01  # Margin as ratio of width (1%)
DEFAULT_FONT_SIZE = 40
DEFAULT_LINE_HEIGHT = 1.0
DEFAULT_DPI = 300
MIN_FONT_SIZE = 4
MAX_FONT_SIZE = 150

# Monospace font characteristics (empirical values)
CHAR_WIDTH_RATIO = 0.6  # Character width ≈ font_size * 0.6
AVG_CHARS_PER_TOKEN = 3.5  # Average characters per token


def get_all_modes():
    """
    Get all mode list, including text_only, image, and all compression ratio modes.
    
    Returns:
        List of mode names
    """
    modes = ["text_only", "image"]
    for ratio in sorted(COMPRESSION_RATIOS):
        modes.append(f"image_ratio{ratio}")
    return modes


def get_flat_filename(filename: str) -> str:
    """
    Convert original filename to flat format (for file naming).
    
    Args:
        filename: Original filename
        
    Returns:
        Flat filename with slashes replaced by underscores
    """
    if filename is None:
        return "unknown"
    return filename.replace("/", "_")


def get_expanded_resolution_list() -> List[int]:
    """
    Generate expanded resolution list with half-step multiples and fractional steps.
    
    Returns:
        List of resolutions including:
        - Fractional steps: 112 * 1/8 (14), 112 * 1/4 (28), 112 * 1/2 (56)
        - Half-step multiples: 112 * 1, 112 * 1.5, 112 * 2, 112 * 2.5, ..., up to 112 * 40
    """
    resolutions = []
    
    # Fractional steps: 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    resolutions.extend([14, 28, 42, 56, 70, 84, 98])
    
    # Quarter-step multiples: 1, 1.25, 1.5, 1.75, 2, ..., up to 40
    # i goes from 4 to 160 (representing 1 to 40 in steps of 0.25)
    for i in range(4, 161):
        res = int(112 * i / 4)
        if res not in resolutions:
            resolutions.append(res)
    
    return sorted(resolutions)



def get_font(font_size: int, font_path: str = None):
    """
    Get monospace font object.

    Args:
        font_size: Font size in pixels
        font_path: Optional font path, if None will search system monospace fonts

    Returns:
        ImageFont object (monospace font)
    """
    try:
        if font_path and os.path.exists(font_path):
            # Use specified font
            font = ImageFont.truetype(font_path, font_size)
            print(f"  Using specified font: {font_path}")
            return font
        else:
            # Try system monospace fonts (sorted by priority)
            monospace_font_paths = [
                # Linux common monospace fonts
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
                "/usr/share/fonts/truetype/courier/Courier_New.ttf",
                "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
                # macOS monospace fonts
                "/System/Library/Fonts/Menlo.ttc",
                "/Library/Fonts/Courier New.ttf",
                "/System/Library/Fonts/Courier.ttc",
                # Windows monospace fonts
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/cour.ttf",
                "C:/Windows/Fonts/courbd.ttf",
                "C:/Windows/Fonts/lucon.ttf",
            ]

            font = None
            for path in monospace_font_paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except Exception:
                        # If loading fails, try next one
                        continue

            if font is None:
                # If all monospace fonts are not found, use PIL default font
                font = ImageFont.load_default()

            return font
    except Exception as e:
        print(f"  Warning: Failed to load font: {e}, using default font")
        return ImageFont.load_default()


# Typographic character replacements (normalize to ASCII)
TYPOGRAPHIC_REPLACEMENTS = {
    "'": "'",  # Left single quote
    "'": "'",  # Right single quote
    '"': '"',  # Left double quote
    '"': '"',  # Right double quote
    "–": "-",  # En dash
    "—": "--",  # Em dash
    "…": "...",  # Ellipsis
}

TAB_SPACES = "    "  # Tab replacement (4 spaces)


def prepare_text_for_rendering(text: str, preserve_newlines: bool = False) -> str:
    """
    Prepare text for rendering.
    
    Steps:
    1. Replace tabs with spaces
    2. Handle newlines (replace with marker in compact mode, keep in normal mode)
    3. Normalize special typographic characters to ASCII

    Args:
        text: Text to process
        preserve_newlines: Whether to preserve newlines (True=normal mode, False=compact mode)
        
    Returns:
        Processed text ready for rendering
    """
    # Replace tabs with spaces
    text = text.replace("\t", TAB_SPACES)

    # Handle newlines
    if not preserve_newlines:
        text = text.replace("\n", NEWLINE_MARKER)

    # Normalize typographic characters
    for original, replacement in TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(original, replacement)

    return text


def crop_whitespace(
    img: PIL_Image.Image, bg_color: str = "white", keep_margin: Tuple[int, int] = (0, 0)
) -> PIL_Image.Image:
    """
    Crop whitespace from image, remove white space on right and bottom.

    Args:
        img: PIL Image object
        bg_color: Background color (for detecting whitespace)
        keep_margin: Margins to keep (left, top), default (0, 0)

    Returns:
        Cropped PIL Image object
    """
    # Convert to grayscale for detection
    gray = img.convert("L")

    # Convert background color to grayscale value
    if bg_color == "white":
        bg_threshold = 240
    elif bg_color == "black":
        bg_threshold = 15
    else:
        # For other colors, calculate grayscale from RGB
        try:
            from PIL import ImageColor

            rgb = ImageColor.getrgb(bg_color)
            bg_threshold = int(sum(rgb) / 3)
        except:
            bg_threshold = 240  # Default to white

    # Create mask: non-background area is 1, background area is 0
    mask = gray.point(lambda p: 0 if p > bg_threshold else 255, mode="1")

    # Get content bounding box
    bbox = mask.getbbox()

    if bbox is None:
        # If no content, return original image
        return img

    # bbox format: (left, top, right, bottom)
    left, top, right, bottom = bbox

    # Keep left and top margins
    left = max(0, left - keep_margin[0])
    top = max(0, top - keep_margin[1])

    # Crop image (keep left and top margins, crop right and bottom whitespace)
    cropped = img.crop((left, top, right, bottom))

    return cropped


def parse_code_with_syntax_highlighting(
    code: str, filename: str = None, language: str = None, theme: str = "light"
) -> List[Tuple[str, str]]:
    """
    Parse code using Pygments and return list of colored tokens.

    Args:
        code: Source code text
        filename: Filename (for automatic language detection)
        language: Language name (e.g. 'python', 'javascript'), if None will auto-detect
        theme: Theme name ('light' or 'modern')

    Returns:
        List of (text, color) tuples, each tuple contains text content and corresponding color (RGB format)
    """
    if not PYGMENTS_AVAILABLE:
        # If Pygments not installed, return monochrome text
        return [(code, "#000000")]

    try:
        # Determine language
        if language:
            lexer = get_lexer_by_name(language)
        elif filename:
            try:
                lexer = guess_lexer_for_filename(filename, code)
                try:
                    from pygments.lexers.special import TextLexer
                except Exception:
                    TextLexer = None

                # If filename-based detection gives TextLexer (e.g. .txt), try content-based detection
                if TextLexer is not None and isinstance(lexer, TextLexer):
                    try:
                        from pygments.lexers import guess_lexer

                        content_lexer = guess_lexer(code)
                        if not isinstance(content_lexer, TextLexer):
                            lexer = content_lexer
                    except Exception:
                        pass
            except:
                try:
                    from pygments.lexers import guess_lexer

                    lexer = guess_lexer(code)
                except:
                    lexer = get_lexer_by_name("python")
        else:
            try:
                from pygments.lexers import guess_lexer

                lexer = guess_lexer(code)
            except:
                lexer = get_lexer_by_name("python")

        # Define color mappings
        if theme == "modern" or theme == "morden":
            # VS Code Light Modern theme
            color_map = {
                # Control Flow Keywords (Purple)
                Token.Keyword: "#AF00DB",             # Default Keyword -> Purple (covers if, return, import, from, etc.)
                Token.Keyword.Namespace: "#AF00DB",   # import/from -> Purple
                
                # Declaration/Storage Keywords (Blue)
                Token.Keyword.Declaration: "#0000FF", # def, class -> Blue
                Token.Keyword.Type: "#0000FF",        # int, str -> Blue
                Token.Keyword.Constant: "#0000FF",    # True, False, None -> Blue
                Token.Operator.Word: "#0000FF",       # and, or, not, is, in -> Blue
                
                # Functions & Builtins (Yellow/Ochre)
                Token.Name.Function: "#795E26",       # Function definitions -> Ochre
                Token.Name.Builtin: "#795E26",        # Built-in functions (open, print) -> Ochre
                Token.Name.Builtin.Pseudo: "#0000FF", # self, cls -> Blue (VS Code treats self as variable or keyword depending on config, but standard is often Blue)
                
                # Classes (Teal)
                Token.Name.Class: "#267F99",          # Class names -> Teal
                
                # Variables (Dark Blue / Light Blue)
                Token.Name: "#000000",                # Default Name -> Black
                Token.Name.Variable: "#001080",       # Variables -> Dark Blue
                Token.Name.Variable.Instance: "#001080",
                Token.Name.Variable.Class: "#001080",
                Token.Name.Variable.Global: "#001080",
                Token.Name.Attribute: "#001080",      # Attributes -> Dark Blue
                
                # Strings (Red)
                Token.String: "#A31515",              # Strings -> Red
                Token.String.Doc: "#008000",          # Docstrings -> Green (VS Code convention)
                Token.String.Interpol: "#A31515",
                Token.String.Escape: "#EE0000",
                
                # Numbers (Green)
                Token.Number: "#098658",              # Numbers -> Green
                Token.Number.Integer: "#098658",
                Token.Number.Float: "#098658",
                Token.Number.Hex: "#098658",
                
                # Comments (Green)
                Token.Comment: "#008000",             # Comments -> Green
                Token.Comment.Single: "#008000",
                Token.Comment.Multiline: "#008000",
                
                # Operators & Punctuation
                Token.Operator: "#000000",            # Operators -> Black
                Token.Punctuation: "#000000",         # Punctuation -> Black
                
                # Others
                Token.Error: "#FF0000",
                Token.Generic.Deleted: "#A31515",
                Token.Generic.Inserted: "#008000",
            }
        else:
            # Default theme (similar to VS Code Classic Light)
            color_map = {
                Token.Keyword: "#0000FF",
                Token.Keyword.Constant: "#0000FF",
                Token.Keyword.Declaration: "#0000FF",
                Token.Keyword.Namespace: "#0000FF",
                Token.Keyword.Pseudo: "#0000FF",
                Token.Keyword.Reserved: "#0000FF",
                Token.Keyword.Type: "#0000FF",
                Token.Name: "#000000",
                Token.Name.Builtin: "#795E26",
                Token.Name.Class: "#267F99",
                Token.Name.Function: "#795E26",
                Token.Name.Namespace: "#000000",
                Token.String: "#A31515",
                Token.String.Doc: "#008000",
                Token.String.Escape: "#A31515",
                Token.String.Interpol: "#A31515",
                Token.String.Other: "#A31515",
                Token.String.Regex: "#811F3F",
                Token.String.Symbol: "#A31515",
                Token.Number: "#098658",
                Token.Number.Bin: "#098658",
                Token.Number.Float: "#098658",
                Token.Number.Hex: "#098658",
                Token.Number.Integer: "#098658",
                Token.Number.Long: "#098658",
                Token.Number.Oct: "#098658",
                Token.Comment: "#008000",
                Token.Comment.Hashbang: "#008000",
                Token.Comment.Multiline: "#008000",
                Token.Comment.Single: "#008000",
                Token.Comment.Special: "#008000",
                Token.Operator: "#000000",
                Token.Operator.Word: "#0000FF",
                Token.Punctuation: "#000000",
                Token.Error: "#FF0000",
                Token.Generic: "#000000",
                Token.Generic.Deleted: "#A31515",
                Token.Generic.Emph: "#000000",
                Token.Generic.Error: "#FF0000",
                Token.Generic.Heading: "#000000",
                Token.Generic.Inserted: "#008000",
                Token.Generic.Output: "#000000",
                Token.Generic.Prompt: "#000000",
                Token.Generic.Strong: "#000000",
                Token.Generic.Subheading: "#000000",
                Token.Generic.Traceback: "#000000",
                Token.Other: "#000000",
                Token.Text: "#000000",
                Token.Text.Whitespace: "#000000",
            }

        # Parse code
        tokens = list(lex(code, lexer))
        result = []

        for token_type, text in tokens:
            # Get color
            color = "#000000"  # Default black
            for token_class, mapped_color in color_map.items():
                if token_type in token_class:
                    color = mapped_color
                    break

            result.append((text, color))

        return result

    except Exception as e:
        # If parsing fails, return monochrome text
        print(f"Warning: Syntax highlighting parsing failed: {e}, using monochrome mode")
        return [(code, "#000000")]


def text_to_image(
    text: str,
    width: int = 800,
    height: int = 1200,
    font_size: int = 10,
    line_height: float = 1.2,
    margin_px: int = 10,
    dpi: int = 300,
    font_path: str = None,
    bg_color: str = "white",
    text_color: str = "black",
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    filename: str = None,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
    theme: str = "light",
) -> List[PIL_Image.Image]:
    """
    Render text as compact images (using PIL direct rendering, fast and precise control).
    Supports automatic pagination, generates multiple images when content exceeds one image.

    Args:
        text: Text to render
        width: Image width in pixels
        height: Image height in pixels
        font_size: Font size in pixels
        line_height: Line height multiplier (e.g., 1.2 means 1.2x font size)
        margin_px: Margin in pixels
        dpi: DPI setting (for metadata, doesn't affect actual pixel dimensions)
        font_path: Font path (optional)
        bg_color: Background color
        text_color: Text color
        preserve_newlines: Whether to preserve newlines (True=normal mode, False=compact mode)
        enable_syntax_highlight: Whether to enable syntax highlighting (requires pygments)
        filename: Filename (for automatic language detection, used only when syntax highlighting is enabled)
        language: Language name (e.g., 'python', 'javascript'), used only when syntax highlighting is enabled
        should_crop_whitespace: Whether to crop whitespace (True=crop, False=keep original size)
        enable_two_column: Whether to enable two-column layout (True=switch to right column when left is full, False=single column)
        enable_bold: Whether to bold text (True=all text bold, False=normal)
        theme: Syntax highlighting theme ('light' or 'modern')

    Returns:
        List of PIL Image objects (may contain multiple images)
    """
    # Get font (for measurement)
    temp_img = PIL_Image.new("RGB", (width, height), color=bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    font = get_font(font_size, font_path)

    # Calculate actual available area
    text_area_width = width - 2 * margin_px
    text_area_height = height - 2 * margin_px

    # Calculate line height in pixels
    line_height_px = int(font_size * line_height)

    # Calculate lines per page
    max_lines_per_page = (
        int(text_area_height / line_height_px) if line_height_px > 0 else 1
    )

    # If syntax highlighting is enabled, parse code to get colored token list
    if enable_syntax_highlight and PYGMENTS_AVAILABLE:
        # Use original text for syntax highlighting (before prepare_text_for_rendering)
        colored_tokens = parse_code_with_syntax_highlighting(
            text, filename=filename, language=language, theme=theme
        )

        # Prepare text (handle tabs and special characters)
        processed_tokens = []
        for token_text, token_color in colored_tokens:
            # Handle tabs
            processed_token_text = token_text.replace("\t", "    ")
            # Handle special characters
            typographic_replacements = {
                "'": "'",
                "'": "'",
                '"': '"',
                '"': '"',
                "–": "-",
                "—": "--",
                "…": "...",
            }
            for original, replacement in typographic_replacements.items():
                processed_token_text = processed_token_text.replace(
                    original, replacement
                )

            # Handle newlines
            if preserve_newlines:
                # Preserve newlines
                processed_tokens.append((processed_token_text, token_color))
            else:
                # Replace newlines with visible marker
                processed_tokens.append(
                    (processed_token_text.replace("\n", NEWLINE_MARKER), token_color)
                )

        # Render using colored tokens
        pages = []
        current_page_lines = 0
        current_x = margin_px
        current_y = margin_px
        current_column = 0  # 0=left column, 1=right column
        max_column_width = 0  # Maximum width of current column
        column_start_x = margin_px  # Starting x position of current column
        # Determine column width based on enable_two_column
        if enable_two_column:
            column_width = (width - 2 * margin_px) // 2  # Available width per column (minus column gap)
            column_gap = 10  # Gap between two columns
        else:
            column_width = width - 2 * margin_px  # Single column mode, use entire width
            column_gap = 0

        # Create first page
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        for token_text, token_color in processed_tokens:
            try:
                from PIL import ImageColor

                rgb_color = ImageColor.getrgb(token_color)
            except:
                rgb_color = ImageColor.getrgb("#000000")  # Default black

            for char in token_text:
                if preserve_newlines and char == "\n":
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1

                    if current_page_lines >= max_lines_per_page:
                        if (
                            enable_two_column
                            and current_column == 0
                            and max_column_width <= (width / 2)
                        ):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(
                                    img, bg_color, keep_margin=(margin_px, margin_px)
                                )
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
                            max_column_width = 0
                    continue

                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]

                if enable_two_column:
                    column_right_bound = (
                        column_start_x + column_width
                        if current_column == 0
                        else width - margin_px
                    )
                else:
                    column_right_bound = width - margin_px
                if (
                    current_x + char_w > column_right_bound
                    and current_x > column_start_x
                ):
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1

                    if current_page_lines >= max_lines_per_page:
                        if (
                            enable_two_column
                            and current_column == 0
                            and max_column_width <= (width / 2)
                        ):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(
                                    img, bg_color, keep_margin=(margin_px, margin_px)
                                )
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
                            max_column_width = 0

                if enable_bold:
                    for dx, dy in [(0, 0), (1, 0)]:
                        draw.text((current_x + dx, current_y + dy), char, font=font, fill=rgb_color)
                else:
                    draw.text((current_x, current_y), char, font=font, fill=rgb_color)
                current_x += char_w
                max_column_width = max(max_column_width, current_x - column_start_x)

        if current_page_lines > 0 or current_x > margin_px:
            if should_crop_whitespace:
                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
            if dpi:
                img.info["dpi"] = (dpi, dpi)
            pages.append(img)

        # Return pages
        result_pages = pages if pages else [img]
        return result_pages

    else:
        processed_text = prepare_text_for_rendering(
            text, preserve_newlines=preserve_newlines
        )
        lines = []

        if preserve_newlines:
            original_lines = processed_text.split("\n")
            for original_line in original_lines:
                current_line = ""
                current_line_width = 0

                for char in original_line:
                    try:
                        char_w = temp_draw.textlength(char, font=font)
                    except:
                        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                        char_w = char_bbox[2] - char_bbox[0]

                    if current_line_width + char_w > text_area_width and current_line:
                        lines.append(current_line)
                        current_line = char
                        current_line_width = char_w
                    else:
                        current_line += char
                        current_line_width += char_w

                if current_line:
                    lines.append(current_line)
        else:
            current_line = ""
            current_line_width = 0

            for char in processed_text:
                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]

                if current_line_width + char_w > text_area_width and current_line:
                    lines.append(current_line)
                    current_line = char
                    current_line_width = char_w
                else:
                    current_line += char
                    current_line_width += char_w

            if current_line:
                lines.append(current_line)

        total_lines = len(lines)
        pages = []

        if enable_two_column:
            column_width = (width - 2 * margin_px) // 2
            column_gap = 10
            left_column_start = margin_px
            right_column_start = width // 2 + column_gap // 2
            left_column_right_bound = left_column_start + column_width
            right_column_right_bound = width - margin_px
        else:
            column_width = width - 2 * margin_px
            column_gap = 0
            left_column_start = margin_px
            right_column_start = margin_px
            left_column_right_bound = width - margin_px
            right_column_right_bound = width - margin_px

        page_start = 0
        while page_start < total_lines:
            img = PIL_Image.new("RGB", (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)

            x = left_column_start
            y = margin_px
            current_page_lines = 0
            max_left_column_width = 0

            while page_start < total_lines and current_page_lines < max_lines_per_page:
                line = lines[page_start]

                current_line_x = x
                line_chars = list(line)
                line_drawn = False

                for char in line_chars:
                    try:
                        char_w = temp_draw.textlength(char, font=font)
                    except:
                        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                        char_w = char_bbox[2] - char_bbox[0]

                    if (
                        current_line_x + char_w > left_column_right_bound
                        and current_line_x > left_column_start
                    ):
                        y += line_height_px
                        current_line_x = left_column_start
                        current_page_lines += 1
                        if current_page_lines >= max_lines_per_page:
                            break

                    if enable_bold:
                        for dx, dy in [(0, 0), (1, 0)]:
                            draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                    else:
                        draw.text((current_line_x, y), char, font=font, fill=text_color)
                    current_line_x += char_w
                    max_left_column_width = max(
                        max_left_column_width, current_line_x - left_column_start
                    )
                    line_drawn = True

                if line_drawn:
                    y += line_height_px
                    x = left_column_start
                    current_page_lines += 1
                    page_start += 1
                else:
                    page_start += 1
                    break

            if (
                enable_two_column
                and current_page_lines >= max_lines_per_page
                and max_left_column_width <= (width / 2)
            ):
                x = right_column_start
                y = margin_px
                current_page_lines = 0

                while (
                    page_start < total_lines and current_page_lines < max_lines_per_page
                ):
                    line = lines[page_start]

                    current_line_x = x
                    line_chars = list(line)
                    line_drawn = False

                    for char in line_chars:
                        try:
                            char_w = temp_draw.textlength(char, font=font)
                        except:
                            char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                            char_w = char_bbox[2] - char_bbox[0]

                        if (
                            current_line_x + char_w > right_column_right_bound
                            and current_line_x > right_column_start
                        ):
                            y += line_height_px
                            current_line_x = right_column_start
                            current_page_lines += 1
                            if current_page_lines >= max_lines_per_page:
                                break

                        if enable_bold:
                            for dx, dy in [(0, 0), (1, 0)]:
                                draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                        else:
                            draw.text((current_line_x, y), char, font=font, fill=text_color)
                        current_line_x += char_w
                        line_drawn = True

                    if line_drawn:
                        y += line_height_px
                        x = right_column_start
                        current_page_lines += 1
                        page_start += 1
                    else:
                        page_start += 1
                        break

            if should_crop_whitespace:
                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))

            if dpi:
                img.info["dpi"] = (dpi, dpi)

            pages.append(img)

        return pages


def text_to_image_stream(
    text: str,
    width: int = 800,
    height: int = 1200,
    font_size: int = 10,
    line_height: float = 1.2,
    margin_px: int = 10,
    dpi: int = 300,
    font_path: str = None,
    bg_color: str = "white",
    text_color: str = "black",
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    filename: str = None,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
    theme: str = "light",
):
    temp_img = PIL_Image.new("RGB", (width, height), color=bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    font = get_font(font_size, font_path)
    text_area_width = width - 2 * margin_px
    text_area_height = height - 2 * margin_px
    line_height_px = int(font_size * line_height)
    max_lines_per_page = int(text_area_height / line_height_px) if line_height_px > 0 else 1
    print(f"  Stream render start: {width}x{height}, font {font_size}, lh {line_height}, two_col {enable_two_column}, bold {enable_bold}, hl {enable_syntax_highlight}, nl {preserve_newlines}")
    import sys
    sys.stdout.flush()
    if enable_syntax_highlight and PYGMENTS_AVAILABLE:
        colored_tokens = parse_code_with_syntax_highlighting(
            text, filename=filename, language=language, theme=theme
        )
        processed_tokens = []
        for token_text, token_color in colored_tokens:
            processed_token_text = token_text.replace("\t", "    ")
            typographic_replacements = {
                "'": "'",
                "'": "'",
                '"': '"',
                '"': '"',
                "–": "-",
                "—": "--",
                "…": "...",
            }
            for original, replacement in typographic_replacements.items():
                processed_token_text = processed_token_text.replace(original, replacement)
            if preserve_newlines:
                processed_tokens.append((processed_token_text, token_color))
            else:
                processed_tokens.append((processed_token_text.replace("\n", NEWLINE_MARKER), token_color))
        current_page_lines = 0
        current_x = margin_px
        current_y = margin_px
        current_column = 0
        max_column_width = 0
        column_start_x = margin_px
        if enable_two_column:
            column_width = (width - 2 * margin_px) // 2
            column_gap = 10
        else:
            column_width = width - 2 * margin_px
            column_gap = 0
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        page_num = 1
        for token_text, token_color in processed_tokens:
            try:
                from PIL import ImageColor
                rgb_color = ImageColor.getrgb(token_color)
            except:
                from PIL import ImageColor
                rgb_color = ImageColor.getrgb("#000000")
            for char in token_text:
                if preserve_newlines and char == "\n":
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1
                    if current_page_lines >= max_lines_per_page:
                        if enable_two_column and current_column == 0 and max_column_width <= (width / 2):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
                            if dpi:
                                img.info["dpi"] = (dpi, dpi)
                            print(f"  Stream page {page_num}: {img.width}x{img.height}, tokens {calculate_image_tokens_qwen3(img.width, img.height)}")
                            sys.stdout.flush()
                            yield img
                            page_num += 1
                            img = PIL_Image.new("RGB", (width, height), color=bg_color)
                            draw = ImageDraw.Draw(img)
                            current_page_lines = 0
                            current_y = margin_px
                            current_column = 0
                            column_start_x = margin_px
                            current_x = margin_px
                            max_column_width = 0
                    continue
                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]
                if enable_two_column:
                    column_right_bound = column_start_x + column_width if current_column == 0 else width - margin_px
                else:
                    column_right_bound = width - margin_px
                if current_x + char_w > column_right_bound and current_x > column_start_x:
                    max_column_width = max(max_column_width, current_x - column_start_x)
                    current_y += line_height_px
                    current_x = column_start_x
                    current_page_lines += 1
                    if current_page_lines >= max_lines_per_page:
                        if enable_two_column and current_column == 0 and max_column_width <= (width / 2):
                            current_column = 1
                            column_start_x = width // 2 + column_gap // 2
                            current_x = column_start_x
                            current_y = margin_px
                            current_page_lines = 0
                            max_column_width = 0
                        else:
                            if should_crop_whitespace:
                                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
                            if dpi:
                                img.info["dpi"] = (dpi, dpi)
                            print(f"  Stream page {page_num}: {img.width}x{img.height}, tokens {calculate_image_tokens_qwen3(img.width, img.height)}")
                            sys.stdout.flush()
                            yield img
                            page_num += 1
                            img = PIL_Image.new("RGB", (width, height), color=bg_color)
                            draw = ImageDraw.Draw(img)
                            current_page_lines = 0
                            current_y = margin_px
                            current_column = 0
                            column_start_x = margin_px
                            current_x = margin_px
                            max_column_width = 0
                if enable_bold:
                    for dx, dy in [(0, 0), (1, 0)]:
                        draw.text((current_x + dx, current_y + dy), char, font=font, fill=rgb_color)
                else:
                    draw.text((current_x, current_y), char, font=font, fill=rgb_color)
                current_x += char_w
                max_column_width = max(max_column_width, current_x - column_start_x)
        if current_page_lines > 0 or current_x > margin_px:
            if should_crop_whitespace:
                img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
            if dpi:
                img.info["dpi"] = (dpi, dpi)
            print(f"  Stream page {page_num}: {img.width}x{img.height}, tokens {calculate_image_tokens_qwen3(img.width, img.height)}")
            sys.stdout.flush()
            yield img
            return
        else:
            print(f"  Stream page {page_num}: {img.width}x{img.height}, tokens {calculate_image_tokens_qwen3(img.width, img.height)}")
            sys.stdout.flush()
            yield img
        return
    processed_text = prepare_text_for_rendering(text, preserve_newlines=preserve_newlines)
    lines = []
    if preserve_newlines:
        original_lines = processed_text.split("\n")
        for original_line in original_lines:
            current_line = ""
            current_line_width = 0
            for char in original_line:
                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]
                if current_line_width + char_w > text_area_width and current_line:
                    lines.append(current_line)
                    current_line = char
                    current_line_width = char_w
                else:
                    current_line += char
                    current_line_width += char_w
            if current_line:
                lines.append(current_line)
    else:
        current_line = ""
        current_line_width = 0
        for char in processed_text:
            try:
                char_w = temp_draw.textlength(char, font=font)
            except:
                char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                char_w = char_bbox[2] - char_bbox[0]
            if current_line_width + char_w > text_area_width and current_line:
                lines.append(current_line)
                current_line = char
                current_line_width = char_w
            else:
                current_line += char
                current_line_width += char_w
        if current_line:
            lines.append(current_line)
    total_lines = len(lines)
    if enable_two_column:
        column_width = (width - 2 * margin_px) // 2
        column_gap = 10
        left_column_start = margin_px
        right_column_start = width // 2 + column_gap // 2
        left_column_right_bound = left_column_start + column_width
        right_column_right_bound = width - margin_px
    else:
        column_width = width - 2 * margin_px
        column_gap = 0
        left_column_start = margin_px
        right_column_start = margin_px
        left_column_right_bound = width - margin_px
        right_column_right_bound = width - margin_px
    page_start = 0
    page_num = 1
    while page_start < total_lines:
        img = PIL_Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        x = left_column_start
        y = margin_px
        current_page_lines = 0
        max_left_column_width = 0
        while page_start < total_lines and current_page_lines < max_lines_per_page:
            line = lines[page_start]
            current_line_x = x
            line_chars = list(line)
            line_drawn = False
            for char in line_chars:
                try:
                    char_w = temp_draw.textlength(char, font=font)
                except:
                    char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                    char_w = char_bbox[2] - char_bbox[0]
                if current_line_x + char_w > left_column_right_bound and current_line_x > left_column_start:
                    y += line_height_px
                    current_line_x = left_column_start
                    current_page_lines += 1
                    if current_page_lines >= max_lines_per_page:
                        break
                if enable_bold:
                    for dx, dy in [(0, 0), (1, 0)]:
                        draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                else:
                    draw.text((current_line_x, y), char, font=font, fill=text_color)
                current_line_x += char_w
                max_left_column_width = max(max_left_column_width, current_line_x - left_column_start)
                line_drawn = True
            if line_drawn:
                y += line_height_px
                x = left_column_start
                current_page_lines += 1
                page_start += 1
            else:
                page_start += 1
                break
        if enable_two_column and current_page_lines >= max_lines_per_page and max_left_column_width <= (width / 2):
            x = right_column_start
            y = margin_px
            current_page_lines = 0
            while page_start < total_lines and current_page_lines < max_lines_per_page:
                line = lines[page_start]
                current_line_x = x
                line_chars = list(line)
                line_drawn = False
                for char in line_chars:
                    try:
                        char_w = temp_draw.textlength(char, font=font)
                    except:
                        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                        char_w = char_bbox[2] - char_bbox[0]
                    if current_line_x + char_w > right_column_right_bound and current_line_x > right_column_start:
                        y += line_height_px
                        current_line_x = right_column_start
                        current_page_lines += 1
                        if current_page_lines >= max_lines_per_page:
                            break
                    if enable_bold:
                        for dx, dy in [(0, 0), (1, 0)]:
                            draw.text((current_line_x + dx, y + dy), char, font=font, fill=text_color)
                    else:
                        draw.text((current_line_x, y), char, font=font, fill=text_color)
                    current_line_x += char_w
                    line_drawn = True
                if line_drawn:
                    y += line_height_px
                    x = right_column_start
                    current_page_lines += 1
                    page_start += 1
                else:
                    page_start += 1
                    break
        if should_crop_whitespace:
            img = crop_whitespace(img, bg_color, keep_margin=(margin_px, margin_px))
        if dpi:
            img.info["dpi"] = (dpi, dpi)
        print(f"  Stream page {page_num}: {img.width}x{img.height}, tokens {calculate_image_tokens_qwen3(img.width, img.height)}")
        import sys
        sys.stdout.flush()
        yield img
        page_num += 1


def optimize_layout_config_dry(
    target_tokens: float,
    previous_configs: List[Tuple[int, int]] = None,
    text_tokens: int = None,
    line_height: float = 1.0,
    text_structure: dict = None,
    compression_ratio: float = None,
    page_limit: int = 100,
    # New parameters for refinement
    text: str = None,
    enable_syntax_highlight: bool = False,
    language: str = None,
    preserve_newlines: bool = False,
    font_path: str = None,
    theme: str = "light",
):
    if previous_configs is None:
        previous_configs = []
    
    # Estimate characters if not provided
    if text_tokens:
        estimated_chars = text_tokens * 3.5 
    else:
        estimated_chars = 10000

    resolutions = get_expanded_resolution_list()
    
    # Dynamic tolerance range for target tokens
    if target_tokens < 50:
        token_min_ratio = 0.5
        token_max_ratio = 2.0
    elif target_tokens < 100:
        token_min_ratio = 0.7
        token_max_ratio = 1.5
    elif target_tokens < 3000:
        token_min_ratio = 0.8
        token_max_ratio = 1.25
    elif target_tokens < 5000:
        token_min_ratio = 0.9
        token_max_ratio = 1.12
    elif target_tokens < 10000:
        token_min_ratio = 0.93
        token_max_ratio = 1.08
    else:
        token_min_ratio = 0.95
        token_max_ratio = 1.05
        
    all_configs = []
    
    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        
        # Calculate valid page counts
        min_pages = math.ceil(target_tokens * token_min_ratio / per_image_tokens)
        max_pages = math.floor(target_tokens * token_max_ratio / per_image_tokens)
        min_pages = max(1, min_pages)
        
        if min_pages > max_pages:
            continue
            
        max_allowed_pages = page_limit
        for pages in range(min_pages, max_pages + 1):
            if pages > max_allowed_pages:
                continue
                
            is_new = (res, pages) not in previous_configs
            
            # Use Area-based estimation for font size and fill rate
            # This is more robust than line-based estimation for dry runs
            # fs = sqrt(Available_Area / (Total_Chars * 0.6))
            margin = res * 0.01
            available_area_per_page = (res - 2 * margin) ** 2
            total_available_area = available_area_per_page * pages
            
            # char_area = (fs * 0.6) * (fs * line_height) = 0.6 * line_height * fs^2
            # fs^2 = total_available_area / (estimated_chars * 0.6 * line_height)
            
            if estimated_chars > 0:
                fs_squared = total_available_area / (estimated_chars * 0.6 * line_height)
                optimal_fs = int(math.sqrt(fs_squared))
            else:
                optimal_fs = 150
                
            # Clamp font size
            optimal_fs = max(4, min(150, optimal_fs))
            
            # Calculate fill rate based on area
            # fill_rate = (estimated_chars * char_area) / total_available_area
            char_area = (optimal_fs * 0.6) * (optimal_fs * line_height)
            text_area = estimated_chars * char_area
            fill_rate = text_area / total_available_area if total_available_area > 0 else 0
            
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
                "per_image_tokens": per_image_tokens,
            })
            
    if not all_configs:
        res = 112
        imgs_pages = 1
        best_fs = 3
        return res, best_fs, imgs_pages

    def _score(c):
        # 1. Token match score (0-1)
        token_diff = abs(c["token_ratio"] - 1.0)
        if target_tokens < 50:
            token_penalty_factor = 1.0
        elif target_tokens < 100:
            token_penalty_factor = 1.5
        elif target_tokens < 3000:
            token_penalty_factor = 2.0
        elif target_tokens < 5000:
            token_penalty_factor = 2.5
        elif target_tokens < 10000:
            token_penalty_factor = 3.0
        else:
            token_penalty_factor = 3.5
            
        token_score = 1.0 / (1.0 + token_diff * token_penalty_factor)
        
        # 2. Fill rate score
        # Mainly used to filter out invalid configs (fs too small or too big)
        fill_rate = c["fill_rate"]
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            if fill_rate >= 0.2:
                fill_score = 0.9
            else:
                fill_score = 0.5
        else:
            if fill_rate >= 0.8: # Relaxed from 0.9
                fill_score = 1.0
            elif fill_rate >= 0.2:
                fill_score = 0.2 + (fill_rate - 0.2) * (0.8 / 0.6)
            elif fill_rate >= 0.1:
                fill_score = 0.05
            else:
                fill_score = 0.01

        # 3. Resolution bonus - ENHANCED
        # User wants to prioritize larger resolution (and fewer pages) when token scores are similar
        resolution_normalized = (c["resolution"] - 112) / (4480 - 112)
        
        # Base bonus 1.0 to 2.0
        resolution_bonus = 1.0 + resolution_normalized * 1.0
        
        # 4. Compression bonus
        if compression_ratio is not None and compression_ratio <= 2.0:
            compression_bonus = 1.2
        else:
            compression_bonus = 1.0

        # Calculate base score
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            score = (token_score ** 2.5) * (fill_score ** 0.2) * resolution_bonus * compression_bonus
        elif target_tokens < 100:
            score = (token_score ** 1.5) * (fill_score ** 0.75) * resolution_bonus * compression_bonus
        else:
            score = token_score * fill_score * resolution_bonus * compression_bonus
            if target_tokens >= 5000 and c.get("font_size", 10) < 8:
                score *= 0.8
        
        # 5. Page penalty (Implicitly handled by resolution bonus? No, let's be explicit)
        # Prefer fewer pages: Penalty for more pages
        # score = score / sqrt(pages)
        score = score / (c["pages"] ** 0.3)

        if target_tokens < 1000:
            if c["pages"] > 2:
                score *= 0.01
                
        if not c["is_new"]:
            score *= 0.95
            
        return score

    for c in all_configs:
        c["score"] = _score(c)
        
    all_configs.sort(key=lambda x: -x["score"])
    
    best = all_configs[0]
    
    if text:
        refined_fs = calculate_max_font_size_at_resolution(
            text,
            best["resolution"],
            best["pages"],
            line_height=line_height,
            font_path=font_path,
            enable_syntax_highlight=enable_syntax_highlight,
            preserve_newlines=preserve_newlines,
            language=language,
            theme=theme
        )
        chosen_fs = refined_fs
        attempts = 0
        while best["resolution"] >= 900 and chosen_fs < 10 and attempts < 3:
            fallback_candidates = [c for c in all_configs if c["pages"] > best["pages"]]
            if not fallback_candidates:
                break
            fallback_candidates.sort(key=lambda x: -x["score"])
            best = fallback_candidates[0]
            chosen_fs = calculate_max_font_size_at_resolution(
                text,
                best["resolution"],
                best["pages"],
                line_height=line_height,
                font_path=font_path,
                enable_syntax_highlight=enable_syntax_highlight,
                preserve_newlines=preserve_newlines,
                language=language,
                theme=theme
            )
            attempts += 1
        return best["resolution"], chosen_fs, best["pages"]

    chosen_fs = best["font_size"]
    attempts = 0
    while best["resolution"] >= 900 and chosen_fs < 10 and attempts < 3:
        fallback_candidates = [c for c in all_configs if c["pages"] > best["pages"]]
        if not fallback_candidates:
            break
        fallback_candidates.sort(key=lambda x: -x["score"])
        best = fallback_candidates[0]
        chosen_fs = best["font_size"]
        attempts += 1
    return best["resolution"], chosen_fs, best["pages"]

def generate_images_for_file(
    filename: str,
    source_code: str,
    base_output_dir: str,
    width: int = 800,
    height: int = 1200,
    font_size: int = 10,
    line_height: float = 1.2,
    dpi: int = 300,
    font_path: str = None,
    unique_id: str = None,
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    enable_bold: bool = False,
) -> List[str]:
    """
    Generate compact images for specified file.

    Args:
        filename: Original filename, e.g., 'src/black/__init__.py'
        source_code: Source code content
        base_output_dir: Base output directory (resolution folder will be created inside)
        width: Image width
        height: Image height
        font_size: Font size
        line_height: Line height
        dpi: DPI setting
        unique_id: Unique identifier (for file naming)
        preserve_newlines: Whether to preserve newlines
        enable_syntax_highlight: Whether to enable syntax highlighting
        language: Language name (e.g., 'python', 'javascript'), auto-detected from filename if None
        should_crop_whitespace: Whether to crop whitespace (True=crop, False=keep original size)
        enable_two_column: Whether to enable two-column layout (True=switch to right when left is full, False=single column)

    Returns:
        List of image file paths
    """
    resolution_parts = [f"{width}x{height}"]
    if enable_syntax_highlight:
        resolution_parts.append("hl")
    if preserve_newlines:
        resolution_parts.append("nl")
    resolution_folder_name = "_".join(resolution_parts)
    resolution_dir = os.path.join(base_output_dir, resolution_folder_name)
    os.makedirs(resolution_dir, exist_ok=True)

    if unique_id is None:
        unique_id = filename.replace("/", "_")

    image_paths = []

    try:
        images = text_to_image(
            source_code,
            width=width,
            height=height,
            font_size=font_size,
            line_height=line_height,
            dpi=dpi,
            font_path=font_path,
            preserve_newlines=preserve_newlines,
            enable_syntax_highlight=enable_syntax_highlight,
            filename=filename,
            language=language,
            should_crop_whitespace=should_crop_whitespace,
            enable_two_column=enable_two_column,
            enable_bold=enable_bold
        )

        if images:
            for page_num, image in enumerate(images, 1):
                image_filename = f"page_{page_num:03d}.png"
                image_path = os.path.join(resolution_dir, image_filename)
                image.save(image_path)
                image_paths.append(os.path.abspath(image_path))
                print(
                    f"  Generated image: {image_filename} ({width}x{height}, font={font_size}px, line-height={line_height})"
                )

            print(f"  Total {len(images)} images generated")
        else:
            raise RuntimeError("Unable to generate images")

    except Exception as e:
        print(f"  Error: Image generation failed: {e}")
        raise

    return image_paths


def calculate_image_tokens_qwen3(width: int, height: int) -> int:
    """
    Calculate image token count using Qwen3 method.
    Formula: (height/16 * width/16)/4, with minimum of 1 token

    Args:
        width: Image width
        height: Image height

    Returns:
        Estimated token count (minimum 1)
    """
    tokens = (width / 16 * height / 16) / 4
    return max(1, int(tokens))  # Ensure at least 1 token to avoid 0 for small images (e.g., 14x14, 28x28)


def find_closest_resolution_prefer_larger(
    target_tokens: int, resolution_list: List[int], tolerance_ratio: float = 1.4
) -> int:
    """
    Find closest resolution to target token count, preferring larger resolutions.

    If multiple resolutions have token counts within reasonable range 
    (max_tokens <= min_tokens * tolerance_ratio), choose the largest resolution.
    
    Args:
        target_tokens: Target token count
        resolution_list: List of available resolutions
        tolerance_ratio: Tolerance ratio for considering resolutions equivalent
        
    Returns:
        Selected resolution
    """
    candidates = []
    for resolution in resolution_list:
        tokens = calculate_image_tokens_qwen3(resolution, resolution)
        diff = abs(tokens - target_tokens)
        candidates.append((resolution, tokens, diff))

    candidates.sort(key=lambda x: x[2])

    if not candidates:
        return resolution_list[0]

    min_diff = candidates[0][2]
    threshold_diff = min_diff * 1.2  # Allow difference within 1.2x of minimum difference

    close_candidates = [c for c in candidates if c[2] <= threshold_diff]

    if len(close_candidates) > 1:
        min_tokens = min(c[1] for c in close_candidates)
        max_tokens = max(c[1] for c in close_candidates)

        if max_tokens <= min_tokens * tolerance_ratio:
            return max(c[0] for c in close_candidates)

    return candidates[0][0]


def resize_images_for_compression(
    images: List[PIL_Image.Image],
    text_tokens: int,
    compression_ratios: List[float] = None,
    renderer_func: Callable[[int, int, int], List[PIL_Image.Image]] = None,
    text_structure: dict = None,
    data_id: str = None,
) -> Dict[float, Tuple[List[PIL_Image.Image], int]]:
    """
    Resize images to target resolution based on compression ratio.
    
    New approach:
    - For ratio=1.0: Dynamically optimize configuration (resolution, pages, font_size) using optimize_layout_config
    - For other ratios: Resize from 1x base images
    
    Args:
        images: List of original images (PIL Image objects, only used if renderer_func is None)
        text_tokens: Text token count
        compression_ratios: List of compression ratios, uses global COMPRESSION_RATIOS if None
        renderer_func: Optional renderer function for dynamic optimization (width, height, font_size) -> images
        text_structure: Text structure info for optimization {'num_lines', 'max_line_chars', 'avg_line_chars'}

    Returns:
        Dictionary where key is compression ratio and value is (resized_images, target_resolution) tuple
    """
    if compression_ratios is None:
        compression_ratios = COMPRESSION_RATIOS

    id_prefix = f"[{data_id}] " if data_id else ""
    print(f"  {id_prefix}Starting image compression/resize for {len(compression_ratios)} ratios...")
    import sys
    sys.stdout.flush()

    resolution_list = get_expanded_resolution_list()
    results = {}
    sorted_ratios = sorted(compression_ratios)
    
    # Use provided images as the 1x base
    base_images = images
    base_resolution = images[0].width if images else 0
    
    # Process all ratios
    for ratio in sorted_ratios:
        if float(ratio) == 0.0:
            os.makedirs("./generated_images", exist_ok=True)
            blank_path = os.path.join("./generated_images", "blank_14x14.png")
            if os.path.exists(blank_path):
                try:
                    blank_img = PIL_Image.open(blank_path).convert("RGB")
                except Exception:
                    blank_img = PIL_Image.new("RGB", (14, 14), color="white")
                    blank_img.save(blank_path)
            else:
                blank_img = PIL_Image.new("RGB", (14, 14), color="white")
                blank_img.save(blank_path)
            results[ratio] = ([blank_img], 14)
            actual_tokens = len(results[ratio][0]) * calculate_image_tokens_qwen3(14, 14)
            print(f"  {id_prefix}Compression ratio 0: Using blank_14x14.png, 1 image, tokens {actual_tokens} (target: 0.0)")
            sys.stdout.flush()
            continue
        image_token_limit = text_tokens / ratio
        
        if ratio == 1.0:
            # For 1.0, just use the base images as is
            # We assume base_images are already optimized for 1.0
            results[1.0] = (base_images, base_resolution)
            
            actual_tokens = len(base_images) * calculate_image_tokens_qwen3(base_resolution, base_resolution)
            print(f"  {id_prefix}Compression ratio 1.0: Using base images {base_resolution}x{base_resolution}, {len(base_images)} images, tokens {actual_tokens} (target: {image_token_limit:.1f})")
            sys.stdout.flush()
        else:
            # For other ratios, resize from base images
            num_images = len(base_images)
            per_image_tokens = image_token_limit / num_images if num_images > 0 else image_token_limit
            
            # Find closest resolution from expanded list
            target_resolution = find_closest_resolution_prefer_larger(
                per_image_tokens, resolution_list, tolerance_ratio=1.4
            )
            
            # Resize all base images
            resized_images = []
            for img in base_images:
                resized_img = img.resize(
                    (target_resolution, target_resolution), PIL_Image.Resampling.LANCZOS
                )
                resized_images.append(resized_img)
            
            results[ratio] = (resized_images, target_resolution)
            actual_tokens = len(resized_images) * calculate_image_tokens_qwen3(target_resolution, target_resolution)
            print(f"  {id_prefix}Compression ratio {ratio}: Resized from 1x base ({base_resolution}) to {target_resolution}x{target_resolution}, "
                  f"{len(resized_images)} images, tokens {actual_tokens} (target: {image_token_limit:.1f})")
            sys.stdout.flush()

    return results



def estimate_initial_font_size(text_tokens: int, resolution: int, line_height: float = DEFAULT_LINE_HEIGHT) -> int:
    """
    Quickly estimate initial font size based on empirical formula.
    
    Uses monospace font characteristics:
    - Character width ≈ font_size * 0.6
    - Each token ≈ 3-4 characters
    - Margin ≈ 1% of resolution
    
    Args:
        text_tokens: Text token count
        resolution: Resolution (square)
        line_height: Line height multiplier
        
    Returns:
        Estimated font size
    """
    # Calculate available area
    margin = resolution * DEFAULT_MARGIN_RATIO
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    available_area = available_width * available_height
    
    # Estimate total characters
    total_chars = text_tokens * AVG_CHARS_PER_TOKEN
    
    if total_chars <= 0:
        return DEFAULT_FONT_SIZE
    
    # Calculate font size from area constraint
    # Character area ≈ (font_size * CHAR_WIDTH_RATIO) * (font_size * line_height)
    # font_size^2 ≈ available_area / (total_chars * CHAR_WIDTH_RATIO * line_height)
    
    fill_factor = 0.95  # Target 95% utilization
    estimated_fs_squared = (available_area * fill_factor) / (total_chars * CHAR_WIDTH_RATIO * line_height)
    estimated_fs = int(math.sqrt(estimated_fs_squared))
    
    # Add 10% margin for readability
    estimated_fs = int(estimated_fs * 1.1)
    
    # Clamp to valid range
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, estimated_fs))


def estimate_page_count(
    text_tokens: int,
    resolution: int,
    font_size: int,
    line_height: float = DEFAULT_LINE_HEIGHT,
) -> int:
    """
    Quickly estimate required number of pages for given configuration (without rendering).
    
    Used for fast binary search in optimize_layout_config to avoid actual rendering.
    
    Args:
        text_tokens: Text token count
        resolution: Resolution (square)
        font_size: Font size
        line_height: Line height multiplier
        
    Returns:
        Estimated page count (999 if configuration is invalid)
    """
    # Calculate available area
    margin = resolution * DEFAULT_MARGIN_RATIO
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    # Validate inputs
    if font_size <= 0 or available_width <= 0 or available_height <= 0:
        return 999
    
    # Calculate character and line metrics
    char_width = font_size * CHAR_WIDTH_RATIO
    line_height_px = font_size * line_height
    
    chars_per_line = max(1, int(available_width / char_width)) if char_width > 0 else 1
    lines_per_page = max(1, int(available_height / line_height_px)) if line_height_px > 0 else 1
    chars_per_page = chars_per_line * lines_per_page
    
    # Estimate total characters with overhead for newlines
    total_chars = text_tokens * AVG_CHARS_PER_TOKEN
    estimated_newlines = total_chars / 50  # ~1 newline per 50 characters in code
    newline_overhead = 1.3 + (estimated_newlines * chars_per_line * 0.3) / total_chars
    effective_chars = total_chars * newline_overhead
    
    # Calculate pages needed
    return max(1, math.ceil(effective_chars / chars_per_page))


def estimate_fill_rate(
    text_tokens: int,
    resolution: int,
    font_size: int,
    line_height: float = DEFAULT_LINE_HEIGHT,
    avg_line_length: int = 80,
) -> float:
    """
    Estimate fill rate of text in given configuration.
    
    Fill rate = average line width / available image width
    Typical code has 60-80 character lines on average.
    
    Args:
        text_tokens: Text token count (unused, kept for API compatibility)
        resolution: Resolution (square)
        font_size: Font size
        line_height: Line height multiplier (unused, kept for API compatibility)
        avg_line_length: Average line length in code, default 80
        
    Returns:
        Fill rate (0.0 - 1.5), closer to 1.0 is optimal
    """
    # Calculate available width
    margin = resolution * DEFAULT_MARGIN_RATIO
    available_width = resolution - 2 * margin
    
    if font_size <= 0 or available_width <= 0:
        return 0.0
    
    # Calculate characters per line
    char_width = font_size * CHAR_WIDTH_RATIO
    chars_per_line = available_width / char_width if char_width > 0 else 1
    
    # Calculate and clamp fill rate
    fill_rate = avg_line_length / chars_per_line if chars_per_line > 0 else 0
    return max(0.0, min(1.5, fill_rate))


def estimate_fill_rate_for_target_pages(
    text_tokens: int,
    resolution: int,
    target_pages: int,
    line_height: float = 1.0,
    avg_line_length: int = 80,
) -> float:
    """
    Estimate actual fill rate when fitting to target page count.
    
    This function first estimates required font size to produce target_pages,
    then calculates fill rate based on that font.
    
    Args:
        text_tokens: Text token count
        resolution: Resolution (square)
        target_pages: Target page count
        line_height: Line height multiplier
        avg_line_length: Average line length in code
        
    Returns:
        Estimated fill rate
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    if available_width <= 0 or available_height <= 0 or target_pages <= 0:
        return 0.0
    
    # Estimate total characters
    total_chars = text_tokens * 3.5
    
    # Characters needed per page
    chars_per_page = total_chars / target_pages
    
    # Page area
    page_area = available_width * available_height
    
    # Character area = char_width * line_height_px = (fs * 0.6) * (fs * line_height)
    # chars_per_page = page_area / char_area
    # char_area = page_area / chars_per_page
    # (fs * 0.6) * (fs * line_height) = page_area / chars_per_page
    # fs^2 = page_area / (chars_per_page * 0.6 * line_height)
    
    if chars_per_page <= 0:
        return 1.5  # Very few characters, definitely can fill
    
    fs_squared = page_area / (chars_per_page * 0.6 * line_height)
    estimated_fs = math.sqrt(fs_squared) if fs_squared > 0 else 4
    estimated_fs = max(4, min(150, estimated_fs))
    
    # Calculate fill rate based on estimated font
    char_width = estimated_fs * 0.6
    chars_per_line = available_width / char_width if char_width > 0 else 1
    fill_rate = avg_line_length / chars_per_line if chars_per_line > 0 else 0
    
    return max(0.0, min(1.5, fill_rate))


def is_token_in_range(
    estimated_pages: int,
    per_image_tokens: int,
    target_tokens: float,
    min_ratio: float = 0.9,
    max_ratio: float = 1.1,
) -> bool:
    """
    Check if actual token count is within target range.
    
    Args:
        estimated_pages: Estimated page count
        per_image_tokens: Tokens per image
        target_tokens: Target token count
        min_ratio: Minimum ratio (default 0.9)
        max_ratio: Maximum ratio (default 1.1)
        
    Returns:
        Whether within range
    """
    if target_tokens <= 0:
        return False
    
    actual_tokens = estimated_pages * per_image_tokens
    ratio = actual_tokens / target_tokens
    
    return min_ratio <= ratio <= max_ratio


def get_token_ratio(
    estimated_pages: int,
    per_image_tokens: int,
    target_tokens: float,
) -> float:
    """Calculate token ratio."""
    if target_tokens <= 0:
        return 999.0
    actual_tokens = estimated_pages * per_image_tokens
    return actual_tokens / target_tokens


def analyze_text_structure(text: str) -> dict:
    """
    Analyze text structure and extract line count, longest line, average line length, etc.
    
    Args:
        text: Original text
        
    Returns:
        dict: {
            'num_lines': Total line count,
            'max_line_chars': Maximum line character count,
            'avg_line_chars': Average line character count,
            'total_chars': Total character count,
        }
    """
    # Handle tabs
    text = text.replace('\t', '    ')
    
    lines = text.split('\n')
    num_lines = len(lines)
    
    if num_lines == 0:
        return {
            'num_lines': 1,
            'max_line_chars': 1,
            'avg_line_chars': 1,
            'total_chars': 1,
        }
    
    line_lengths = [len(line) for line in lines]
    max_line_chars = max(line_lengths) if line_lengths else 1
    avg_line_chars = sum(line_lengths) / len(line_lengths) if line_lengths else 1
    total_chars = sum(line_lengths)
    
    # Ensure minimum values
    max_line_chars = max(1, max_line_chars)
    avg_line_chars = max(1, avg_line_chars)
    
    return {
        'num_lines': num_lines,
        'max_line_chars': max_line_chars,
        'avg_line_chars': avg_line_chars,
        'total_chars': total_chars,
    }


def dry_run_layout(
    processed_tokens: List[Tuple[str, str]],
    width: int,
    height: int,
    font_size: int,
    line_height: float,
    margin_px: int,
    enable_two_column: bool,
    font_path: str = None,
    should_crop_whitespace: bool = False,
    preserve_newlines: bool = False,
) -> Tuple[int, float]:
    """
    Simulate layout to calculate page count and last page fill rate.
    optimized for speed (no drawing).
    """
    if font_size <= 0:
        return 9999, 0.0

    # Setup
    font = get_font(font_size, font_path)
    # Use a dummy image for text measurement
    dummy_img = PIL_Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(dummy_img)

    text_area_width = width - 2 * margin_px
    text_area_height = height - 2 * margin_px
    line_height_px = int(font_size * line_height)
    max_lines_per_page = int(text_area_height / line_height_px) if line_height_px > 0 else 1

    current_page_lines = 0
    current_x = margin_px
    current_y = margin_px
    current_column = 0
    max_column_width = 0
    column_start_x = margin_px
    
    if enable_two_column:
        column_width = (width - 2 * margin_px) // 2
        column_gap = 10
    else:
        column_width = width - 2 * margin_px
        column_gap = 0
    
    pages_count = 1
    
    # Simulate layout
    for token_text, _ in processed_tokens:
        for char in token_text:
            if preserve_newlines and char == "\n":
                max_column_width = max(max_column_width, current_x - column_start_x)
                current_y += line_height_px
                current_x = column_start_x
                current_page_lines += 1

                if current_page_lines >= max_lines_per_page:
                    if enable_two_column and current_column == 0 and max_column_width <= (width / 2):
                        current_column = 1
                        column_start_x = width // 2 + column_gap // 2
                        current_x = column_start_x
                        current_y = margin_px
                        current_page_lines = 0
                        max_column_width = 0
                    else:
                        pages_count += 1
                        current_page_lines = 0
                        current_y = margin_px
                        current_column = 0
                        column_start_x = margin_px
                        current_x = margin_px
                        max_column_width = 0
                continue

            try:
                char_w = temp_draw.textlength(char, font=font)
            except:
                char_bbox = temp_draw.textbbox((0, 0), char, font=font)
                char_w = char_bbox[2] - char_bbox[0]

            if enable_two_column:
                column_right_bound = column_start_x + column_width if current_column == 0 else width - margin_px
            else:
                column_right_bound = width - margin_px
            
            if current_x + char_w > column_right_bound and current_x > column_start_x:
                max_column_width = max(max_column_width, current_x - column_start_x)
                current_y += line_height_px
                current_x = column_start_x
                current_page_lines += 1

                if current_page_lines >= max_lines_per_page:
                    if enable_two_column and current_column == 0 and max_column_width <= (width / 2):
                        current_column = 1
                        column_start_x = width // 2 + column_gap // 2
                        current_x = column_start_x
                        current_y = margin_px
                        current_page_lines = 0
                        max_column_width = 0
                    else:
                        pages_count += 1
                        current_page_lines = 0
                        current_y = margin_px
                        current_column = 0
                        column_start_x = margin_px
                        current_x = margin_px
                        max_column_width = 0
            
            current_x += char_w
            max_column_width = max(max_column_width, current_x - column_start_x)

    # Calculate fill rate of the last page (vertical only for simplicity, or area?)
    # Vertical fill is good proxy for "close to bottom"
    if enable_two_column:
        # Complex to estimate fill rate for two columns, simplified:
        # If we are in second column, it's (lines in col1 + lines in col2) / (2 * max_lines) ?
        # Or just use the Y position relative to height
        if current_column == 1:
            # We are in second column, so first column is full
            # total lines filled = max_lines_per_page + current_page_lines
            # total capacity = 2 * max_lines_per_page
            fill_rate = (max_lines_per_page + current_page_lines) / (2 * max_lines_per_page)
        else:
            # First column only
            fill_rate = current_page_lines / (2 * max_lines_per_page)
    else:
        fill_rate = current_page_lines / max_lines_per_page if max_lines_per_page > 0 else 1.0
        
    return pages_count, fill_rate


def calculate_max_font_size_at_resolution(
    text: str,
    resolution: int,
    target_pages: int,
    line_height: float = 1.0,
    font_path: str = None,
    enable_syntax_highlight: bool = False,
    preserve_newlines: bool = False,
    language: str = None,
    theme: str = "light",
) -> int:
    """
    Calculate maximum font size using binary search to fit text into target_pages at resolution.
    Goal: Last line close to bottom of last page.
    """
    # Pre-process text/tokens once
    if enable_syntax_highlight and PYGMENTS_AVAILABLE:
        colored_tokens = parse_code_with_syntax_highlighting(
            text, filename=None, language=language, theme=theme
        )
    else:
        # Mock tokens for plain text
        colored_tokens = [(text, "#000000")]

    processed_tokens = []
    for token_text, token_color in colored_tokens:
        processed_token_text = token_text.replace("\t", "    ")
        typographic_replacements = {
            "'": "'", "'": "'", '"': '"', '"': '"', "–": "-", "—": "--", "…": "..."
        }
        for original, replacement in typographic_replacements.items():
            processed_token_text = processed_token_text.replace(original, replacement)
        
        if preserve_newlines:
             processed_tokens.append((processed_token_text, token_color))
        else:
             processed_tokens.append((processed_token_text.replace("\n", NEWLINE_MARKER), token_color))

    # Binary Search
    low = 4
    high = 150
    best_fs = 4
    margin_px = int(resolution * 0.01)
    
    # Heuristic optimization: Start from area-based estimate to narrow range?
    # But binary search is fast enough (log2(150) ~ 8 steps)
    
    while low <= high:
        mid = (low + high) // 2
        pages, fill_rate = dry_run_layout(
            processed_tokens,
            resolution,
            resolution,
            mid,
            line_height,
            margin_px,
            enable_two_column=False, # Assuming single column for simplicity unless specified
            font_path=font_path,
            preserve_newlines=preserve_newlines
        )
        
        if pages <= target_pages:
            # Fits in target pages, try larger font
            best_fs = mid
            low = mid + 1
        else:
            # Too many pages, font too big
            high = mid - 1
            
    return best_fs


def calculate_optimal_font_size(
    resolution: int,
    pages: int,
    num_lines: int,
    max_line_chars: int,
    line_height: float = 1.0,
) -> int:
    """
    Calculate optimal font size based on resolution, page count, line count, and longest line.
    
    Font size constraints:
    - Height constraint: font_size * line_height * num_lines <= resolution * pages (all lines fit)
    - Width constraint: font_size * 0.6 * max_line_chars <= resolution (longest line doesn't wrap)
    
    Optimal font = min(height limit, width limit)
    
    Args:
        resolution: Resolution (square)
        pages: Page count
        num_lines: Total line count
        max_line_chars: Maximum line character count
        line_height: Line height multiplier
        
    Returns:
        Optimal font size
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    # Height constraint: all lines fit
    # font_size * line_height * num_lines <= available_height * pages
    # font_size <= (available_height * pages) / (num_lines * line_height)
    fs_height_limit = (available_height * pages) / (num_lines * line_height) if num_lines > 0 else 150
    
    # Width constraint: longest line doesn't wrap
    # font_size * 0.6 * max_line_chars <= available_width
    # font_size <= available_width / (max_line_chars * 0.6)
    fs_width_limit = available_width / (max_line_chars * 0.6) if max_line_chars > 0 else 150
    
    # Take minimum of two constraints
    optimal_fs = min(fs_height_limit, fs_width_limit)
    optimal_fs = max(4, min(150, int(optimal_fs)))
    
    return optimal_fs


def calculate_fill_rate(
    font_size: int,
    resolution: int,
    pages: int,
    num_lines: int,
    avg_line_chars: int,
    line_height: float = 1.0,
) -> float:
    """
    Calculate fill rate.
    
    Vertical fill rate = font_size * line_height * num_lines / (available_height * pages)
    Horizontal fill rate = font_size * 0.6 * avg_line_chars / available_width
    Total fill rate = min(vertical, horizontal)
    
    Args:
        font_size: Font size
        resolution: Resolution
        pages: Page count
        num_lines: Total line count
        avg_line_chars: Average line character count
        line_height: Line height multiplier
        
    Returns:
        Fill rate (0.0 - 1.5)
    """
    margin = resolution * 0.01
    available_width = resolution - 2 * margin
    available_height = resolution - 2 * margin
    
    # Vertical fill rate
    total_text_height = font_size * line_height * num_lines
    total_available_height = available_height * pages
    vertical_fill = total_text_height / total_available_height if total_available_height > 0 else 0
    
    # Horizontal fill rate
    avg_line_width = font_size * 0.6 * avg_line_chars
    horizontal_fill = avg_line_width / available_width if available_width > 0 else 0
    
    # Take minimum as total fill rate
    fill_rate = min(vertical_fill, horizontal_fill)
    
    return min(1.5, fill_rate)


def optimize_layout_config(
    target_tokens: float,
    renderer_callback: Callable[[int, int, int], List[PIL_Image.Image]],
    previous_configs: List[Tuple[int, int]] = None,
    text_tokens: int = None,
    line_height: float = 1.0,
    text_structure: dict = None,
    compression_ratio: float = None,
    page_limit: int = 100,
) -> Tuple[List[PIL_Image.Image], int, int]:
    """
    Find optimal layout configuration (resolution, font size, page count) to fit target token count.
    
    Core logic:
    1. Calculate valid page counts for each resolution based on dynamic tolerance range of target_tokens
    2. For each (resolution, pages) config, calculate optimal font size based on text lines and resolution
    3. Mathematically calculate fill rate
    4. Use comprehensive scoring system (token match + fill rate + resolution + compression) to select best config
    5. Only perform actual rendering for final selected configuration
    
    Args:
        target_tokens: Target total token count
        renderer_callback: Callback function accepting (width, height, font_size) and returning image list
        previous_configs: List of previously used configs (resolution, image_count) to avoid duplication
        text_tokens: Original text token count
        line_height: Line height multiplier (default 1.0)
        text_structure: Text structure info {'num_lines', 'max_line_chars', 'avg_line_chars'}
        compression_ratio: Compression ratio (for adjusting resolution weight)
        page_limit: Maximum page limit
        
    Returns:
        Tuple of (best_images, best_resolution, best_font_size)
    """
    if previous_configs is None:
        previous_configs = []
    
    # If no text structure info, use default estimation
    if text_structure is None:
        # Estimate from text_tokens
        estimated_chars = text_tokens * 3.5 if text_tokens else 10000
        text_structure = {
            'num_lines': int(estimated_chars / 60),  # Assume 60 characters per line on average
            'max_line_chars': 120,  # Assume longest line is 120 characters
            'avg_line_chars': 60,   # Assume average of 60 characters
        }
    
    num_lines = text_structure['num_lines']
    max_line_chars = text_structure['max_line_chars']
    avg_line_chars = text_structure['avg_line_chars']
    
    # Resolution list: use expanded list with half-step and fractional multiples
    resolutions = get_expanded_resolution_list()
    
    # Fill rate threshold
    FILL_RATE_THRESHOLD = 0.90  # 90%
    
    # Dynamically adjust tolerance range
    if target_tokens < 50:
        token_min_ratio = 0.5
        token_max_ratio = 2.0
    elif target_tokens < 100:
        token_min_ratio = 0.7
        token_max_ratio = 1.5
    elif target_tokens < 3000:
        token_min_ratio = 0.8
        token_max_ratio = 1.25
    elif target_tokens < 5000:
        token_min_ratio = 0.9
        token_max_ratio = 1.12
    elif target_tokens < 10000:
        token_min_ratio = 0.93
        token_max_ratio = 1.08
    else:
        token_min_ratio = 0.95
        token_max_ratio = 1.05
    
    # ===== Step 1: Construct all valid (resolution, pages) configs =====
    all_configs = []
    
    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        
        # Calculate valid page counts based on dynamic tolerance range
        min_pages = math.ceil(target_tokens * token_min_ratio / per_image_tokens)
        max_pages = math.floor(target_tokens * token_max_ratio / per_image_tokens)
        
        # Filter pages >= 1
        min_pages = max(1, min_pages)
        
        if min_pages > max_pages:
            continue  # This resolution has no valid page counts
        
        # For each valid page count (dynamically adjust maximum page limit)
        max_allowed_pages = page_limit
        for pages in range(min_pages, max_pages + 1):
            if pages > max_allowed_pages:
                continue

            # Check if it's a new configuration
            is_new = (res, pages) not in previous_configs
            
            # ===== Step 2: Calculate optimal font size =====
            optimal_fs = calculate_optimal_font_size(
                res, pages, num_lines, max_line_chars, line_height
            )
            
            # ===== Step 3: Calculate fill rate =====
            fill_rate = calculate_fill_rate(
                optimal_fs, res, pages, num_lines, avg_line_chars, line_height
            )
            
            # Calculate actual tokens
            actual_tokens = pages * per_image_tokens
            token_ratio = actual_tokens / target_tokens if target_tokens > 0 else 1.0
            
            # Filter rules (keep more candidates, only filter extreme cases)
            if target_tokens >= 10000:
                if fill_rate < 0.03 and optimal_fs < 6:
                    continue
            elif target_tokens >= 5000:
                if fill_rate < 0.1 and optimal_fs < 5:
                    continue
            elif target_tokens >= 3000:
                if fill_rate < 0.1 and optimal_fs < 5:
                    continue
            
            all_configs.append({
                'resolution': res,
                'pages': pages,
                'font_size': optimal_fs,
                'fill_rate': fill_rate,
                'tokens': actual_tokens,
                'token_ratio': token_ratio,
                'is_new': is_new,
                'per_image_tokens': per_image_tokens,
            })
    
    if not all_configs:
        # No valid configuration, use fallback strategy: 112x112x1, font>=3
        print("Warning: No valid layout found for target tokens. Using fallback strategy (112x112x1).")
        res = 112
        
        # Binary search: find maximum font that fits in 1 page (minimum 3)
        low = 3
        high = 150
        best_fs = 3
        
        # First check if minimum value works (or if minimum is all we can do)
        min_imgs = renderer_callback(res, res, 3)
        if len(min_imgs) > 1:
            print(f"  [Fallback] Text too long for 112x112x1 even at font 3. Using font 3 ({len(min_imgs)} pages).")
            min_imgs = min_imgs[:page_limit]
            return min_imgs, res, 3
            
        while low <= high:
            mid = (low + high) // 2
            imgs = renderer_callback(res, res, mid)
            if len(imgs) == 1:
                best_fs = mid
                low = mid + 1  # Try larger font
            else:
                high = mid - 1  # Font too large, causes pagination, need to reduce
        
        imgs = renderer_callback(res, res, best_fs)
        print(f"  [Fallback] Using 112x112x1 with font {best_fs}")
        return imgs, res, best_fs
    
    # ===== Step 4: Use comprehensive scoring system to select best config =====
    def calculate_config_score(c, compression_ratio=None):
        """
        Calculate comprehensive score for configuration (higher is better).
        
        Factors considered:
        1. Token match: closeness of actual tokens to target tokens
        2. Fill rate: 90%+ same grade, below 20% heavy penalty
        3. Resolution: prefer larger resolution when fill rates are similar
        4. Compression: prioritize resolution more for small compression ratios
        """
        # 1. Token match score (0-1)
        token_diff = abs(c['token_ratio'] - 1.0)
        if target_tokens < 50:
            token_penalty_factor = 1.0
        elif target_tokens < 100:
            token_penalty_factor = 1.5
        elif target_tokens < 3000:
            token_penalty_factor = 2.0
        elif target_tokens < 5000:
            token_penalty_factor = 2.5
        elif target_tokens < 10000:
            token_penalty_factor = 3.0
        else:
            token_penalty_factor = 3.5
        
        token_score = 1.0 / (1.0 + token_diff * token_penalty_factor)
        
        # 2. Fill rate score (0-1)
        fill_rate = c['fill_rate']
        
        # For small token scenarios, reduce fill rate weight significantly
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            # Small token/high compression: fill rate just needs to not be extremely low (adjust by font to fill)
            if fill_rate >= 0.2:
                fill_score = 0.9  # Above 20% considered acceptable
            else:
                fill_score = 0.5  # Below 20% light penalty (not severe penalty)
        else:
            # Normal/large token scenario: fill rate very important
            if fill_rate >= 0.9:
                fill_score = 1.0  # Above 90% considered same grade
            elif fill_rate >= 0.2:
                # 20%-90% linearly mapped to 0.2-1.0
                fill_score = 0.2 + (fill_rate - 0.2) * (0.8 / 0.7)
            elif fill_rate >= 0.1:
                # 10%-20%: severe penalty
                fill_score = 0.05
            else:
                # <10%: extreme penalty (almost excluded)
                fill_score = 0.01
        
        resolution_normalized = (c['resolution'] - 112) / (4480 - 112)
        resolution_bonus = 1.0 + resolution_normalized * (0.5 if target_tokens >= 3000 else 0.3)
        
        # 4. Impact of compression ratio on resolution weight
        if compression_ratio is not None and compression_ratio <= 2.0:
            # Small compression ratios (0.5x, 1x, 1.5x, 2x): increase resolution weight
            compression_bonus = 1.2
        else:
            compression_bonus = 1.0
        
        # 5. Dynamically adjust weights: for small tokens or high compression, token match is more important
        if target_tokens < 50 or (compression_ratio is not None and compression_ratio >= 4.0):
            # Small token or high compression scenario: token match dominates, fill_score has minimal impact
            # Formula: token_score^2.5 * fill_score^0.2 (token dominant, fill almost no impact)
            score = (token_score ** 2.5) * (fill_score ** 0.2) * resolution_bonus * compression_bonus
        elif target_tokens < 100:
            score = (token_score ** 1.5) * (fill_score ** 0.75) * resolution_bonus * compression_bonus
        else:
            score = token_score * fill_score * resolution_bonus * compression_bonus
            if target_tokens >= 5000 and c.get('font_size', 10) < 8:
                score *= 0.8
        
        # 6. Page count penalty (Strict User Constraint)
        # "For tokens < 1000, keep pages within 2"
        if target_tokens < 1000:
            if c['pages'] > 2:
                # Severe penalty to force selection of <= 2 pages if possible
                score *= 0.01 
            else:
                # Slight bonus for fitting in fewer pages (implies higher resolution/better fit)
                # But prefer single page if possible? No, user just said <= 2.
                pass
                
            # Enhance resolution bonus specifically for this case "prefer larger resolution"
            resolution_bonus = 1.0 + resolution_normalized * 1.0  # Increase weight of resolution
            score = score * (resolution_bonus / (1.0 + resolution_normalized * (0.5 if target_tokens >= 3000 else 0.3))) # Replace old bonus with new stronger one effectively

        return score
    
    # Calculate comprehensive scores for all configs (including already used configs)
    for c in all_configs:
        c['score'] = calculate_config_score(c, compression_ratio=compression_ratio)
        # Give slight penalty to used configs (-5%), but don't completely exclude
        if not c['is_new']:
            c['score'] *= 0.95
    
    # Sort by score (descending)
    all_configs.sort(key=lambda x: -x['score'])
    
    # Selection strategy: select top 5 highest scoring for actual rendering validation
    # Prioritize new configs, but can select old configs if they score significantly higher
    selected = all_configs[:min(5, len(all_configs))]
    
    # Select highest scoring from selected as preliminary best config
    best = selected[0]
    
    # ===== Step 5: Actual rendering =====
    imgs = renderer_callback(best['resolution'], best['resolution'], best['font_size'])
    actual_pages = len(imgs)
    
    # If actual pages don't match expected, adjust font size
    if actual_pages != best['pages']:
        # Binary search for appropriate font size
        low_fs = 4
        high_fs = 150
        target_pages = best['pages']
        best_fs = best['font_size']
        best_imgs = imgs
        
        while low_fs <= high_fs:
            mid_fs = (low_fs + high_fs) // 2
            test_imgs = renderer_callback(best['resolution'], best['resolution'], mid_fs)
            
            if len(test_imgs) <= target_pages:
                best_fs = mid_fs
                best_imgs = test_imgs
                low_fs = mid_fs + 1
            else:
                high_fs = mid_fs - 1
        
        imgs = best_imgs
        best['font_size'] = best_fs
        actual_pages = len(imgs)
    
    # ===== Step 6: Try larger font to improve readability =====
    # Even if page count matches expected, should try larger font (as long as pages don't increase)
    current_best_fs = best['font_size']
    current_best_imgs = imgs
    
    # Use binary search to find maximum usable font
    # More efficient than linear attempt, can also find larger font
    low_fs = current_best_fs
    high_fs = 150
    target_pages = actual_pages
    
    # First quickly test a larger font to see if there's room for improvement
    test_fs = min(current_best_fs + 20, 150)
    test_imgs = renderer_callback(best['resolution'], best['resolution'], test_fs)
    if len(test_imgs) <= target_pages:
        # Significant room for improvement, use binary search
        low_fs = test_fs
        current_best_fs = test_fs
        current_best_imgs = test_imgs
    
    # Binary search for maximum usable font
    while low_fs < high_fs - 1:
        mid_fs = (low_fs + high_fs) // 2
        test_imgs = renderer_callback(best['resolution'], best['resolution'], mid_fs)
        
        if len(test_imgs) <= target_pages:
            current_best_fs = mid_fs
            current_best_imgs = test_imgs
            low_fs = mid_fs
        else:
            high_fs = mid_fs
    
    return current_best_imgs, best['resolution'], current_best_fs


def _optimize_layout_config_slow(
    target_tokens: float,
    renderer_callback: Callable[[int, int, int], List[PIL_Image.Image]],
    previous_configs: List[Tuple[int, int]],
    line_height: float = 1.0,
) -> Tuple[List[PIL_Image.Image], int, int]:
    """
    Original slow version (used when text_tokens not available).
    Actually calls rendering function for search.
    """
    # Resolution list: only try a few key resolutions to speed up
    resolutions = [112 * i for i in [20, 16, 12, 8, 4, 2, 1]]
    
    min_fs = 4
    max_fs = 150
    
    strict_candidates = []
    relaxed_candidates = []

    for res in resolutions:
        per_image_tokens = calculate_image_tokens_qwen3(res, res)
        token_limit = target_tokens * 1.25
        
        imgs_min = renderer_callback(res, res, min_fs)
        min_needed = len(imgs_min)
        
        if min_needed == 0:
            continue
        
        current_tokens = min_needed * per_image_tokens
        
        if current_tokens <= token_limit:
            max_images_limit = int(token_limit // per_image_tokens)
            
            low = min_fs
            high = max_fs
            curr_best_fs = min_fs
            curr_best_imgs = imgs_min
            
            while low <= high:
                mid = (low + high) // 2
                if mid == low:
                    if high > low:
                        imgs_high = renderer_callback(res, res, high)
                        if len(imgs_high) <= max_images_limit:
                            curr_best_fs = high
                            curr_best_imgs = imgs_high
                    break
                
                imgs = renderer_callback(res, res, mid)
                if len(imgs) <= max_images_limit:
                    curr_best_fs = mid
                    curr_best_imgs = imgs
                    low = mid + 1
                else:
                    high = mid - 1
            
            is_new = (res, len(curr_best_imgs)) not in previous_configs
            score = res * 1000 + curr_best_fs
            strict_candidates.append((score, curr_best_fs, res, curr_best_imgs, is_new))
            
        else:
            limit_images = min_needed
            
            low = min_fs
            high = max_fs
            curr_best_fs = min_fs
            curr_best_imgs = imgs_min
            
            while low <= high:
                mid = (low + high) // 2
                if mid == low:
                    if high > low:
                        imgs_high = renderer_callback(res, res, high)
                        if len(imgs_high) <= limit_images:
                            curr_best_fs = high
                            curr_best_imgs = imgs_high
                    break
                
                imgs = renderer_callback(res, res, mid)
                if len(imgs) <= limit_images:
                    curr_best_fs = mid
                    curr_best_imgs = imgs
                    low = mid + 1
                else:
                    high = mid - 1
            
            is_new = (res, len(curr_best_imgs)) not in previous_configs
            total_tokens = len(curr_best_imgs) * per_image_tokens
            relaxed_candidates.append((total_tokens, curr_best_fs, res, curr_best_imgs, is_new))

    # Decision phase
    strict_new = [c for c in strict_candidates if c[4]]
    if strict_new:
        strict_new.sort(key=lambda x: x[0], reverse=True)
        best = strict_new[0]
        return best[3], best[2], best[1]
        
    if strict_candidates:
        strict_candidates.sort(key=lambda x: x[0], reverse=True)
        best = strict_candidates[0]
        return best[3], best[2], best[1]
        
    if relaxed_candidates:
        relaxed_candidates.sort(key=lambda x: (x[0], -x[1], -x[2]))
        best_overall = relaxed_candidates[0]
        
        relaxed_new = [c for c in relaxed_candidates if c[4]]
        if relaxed_new:
            best_new = relaxed_new[0]
            if best_new[0] <= best_overall[0] * 1.1:
                return best_new[3], best_new[2], best_new[1]
            else:
                return best_overall[3], best_overall[2], best_overall[1]
        else:
            return best_overall[3], best_overall[2], best_overall[1]

    print("Warning: No valid layout found for target tokens. Using fallback strategy.")
    # Fallback: 112x112, 1 image, font size >= 3 (max possible)
    res = 112
    limit_images = 1
    
    # Binary search for max font size that fits in 1 image (or min 3)
    low = 3
    high = 150
    best_fs = 3
    
    # First check if font=3 can fit (or at least this is the best we can do)
    imgs_min = renderer_callback(res, res, 3)
    best_imgs = imgs_min[:1]  # Force 1 image if needed, but renderer returns list
    
    # If renderer returns multiple images, font=3 can't fit all text
    # But according to requirement "112*112*1", we can only take first image
    # And "minimum font is 3... select maximum font that fills the canvas"
    # If font=3 is already too much, then only font=3 is possible
    # If font=3 doesn't exceed (imgs_min length is 1), we can try larger font
    
    if len(imgs_min) <= 1:
        # Try to find larger font
        while low <= high:
            mid = (low + high) // 2
            if mid == low:
                if high > low:
                    imgs_high = renderer_callback(res, res, high)
                    if len(imgs_high) <= 1:
                        best_fs = high
                        best_imgs = imgs_high
                break
            
            imgs = renderer_callback(res, res, mid)
            if len(imgs) <= 1:
                best_fs = mid
                best_imgs = imgs
                low = mid + 1
            else:
                high = mid - 1
    
    # Ensure only 1 image is returned
    if len(best_imgs) > 1:
        best_imgs = best_imgs[:1]
        
    return best_imgs, res, best_fs


def generate_compressed_images_dynamic(
    text_tokens: int,
    renderer_func: Callable[[int, int, int], List[PIL_Image.Image]],
    compression_ratios: List[float] = None,
    text_structure: dict = None,
    data_id: str = None,
    page_limit: int = 100,
) -> Dict[float, Tuple[List[PIL_Image.Image], int, int]]:
    """
    Dynamically generate images based on compression ratio, rather than simple resize.
    
    Args:
        text_tokens: Text token count
        renderer_func: Rendering function
        compression_ratios: List of compression ratios
        text_structure: Text structure info {'num_lines', 'max_line_chars', 'avg_line_chars'}
        data_id: Data identifier for logging
        page_limit: Maximum page limit
    
    Returns:
        Dict[ratio, (images, resolution, font_size)]
    """
    if compression_ratios is None:
        compression_ratios = COMPRESSION_RATIOS
    
    results = {}
    used_configs = []  # List of (res, image_count)
    
    # Sort by compression ratio
    sorted_ratios = sorted(compression_ratios)
    
    for ratio in sorted_ratios:
        ratio_start_time = time.time()
        if float(ratio) == 0.0:
            os.makedirs("./generated_images", exist_ok=True)
            blank_path = os.path.join("./generated_images", "blank_14x14.png")
            if os.path.exists(blank_path):
                try:
                    img = PIL_Image.open(blank_path).convert("RGB")
                except Exception:
                    img = PIL_Image.new("RGB", (14, 14), color="white")
                    img.save(blank_path)
            else:
                img = PIL_Image.new("RGB", (14, 14), color="white")
                img.save(blank_path)
            imgs, res, fs = [img], 14, 0
            results[ratio] = (imgs, res, fs)
            ratio_elapsed = time.time() - ratio_start_time
            actual_tokens = len(imgs) * calculate_image_tokens_qwen3(res, res)
            id_prefix = f"[{data_id}] " if data_id else ""
            print(f"  {id_prefix}Ratio 0: Res {res}x{res}, Count {len(imgs)}, Font {fs}, Fill 0%, Tokens {actual_tokens} (Target 0.0) [Time: {ratio_elapsed:.3f}s]")
            import sys
            sys.stdout.flush()
            continue
        target_tokens = text_tokens / ratio
        
        imgs, res, fs = optimize_layout_config(
            target_tokens,
            renderer_func,
            previous_configs=used_configs,
            text_tokens=text_tokens,
            line_height=1.0,
            text_structure=text_structure,
            compression_ratio=ratio,
            page_limit=page_limit,
        )
        
        results[ratio] = (imgs, res, fs)
        # Record used configuration to avoid subsequent duplicate selection
        used_configs.append((res, len(imgs)))
        
        ratio_elapsed = time.time() - ratio_start_time
        actual_tokens = len(imgs) * calculate_image_tokens_qwen3(res, res)
        
        # Calculate actual fill rate
        if text_structure:
            fill_rate = calculate_fill_rate(
                fs, res, len(imgs), 
                text_structure['num_lines'], 
                text_structure['avg_line_chars'], 
                1.0
            )
        else:
            fill_rate = estimate_fill_rate(text_tokens, res, fs, 1.0)
        
        # Build base log info
        id_prefix = f"[{data_id}] " if data_id else ""
        base_log = f"  {id_prefix}Ratio {ratio}: Res {res}x{res}, Count {len(imgs)}, Font {fs}, Fill {fill_rate:.0%}, Tokens {actual_tokens} (Target {target_tokens:.1f}) [Time: {ratio_elapsed:.3f}s]"
        
        # Check warning conditions
        warnings = []
        
        # 1. Severe token imbalance: deviation exceeds 20%
        token_diff_ratio = abs(actual_tokens - target_tokens) / target_tokens
        if token_diff_ratio > 0.2:
            warnings.append(f"⚠️ Token imbalance: {token_diff_ratio:.1%}")
        
        # 2. Fill rate too low: below 10%
        if fill_rate < 0.1:
            warnings.append(f"⚠️ Fill rate too low: {fill_rate:.1%}")
        
        # 3. Font too small: below 8
        if fs < 8:
            warnings.append(f"⚠️ Font too small: {fs}")
        
        # Output log (including warnings)
        if warnings:
            print(f"{base_log} {' '.join(warnings)}")
        else:
            print(base_log)
        import sys
        sys.stdout.flush()
    
    return results


def calculate_image_tokens_from_paths(image_paths: List[str]) -> int:
    """
    Calculate image token count from image paths (using patch estimation method).
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Total token count
    """
    total_tokens = 0

    BASE_IMAGE_TOKENS = 170
    PATCH_SIZE = 14

    for image_path in image_paths:
        try:
            with PIL_Image.open(image_path) as img:
                width, height = img.size

            num_patches_w = math.ceil(width / PATCH_SIZE)
            num_patches_h = math.ceil(height / PATCH_SIZE)
            total_patches = num_patches_w * num_patches_h

            image_tokens = BASE_IMAGE_TOKENS + min(total_patches * 2, 100)
            total_tokens += int(image_tokens)
        except Exception as e:
            print(f"  Warning: Error calculating tokens for image {image_path}: {e}, using default value")
            total_tokens += BASE_IMAGE_TOKENS

    return total_tokens


def calculate_image_tokens_with_processor(
    image_paths: List[str], processor: Optional[AutoProcessor] = None
) -> Optional[int]:
    """
    Calculate image token count using AutoProcessor.
    
    Args:
        image_paths: List of image file paths
        processor: AutoProcessor instance
        
    Returns:
        Total token count, or None if processor unavailable
    """
    if not TRANSFORMERS_AVAILABLE:
        return None

    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-235B-A22B-Instruct", trust_remote_code=True
            )
        except Exception as e:
            print(f"  Warning: AutoProcessor loading failed, trying force_download=True...: {e}")
            try:
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    trust_remote_code=True,
                    force_download=True,
                )
            except Exception as e2:
                print(f"  Warning: AutoProcessor loading failed again: {e2}")
                return None

    total_tokens = 0

    for image_path in image_paths:
        try:
            image = PIL_Image.open(image_path).convert("RGB")

            # Construct messages (only images, no text)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": ""},  # Empty text, only calculate image tokens
                    ],
                }
            ]

            # Process using processor
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Extract token information from inputs
            image_tokens = 0

            # Method 1: Use image_grid_thw for calculation (most accurate)
            if "image_grid_thw" in inputs:
                grid_info = inputs["image_grid_thw"]
                # grid_info shape: [1, 3] -> [num_images, height_grid, width_grid]
                num_images = grid_info[0][0].item()
                height_grid = grid_info[0][1].item()
                width_grid = grid_info[0][2].item()
                image_tokens = height_grid * width_grid
            # Method 2: Use first dimension of pixel_values
            elif "pixel_values" in inputs:
                image_tokens = inputs["pixel_values"].shape[0]
            else:
                # Fallback method: use patch estimation
                image_tokens = calculate_image_tokens_from_paths([image_path])

            total_tokens += image_tokens

        except Exception as e:
            print(f"  Warning: Error calculating tokens for image {image_path} with Processor: {e}")
            # Use default estimation
            total_tokens += calculate_image_tokens_from_paths([image_path])

    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Compact image generation tool")
    parser.add_argument("--filename", type=str, default=None, help="Original filename")
    parser.add_argument("--txt-file", type=str, default=None, help="Text file")
    parser.add_argument(
        "--output-dir", type=str, default="./generated_images", help="Output directory"
    )
    parser.add_argument("--width", type=int, default=2240, help="Width (default: 2240)")
    parser.add_argument("--height", type=int, default=2240, help="Height (default: 2240)")
    parser.add_argument("--font-size", type=int, default=40, help="Font size (default: 40)")
    parser.add_argument("--line-height", type=float, default=1.0, help="Line height (default: 1.0)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI")
    parser.add_argument(
        "--preserve-newlines",
        action="store_true",
        default=True,
        help="Preserve newlines (default: True)",
    )
    parser.add_argument(
        "--enable-syntax-highlight", action="store_true", help="Enable syntax highlighting"
    )
    parser.add_argument("--crop-whitespace", action="store_true", help="Crop whitespace")
    parser.add_argument("--enable-two-column", action="store_true", help="Enable two-column layout")
    parser.add_argument(
        "--resize-mode", action="store_true", default=True, help="Enable resize mode (default: True)"
    )
    parser.add_argument(
        "--no-resize-mode",
        action="store_false",
        dest="resize_mode",
        help="Disable resize mode",
    )
    parser.add_argument("--enable-bold", action="store_true", help="Enable bold text")

    args = parser.parse_args()

    if args.txt_file:
        with open(args.txt_file, "r") as f:
            source_code = f.read()
        filename = os.path.basename(args.txt_file)
    elif args.filename:
        if os.path.exists(args.filename):
            with open(args.filename, "r") as f:
                source_code = f.read()
            filename = args.filename
        else:
            print(
                f"File not found: {args.filename}. "
            )
            return
    else:
        print("Please provide --filename or --txt-file")
        return

    image_paths = generate_images_for_file(
        filename,
        source_code,
        args.output_dir,
        args.width,
        args.height,
        args.font_size,
        args.line_height,
        args.dpi,
        preserve_newlines=args.preserve_newlines,
        enable_syntax_highlight=args.enable_syntax_highlight,
        should_crop_whitespace=args.crop_whitespace,
        enable_two_column=args.enable_two_column,
        enable_bold=args.enable_bold,
    )
    resolution_parts = [f"{args.width}x{args.height}"]
    if args.enable_syntax_highlight:
        resolution_parts.append("hl")
    if args.preserve_newlines:
        resolution_parts.append("nl")
    resolution_dir = os.path.join(args.output_dir, "_".join(resolution_parts))
    if args.resize_mode and image_paths:
        try:
            text_tokens = get_text_tokens(source_code)
        except Exception:
            text_tokens = max(1, len(source_code) // 4)
        original_images = []
        for p in image_paths:
            try:
                img = PIL_Image.open(p)
                original_images.append(img.copy())
                img.close()
            except Exception:
                pass
        for compression_ratio in COMPRESSION_RATIOS:
            resized = resize_images_for_compression(
                original_images,
                text_tokens,
                compression_ratios=[compression_ratio],
            )
            resized_images, target_resolution = resized.get(compression_ratio, ([], None))
            if not resized_images or not target_resolution:
                continue
            resize_output_dir = os.path.join(
                resolution_dir, f"resize_ratio{compression_ratio}_{target_resolution}x{target_resolution}"
            )
            os.makedirs(resize_output_dir, exist_ok=True)
            for i, img in enumerate(resized_images, 1):
                path = os.path.join(resize_output_dir, f"page_{i:03d}.png")
                try:
                    img.save(path)
                except Exception:
                    continue


if __name__ == "__main__":
    main()
    main()
