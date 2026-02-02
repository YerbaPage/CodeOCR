# Constants definition

NEWLINE_MARKER = "⏎"  # Visual marker for newlines in compact mode
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

# Typographic character replacements
TYPOGRAPHIC_REPLACEMENTS = {
    "'": "'",
    "'": "'",
    '"': '"',
    '"': '"',
    "–": "-",
    "—": "--",
    "…": "...",
}

TAB_SPACES = "    "  # Tab replacement (4 spaces)
