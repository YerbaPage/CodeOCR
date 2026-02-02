# Syntax highlighting module

from typing import List, Tuple

# Try to import pygments
try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename, guess_lexer
    from pygments.token import Token
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

# VS Code Light Modern theme colors
MODERN_THEME = {
    "Keyword": "#AF00DB",
    "Keyword.Declaration": "#0000FF",
    "Keyword.Type": "#0000FF",
    "Keyword.Constant": "#0000FF",
    "Operator.Word": "#0000FF",
    "Name.Function": "#795E26",
    "Name.Builtin": "#795E26",
    "Name.Class": "#267F99",
    "Name": "#000000",
    "Name.Variable": "#001080",
    "Name.Attribute": "#001080",
    "String": "#A31515",
    "String.Doc": "#008000",
    "Number": "#098658",
    "Comment": "#008000",
    "Operator": "#000000",
    "Punctuation": "#000000",
    "Error": "#FF0000",
}

# Classic Light theme colors
LIGHT_THEME = {
    "Keyword": "#0000FF",
    "Name.Function": "#795E26",
    "Name.Builtin": "#795E26",
    "Name.Class": "#267F99",
    "Name": "#000000",
    "String": "#A31515",
    "String.Doc": "#008000",
    "Number": "#098658",
    "Comment": "#008000",
    "Operator": "#000000",
    "Punctuation": "#000000",
    "Error": "#FF0000",
}


def _get_color_for_token(token_type, theme_map: dict) -> str:
    """Get color for token type."""
    token_str = str(token_type)
    for key, color in theme_map.items():
        if key in token_str:
            return color
    return "#000000"


def parse_code_with_syntax_highlighting(
    code: str, 
    filename: str = None, 
    language: str = None, 
    theme: str = "light"
) -> List[Tuple[str, str]]:
    """
    Parse code using Pygments and return colored token list.
    
    Args:
        code: Source code text
        filename: Filename (for automatic language detection)
        language: Language name (e.g. 'python')
        theme: Theme name ('light' or 'modern')
    
    Returns:
        List of (text, color) tuples
    """
    if not PYGMENTS_AVAILABLE:
        return [(code, "#000000")]
    
    # Get lexer
    lexer = None
    if language:
        lexer = get_lexer_by_name(language)
    elif filename:
        lexer = guess_lexer_for_filename(filename, code)
    else:
        lexer = guess_lexer(code)
    
    if lexer is None:
        lexer = get_lexer_by_name("python")
    
    # Select theme
    theme_map = MODERN_THEME if theme == "modern" else LIGHT_THEME
    
    # Parse and colorize
    tokens = list(lex(code, lexer))
    result = []
    for token_type, text in tokens:
        color = _get_color_for_token(token_type, theme_map)
        result.append((text, color))
    
    return result
