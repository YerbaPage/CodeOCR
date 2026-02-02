try:
    import datasets
    import editdistance
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed")

DEFAULT_MAX_WORKERS = 20
DEFAULT_NUM_EXAMPLES = 200
DEFAULT_FILTER_CURRENT_LINES_MAX = 50
DEFAULT_FILTER_BACKGROUND_TOKENS_MIN = 3000
DEFAULT_RAG_WINDOW_SIZE = 80
DEFAULT_RAG_OVERLAP = 40
DEFAULT_RAG_TOP_K = 3
