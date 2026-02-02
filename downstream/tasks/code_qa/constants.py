try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed, Function RAG functionality unavailable")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed, Function RAG functionality unavailable")

DEFAULT_LQA_FILES = ["32K", "64K"]
DEFAULT_NUM_EXAMPLES_PER_FILE = 200
DEFAULT_MAX_WORKERS = 20
DEFAULT_RAG_TOP_K = 3
RAG_GPU_DEVICE = None
