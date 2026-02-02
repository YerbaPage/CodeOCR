import torch

from llm_utils import (
    get_appropriate_device,
    load_model_with_retry,
    TRANSFORMERS_AVAILABLE,
)
from tasks.code_completion.task import TORCH_AVAILABLE


class RuntimeContext:
    def __init__(self):
        self.processor = None
        self.qwen_tokenizer = None
        self.embed_model = None
        self.embed_tokenizer = None
        self.rag_gpu_device = None


def initialize_processors(ctx: RuntimeContext) -> None:
    if not TRANSFORMERS_AVAILABLE:
        return

    from transformers import AutoProcessor, AutoTokenizer

    print("Loading AutoProcessor and Qwen tokenizer...")
    ctx.processor = load_model_with_retry(
        AutoProcessor,
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        trust_remote_code=True,
    )
    ctx.qwen_tokenizer = load_model_with_retry(
        AutoTokenizer,
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True,
    )


def initialize_embedding_model(ctx: RuntimeContext, embed_model_name: str) -> None:
    if not TORCH_AVAILABLE:
        print("Warning: torch not available, cannot load embedding model")
        return

    from transformers import AutoTokenizer, AutoModel

    print(f"\nLoading embedding model: {embed_model_name}")
    device = get_appropriate_device()
    print(f"Using device: {device}")

    ctx.rag_gpu_device = device
    ctx.embed_tokenizer = AutoTokenizer.from_pretrained(
        embed_model_name, trust_remote_code=True
    )
    ctx.embed_model = AutoModel.from_pretrained(
        embed_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    ).to(device)
    ctx.embed_model.eval()
