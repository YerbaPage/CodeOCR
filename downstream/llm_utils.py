import os
import json
import base64
import re
import time
import torch
import numpy as np
import tiktoken
from openai import OpenAI, AzureOpenAI
from typing import Tuple, Dict, List, Optional, Union

try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# API Configuration
API_BASE_URL = "https://aihubmix.com/v1"
RETRY = 20


def get_config() -> Dict:
    """Load configuration from config.json file."""
    return json.load(open("config.json", "r"))


try:
    config = get_config()
    if config and isinstance(config, dict):
        API_KEY = config.get("api_key")
        API_BASE_URL = config.get("base_url")
        AZURE_ENDPOINT = config.get("azure_endpoint")
        AZURE_API_VERSION = config.get("azure_api_version")
        CLIENT_TYPE = "Azure" if config.get("azure_endpoint") else "OpenAI"
    else:
        API_KEY = os.environ.get("OPENAI_API_KEY", "")
        AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
except Exception:
    API_KEY = ""
    AZURE_ENDPOINT = ""
    AZURE_API_VERSION = ""

# Evaluator model mappings
EVALUATOR_MODELS = {
    "qwen3-vl-235b-a22b-instruct": "gpt-5.1-mini",
    "gpt-5.1": "qwen3-235b-a22b-instruct-2507",
    "gemini-2.5-pro": "qwen3-235b-a22b-instruct-2507",
    "claude-sonnet-4-5": "qwen3-235b-a22b-instruct-2507",
    "glm-4.5v": "qwen3-235b-a22b-instruct-2507",
    "DeepSeek-OCR": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-mini": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-mini-2025-08-07": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-2025-08-07": "qwen3-235b-a22b-instruct-2507",
}


def create_client(client_type: str = "OpenAI", **kwargs) -> Union[OpenAI, AzureOpenAI]:
    """
    Create OpenAI or AzureOpenAI client.

    Args:
        client_type: Client type, "OpenAI" or "Azure"
        **kwargs: Additional parameters to override default config

    Returns:
        OpenAI or AzureOpenAI client instance
    """
    if client_type == "Azure":
        return AzureOpenAI(
            api_key=kwargs.get("api_key", API_KEY),
            api_version=kwargs.get("api_version", AZURE_API_VERSION),
            azure_endpoint=kwargs.get("azure_endpoint", AZURE_ENDPOINT),
        )
    else:
        return OpenAI(
            base_url=kwargs.get("base_url", API_BASE_URL),
            api_key=kwargs.get("api_key", API_KEY),
        )


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image) -> str:
    """Encode PIL Image object to base64 string."""
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_prompt(prompt_path: str) -> Dict:
    """Load prompt configuration from JSON file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_api_kwargs(model_name: str, max_tokens: int) -> Dict:
    """Prepare API kwargs based on model type."""
    kwargs = {}
    
    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = 0.0
        kwargs["stream"] = False
        kwargs["max_tokens"] = max_tokens
    else:
        kwargs["max_completion_tokens"] = max_tokens
    
    # Model-specific token limits
    if model_name.startswith("DeepSeek") or model_name.startswith("glm"):
        kwargs["max_tokens"] = min(4096, max_tokens)
    
    # Model-specific extra parameters
    if model_name.startswith("qwen"):
        kwargs["extra_body"] = {"thinking_budget": 1024}
    
    if CLIENT_TYPE == "Azure":
        kwargs["extra_headers"] = {"X-TT-LOGID": "${your_logid}"}
        kwargs["extra_body"] = {
            "thinking": {"include_thoughts": False, "budget_tokens": 1024}
        }
    
    return kwargs


def _get_client_for_model(client: Union[OpenAI, AzureOpenAI], model_name: str) -> Union[OpenAI, AzureOpenAI]:
    """Get appropriate client for model (handles Qwen special case)."""
    config = get_config()
    if config.get("qwen_api_key") and config.get("qwen_base_url") and model_name.startswith("qwen"):
        return OpenAI(
            base_url=config.get("qwen_base_url"),
            api_key=config.get("qwen_api_key"),
        )
    return client


def call_llm_with_images(
    client: Union[OpenAI, AzureOpenAI],
    model_name: str,
    images: Union[List[str], List],
    system_prompt: str,
    user_prompt: str,
    retry_on_empty: bool = True,
    client_type: str = "OpenAI",
    max_tokens: int = 6144,
    data_id: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Call LLM API for image recognition.

    Args:
        client: OpenAI or AzureOpenAI client
        model_name: Model name
        images: List of image paths or PIL Image objects
        system_prompt: System prompt
        user_prompt: User prompt
        retry_on_empty: Whether to retry if response is empty
        client_type: Client type ("OpenAI" or "Azure")
        max_tokens: Maximum tokens to generate
        data_id: Optional data identifier for logging

    Returns:
        Tuple of (generated_text, token_info) containing the generated text and token statistics
    """
    from PIL import Image as PIL_Image
    
    # Encode images to base64
    base64_images = []
    for img in images:
        if isinstance(img, PIL_Image.Image):
            base64_images.append(encode_pil_image_to_base64(img))
        elif isinstance(img, str):
            base64_images.append(encode_image_to_base64(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": user_prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    }
                    for image in base64_images
                ]
            ),
        },
    ]
    
    # Get appropriate client
    client = _get_client_for_model(client, model_name)
    
    # Retry loop
    for attempt in range(RETRY if retry_on_empty else 1):
        try:
            # Prepare API arguments
            kwargs = {
                "model": model_name,
                "messages": messages,
                **_prepare_api_kwargs(model_name, max_tokens)
            }
            
            response = client.chat.completions.create(**kwargs)
            generated_text = response.choices[0].message.content
            usage = response.usage

            # Check for empty response
            if not generated_text or not generated_text.strip():
                if retry_on_empty and attempt < RETRY - 1:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}Warning: LLM returned empty response, retrying (attempt {attempt + 1})...")
                    time.sleep(1)
                    continue
                else:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}Warning: LLM returned empty response after {attempt + 1} attempts")
                    kwargs.pop("messages", None)
                    return "", {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "api_kwargs": kwargs
                    }
            
            # Success
            kwargs.pop("messages", None)
            return generated_text, {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "api_kwargs": kwargs
            }
            
        except Exception as e:
            if attempt < RETRY - 1 and retry_on_empty:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"  {prefix}Warning: Error calling model {model_name}, retrying: {e}, attempt {attempt + 1}")
                time.sleep(2)
                continue
            else:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"{prefix}Error calling model {model_name}: {e}, attempt {attempt + 1}")
                kwargs.pop("messages", None)
                return "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_kwargs": kwargs
                }

    # Fallback if all attempts failed
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_kwargs": {}}


def call_llm_with_text_only(
    client: Union[OpenAI, AzureOpenAI],
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    retry_on_empty: bool = True,
    client_type: str = "OpenAI",
    max_tokens: int = 6144,
    data_id: Optional[str] = None,
) -> Tuple[str, Dict]:
    """Call LLM API with text only (no images).

    Args:
        client: OpenAI or AzureOpenAI client
        model_name: Model name
        system_prompt: System prompt
        user_prompt: User prompt
        retry_on_empty: Whether to retry if response is empty
        client_type: Client type
        max_tokens: Maximum tokens to generate
        data_id: Optional data identifier for logging

    Returns:
        (generated_text, token_info): Generated text and token information
    """
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get appropriate client
    client = _get_client_for_model(client, model_name)

    # Retry loop
    for attempt in range(RETRY if retry_on_empty else 1):
        try:
            # Prepare API arguments
            kwargs = {
                "model": model_name,
                "messages": messages,
                **_prepare_api_kwargs(model_name, max_tokens)
            }
            
            response = client.chat.completions.create(**kwargs)
            generated_text = response.choices[0].message.content
            usage = response.usage

            # Check for empty response
            if not generated_text or not generated_text.strip():
                if retry_on_empty and attempt < RETRY - 1:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}Warning: LLM returned empty response, retrying (attempt {attempt+1})...")
                    time.sleep(1)
                    continue
                else:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}Warning: LLM returned empty response after {attempt+1} attempts")
                    return "", {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "api_kwargs": kwargs
                    }

            # Success
            return generated_text, {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "api_kwargs": kwargs
            }
            
        except Exception as e:
            if attempt < RETRY - 1 and retry_on_empty:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"  {prefix}Warning: Error calling model {model_name}, retrying: {e}, attempt {attempt+1}")
                time.sleep(2)
                continue
            else:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"{prefix}Error calling model {model_name}: {e}, attempt {attempt+1}")
                return "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_kwargs": kwargs
                }


def get_text_tokens(text: str) -> int:
    """Calculate token count for text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def get_text_tokens_qwen(text: str, tokenizer=None) -> Optional[int]:
    """Calculate token count for text using Qwen tokenizer."""
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True
            )
        except ImportError:
            return None
        except Exception:
            return None

    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"  Warning: Failed to calculate tokens using Qwen tokenizer: {e}")
        return None


def call_llm_with_logit_bias(client, eval_model, query: str, options: list[str]):
    try:
        tokenizer_model = (
            "gemini-2.5-pro" if "gemini-2.5-pro" in eval_model else "gpt-4"
        )
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        print(
            f"Warning: Cannot automatically map {tokenizer_model} to tokenizer. Using default cl100k_base tokenizer."
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    logit_bias = dict()
    for opt in options:
        tok_ids = tokenizer.encode(opt)
        assert len(tok_ids) == 1, "Only single token options are supported"
        logit_bias[tok_ids[0]] = 100
    kwargs = {
        "model": eval_model,
        "messages": [
            {"role": "system", "content": "You are a code quality assesing engine."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 1,
        "temperature": 0.3,
        "n": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "logit_bias": logit_bias,
    }
    if isinstance(eval_model, str) and "deepseek" in eval_model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    completion = client.chat.completions.create(**kwargs)

    logprobs = np.full(2, np.nan)
    choice = completion.choices[0]
    opt_to_idx = {t: n for n, t in enumerate(options)}
    min_lp = 0
    for logprob_item in choice.logprobs.content[0].top_logprobs:
        tok = logprob_item.token
        lp = logprob_item.logprob
        min_lp = min(min_lp, lp)
        if tok in opt_to_idx:
            logprobs[opt_to_idx[tok]] = lp
    logprobs[np.isnan(logprobs)] = (
        min_lp - 2.3
    )  # approximately 10 times less than the minimal one
    usage = completion.usage
    assert not np.isnan(logprobs).any()
    return torch.from_numpy(logprobs), {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }


def build_folder(base_dir: str, folder_parts: list[str], **kwargs) -> str:
    """Build full path from base directory and folder parts."""
    if kwargs.get('enable_syntax_highlight'):
        folder_parts.append("hl")
    if kwargs.get('preserve_newlines'):
        folder_parts.append("nl")
    if kwargs.get('enable_bold'):
        folder_parts.append("bold")
    return os.path.join(base_dir, "_".join(folder_parts))


# ============================================================================
# Device and Model Utilities (merged from shared_utils.py)
# ============================================================================

def get_appropriate_device():
    """
    Get appropriate compute device with priority order: CUDA > MPS > CPU.
    
    Returns:
        torch.device: The selected compute device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_embedding(text: str, model, tokenizer, device):
    """
    Calculate text embedding vector (using mean pooling).
    
    Args:
        text: Input text
        model: Embedding model
        tokenizer: Tokenizer
        device: Compute device
        
    Returns:
        Embedding tensor
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
    return embedding


def split_code_by_functions_standalone(code: str, language: str = "python") -> List[str]:
    """
    Split code into chunks based on function and class definitions for various languages.
    Standalone version that doesn't require CodeCompressor instance.
    
    Args:
        code: The code to split
        language: Programming language of the code (python, cpp, java, typescript, rust, go)
        
    Returns:
        List of code chunks, each containing a function, class, or class method
    """
    # Language-specific patterns for function/class definitions
    patterns = {
        "python": r"^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)",
        "py": r"^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)",
        "java": r"^(\s*(public|private|protected)?\s*(static)?\s*(class|interface|enum|void|int|String|boolean|long|double|float|char|byte|short)\s+\w+)",
        "cpp": r"^(\s*(class|struct|void|int|bool|auto|template)\s+\w+)",
        "typescript": r"^(\s*(export\s+)?(class|interface|function|async\s+function|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\()\s*\w*)",
        "ts": r"^(\s*(export\s+)?(class|interface|function|async\s+function|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\()\s*\w*)",
        "javascript": r"^(\s*(export\s+)?(class|function|async\s+function|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\()\s*\w*)",
        "js": r"^(\s*(export\s+)?(class|function|async\s+function|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\()\s*\w*)",
        "rust": r"^(\s*(pub\s+)?(fn|struct|enum|impl|trait)\s+\w+)",
        "go": r"^(\s*(func|type)\s+\w+)",
    }
    
    pattern = patterns.get(language.lower(), patterns["python"])
    
    lines = code.split("\n")
    chunks = []
    current_chunk_lines = []
    
    for line in lines:
        if re.match(pattern, line, re.MULTILINE):
            # Found a new definition, save current chunk if exists
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
    
    # Don't forget the last chunk
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
    
    # Filter out empty chunks
    chunks = [c for c in chunks if c.strip()]
    
    return chunks


def function_rag_retrieve(
    background_code: str,
    query_code: str,
    model,
    tokenizer,
    device,
    language: str,
    top_k: int
) -> str:
    """
    Use function-level chunking to retrieve top_k similar functions.

    Args:
        background_code: Background code context
        query_code: Query code
        model: Embedding model
        tokenizer: Tokenizer
        device: Compute device
        language: Programming language
        top_k: Number of top similar functions to retrieve

    Returns:
        Combined relevant code blocks
    """
    # Split background code into function-level chunks
    chunks = split_code_by_functions_standalone(background_code, language)
    
    if not chunks:
        return background_code
    
    # Calculate query embedding
    query_embedding = compute_embedding(query_code, model, tokenizer, device)
    
    # Calculate similarity for each chunk
    similarities = []
    for chunk in chunks:
        chunk_embedding = compute_embedding(chunk, model, tokenizer, device)
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding, chunk_embedding, dim=1
        ).item()
        similarities.append((similarity, chunk))
    
    # Sort by similarity (descending) and take top_k
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in similarities[:top_k]]
    
    return "\n\n".join(top_chunks)


def find_idle_gpu(threshold_memory_gb: float = 1.0) -> Optional[int]:
    """
    Find an idle GPU with available memory below the threshold.
    
    Args:
        threshold_memory_gb: Memory threshold in GB
        
    Returns:
        GPU index or None if no idle GPU found
    """
    if not torch.cuda.is_available():
        return None

    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
        used_memory = max(allocated_memory, reserved_memory)

        if used_memory < threshold_memory_gb:
            print(
                f"  Found idle GPU {i}: {torch.cuda.get_device_name(i)}, "
                f"Total memory: {total_memory:.2f} GB, Used memory: {used_memory:.2f} GB"
            )
            return i
    return None


def load_model_with_retry(model_class, model_name: str, **kwargs):
    """
    Load a model with retry logic (tries force_download=True if first attempt fails).
    
    Args:
        model_class: Model class (AutoProcessor, AutoTokenizer, etc.)
        model_name: Model identifier
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model instance
    """
    try:
        return model_class.from_pretrained(model_name, **kwargs)
    except Exception as e:
        print(f"  Model loading failed, trying force_download=True: {e}")
        return model_class.from_pretrained(model_name, force_download=True, **kwargs)

