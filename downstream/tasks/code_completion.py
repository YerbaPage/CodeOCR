import os
import json
import re
from typing import List, Tuple, Dict, Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Local imports
from text_to_image import (
    text_to_image,
    resize_images_for_compression,
    generate_compressed_images_dynamic,
    analyze_text_structure,
    calculate_image_tokens_from_paths,
    calculate_image_tokens_with_processor,
    COMPRESSION_RATIOS,
    text_to_image_stream,
    optimize_layout_config_dry,
    find_closest_resolution_prefer_larger,
    get_expanded_resolution_list,
    calculate_image_tokens_qwen3,
    calculate_fill_rate,
)
from llm_utils import (
    get_config,
    build_folder,
    create_client,
    call_llm_with_images,
    call_llm_with_text_only,
    get_text_tokens,
    get_text_tokens_qwen,
    get_appropriate_device,
)



try:
    import datasets
    import editdistance
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed")

# Constants
DEFAULT_MAX_WORKERS = 20
DEFAULT_NUM_EXAMPLES = 200
DEFAULT_FILTER_CURRENT_LINES_MAX = 50
DEFAULT_FILTER_BACKGROUND_TOKENS_MIN = 3000
DEFAULT_RAG_WINDOW_SIZE = 80
DEFAULT_RAG_OVERLAP = 40
DEFAULT_RAG_TOP_K = 3

# Global variables
RAG_GPU_DEVICE = None


def compute_ES(target: str, prediction: str) -> float:
    """
    Calculate edit similarity score.
    
    Args:
        target: Target/reference text
        prediction: Predicted text
        
    Returns:
        Edit similarity score (0-100)
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("editdistance not installed, cannot calculate ES")

    target_lines = [line.strip()
                    for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    prediction_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ][: len(target_lines)]
    prediction_str = '\n'.join(prediction_lines)

    if not target_str and not prediction_str:
        return 100.0
    if not target_str or not prediction_str:
        return 0.0

    return (1 - (editdistance.eval(target_str, prediction_str) /
                 max(len(target_str), len(prediction_str)))) * 100


def compute_EM(target: str, prediction: str) -> float:
    """
    Calculate exact match score.
    
    Args:
        target: Target/reference text
        prediction: Predicted text
        
    Returns:
        Exact match score (0 or 100)
    """
    target_lines = [line.strip()
                    for line in target.splitlines() if line.strip()]
    prediction_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ][: len(target_lines)]

    if len(target_lines) != len(prediction_lines):
        return 0.0
    return (int(target_lines == prediction_lines)) * 100


def find_last_func_or_class_start(code_string: str) -> Optional[int]:
    """
    Find the start line of the last function or class definition (1-based).
    
    Args:
        code_string: Source code string
        
    Returns:
        1-based line number of the start of last function/class, or None if not found
    """
    lines = code_string.splitlines()
    if not lines:
        return None

    last_def_line_index = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped_line = lines[i].lstrip()
        if re.match(r'^(def|async\s+def|class)\s+', stripped_line):
            last_def_line_index = i
            break

    if last_def_line_index != -1:
        start_line_index = last_def_line_index
        # Include decorators
        for i in range(last_def_line_index - 1, -1, -1):
            stripped_line = lines[i].lstrip()
            if stripped_line.startswith('@'):
                start_line_index = i
            elif stripped_line == '' or stripped_line.startswith('#'):
                continue
            else:
                break
        return start_line_index + 1

    return None


def split_context_ast(code_string: str) -> Tuple[str, str]:
    """
    Split code context into background and current function/class context.
    
    Args:
        code_string: Source code string
        
    Returns:
        Tuple of (background_code, current_function_code)
    """
    lines = code_string.splitlines()
    split_line_1_based = find_last_func_or_class_start(code_string)

    if split_line_1_based is not None and split_line_1_based > 0:
        background_lines = lines[:split_line_1_based - 1]
        current_func_lines = lines[split_line_1_based - 1:]
        return '\n'.join(background_lines), '\n'.join(current_func_lines)
    else:
        return "", code_string


def load_completion_data(
    path: str = "microsoft/LCC_python",
    split: str = "test",
    num_examples: int = 100,
    filter_current_lines_max: int = 50,
    filter_background_tokens_min: int = 3000,
    test_single: bool = False,
    language: Optional[str] = None,
):
    """
    Load code completion dataset.

    Args:
        test_single: If True, only load small amount of data for filtering (for testing)

    Returns:
        dataset: Filtered dataset
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets not installed, cannot load data")

    print(f"Loading dataset: {path} ({split} split)...")
    dataset = datasets.load_dataset(path, split=split)
    if test_single:
        dataset = dataset.select(range(min(50, len(dataset))))
    else:
        dataset = dataset.select(range(min(num_examples * 10, len(dataset))))
    original_size = len(dataset)

    # Initialize tokenizer
    if TRANSFORMERS_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct")
    else:
        raise RuntimeError("transformers error")

    def add_split_context(example):
        code = example['context']
        lang = (language or "python").lower()
        if lang == "python":
            background, current_func = split_context_ast(code)
        else:
            chunks = split_code_by_functions_standalone(code, lang)
            if chunks:
                current_chunk = chunks[-1]
                pos = code.rfind(current_chunk)
                if pos != -1:
                    background = code[:pos]
                    current_func = code[pos:pos + len(current_chunk)]
                else:
                    background, current_func = "", code
            else:
                background, current_func = "", code
        example['background_context'] = background
        example['current_function_context'] = current_func
        return example

    processed_dataset = dataset.map(add_split_context, num_proc=4)

    # Filter dataset
    filtered_dataset_list = []
    print(
        f"Filtering dataset: keeping examples with current_func lines <= {filter_current_lines_max} and background tokens >= {filter_background_tokens_min}...")

    from tqdm import tqdm
    for example in tqdm(processed_dataset):
        curr_ctx = example['current_function_context']
        bg_ctx = example['background_context']

        curr_line_count = len(curr_ctx.splitlines())

        bg_token_count = 0
        if bg_ctx and bg_ctx.strip():
            bg_token_count = len(tokenizer.encode(
                bg_ctx, add_special_tokens=False))

        if curr_line_count <= filter_current_lines_max and bg_token_count >= filter_background_tokens_min:
            filtered_dataset_list.append(example)

    filtered_dataset = datasets.Dataset.from_list(filtered_dataset_list)
    if len(filtered_dataset) >= num_examples:
        selected_dataset = filtered_dataset.select(range(num_examples))
    else:
        selected_dataset = processed_dataset.select(range(min(num_examples, len(processed_dataset))))

    print(
        f"Filtering complete. Original size: {original_size}, Filtered size: {len(filtered_dataset)}, Kept: {len(selected_dataset)} examples")

    return selected_dataset


# ========== RAG Related Functions ==========

def chunk_sliding_window(code: str, window_size: int, overlap: int) -> List[str]:
    """Split code into overlapping chunks using sliding window."""
    lines = code.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("Overlap size must be smaller than window size.")

    while True:
        end = min(start + window_size, len(lines))
        chunk_lines = lines[start:end]
        if not chunk_lines:
            break
        chunks.append("\n".join(chunk_lines))
        if end == len(lines):
            break
        next_start = start + stride
        if next_start >= len(lines):
            final_start = max(0, len(lines) - window_size)
            if final_start > start:
                final_chunk_lines = lines[final_start:]
                chunks.append("\n".join(final_chunk_lines))
            break
        start = next_start

    if not chunks and lines:
        return ["\n".join(lines)]

    # Deduplicate while maintaining order
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks


def compute_embedding(text: str, model, tokenizer, device) -> torch.Tensor:
    """Calculate text embedding vector (using mean pooling)."""
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("torch and transformers must be installed to use RAG functionality")

    if not text.strip():
        # Get model device (supports multi-GPU cases)
        model_device = next(model.parameters()).device
        return torch.zeros(model.config.hidden_size).to(model_device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)

    # Get model device (supports multi-GPU cases)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling on last hidden layer
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, window_size: int, overlap: int, top_k: int) -> str:
    """Use RAG to retrieve code blocks most relevant to query code."""
    if not background_code.strip():
        return ""

    chunks = chunk_sliding_window(background_code, window_size, overlap)
    if not chunks:
        return ""

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(
                chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Calculate cosine similarity
    similarities = torch.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Get top_k indices
    top_k_indices = torch.topk(similarities, k=min(
        top_k, len(valid_chunks)), dim=0).indices

    # Retrieve relevant chunks and sort by original position
    retrieved_chunk_contents = [valid_chunks[i]
                                for i in top_k_indices.tolist()]

    # Find original start lines to sort chronologically (approximate)
    chunk_start_lines = {}
    lines = background_code.splitlines()
    chunk_map_from_sliding = chunk_sliding_window(
        background_code, window_size, overlap)
    start_line_num = 0
    stride = window_size - overlap
    for i, chunk_content in enumerate(chunk_map_from_sliding):
        chunk_start_lines[chunk_content] = start_line_num
        start_line_num += stride

    sorted_relevant_chunks = sorted(
        retrieved_chunk_contents,
        key=lambda content: chunk_start_lines.get(content, float('inf'))
    )

    # Combine relevant chunks
    combined_code = "\n\n".join(sorted_relevant_chunks)

    return combined_code


# Add function-level code splitting functionality
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
    # Define regex patterns for different languages
    patterns = {
        # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
        "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
        # C++: Improved to better handle multi-line declarations
        "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Java: Improved for multi-line method declarations
        "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # TypeScript: Enhanced to handle multi-line methods and arrow functions
        "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Rust: Improved for multi-line function declarations
        "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Go: Improved for multi-line function declarations
        "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
    }
    
    # Use default Python pattern if language not supported
    if language.lower() not in patterns:
        language = "python"
    
    function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
    matches = list(function_pattern.finditer(code))
    
    if not matches:
        return [code] if code.strip() else []  # No matches, return whole code if not empty
        
    result_chunks = []
    
    # Add code before first match if exists
    if matches[0].start() > 0:
        pre_code = code[:matches[0].start()].strip()
        if pre_code:
            result_chunks.append(pre_code)
    
    # Process each match
    for i, match in enumerate(matches):
        start = match.start()
        
        # End is either start of next match or end of code
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(code)
        
        chunk = code[start:end].strip()
        if chunk:
            result_chunks.append(chunk)
    
    return result_chunks


def function_rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, language: str, top_k: int) -> str:
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
    if not background_code.strip():
        return ""  # Return empty if no background context

    # Split code using function-level chunking
    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""  # Return empty if chunking result is empty

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Calculate cosine similarity
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Get top_k indices
    top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_chunks)), dim=0).indices

    # Retrieve relevant chunks
    retrieved_chunks = [valid_chunks[i] for i in top_k_indices.tolist()]

    # Combine relevant chunks (sorted by similarity score)
    combined_code = "\n\n".join(retrieved_chunks)

    return combined_code


def load_code_completion_rag_results(
    result_dir: str,
    mode: str
) -> Dict:
    """
    Load results for specified mode from code_completion_rag result directory.
    """
    if not os.path.exists(result_dir):
        return None

    results = []

    # Determine file suffix based on mode
    if mode == 'text_only':
        file_pattern = '*_text_only.jsonl'
    elif mode == 'image':
        file_pattern = '*_original.jsonl'
    elif mode.startswith('image_ratio'):
        # Extract compression ratio
        ratio_str = mode.replace('image_ratio', '')
        file_pattern = f'*_ratio{ratio_str}.jsonl'
    else:
        return None

    # Find matching files
    import glob
    result_files = glob.glob(os.path.join(result_dir, file_pattern))

    if not result_files:
        return None

    # Read all result files
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        if 'es' in result and 'em' in result:
                            results.append({
                                'es': result.get('es', 0.0),
                                'em': result.get('em', 0.0)
                            })
        except Exception as e:
            print(f"  Warning: Error reading result file {result_file}: {e}")
            continue

    if not results:
        return None

    # Calculate average ES and EM
    avg_es = sum(r['es'] for r in results) / len(results) if results else 0.0
    avg_em = sum(r['em'] for r in results) / len(results) if results else 0.0

    return {
        'average_es': avg_es,
        'average_em': avg_em,
        'num_examples': len(results),
        'mode': mode
    }

from PIL import Image as PIL_Image


def extract_code(model_output: str):
    """
    Extract code from model output, handling various edge cases.
    
    Args:
        model_output: Raw model output
        
    Returns:
        Extracted code or original output (if extraction fails)
    """
    try:
        # Handle empty output
        if not model_output or not model_output.strip():
            return ""
            
        outputlines = model_output.split("\n")
        
        # Find all lines containing ```
        if "```" in model_output:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
            
            # Need at least two markers to extract code block
            if len(indexlines) >= 2:
                # Use last two markers to extract code block
                start_idx = indexlines[-2] + 1
                end_idx = indexlines[-1]
                
                # Boundary check, ensure indices are valid
                if (start_idx < len(outputlines) and 
                    end_idx <= len(outputlines) and 
                    start_idx < end_idx):
                    extracted = "\n".join(outputlines[start_idx:end_idx]).strip()
                    return extracted if extracted else model_output.strip()
                    
            # If standard method fails, try regex method
            import re
            code_match = re.search(r"```(?:python|Python)?\s*\n(.*?)\n?```", model_output, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
                
        # Compatible with special wrapper markers
        if "<|begin_of_box|>" in model_output and "<|end_of_box|>" in model_output:
            import re
            m = re.search(r"<\|begin_of_box\|>([\s\S]*?)<\|end_of_box\|>", model_output)
            if m:
                return m.group(1).strip()
        # If no code block markers found, return cleaned output
        return model_output.strip()
        
    except Exception as e:
        # On any error, return original output
        print(f"Warning: extract_code failed with error: {e}")
        return model_output.strip() if model_output else ""


def run_code_completion_rag(
    model_name: str,
    config_name: str,
    width: int,
    height: int,
    font_size: int,
    line_height: float = 1.2,
    dpi: int = 300,
    font_path: str = None,
    output_dir: str = "./llm_outputs",
    processor=None,
    qwen_tokenizer=None,
    embed_model=None,
    embed_tokenizer=None,
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    dataset_path: str = "microsoft/LCC_python",
    dataset_split: str = "test",
    num_examples: int = 100,
    filter_current_lines_max: int = 50,
    filter_background_tokens_min: int = 3000,
    max_new_tokens: int = 128,
    test_single: bool = False,
    preserve_newlines: bool = False,
    enable_syntax_highlight: bool = False,
    language: str = None,
    should_crop_whitespace: bool = False,
    enable_two_column: bool = False,
    resize_mode: bool = False,
    client_type: str = "OpenAI",
    rag_mode: str = "function_rag",  # Add rag_mode parameter, default is function_rag
    no_context_mode: bool = False,
    enable_bold: bool = False,
    theme: str = "",
    extreme_mode: bool = False,
) -> Tuple[Dict, List[Dict], str]:
    """
    Run code_completion_rag task.
    
    Args:
        model_name: Model name
        config_name: Configuration name
        width: Image width
        height: Image height
        font_size: Font size
        line_height: Line height
        dpi: DPI
        font_path: Font path
        output_dir: Output directory
        processor: Qwen processor
        qwen_tokenizer: Qwen tokenizer
        embed_model: Embedding model for RAG
        embed_tokenizer: Embedding tokenizer
        rag_window_size: RAG sliding window size
        rag_overlap: RAG overlap size
        rag_top_k: RAG top-k retrieval
        dataset_path: Dataset path
        dataset_split: Dataset split
        num_examples: Number of examples
        filter_current_lines_max: Maximum lines in current function
        filter_background_tokens_min: Minimum tokens in background
        max_new_tokens: Maximum new tokens
        test_single: Test single example
        preserve_newlines: Preserve newlines
        enable_syntax_highlight: Enable syntax highlighting
        language: Programming language
        should_crop_whitespace: Crop whitespace
        enable_two_column: Enable two-column layout
        resize_mode: Enable resize mode
        client_type: Client type
        rag_mode: RAG mode (function_rag or sliding_window)
        no_context_mode: No context mode
        enable_bold: Enable bold
        theme: Theme
        
    Returns:
        Tuple of (results_dict, token_stats, result_dir)
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError(
            "datasets or editdistance not installed, cannot run code_completion_rag task")

    if not embed_model or not embed_tokenizer:
        raise ValueError("code_completion_rag task requires embedding model, but not initialized")

    if not TORCH_AVAILABLE:
        raise RuntimeError("code_completion_rag task requires torch, but not installed")

    print("=" * 60)
    print(f"Code Completion RAG Task (mode: {rag_mode})")
    print("=" * 60)

    print("\nStep 1: Loading code completion dataset...")
    print("-" * 60)
    dataset = load_completion_data(
        path=dataset_path,
        split=dataset_split,
        num_examples=num_examples,
        filter_current_lines_max=filter_current_lines_max,
        filter_background_tokens_min=filter_background_tokens_min,
        test_single=test_single,
        language=language
    )
    print(f"✓ Successfully loaded {len(dataset)} examples")

    client = create_client(client_type, **get_config())

    global RAG_GPU_DEVICE
    if RAG_GPU_DEVICE is None:
        # Use improved device detection method, supports Apple MPS
        RAG_GPU_DEVICE = get_appropriate_device()
    device = RAG_GPU_DEVICE

    os.makedirs(output_dir, exist_ok=True)
    lh_str = str(line_height).replace('.', '_')
    folder_parts = [
        f"code_completion_rag_{model_name.replace('/', '_slash_')}",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}"
    ]
    folder_kwargs = {
        "enable_syntax_highlight": enable_syntax_highlight,
        "preserve_newlines": preserve_newlines,
        "enable_bold": enable_bold,
    }
    result_dir = build_folder(
        output_dir,
        folder_parts,
        **folder_kwargs
    )
    os.makedirs(result_dir, exist_ok=True)

    all_results = []
    all_ratios = ["text_only", "image"] + [
        f"image_ratio{r}" for r in COMPRESSION_RATIOS
    ]
    all_token_stats = {
        "total": 0,
        "empty": {r: 0 for r in all_ratios},
        "prompt_tokens": {r: 0 for r in all_ratios},
        "completion_tokens": {r: 0 for r in all_ratios},
        "total_tokens": {r: 0 for r in all_ratios},
    }
    total_es = 0.0
    total_em = 0.0
    valid_scores = 0
    results_lock = Lock()
    stats_lock = Lock()

    if test_single:
        dataset = dataset.select(range(1))
        print(f"\nStep 2: Test mode - processing only 1 example...")
    else:
        print(f"\nStep 2: Processing {len(dataset)} examples (concurrency: 20)...")
    print("-" * 60)

    images_base_dir = "./generated_images/comp_rag"
    images_folder_parts = [
        "java" if language == "java" else "python",
        f"{width}x{height}",
        f"font{font_size}",
        f"lh{lh_str}"
    ]
    if theme:
        images_folder_parts.append(theme)
    images_dir = build_folder(
        images_base_dir,
        images_folder_parts,
        **folder_kwargs
    )
    os.makedirs(images_dir, exist_ok=True)

    def process_single_example(example_data):
        nonlocal total_es, total_em, valid_scores
        i, example = example_data
        try:

            def count_tk(_task, _tk):
                all_token_stats["total"] += 1
                all_token_stats["empty"][_task] += (
                    1 if not _tk.get("completion_tokens", 0) else 0
                )
                all_token_stats["prompt_tokens"][_task] += _tk.get("prompt_tokens", 0)
                all_token_stats["completion_tokens"][_task] += _tk.get(
                    "completion_tokens", 0
                )
                all_token_stats["total_tokens"][_task] += _tk.get("total_tokens", 0)

            background_ctx = example['background_context']
            current_func_ctx = example['current_function_context']
            ground_truth = example['gt']
            example_id = example.get('id', i)

            if not no_context_mode:
                # Choose different retrieval methods based on rag_mode
                if rag_mode == "function_rag":
                    # Use function-level RAG retrieval
                    retrieved_context = function_rag_retrieve(
                        background_ctx,
                        current_func_ctx,
                        embed_model,
                        embed_tokenizer,
                        device,
                        language or "python",
                        rag_top_k,
                    )
                else:
                    # Use original sliding window RAG retrieval
                    retrieved_context = rag_retrieve(
                        background_ctx,
                        current_func_ctx,
                        embed_model,
                        embed_tokenizer,
                        device,
                        rag_window_size,
                        rag_overlap,
                        rag_top_k,
                    )
            else:
                retrieved_context = ""

            if not no_context_mode and not retrieved_context.strip():
                print(f"  Example {example_id}: Warning - Retrieved context is empty")
                return example_id, True, None

            user_prompt = f"Next line for the following code:\n{current_func_ctx}"
            system_prompt = "You are a code completion assistant. You must output ONLY the next single line of code that logically follows the given context in images. Do NOT output more than one line. Do NOT output explanations, comments, markdown, or repeat the input. Preserve indentation exactly. Your entire response must contain exactly one single line of code."

            print(f"  Example {example_id}: Testing text-only version...")
            user_prompt_text_only = f"Next line for the following code:\n{current_func_ctx}\n\nContext:\n```python\n{retrieved_context}\n```"
            system_prompt_text_only = "You are a code completion assistant. You must output ONLY the next single line of code that logically follows the given context. Do NOT output more than one line. Do NOT output explanations, comments, markdown, or repeat the input. Preserve indentation exactly. Your entire response must contain exactly one single line of code."

            # thread_client_text = create_client(client_type)
            response_text_text_only, token_info_text_only = call_llm_with_text_only(
                client, model_name, system_prompt_text_only, user_prompt_text_only
            )

            es_text_only = 0.0
            em_text_only = 0.0
            cleaned_out = ""
            if response_text_text_only and ground_truth:
                try:
                    cleaned_out = extract_code(response_text_text_only)
                    es_text_only = compute_ES(ground_truth, cleaned_out)
                    em_text_only = compute_EM(ground_truth, cleaned_out)
                    with stats_lock:
                        total_es += es_text_only
                        total_em += em_text_only
                        valid_scores += 1
                except Exception as e:
                    print(f"  Example {example_id}: Text evaluation error: {e}")

            prompt_text_text_only = system_prompt_text_only + "\n" + user_prompt_text_only
            prompt_text_tokens_text_only = get_text_tokens(
                prompt_text_text_only)
            background_tokens = get_text_tokens(background_ctx)
            current_func_tokens = get_text_tokens(current_func_ctx)
            completion_tokens_text_only = get_text_tokens(
                response_text_text_only)
            api_prompt_tokens_text_only = token_info_text_only['prompt_tokens']

            background_tokens_qwen = get_text_tokens_qwen(
                background_ctx, qwen_tokenizer)
            current_func_tokens_qwen = get_text_tokens_qwen(
                current_func_ctx, qwen_tokenizer)
            retrieved_tokens_qwen = get_text_tokens_qwen(
                retrieved_context, qwen_tokenizer)

            result_text_only = {
                'id': example_id,
                'gt': ground_truth,
                'output': response_text_text_only,
                'cleaned_output': cleaned_out,  # Add extracted code
                'api_kwargs': token_info_text_only.get('api_kwargs', {}),
                'es': es_text_only,
                'em': em_text_only,
                'background_context': background_ctx,
                'current_function_context': current_func_ctx,
                'retrieved_context': retrieved_context,
                'prompt': user_prompt_text_only,
                'tokens': token_info_text_only,
                'compression_ratio': None,
                'resolution': 'text_only',
                'mode': 'text_only'
            }

            result_file_text_only = os.path.join(
                result_dir, f"example_{example_id:05d}_text_only.jsonl")
            with open(result_file_text_only, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result_text_only, ensure_ascii=False) + "\n")

            token_stat_text_only = {
                'model': model_name,
                'config': config_name,
                'example_id': example_id,
                'task': 'code_completion_rag',
                'mode': 'text_only',
                'image_tokens': 0,
                'image_tokens_processor': None,
                'background_tokens': background_tokens,
                'background_tokens_qwen': background_tokens_qwen,
                'retrieved_tokens_qwen': retrieved_tokens_qwen,
                'current_func_tokens': current_func_tokens,
                'current_func_tokens_qwen': current_func_tokens_qwen,
                'prompt_text_tokens': prompt_text_tokens_text_only,
                'completion_tokens': completion_tokens_text_only,
                'api_prompt_tokens': api_prompt_tokens_text_only,
                'api_image_tokens_estimate': 0,
                'total_tokens': token_info_text_only['total_tokens'],
                'num_images': 0,
                'es': es_text_only,
                'em': em_text_only,
                'compression_ratio': None,
                'resolution': 'text_only'
            }

            with results_lock:
                all_results.append(result_text_only)
            with stats_lock:
                count_tk("text_only", token_info_text_only)

            if no_context_mode:
                return example_id, True, None

            def test_with_images(image_paths, compression_ratio=None, resolution=None):
                # thread_client = create_client(client_type)

                if image_paths:
                    response_text, token_info = call_llm_with_images(
                        client, model_name, image_paths, system_prompt, user_prompt
                    )
                else:
                    response_text, token_info = call_llm_with_images(
                        client, model_name, [], system_prompt, user_prompt
                    )

                es = 0.0
                em = 0.0
                cleaned_response = ""
                if response_text and ground_truth:
                    try:
                        # Extract code using extract_code before evaluation
                        cleaned_response = extract_code(response_text)
                        es = compute_ES(ground_truth, cleaned_response)
                        em = compute_EM(ground_truth, cleaned_response)
                    except Exception as e:
                        print(f"  Example {example_id}: Evaluation error: {e}")

                prompt_text = user_prompt
                prompt_text_tokens = get_text_tokens(prompt_text)
                if image_paths:
                    image_tokens = calculate_image_tokens_from_paths(
                        image_paths)
                    image_tokens_processor = calculate_image_tokens_with_processor(
                        image_paths, processor)
                else:
                    image_tokens = 0
                    image_tokens_processor = None

                background_tokens = get_text_tokens(background_ctx)
                current_func_tokens = get_text_tokens(current_func_ctx)
                completion_tokens = get_text_tokens(response_text)
                api_prompt_tokens = token_info['prompt_tokens']
                api_image_tokens_estimate = max(
                    0, api_prompt_tokens - prompt_text_tokens)

                background_tokens_qwen = get_text_tokens_qwen(
                    background_ctx, qwen_tokenizer)
                current_func_tokens_qwen = get_text_tokens_qwen(
                    current_func_ctx, qwen_tokenizer)

                return {
                    "response_text": response_text,
                    "token_info": token_info,
                    "es": es,
                    "em": em,
                    "cleaned_response": cleaned_response,
                    "api_kwargs": token_info.get('api_kwargs', {}),
                    "image_tokens": image_tokens,
                    "image_tokens_processor": image_tokens_processor,
                    "background_tokens": background_tokens,
                    "background_tokens_qwen": background_tokens_qwen,
                    "current_func_tokens": current_func_tokens,
                    "current_func_tokens_qwen": current_func_tokens_qwen,
                    "prompt_text_tokens": prompt_text_tokens,
                    "completion_tokens": completion_tokens,
                    "api_prompt_tokens": api_prompt_tokens,
                    "api_image_tokens_estimate": api_image_tokens_estimate,
                    "compression_ratio": compression_ratio,
                    "resolution": resolution,
                    "mode": (
                        "image"
                        if compression_ratio is None
                        else f"image_ratio{compression_ratio}"
                    ),
                }

            if resize_mode:
                # Calculate text tokens
                if qwen_tokenizer is not None:
                    text_tokens = get_text_tokens_qwen(
                        retrieved_context, qwen_tokenizer)
                    if text_tokens is None:
                        text_tokens = get_text_tokens(retrieved_context)
                else:
                    text_tokens = get_text_tokens(retrieved_context)

                # Choose compression method based on extreme_mode
                if extreme_mode:
                    # Extreme mode: dynamically re-render at different resolutions
                    text_structure = analyze_text_structure(retrieved_context)
                    
                    def renderer_func(w, h, fs):
                        margin = int(w * 0.01)
                        return text_to_image(
                            retrieved_context,
                            width=w,
                            height=h,
                            font_size=fs,
                            line_height=line_height,
                            margin_px=margin,
                            dpi=dpi,
                            font_path=font_path,
                            preserve_newlines=preserve_newlines,
                            enable_syntax_highlight=enable_syntax_highlight,
                            filename=None,
                            language=language or "python",
                            should_crop_whitespace=should_crop_whitespace,
                            enable_two_column=enable_two_column,
                            enable_bold=enable_bold,
                            theme=theme,
                        )
                    
                    resized_results = generate_compressed_images_dynamic(
                        text_tokens,
                        renderer_func,
                        compression_ratios=COMPRESSION_RATIOS,
                        text_structure=text_structure,
                        data_id=f"example_{example_id}",
                    )
                    
                    for compression_ratio in COMPRESSION_RATIOS:
                        resized_images, target_resolution, target_font_size = resized_results.get(
                            compression_ratio, ([], None, None)
                        )
                        if not resized_images:
                            continue
                        
                        resized_image_paths = []
                        for page_num, resized_img in enumerate(resized_images, 1):
                            image_filename = f"example_{example_id:05d}_ratio{compression_ratio}_{target_resolution}x{target_resolution}_fs{target_font_size}_page_{page_num:03d}.png"
                            image_path = os.path.join(images_dir, image_filename)
                            resized_img.save(image_path)
                            resized_image_paths.append(os.path.abspath(image_path))
                        
                        compressed_result = test_with_images(
                            resized_image_paths,
                            compression_ratio=compression_ratio,
                            resolution=f"{target_resolution}x{target_resolution}")
                        
                        with stats_lock:
                            total_es += compressed_result['es']
                            total_em += compressed_result['em']
                            valid_scores += 1
                        
                        # Save results (rest of the code remains the same)
                        compressed_result_data = {
                            "id": example_id,
                            "gt": ground_truth,
                            "output": compressed_result["response_text"],
                            "cleaned_response": compressed_result["cleaned_response"],
                            "es": compressed_result["es"],
                            "em": compressed_result["em"],
                            "background_context": background_ctx,
                            "current_function_context": current_func_ctx,
                            "retrieved_context": retrieved_context,
                            "prompt": user_prompt,
                            "tokens": compressed_result["token_info"],
                            "compression_ratio": compression_ratio,
                            "resolution": f"{target_resolution}x{target_resolution}",
                            "mode": f"image_ratio{compression_ratio}",
                        }
                        
                        compressed_result_file = os.path.join(
                            result_dir, f"example_{example_id:05d}_ratio{compression_ratio}.jsonl")
                        with open(compressed_result_file, 'w', encoding='utf-8') as f:
                            f.write(json.dumps(compressed_result_data, ensure_ascii=False) + "\n")
                        
                        compressed_token_stat = {
                            'model': model_name,
                            'config': config_name,
                            'example_id': example_id,
                            'task': 'code_completion_rag',
                            'mode': f'image_ratio{compression_ratio}',
                            'image_tokens': compressed_result['image_tokens'],
                            'image_tokens_processor': compressed_result['image_tokens_processor'],
                            'background_tokens': compressed_result['background_tokens'],
                            'background_tokens_qwen': compressed_result['background_tokens_qwen'],
                            'current_func_tokens': compressed_result['current_func_tokens'],
                            'current_func_tokens_qwen': compressed_result['current_func_tokens_qwen'],
                            'prompt_text_tokens': compressed_result['prompt_text_tokens'],
                            'completion_tokens': compressed_result['completion_tokens'],
                            'api_prompt_tokens': compressed_result['api_prompt_tokens'],
                            'api_image_tokens_estimate': compressed_result['api_image_tokens_estimate'],
                            'total_tokens': compressed_result['token_info']['total_tokens'],
                            'num_images': len(resized_image_paths),
                            'es': compressed_result['es'],
                            'em': compressed_result['em'],
                            'compression_ratio': compression_ratio,
                            'resolution': f"{target_resolution}x{target_resolution}"
                        }
                        
                        with results_lock:
                            all_results.append(compressed_result_data)
                        with stats_lock:
                            count_tk(
                                f"image_ratio{compression_ratio}",
                                compressed_result["token_info"],
                            )
                else:
                    # Standard mode: find 1x config, then resize to each ratio
                    text_structure = analyze_text_structure(retrieved_context)
                    res_1x, fs_1x, _ = optimize_layout_config_dry(
                        target_tokens=text_tokens,
                        previous_configs=[],
                        text_tokens=text_tokens,
                        line_height=line_height,
                        text_structure=text_structure,
                        compression_ratio=1.0,
                        page_limit=100,
                        text=retrieved_context,
                        enable_syntax_highlight=enable_syntax_highlight,
                        language=language or "python",
                        preserve_newlines=preserve_newlines,
                        font_path=font_path,
                        theme=theme,
                    )
                    margin_1x = int(res_1x * 0.01)
                    base_images = []
                    base_paths = []
                    import time
                    start_time = time.time()
                    page_idx = 0
                    for img in text_to_image_stream(
                        retrieved_context,
                        width=res_1x,
                        height=res_1x,
                        font_size=fs_1x,
                        line_height=line_height,
                        margin_px=margin_1x,
                        dpi=dpi,
                        font_path=font_path,
                        preserve_newlines=preserve_newlines,
                        enable_syntax_highlight=enable_syntax_highlight,
                        filename=None,
                        language=language or "python",
                        should_crop_whitespace=should_crop_whitespace,
                        enable_two_column=enable_two_column,
                        enable_bold=enable_bold,
                        theme=theme,
                    ):
                        page_idx += 1
                        bp = os.path.join(
                            images_dir,
                            f"example_{example_id:05d}_ratio{1.0}_{res_1x}x{res_1x}_fs{fs_1x}_page_{page_idx:03d}.png",
                        )
                        img.save(bp)
                        base_images.append(img)
                        base_paths.append(bp)
                        fr = calculate_fill_rate(fs_1x, res_1x, 1, text_structure["num_lines"], int(text_structure["avg_line_chars"]), line_height)
                        tk = calculate_image_tokens_qwen3(res_1x, res_1x)
                        tt = text_tokens / 1.0
                        et = time.time() - start_time
                        print(f"[example_{example_id:05d}_1x] Ratio 1.0: Res {res_1x}x{res_1x}, Count {page_idx}, Font {fs_1x}, Fill {int(fr*100)}%, Tokens {tk} (Target {tt:.1f}) [Time: {et:.3f}s]")
                    resolution_list = get_expanded_resolution_list()
                    for compression_ratio in COMPRESSION_RATIOS:
                        if float(compression_ratio) == 0.0:
                            os.makedirs("./generated_images", exist_ok=True)
                            blank_path = os.path.join("./generated_images", "blank_14x14.png")
                            if not os.path.exists(blank_path):
                                PIL_Image.new("RGB", (14, 14), color="white").save(blank_path)
                            resized_image_paths = [os.path.abspath(blank_path)]
                            compressed_result = test_with_images(
                                resized_image_paths,
                                compression_ratio=compression_ratio,
                                resolution="14x14")
                            with stats_lock:
                                total_es += compressed_result['es']
                                total_em += compressed_result['em']
                                valid_scores += 1
                            compressed_result_data = {
                                "id": example_id,
                                "gt": ground_truth,
                                "output": compressed_result["response_text"],
                                "cleaned_response": compressed_result["cleaned_response"],
                                "es": compressed_result["es"],
                                "em": compressed_result["em"],
                                "background_context": background_ctx,
                                "current_function_context": current_func_ctx,
                                "retrieved_context": retrieved_context,
                                "prompt": user_prompt,
                                "tokens": compressed_result["token_info"],
                                "compression_ratio": compression_ratio,
                                "resolution": "14x14",
                                "mode": f"image_ratio{compression_ratio}",
                            }
                            compressed_result_file = os.path.join(
                                result_dir, f"example_{example_id:05d}_ratio{compression_ratio}.jsonl")
                            with open(compressed_result_file, 'w', encoding='utf-8') as f:
                                f.write(json.dumps(compressed_result_data, ensure_ascii=False) + "\n")
                            with results_lock:
                                all_results.append(compressed_result_data)
                            with stats_lock:
                                count_tk(
                                    f"image_ratio{compression_ratio}",
                                    compressed_result["token_info"],
                                )
                            continue
                        image_token_limit = text_tokens / compression_ratio
                        num_images = len(base_images)
                        per_image_tokens = image_token_limit / num_images if num_images > 0 else image_token_limit
                        target_res = find_closest_resolution_prefer_larger(per_image_tokens, resolution_list, tolerance_ratio=1.4)
                        fs_scaled = int(fs_1x * (target_res / res_1x)) if res_1x > 0 else fs_1x
                        resized_image_paths = []
                        start_time_ratio = time.time()
                        for i, bp in enumerate(base_paths, 1):
                            try:
                                with PIL_Image.open(bp) as im:
                                    resized_img = im.resize((target_res, target_res), PIL_Image.Resampling.LANCZOS)
                                rp = os.path.join(images_dir, f"example_{example_id:05d}_ratio{compression_ratio}_{target_res}x{target_res}_fs{fs_scaled}_page_{i:03d}.png")
                                resized_img.save(rp)
                                resized_image_paths.append(os.path.abspath(rp))
                                fr = calculate_fill_rate(fs_scaled, target_res, 1, text_structure["num_lines"], int(text_structure["avg_line_chars"]), line_height)
                                tk = calculate_image_tokens_qwen3(target_res, target_res)
                                tt = text_tokens / compression_ratio
                                et = time.time() - start_time_ratio
                                print(f"[example_{example_id:05d}_ratio{compression_ratio}] Ratio {compression_ratio}: Res {target_res}x{target_res}, Count {i}, Font {fs_scaled}, Fill {int(fr*100)}%, Tokens {tk} (Target {tt:.1f}) [Time: {et:.3f}s]")
                            except Exception:
                                continue
                        compressed_result = test_with_images(
                            resized_image_paths,
                            compression_ratio=compression_ratio,
                            resolution=f"{target_res}x{target_res}")
                        with stats_lock:
                            total_es += compressed_result['es']
                            total_em += compressed_result['em']
                            valid_scores += 1
                        compressed_result_data = {
                            "id": example_id,
                            "gt": ground_truth,
                            "output": compressed_result["response_text"],
                            "cleaned_response": compressed_result["cleaned_response"],
                            "es": compressed_result["es"],
                            "em": compressed_result["em"],
                            "background_context": background_ctx,
                            "current_function_context": current_func_ctx,
                            "retrieved_context": retrieved_context,
                            "prompt": user_prompt,
                            "tokens": compressed_result["token_info"],
                            "compression_ratio": compression_ratio,
                            "resolution": f"{target_res}x{target_res}",
                            "mode": f"image_ratio{compression_ratio}",
                        }
                        compressed_result_file = os.path.join(
                            result_dir, f"example_{example_id:05d}_ratio{compression_ratio}.jsonl")
                        with open(compressed_result_file, 'w', encoding='utf-8') as f:
                            f.write(json.dumps(compressed_result_data, ensure_ascii=False) + "\n")
                        with results_lock:
                            all_results.append(compressed_result_data)
                        with stats_lock:
                            count_tk(
                                f"image_ratio{compression_ratio}",
                                compressed_result["token_info"],
                            )

            return example_id, True, None

        except Exception as e:
            error_msg = f"Example {i}: Processing failed: {e}"
            print(f"  {error_msg}")
            import traceback
            traceback.print_exc()
            return i, False, str(e)

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_example = {
            executor.submit(process_single_example, (i, example)): (i, example)
            for i, example in enumerate(dataset)
        }

        for future in tqdm(as_completed(future_to_example), total=len(future_to_example), desc="Processing examples"):
            example_id, success, error = future.result()

    avg_es = (total_es / valid_scores) if valid_scores > 0 else 0.0
    avg_em = (total_em / valid_scores) if valid_scores > 0 else 0.0

    total_compression_ratios = []
    total_text_tokens = 0
    total_image_tokens = 0
    valid_compression_count = 0

    # Calculate compression ratio information
    for mode in ["text_only", "image"] + [
        f"image_ratio{r}" for r in COMPRESSION_RATIOS
    ]:
        text_tokens = all_token_stats["prompt_tokens"][mode]
        completion_tokens = all_token_stats["completion_tokens"][mode]

        # Here we assume text_tokens and completion_tokens can represent relevant information
        # In actual application, can adjust according to specific needs
        if text_tokens > 0:
            total_text_tokens += text_tokens
        if completion_tokens > 0:
            total_image_tokens += completion_tokens

        if completion_tokens > 0 and text_tokens > 0:
            compression_ratio = text_tokens / completion_tokens
            total_compression_ratios.append(compression_ratio)
            valid_compression_count += 1

    avg_compression_ratio = (sum(total_compression_ratios) /
                             len(total_compression_ratios)) if total_compression_ratios else 0.0
    overall_compression_ratio = (
        total_text_tokens / total_image_tokens) if total_image_tokens > 0 else 0.0

    scores = {
        "model_name": model_name,
        "method": "code_completion_rag",
        "num_examples_total": len(dataset),
        "num_examples_scored": valid_scores,
        "average_es": avg_es,
        "average_em": avg_em,
        "average_compression_ratio": avg_compression_ratio,
        "overall_compression_ratio": overall_compression_ratio,
        "token_statistics": {
            "total_text_tokens": total_text_tokens,
            "total_image_tokens": total_image_tokens,
            "valid_compression_count": valid_compression_count,
            "all_token_stats": all_token_stats,
        },
        "parameters": {
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "filter_current_lines_max": filter_current_lines_max,
            "filter_background_tokens_min": filter_background_tokens_min,
            "rag_window_size": rag_window_size,
            "rag_overlap": rag_overlap,
            "rag_top_k": rag_top_k,
            "width": width,
            "height": height,
            "font_size": font_size,
            "line_height": line_height,
            "max_new_tokens": max_new_tokens,
        },
    }

    score_file = os.path.join(result_dir, "SCORES.json")
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Processing complete")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Successfully evaluated: {valid_scores}")
    print(f"  Average ES: {avg_es:.2f}")
    print(f"  Average EM: {avg_em:.2f}")
    print(
        f"  Average Token compression ratio: {avg_compression_ratio:.4f} (text_tokens / image_tokens)")
    print(
        f"  Overall Token compression ratio: {overall_compression_ratio:.4f} (total text_tokens / total image_tokens)")
    print(f"  Result directory: {result_dir}")
    print(f"  Score file: {score_file}")

    results_dict = {
        'code_completion_rag': {
            'results': all_results,
            'scores': scores
        }
    }

    return results_dict, all_token_stats, result_dir
