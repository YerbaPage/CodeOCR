import re
from typing import List, Tuple, Optional

from task_utils import split_code_by_functions_standalone
from .constants import DATASETS_AVAILABLE, TRANSFORMERS_AVAILABLE


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

    import datasets
    from transformers import AutoTokenizer

    print(f"Loading dataset: {path} ({split} split)...")
    dataset = datasets.load_dataset(path, split=split)
    if test_single:
        dataset = dataset.select(range(min(50, len(dataset))))
    else:
        dataset = dataset.select(range(min(num_examples * 10, len(dataset))))
    original_size = len(dataset)

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
