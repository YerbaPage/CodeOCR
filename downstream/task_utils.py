import ast
import json
import re
from typing import Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("torch and transformers must be installed to use RAG functionality")

    if not text.strip():
        model_device = next(model.parameters()).device
        return torch.zeros(model.config.hidden_size).to(model_device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def split_python_code_by_ast(code: str) -> List[str]:
    """
    Split Python code using AST.

    Args:
        code: Python source code

    Returns:
        List of code chunks (functions and classes)
    """
    try:
        tree = ast.parse(code)
    except:
        return []

    lines = code.splitlines(keepends=True)
    if not lines:
        return []

    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_lineno = node.lineno
            if node.decorator_list:
                start_lineno = min(start_lineno, node.decorator_list[0].lineno)
            nodes.append((start_lineno, node))

    if not nodes:
        return [code]

    nodes.sort(key=lambda x: x[0])

    chunks = []

    first_start = nodes[0][0]
    if first_start > 1:
        chunks.append("".join(lines[:first_start-1]))

    for i in range(len(nodes)):
        start_lineno, _ = nodes[i]

        if i < len(nodes) - 1:
            next_start_lineno, _ = nodes[i+1]
            end_lineno = next_start_lineno - 1
        else:
            end_lineno = len(lines)

        chunk = "".join(lines[start_lineno-1: end_lineno])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def split_code_by_functions_standalone(code: str, language: str = "python") -> List[str]:
    """
    Split code into chunks based on function and class definitions.

    Args:
        code: Code to split
        language: Programming language of the code

    Returns:
        List of code chunks
    """
    if language.lower() == "python":
        try:
            chunks = split_python_code_by_ast(code)
            if chunks:
                return chunks
        except Exception as e:
            print(
                f"Warning: AST parsing failed for Python code, falling back to regex: {e}")

    patterns = {
        "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
        "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
        "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
    }

    if language.lower() not in patterns:
        language = "python"

    function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
    matches = list(function_pattern.finditer(code))

    if not matches:
        return [code] if code.strip() else []

    result_chunks = []

    if matches[0].start() > 0:
        pre_code = code[:matches[0].start()].strip()
        if pre_code:
            result_chunks.append(pre_code)

    for i, match in enumerate(matches):
        start = match.start()
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
        background_code: Background code (repo_text)
        query_code: Query code (question)
        model: Embedding model
        tokenizer: Tokenizer
        device: Compute device
        language: Programming language
        top_k: Number of most similar chunks to return

    Returns:
        Concatenated string of retrieved relevant code blocks
    """
    if not background_code.strip():
        return ""

    chunks = split_code_by_functions_standalone(background_code, language)
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

    chunk_embeddings_tensor = torch.stack(chunk_embeddings)
    similarities = torch.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)
    top_k_indices = torch.topk(similarities, k=min(
        top_k, len(valid_chunks)), dim=0).indices

    retrieved_chunks = [valid_chunks[i] for i in top_k_indices.tolist()]
    combined_code = "\n\n".join(retrieved_chunks)

    return combined_code


def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """
    Extract JSON object from model response.

    Args:
        response_text: Model response text

    Returns:
        Parsed JSON object or None
    """
    if not response_text:
        return None

    cleaned = response_text.strip()

    box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", cleaned, re.DOTALL)
    if box_match:
        cleaned = box_match.group(1).strip()

    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return None
