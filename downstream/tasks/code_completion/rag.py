from typing import List

from task_utils import compute_embedding, split_code_by_functions_standalone
from .constants import TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE

try:
    import torch
except ImportError:
    torch = None


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

    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks


def rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, window_size: int, overlap: int, top_k: int) -> str:
    """Use RAG to retrieve code blocks most relevant to query code."""
    if not background_code.strip():
        return ""

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("torch and transformers must be installed to use RAG functionality")

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

    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    similarities = torch.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    top_k_indices = torch.topk(similarities, k=min(
        top_k, len(valid_chunks)), dim=0).indices

    retrieved_chunk_contents = [valid_chunks[i]
                                for i in top_k_indices.tolist()]

    chunk_start_lines = {}
    chunk_map_from_sliding = chunk_sliding_window(
        background_code, window_size, overlap)
    start_line_num = 0
    stride = window_size - overlap
    for chunk_content in chunk_map_from_sliding:
        chunk_start_lines[chunk_content] = start_line_num
        start_line_num += stride

    sorted_relevant_chunks = sorted(
        retrieved_chunk_contents,
        key=lambda content: chunk_start_lines.get(content, float('inf'))
    )

    combined_code = "\n\n".join(sorted_relevant_chunks)

    return combined_code


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
        return ""

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("torch and transformers must be installed to use RAG functionality")

    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_chunks)), dim=0).indices

    retrieved_chunks = [valid_chunks[i] for i in top_k_indices.tolist()]

    combined_code = "\n\n".join(retrieved_chunks)

    return combined_code
