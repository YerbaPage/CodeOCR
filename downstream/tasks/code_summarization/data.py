from typing import Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from llm_utils import get_config


def collect_good_context(row):
    return row["relevant_code_context"]


def trim_context(context, tokenizer, max_len):
    tokenized_context = tokenizer.encode(context, max_length=512_000, truncation=True)
    tokenized_context = tokenized_context[:max_len]
    text_tks = len(tokenized_context)
    detokenized_context = tokenizer.decode(tokenized_context)
    return detokenized_context, text_tks


def prepare_code_context(row, max_context_toks, tokenizer):
    context = collect_good_context(row)
    tokenized_context = tokenizer.encode(context, max_length=512_000, truncation=True)
    text_tks = len(tokenized_context)
    return context, text_tks


def load_code_summarization_data() -> Tuple:
    sum_config = get_config()["summarization"]
    dataset = load_dataset("JetBrains-Research/lca-module-summarization")["test"]
    tokenizer = AutoTokenizer.from_pretrained(
        sum_config.get("tokenizer", "meta-llama/Llama-2-7b-chat-hf")
    )
    return dataset, tokenizer, sum_config
