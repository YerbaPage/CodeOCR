import os
import re
from typing import Dict, List, Optional, Tuple


def _get_file_id_sort_key(file_path: str) -> int:
    name = os.path.basename(file_path)
    match = re.search(r"(\d+)(?=\.\w+$)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return float("inf")


def _split_pair(text: str) -> Optional[Tuple[str, str]]:
    lines = text.splitlines()
    parts = []
    current = []
    empty_run = 0
    for line in lines:
        if len(parts) >= 2:
            break
        if line.strip() == "":
            empty_run += 1
        else:
            if empty_run >= 2 and current:
                parts.append("\n".join(current))
                if len(parts) >= 2:
                    break
                current = []
            empty_run = 0
            current.append(line)
    if current and len(parts) < 2:
        parts.append("\n".join(current))
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def load_code_clone_detection_pairs(
    base_dir: str,
    dataset_type: str = "true",
    lang: str = "py",
    difficulty: str = "prompt_2",
    tier: str = "T4",
    num_examples: int = 200,
) -> List[Dict]:
    pairs = []
    if lang is None:
        lang = "py"
    ext = "py" if lang.lower() in ["py", "python"] else lang.lower()
    if dataset_type == "true":
        root = os.path.join(base_dir, "true_semantic_clones", ext, difficulty, tier)
        if not os.path.isdir(root):
            return []
        all_files = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(f".{ext}")
        ]
        files = sorted(all_files, key=_get_file_id_sort_key)

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                split = _split_pair(content)
                if not split:
                    continue
                left, right = split
                pairs.append({"file": fp, "left": left, "right": right, "label": True})
            except Exception:
                continue
    else:
        root = os.path.join(base_dir, "false_semantic_clones", ext)
        if not os.path.isdir(root):
            return []
        all_files = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(f".{ext}")
        ]
        files = sorted(all_files, key=_get_file_id_sort_key)

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                split = _split_pair(content)
                if not split:
                    continue
                left, right = split
                pairs.append({"file": fp, "left": left, "right": right, "label": False})
            except Exception:
                continue
    if len(pairs) > num_examples:
        pairs = pairs[:num_examples]
    return pairs


def build_balanced_dataset(
    base_dir: str,
    lang: str = "py",
    difficulty: str = "prompt_2",
    tier: str = "T4",
    num_total: int = 200,
) -> List[Dict]:
    pos_pairs = load_code_clone_detection_pairs(
        base_dir, "true", lang, difficulty, tier, num_examples=num_total
    )
    positives = pos_pairs[: num_total // 2]
    rev_pairs = list(reversed(pos_pairs))
    neg_count = num_total // 2
    negatives = []
    for i in range(min(len(pos_pairs), len(rev_pairs), neg_count)):
        left_src = pos_pairs[i]
        right_src = rev_pairs[i]
        if left_src["file"] == right_src["file"]:
            j = (i + 1) % len(rev_pairs)
            right_src = rev_pairs[j]
        negatives.append(
            {
                "file": f"{left_src['file']}___vs___{right_src['file']}",
                "left": left_src["left"],
                "right": right_src["right"],
                "label": False,
            }
        )
    dataset = positives + negatives
    import random

    random.seed(42)
    random.shuffle(dataset)
    return dataset
