import json
import os
import re
from typing import Dict, List, Optional


def load_code_qa_data(
    data_dir: str = "./dataset/code_qa",
) -> List[Dict]:
    """
    Load dataset

    Args:
        data_dir:QA data directory

    Returns:
        List of data items, each containing source_file field to identify origin
    """
    file_path = os.path.join(data_dir, "qa_dataset_test_no_comments.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: File format error (not a list): {file_path}")
            return []

        filtered_data = []
        for item in data:
            context = item.get("context", "")
            if context and context.strip():
                filtered_data.append(item)

        print(f"✓ Loaded {len(data)} original entries from {file_path}")
        print(f"✓ After filtering (non-empty context): {len(filtered_data)} entries")

        selected_data = filtered_data[:200]
        print(f"✓ Selected first 200 entries in fixed order: {len(selected_data)} entries")

        converted_data = []
        for item in selected_data:
            options = item.get("options", {})
            options_text = ""
            for key in sorted(options.keys()):
                options_text += f"{key}. {options[key]}\n"

            converted_item = {
                "repo": item.get("repo", ""),
                "prompt": item.get("context", ""),
                "question": f"{item.get('question', '')}\n\n{options_text}",
                "correct_letter": item.get("answer", ""),
                "repo_text": item.get("context", ""),
                "prompt_goal": item.get("question", ""),
                "is_hard": item.get("is_hard", False),
                "source_file": item.get("repo", "unknown"),
            }
            converted_data.append(converted_item)

        return converted_data

    except Exception as e:
        print(f"Warning: Failed to load file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_answer_letter(response: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, D, etc.) from model response.

    Args:
        response: Model response text

    Returns:
        Extracted answer letter (uppercase), or None
    """
    if not response:
        return None

    response = response.strip()

    try:
        if "{" in response and "}" in response:
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                obj = json.loads(json_match.group())
                answer = obj.get("answer", obj.get(
                    "Answer", obj.get("ANSWER", "")))
                if answer and len(answer) == 1 and answer.upper() in "ABCDEFGH":
                    return answer.upper()
    except:
        pass

    if len(response) == 1 and response.upper() in "ABCDEFGH":
        return response.upper()

    patterns = [
        r"[Aa]nswer[\s:]*([A-Ha-h])\b",
        r"[Tt]he answer is[\s:]*([A-Ha-h])\b",
        r"[Cc]orrect answer[\s:]*([A-Ha-h])\b",
        r"\b([A-Ha-h])\s*[\.:\)]\s*$",
        r"^([A-Ha-h])\s*[\.:\)]",
        r"\b([A-Ha-h])\s+is correct",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()

    matches = re.findall(r'\b([A-Ha-h])\b', response)
    if matches:
        return matches[-1].upper()

    return None
