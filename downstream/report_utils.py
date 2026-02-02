import os
import json
from typing import List, Dict
from text_to_image import get_all_modes, COMPRESSION_RATIOS

def get_flat_filename(filename: str) -> str:
    """Convert original filename to flat format."""
    if filename is None:
        return "unknown"
    return filename.replace('/', '_')

def generate_font_size_summary(
    all_results: List[Dict],
    model_name: str,
    filename: str,
    output_dir: str = './evaluation_results',
    code_completion_rag_results: Dict = None
) -> Dict:
    """
    Generate evaluation result summary for different font sizes.
    """
    summary = {
        'model': model_name,
        'filename': filename,
        'font_sizes': {}
    }

    for result in all_results:
        font_size = result.get('font_size')
        if font_size is None:
            continue
        config_name = result.get('config_name')
        evaluation_results = result.get('evaluation_results', {})

        if font_size not in summary['font_sizes']:
            summary['font_sizes'][font_size] = {
                'config_name': config_name,
                'evaluation_results': {}
            }
        # Merge evaluation_results by task and mode keys
        existing_eval = summary['font_sizes'][font_size]['evaluation_results']
        for task_key, task_eval in evaluation_results.items():
            if task_key not in existing_eval or not isinstance(existing_eval.get(task_key), dict):
                existing_eval[task_key] = {}
            # task_eval could be a dict of modes or a scalar (e.g., accuracy)
            if isinstance(task_eval, dict):
                for mode_key, mode_val in task_eval.items():
                    existing_eval[task_key][mode_key] = mode_val
            else:
                # For scalar values, store under a generic key
                existing_eval[task_key] = task_eval

    flat_filename = get_flat_filename(filename)
    summary_file = os.path.join(
        output_dir,
        f"{model_name}_COMPACT_font_summary_{flat_filename}.json"
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nFont size summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("Font Size Evaluation Summary")
    print("=" * 80)

    tasks = set()
    for result in all_results:
        tasks.update(result['evaluation_results'].keys())

    tasks = sorted(list(tasks))

    base_modes = ['text_only', 'no_context', 'image']
    mode_labels = {
        'text_only': 'Text Only',
        'no_context': 'No Context',
        'image': 'Image',
    }
    ratio_modes_set = set()
    for result in all_results:
        evaluation_results = result.get('evaluation_results', {})
        for task_key, task_eval in evaluation_results.items():
            if isinstance(task_eval, dict):
                for k in task_eval.keys():
                    if isinstance(k, str) and k.startswith('image_ratio'):
                        ratio_modes_set.add(k)
    def _ratio_sort_key(s):
        try:
            r = float(s.replace('image_ratio', ''))
            return r
        except Exception:
            return float('inf')
    union_set = set(ratio_modes_set) | {f'image_ratio{r}' for r in sorted(COMPRESSION_RATIOS)}
    mode_order = base_modes + sorted(list(union_set), key=_ratio_sort_key)
    for mode_key in union_set:
        try:
            ratio = float(mode_key.replace('image_ratio', ''))
            mode_labels[mode_key] = f'Compress {int(ratio)}x' if float(ratio).is_integer() else f'Compress {ratio}x'
        except Exception:
            mode_labels[mode_key] = mode_key

    results_by_font_size = {}
    for result in all_results:
        font_size = result['font_size']
        if font_size not in results_by_font_size:
            results_by_font_size[font_size] = []
        results_by_font_size[font_size].append(result)

    def extract_result_text(eval_data, task):
        # For some tasks, eval_data may be numeric type instead of dictionary
        # Check special task types first
        if task == 'code_qa':
            # Direct numeric type, represents accuracy
            if isinstance(eval_data, (int, float)):
                return f"{eval_data:.3f}"
            # If dictionary containing accuracy key
            elif isinstance(eval_data, dict) and 'accuracy' in eval_data:
                accuracy = eval_data.get('accuracy', 0.0)
                correct = eval_data.get('correct', 0)
                total = eval_data.get('total', 0)
                return f"Acc:{accuracy:.3f}({correct}/{total})"
            return "N/A"
        elif task == 'code_summarization':
            # Direct numeric type, represents average score
            if isinstance(eval_data, (int, float)):
                return f"{eval_data:.3f}"
            return "N/A"
        elif task == 'code_summarization':
            # Support displaying both BLEU and EM metrics
            if isinstance(eval_data, (int, float)):
                return f"{eval_data:.3f}"
            elif isinstance(eval_data, dict):
                bleu_val = eval_data.get('bleu')
                em_val = eval_data.get('em')
                llm_val = eval_data.get('llm')
                if bleu_val is not None and em_val is not None:
                    if llm_val is not None:
                        return f"BLEU:{bleu_val:.2f} EM:{em_val:.2f} LLM:{llm_val:.2f}"
                    return f"BLEU:{bleu_val:.2f} EM:{em_val:.2f}"
                if llm_val is not None and bleu_val is None and em_val is None:
                    return f"LLM:{llm_val:.2f}"
                if bleu_val is not None:
                    return f"{bleu_val:.3f}"
            return "N/A"
        
        # For other tasks, require dictionary type
        if not isinstance(eval_data, dict):
            return "N/A"

        if task == 'code_completion_rag':
            if 'average_es' in eval_data and 'average_em' in eval_data:
                es = eval_data.get('average_es', 0.0)
                em = eval_data.get('average_em', 0.0)
                return f"ES:{es:.2f} EM:{em:.2f}"
            elif 'es' in eval_data and 'em' in eval_data:
                es = eval_data.get('es', 0.0)
                em = eval_data.get('em', 0.0)
                return f"ES:{es:.2f} EM:{em:.2f}"
            return "N/A"
        elif task == 'SearchingNeedle':
            needle_results = []
            for key, value in eval_data.items():
                if isinstance(value, dict) and ('verdict' in value or 'best_similarity' in value):
                    verdict = value.get('verdict', 'unknown')
                    similarity = value.get('best_similarity', 0)
                    if verdict == 'best_match':
                        needle_results.append(f"✓{similarity:.2f}")
                    else:
                        needle_results.append(f"✗{similarity:.2f}")
            return ', '.join(needle_results) if needle_results else 'N/A'
        elif task == 'code_clone_detection':
            if 'acc' in eval_data:
                acc = eval_data.get('acc', 0.0)
                f1 = eval_data.get('f1', 0.0)
                return f"Acc:{acc:.2f} F1:{f1:.2f}"
            return "N/A"

        return "N/A"

    def get_mode_data(task_eval, mode_key):
        if not isinstance(task_eval, dict):
            return None

        if mode_key in task_eval:
            return task_eval.get(mode_key)
        elif mode_key == 'image' and 'image' in task_eval:
            return task_eval.get('image')
        elif ('verdict' in task_eval or 'metrics' in task_eval or
              any(key in task_eval for key in ['best_similarity', 'best_target'])):
            if mode_key.startswith('image_ratio') or mode_key == 'image':
                return task_eval
        elif task_eval and mode_key == 'image':
            return task_eval

        return None

    header = f"{ 'Task':<20}"
    for font_size in sorted(results_by_font_size.keys()):
        header += f"  {font_size}px"
        for mode_key in mode_order:
            label = mode_labels.get(mode_key, mode_key)
            header += f" ({label})"
    print(header)
    print("-" * 120)

    if code_completion_rag_results:
        if 'code_completion_rag' not in tasks:
            tasks.append('code_completion_rag')

        for font_size in results_by_font_size.keys():
            font_results = results_by_font_size[font_size]
            for result in font_results:
                if 'code_completion_rag' not in result['evaluation_results']:
                    result['evaluation_results']['code_completion_rag'] = {}

                if font_size in code_completion_rag_results:
                    rag_result = code_completion_rag_results[font_size]
                    for mode_key in mode_order:
                        if mode_key in rag_result:
                            if mode_key not in result['evaluation_results']['code_completion_rag']:
                                result['evaluation_results']['code_completion_rag'][mode_key] = rag_result[mode_key]

    for task in tasks:
        row = f"{task:<20}"
        for font_size in sorted(results_by_font_size.keys()):
            font_results = results_by_font_size[font_size]

            if task == "code_summarization":
                # TODO: currently simplified handling
                task_eval = font_results[0]["evaluation_results"].get(task, {})
                for mode_key in mode_order:
                    mode_data = task_eval.get(mode_key)
                    if mode_data:
                        row += f"  {mode_data:.3f}({mode_key})"
                continue

            mode_data = {}
            for result in font_results:
                task_eval = result['evaluation_results'].get(task, {})
                if not isinstance(task_eval, dict) or not task_eval:
                    continue

                compression_ratio = result.get('compression_ratio')

                if task == 'code_completion_rag':
                    for mode_key in mode_order:
                        if mode_key in task_eval and mode_key not in mode_data:
                            mode_data[mode_key] = task_eval[mode_key]

                if task == 'code_clone_detection':
                    for mode_key in mode_order:
                        if mode_key in task_eval and mode_key not in mode_data:
                            mode_data[mode_key] = task_eval[mode_key]

                if task == 'code_summarization':
                    for mode_key in mode_order:
                        if mode_key in task_eval and mode_key not in mode_data:
                            mode_data[mode_key] = task_eval[mode_key]

                if task == 'code_qa':
                    for mode_key in mode_order:
                        if mode_key in task_eval and mode_key not in mode_data:
                            mode_data[mode_key] = task_eval[mode_key]

                if 'text_only' not in mode_data:
                    if 'text_only' in task_eval:
                        text_only_data = task_eval.get('text_only')
                        if text_only_data:
                            mode_data['text_only'] = text_only_data
                    text_only_task_key = f"{task}_text_only"
                    if text_only_task_key in result['evaluation_results']:
                        text_only_data = result['evaluation_results'][text_only_task_key]
                        if isinstance(text_only_data, dict) and text_only_data:
                            mode_data['text_only'] = text_only_data

                if compression_ratio is None:
                    current_mode = 'image'
                elif compression_ratio in COMPRESSION_RATIOS:
                    current_mode = f'image_ratio{compression_ratio}'
                else:
                    current_mode = f'image_ratio{compression_ratio}'

                if current_mode not in mode_data:
                    if current_mode.startswith('image_ratio'):
                        if 'image' in task_eval:
                            eval_data = task_eval.get('image')
                        elif 'verdict' in task_eval or 'metrics' in task_eval or any(key in task_eval for key in ['best_similarity', 'best_target']):
                            eval_data = task_eval
                        else:
                            eval_data = None
                    else:
                        eval_data = get_mode_data(task_eval, current_mode)

                    if eval_data:
                        mode_data[current_mode] = eval_data

            for mode_key in mode_order:
                eval_data = mode_data.get(mode_key)
                result_text = extract_result_text(eval_data, task)
                row += f"  {result_text:<15}"

        print(row)

    print("=" * 120)

    return summary
