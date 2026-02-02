import json
import os
from typing import List

from report_utils import generate_font_size_summary
from tasks.code_clone_detection.task import (
    evaluate_code_clone_detection_results,
    run_code_clone_detection_task,
)
from tasks.code_completion.task import (
    load_code_completion_rag_results,
    run_code_completion_rag,
)
from tasks.code_qa.task import (
    evaluate_code_qa_results,
    print_code_qa_results,
    run_code_qa_task,
)
from tasks.code_summarization.task import (
    run_all_evaluations,
    run_code_summarization_task,
)
from text_to_image import get_all_modes

from pipeline_context import RuntimeContext


def run_code_completion_task(args, ctx: RuntimeContext, models_to_test: List[str], timestamp: str):
    if not ctx.embed_model:
        print("Skipping code_completion_rag: embedding model not available")
        return

    comp_tk_all = {}
    comp_output_dir = os.path.join("./llm_outputs", "comp_rag", timestamp)

    for model_name in models_to_test:
        font_size = args.font_size[0]
        config_name = f"COMPACT_font{font_size}"
        try:
            completion_results, completion_token_stats, result_dir = run_code_completion_rag(
                model_name,
                config_name,
                args.width,
                args.height,
                font_size,
                args.line_height,
                args.dpi,
                args.font_path,
                output_dir=comp_output_dir,
                processor=ctx.processor,
                qwen_tokenizer=ctx.qwen_tokenizer,
                embed_model=ctx.embed_model,
                embed_tokenizer=ctx.embed_tokenizer,
                rag_window_size=args.rag_window_size,
                rag_overlap=args.rag_overlap,
                rag_top_k=args.rag_top_k,
                dataset_path=args.dataset_path,
                dataset_split=args.dataset_split,
                num_examples=1 if args.test_single else args.num_examples or 200,
                filter_current_lines_max=args.filter_current_lines_max,
                filter_background_tokens_min=args.filter_background_tokens_min,
                max_new_tokens=args.max_new_tokens,
                test_single=args.test_single,
                preserve_newlines=args.preserve_newlines,
                enable_syntax_highlight=args.enable_syntax_highlight,
                language=args.language,
                should_crop_whitespace=args.crop_whitespace,
                enable_two_column=args.enable_two_column,
                resize_mode=args.resize_mode,
                client_type=args.client_type,
                rag_mode="function_rag",
                no_context_mode=args.no_context_mode,
                enable_bold=args.enable_bold,
                theme=args.theme,
                extreme_mode=args.extreme_mode,
            )

            if completion_token_stats:
                comp_tk_all[model_name] = completion_token_stats

            _process_completion_results(args, result_dir, font_size, config_name, model_name)

        except Exception as e:
            print(f"Code Completion RAG task failed: {e}")

    if comp_tk_all:
        os.makedirs(comp_output_dir, exist_ok=True)
        with open(os.path.join(comp_output_dir, f"tk_stats_{timestamp}.json"), "w") as f:
            json.dump(comp_tk_all, f, indent=2)


def _process_completion_results(args, result_dir, font_size, config_name, model_name):
    if not result_dir or not os.path.exists(result_dir):
        return

    rag_results_by_mode = {}
    for mode in get_all_modes():
        mode_result = load_code_completion_rag_results(result_dir, mode)
        if mode_result:
            rag_results_by_mode[mode] = mode_result

    if not rag_results_by_mode:
        return

    evaluation_results = {
        mode: {
            "average_es": stats.get("average_es", 0.0),
            "average_em": stats.get("average_em", 0.0),
            "num_examples": stats.get("num_examples", 0),
        }
        for mode, stats in rag_results_by_mode.items()
    }

    eval_result_file = os.path.join(result_dir, "evaluation_results.json")
    with open(eval_result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "code_completion_rag",
                "evaluation_results": evaluation_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    virtual_results = [
        {
            "font_size": font_size,
            "config_name": config_name,
            "filename": "code_completion_rag",
            "compression_ratio": None,
            "resolution": f"{args.width}x{args.height}",
            "evaluation_results": {"code_completion_rag": rag_results_by_mode},
        }
    ]
    generate_font_size_summary(virtual_results, model_name, "code_completion_rag")


def run_summarization_task(args, ctx: RuntimeContext, models_to_test: List[str], timestamp: str):
    print("\n=== Running Code Summarization Task ===")
    base_output_dir = os.path.join("./llm_outputs", "code_summarization", timestamp)
    sum_res_dir = None

    for model_name in models_to_test:
        print(f"\n--- Running model: {model_name} ---")
        try:
            summary_results, sum_res_dir = run_code_summarization_task(
                model_name=model_name,
                output_dir=base_output_dir,
                width=args.width,
                height=args.height,
                font_size=args.font_size[0] if args.font_size else 40,
                line_height=args.line_height,
                dpi=args.dpi,
                font_path=args.font_path,
                preserve_newlines=args.preserve_newlines,
                enable_syntax_highlight=args.enable_syntax_highlight,
                language=args.language,
                should_crop_whitespace=args.crop_whitespace,
                enable_two_column=args.enable_two_column,
                resize_mode=args.resize_mode,
                processor=ctx.processor,
                qwen_tokenizer=ctx.qwen_tokenizer,
                client_type=args.client_type,
                num_examples=args.num_examples if args.num_examples else 139,
                test_single=args.test_single,
                no_context_mode=args.no_context_mode,
                enable_bold=args.enable_bold,
                extreme_mode=args.extreme_mode,
                enable_text_only_test=False,
            )

            with open(os.path.join(base_output_dir, "summary_out_and_tokens.json"), "w") as f:
                json.dump(summary_results, f, indent=4)

        except Exception as e:
            print(f"Code Summarization task failed (model: {model_name}): {e}")
            import traceback

            traceback.print_exc()

    print("\n=== Running Code Summarization unified evaluation ===")
    summary_report = run_all_evaluations(
        models_to_test=models_to_test,
        base_output_dir=base_output_dir,
        model_output_dir=sum_res_dir,
    )
    _process_summarization_results(args, summary_report, models_to_test)


def _process_summarization_results(args, summary_report, models_to_test):
    if not summary_report or "all_evaluations" not in summary_report:
        return

    for model_name in models_to_test:
        model_eval_data = summary_report["all_evaluations"].get(model_name)
        if not model_eval_data or "evaluation_results" not in model_eval_data:
            continue

        evaluation_results = model_eval_data["evaluation_results"]
        average_scores = {
            mode: stats.get("mean_score", 0.0)
            for mode, stats in evaluation_results.items()
            if stats.get("count", 0) > 0
        }

        if average_scores:
            processed_results = [
                {
                    "font_size": args.font_size[0] if args.font_size else 40,
                    "config_name": "code_summarization",
                    "filename": "code_summarization",
                    "compression_ratio": None,
                    "resolution": "N/A",
                    "evaluation_results": {"code_summarization": average_scores},
                }
            ]
            generate_font_size_summary(processed_results, model_name, "code_summarization")


def run_clone_detection_task(
    args,
    ctx: RuntimeContext,
    models_to_test: List[str],
    timestamp: str,
    enable_text_only_test: bool,
    target_ratios,
):
    print("\n=== Running Code Clone Detection Task ===")
    base_output_dir = os.path.join("./llm_outputs", "code_clone_detection", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    for model_name in models_to_test:
        try:
            font_size = args.font_size[0] if args.font_size else 40
            sc_results, sc_token_stats, res_dir = run_code_clone_detection_task(
                model_name=model_name,
                output_dir=base_output_dir,
                width=args.width,
                height=args.height,
                font_size=font_size,
                line_height=args.line_height,
                dpi=args.dpi,
                font_path=args.font_path,
                preserve_newlines=args.preserve_newlines,
                resize_mode=args.resize_mode,
                base_dir="./dataset/code_clone_detection/standalone",
                dataset_type=args.code_clone_detection_type,
                lang=args.language,
                difficulty=args.code_clone_detection_difficulty,
                tier=args.code_clone_detection_tier,
                num_examples=args.code_clone_detection_num_examples,
                processor=ctx.processor,
                qwen_tokenizer=ctx.qwen_tokenizer,
                enable_text_only_test=enable_text_only_test,
                client_type=args.client_type,
                enable_syntax_highlight=args.enable_syntax_highlight,
                enable_bold=args.enable_bold,
                separate_mode=args.code_clone_detection_separate_mode,
                evaluation_only_file=args.code_clone_detection_evaluation_only_file,
                existing_results_dir=args.code_clone_detection_existing_results_dir,
                target_ratios=target_ratios,
                theme=args.theme,
                max_workers=args.max_workers,
                extreme_mode=args.extreme_mode,
            )

            eval_res = evaluate_code_clone_detection_results(sc_results)
            _process_clone_detection_results(
                args, eval_res, font_size, model_name, base_output_dir, timestamp, sc_token_stats
            )

        except Exception as e:
            print(f"Code Clone Detection task failed: {e}")


def _process_clone_detection_results(
    args, eval_res, font_size, model_name, base_output_dir, timestamp, sc_token_stats
):
    modes_dict = eval_res.get("code_clone_detection", {})
    virtual_results = []

    def _add_virtual(mode_key, metrics, ratio_val=None):
        virtual_results.append(
            {
                "font_size": font_size,
                "config_name": f"COMPACT_font{font_size}",
                "filename": "code_clone_detection",
                "compression_ratio": ratio_val,
                "resolution": f"{args.width}x{args.height}",
                "evaluation_results": {"code_clone_detection": {mode_key: metrics}},
            }
        )

    if "text_only" in modes_dict:
        _add_virtual("text_only", modes_dict["text_only"], None)
    if "image" in modes_dict:
        _add_virtual("image", modes_dict["image"], None)

    for k, v in modes_dict.items():
        if isinstance(k, str) and k.startswith("image_ratio"):
            try:
                r_str = k.replace("image_ratio", "")
                r_val = float(r_str)
                if r_val.is_integer():
                    r_val = int(r_val)
            except Exception:
                r_val = r_str
            _add_virtual(k, v, r_val)

    if not virtual_results:
        virtual_results = [
            {
                "font_size": font_size,
                "config_name": f"COMPACT_font{font_size}",
                "filename": "code_clone_detection",
                "compression_ratio": None,
                "resolution": f"{args.width}x{args.height}",
                "evaluation_results": eval_res,
            }
        ]

    generate_font_size_summary(virtual_results, model_name, "code_clone_detection")

    if sc_token_stats:
        with open(
            os.path.join(base_output_dir, f"tk_stats_{timestamp}_{model_name}.json"), "w"
        ) as f:
            json.dump(sc_token_stats, f, indent=2)


def run_code_qa_task_wrapper(
    args,
    ctx: RuntimeContext,
    models_to_test: List[str],
    timestamp: str,
    enable_text_only_test: bool,
    enable_image_test: bool,
):
    print("\n=== Running CodeQA Task ===")
    base_output_dir = os.path.join("./llm_outputs", "code_qa", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    code_qa_file_list = [f.strip() for f in args.codeqa_files.split(",")]

    for model_name in models_to_test:
        try:
            font_size = args.font_size[0] if args.font_size else 40
            nlcb_results, nlcb_token_stats, res_dir = run_code_qa_task(
                model_name=model_name,
                output_dir=base_output_dir,
                width=args.width,
                height=args.height,
                font_size=font_size,
                line_height=args.line_height,
                dpi=args.dpi,
                font_path=args.font_path,
                preserve_newlines=args.preserve_newlines,
                resize_mode=args.resize_mode,
                no_context_mode=args.no_context_mode,
                qa_dir=args.code_qa_dir,
                processor=ctx.processor,
                qwen_tokenizer=ctx.qwen_tokenizer,
                enable_text_only_test=enable_text_only_test,
                enable_image_test=enable_image_test,
                client_type=args.client_type,
                enable_syntax_highlight=args.enable_syntax_highlight,
                language=args.language,
                enable_bold=args.enable_bold,
                theme=args.theme,
                test_single=args.test_single,
                embed_model=ctx.embed_model,
                embed_tokenizer=ctx.embed_tokenizer,
                extreme_mode=args.extreme_mode,
            )

            eval_res = evaluate_code_qa_results(nlcb_results)
            print_code_qa_results(eval_res, model_name)
            _process_code_qa_results(
                args,
                eval_res,
                font_size,
                model_name,
                base_output_dir,
                timestamp,
                res_dir,
                nlcb_token_stats,
            )

        except Exception as e:
            print(f"CodeQA task failed: {e}")
            import traceback

            traceback.print_exc()


def _process_code_qa_results(
    args, eval_res, font_size, model_name, base_output_dir, timestamp, res_dir, nlcb_token_stats
):
    accuracy_scores = {}
    if "code_qa" in eval_res:
        for mode, stats in eval_res["code_qa"].items():
            accuracy_scores[mode] = stats.get("accuracy", 0.0)

    virtual = [
        {
            "font_size": font_size,
            "config_name": f"font{font_size}",
            "filename": "code_qa",
            "compression_ratio": None,
            "resolution": f"{args.width}x{args.height}",
            "evaluation_results": {"code_qa": accuracy_scores},
        }
    ]
    generate_font_size_summary(virtual, model_name, "code_qa")

    if nlcb_token_stats:
        with open(
            os.path.join(base_output_dir, f"tk_stats_{timestamp}_{model_name}.json"), "w"
        ) as f:
            json.dump(nlcb_token_stats, f, indent=2)

    eval_file = os.path.join(res_dir, "evaluation_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to: {eval_file}")
