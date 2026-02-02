#!/usr/bin/env python3
"""Main pipeline for running downstream code understanding tasks."""
from datetime import datetime
import sys

from pipeline_args import create_argument_parser
from pipeline_context import RuntimeContext, initialize_embedding_model, initialize_processors
from pipeline_tasks import (
    run_clone_detection_task,
    run_code_completion_task,
    run_code_qa_task_wrapper,
    run_summarization_task,
)


def get_timestamp() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    target_ratios = None
    if args.resize_ratios:
        try:
            target_ratios = [float(r.strip()) for r in args.resize_ratios.split(",")]
        except ValueError:
            print("Error parsing --resize-ratios. Please use comma separated numbers, e.g., '0.5,0.3'")
            sys.exit(1)

    enable_text_only_test = not args.disable_text_only_test
    enable_image_test = not args.disable_image_test

    selected_tasks = [t.strip() for t in args.tasks.split(",")]

    if args.models:
        models_to_test = [m.strip() for m in args.models.split(",")]
    else:
        models_to_test = [args.model]

    ctx = RuntimeContext()

    try:
        initialize_processors(ctx)
    except Exception as e:
        print(f"Warning: Processor initialization failed: {e}")

    if args.run_tasks and ("code_completion_rag" in selected_tasks or "code_qa" in selected_tasks):
        try:
            initialize_embedding_model(ctx, args.embed_model_name)
        except Exception as e:
            print(f"Warning: Embedding model initialization failed: {e}")

    timestamp = get_timestamp()

    if "code_completion_rag" in selected_tasks:
        run_code_completion_task(args, ctx, models_to_test, timestamp)

    if "code_summarization" in selected_tasks:
        run_summarization_task(args, ctx, models_to_test, timestamp)

    if "code_clone_detection" in selected_tasks:
        run_clone_detection_task(
            args, ctx, models_to_test, timestamp, enable_text_only_test, target_ratios
        )

    if "code_qa" in selected_tasks:
        run_code_qa_task_wrapper(
            args, ctx, models_to_test, timestamp, enable_text_only_test, enable_image_test
        )


if __name__ == "__main__":
    main()
