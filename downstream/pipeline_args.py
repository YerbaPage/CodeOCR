import argparse

from tasks.code_qa.task import DEFAULT_LQA_FILES, DEFAULT_NUM_EXAMPLES_PER_FILE

ALL_TASKS = [
    "code_completion_rag",
    "code_summarization",
    "code_clone_detection",
    "code_qa",
]
DEFAULT_TASKS = ALL_TASKS
DEFAULT_MAX_WORKERS = 20


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image generation and multi-task evaluation pipeline"
    )

    parser.add_argument(
        "--width", type=int, default=2240, help="Image width in pixels (default: 2240)"
    )
    parser.add_argument(
        "--height", type=int, default=2240, help="Image height in pixels (default: 2240)"
    )
    parser.add_argument(
        "--font-size",
        type=int,
        nargs="+",
        default=[40],
        help="Font size list in pixels, can specify multiple (default: 40)",
    )
    parser.add_argument(
        "--line-height", type=float, default=1.0, help="Line height multiplier (default: 1.0)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI setting (default: 300)")
    parser.add_argument("--font-path", type=str, default=None, help="Specify font file path")
    parser.add_argument(
        "--preserve-newlines",
        action="store_true",
        help="Preserve newline characters (default: True)",
    )
    parser.add_argument(
        "--enable-syntax-highlight", action="store_true", help="Enable code syntax highlighting"
    )
    parser.add_argument("--language", type=str, default="py", help="Programming language name")
    parser.add_argument("--theme", type=str, default="light", help="Syntax highlighting theme (light/modern)")
    parser.add_argument("--crop-whitespace", action="store_true", help="Crop whitespace from images")
    parser.add_argument("--enable-two-column", action="store_true", help="Enable two-column layout")
    parser.add_argument(
        "--resize-mode",
        action="store_true",
        default=True,
        help="Enable resize mode (default: True)",
    )
    parser.add_argument(
        "--no-resize-mode",
        action="store_false",
        dest="resize_mode",
        help="Disable resize mode",
    )
    parser.add_argument(
        "--resize-ratios",
        type=str,
        default=None,
        help="Specify resize ratio list (comma-separated), e.g., 0.5,0.3",
    )
    parser.add_argument(
        "--extreme-mode",
        action="store_true",
        help="Enable extreme mode: dynamically re-render images at different resolutions instead of simple resize",
    )

    parser.add_argument(
        "--run-tasks",
        action="store_true",
        default=True,
        help="Enable task and evaluation mode (default: True)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help=f'Task list to run (comma-separated), default: {",".join(DEFAULT_TASKS)}',
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5", help="Model name, default: gpt-5"
    )
    parser.add_argument(
        "--models", type=str, default=None, help="Model name list (comma-separated)"
    )
    parser.add_argument(
        "--client-type",
        type=str,
        default="OpenAI",
        choices=["OpenAI", "Azure"],
        help="API client type: OpenAI (default) or Azure",
    )
    parser.add_argument("--code-clone-detection-type", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--code-clone-detection-lang", type=str, default="py")
    parser.add_argument("--code-clone-detection-difficulty", type=str, default="prompt_2")
    parser.add_argument("--code-clone-detection-tier", type=str, default="T4")
    parser.add_argument("--code-clone-detection-num-examples", type=int, default=200)
    parser.add_argument("--code-clone-detection-separate-mode", action="store_true", help="Code clone detection task: separate rendering mode (render A/B separately)")
    parser.add_argument("--code-clone-detection-evaluation-only-file", type=str, default=None, help="Code clone detection task: evaluation-only mode, specify result file path")
    parser.add_argument("--code-clone-detection-existing-results-dir", type=str, default=None, help="Code clone detection task: specify existing results directory")

    parser.add_argument("--codeqa-dir", type=str, default="./dataset/code_qa", help="CodeQA: data directory")
    parser.add_argument("--codeqa-files", type=str, default=",".join(DEFAULT_LQA_FILES), help="CodeQA: file list to load (comma-separated), default: 32K,64K")
    parser.add_argument("--codeqa-num-examples", type=int, default=DEFAULT_NUM_EXAMPLES_PER_FILE, help="CodeQA: number of examples per file, default: 100")

    parser.add_argument(
        "--rag-window-size", type=int, default=80, help="RAG sliding window size"
    )
    parser.add_argument(
        "--rag-overlap", type=int, default=40, help="RAG sliding window overlap size"
    )
    parser.add_argument("--rag-top-k", type=int, default=3, help="RAG retrieval top-k")
    parser.add_argument(
        "--embed-model-name",
        type=str,
        default="microsoft/unixcoder-base",
        help="Embedding model name",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="microsoft/LCC_python",
        help="Code completion dataset path",
    )
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num-examples", type=int, default=0, help="Number of examples to process")
    parser.add_argument(
        "--filter-current-lines-max",
        type=int,
        default=50,
        help="Filter condition: maximum lines in current function",
    )
    parser.add_argument(
        "--filter-background-tokens-min",
        type=int,
        default=3000,
        help="Filter condition: minimum tokens in background code",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--test-single", action="store_true", help="Test mode: process only the first item"
    )
    parser.add_argument(
        "--disable-text-only-test", action="store_true", help="Disable text-only test"
    )
    parser.add_argument(
        "--disable-image-test", action="store_true", help="Disable image test"
    )
    parser.add_argument(
        "--llm-for-ocr-eval", action="store_true", help="Use LLM for OCR evaluation"
    )
    parser.add_argument(
        "--no-context-mode", action="store_true", help="Use no-context mode"
    )
    parser.add_argument("--enable-bold", action="store_true", help="Enable bold text")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of workers for parallel processing (default: {DEFAULT_MAX_WORKERS})",
    )

    return parser
