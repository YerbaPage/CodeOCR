import importlib.util
import os

_legacy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code_qa.py"))
_spec = importlib.util.spec_from_file_location("tasks._code_qa_legacy", _legacy_path)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

DEFAULT_LQA_FILES = _legacy.DEFAULT_LQA_FILES
DEFAULT_NUM_EXAMPLES_PER_FILE = _legacy.DEFAULT_NUM_EXAMPLES_PER_FILE
run_code_qa_task = _legacy.run_code_qa_task
evaluate_code_qa_results = _legacy.evaluate_code_qa_results
print_code_qa_results = _legacy.print_code_qa_results
