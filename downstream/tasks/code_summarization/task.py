import importlib.util
import os

_legacy_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "code_summarization.py")
)
_spec = importlib.util.spec_from_file_location(
    "tasks._code_summarization_legacy", _legacy_path
)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

run_code_summarization_task = _legacy.run_code_summarization_task
run_all_evaluations = _legacy.run_all_evaluations
