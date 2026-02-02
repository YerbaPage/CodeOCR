import importlib.util
import os

_legacy_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "code_completion.py")
)
_spec = importlib.util.spec_from_file_location(
    "tasks._code_completion_legacy", _legacy_path
)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

run_code_completion_rag = _legacy.run_code_completion_rag
load_code_completion_rag_results = _legacy.load_code_completion_rag_results
TORCH_AVAILABLE = _legacy.TORCH_AVAILABLE
