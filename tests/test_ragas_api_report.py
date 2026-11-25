import ast
from pathlib import Path
from types import SimpleNamespace


def test_public_job_hides_ragas_results_but_keeps_review_fields():
    job = {
        "id": "job",
        "status": "awaiting_approval",
        "findings": [],
        "ragas_task_id": "task",
        "ragas_scores": {"faithfulness": 0.8},
        "ragas_status": "completed",
        "ragas_error": None,
        "ragas_metrics": {"faithfulness": {"status": "completed"}},
        "ragas_evaluation": {"engine": "RAGAS"},
        "ragas_inputs": {"endpoint": {"api_key": "private"}},
    }
    tree = ast.parse(Path("backend/api.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_public_job")
    ns = {"Any": object, "time": SimpleNamespace(time=lambda: 100), "_jobs": {}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "api.py", "exec"), ns)

    result = ns["_public_job"](job)

    assert result["id"] == "job"
    assert result["status"] == "awaiting_approval"
    assert not any(key.startswith("ragas") for key in result)
