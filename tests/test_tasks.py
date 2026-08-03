import os
import json
import pytest
import subprocess
from kug_eval.tasks.registry import TaskRegistry, get_all_tasks
from kug_eval.data.dataset import load_task_items_from_jsonl


def test_task_registry_categories_and_items():
    categories = TaskRegistry.list_categories()
    assert len(categories) == 5
    assert "car_wash" in categories
    assert "reversal" in categories
    assert "multi_hop" in categories
    assert "counterfactual" in categories
    assert "set_intersection" in categories

    all_items = get_all_tasks()
    assert len(all_items) >= 12
    for item in all_items:
        assert item.id
        assert item.document
        assert item.query
        assert item.target_entity
        assert item.category in categories


def test_benchmark_jsonl_file():
    benchmark_file = "data/tasks/sota_generalization_benchmark.jsonl"
    assert os.path.exists(benchmark_file)

    items = load_task_items_from_jsonl(benchmark_file, strict=True)
    assert len(items) == 12
    cat_counts = {}
    for it in items:
        cat_counts[it.category] = cat_counts.get(it.category, 0) + 1

    assert len(cat_counts) == 5


def test_evaluate_sota_cli_mock():
    cmd = [
        "python",
        "scripts/evaluate_sota.py",
        "--data_path",
        "data/tasks/sota_generalization_benchmark.jsonl",
        "--model_name",
        "gemini-3.6-flash",
        "--mock",
        "--out_dir",
        "outputs/test_cli_eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0
    assert "EVALUATION SUMMARY" in res.stdout
    assert os.path.exists("outputs/test_cli_eval/gemini-3.6-flash_results.json")
    assert os.path.exists("outputs/test_cli_eval/kug_sota_summary.png")
