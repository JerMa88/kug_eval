import os
import tempfile
import pytest
import torch
import torch.nn as nn
from kug_eval.data.schema import GeneralizationTaskItem
from kug_eval.evaluation.metrics import (
    normalize_answer,
    exact_match_score,
    compute_kug_metrics,
    plot_kug_diagnostics,
)
from kug_eval.evaluation.evaluator import (
    APIModelEvaluator,
    LocalModelEvaluator,
    evaluate_dataset,
)


class MockLocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))

    def generate(self, input_ids, **kwargs):
        # Always return input_ids concatenated with token ID 10 (Target token)
        target_token = torch.tensor([[10]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, target_token], dim=-1)


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "Drive"


def test_normalize_answer_and_exact_match():
    assert normalize_answer("The Drive!") == "drive"
    assert normalize_answer("a car wash ") == "car wash"
    assert exact_match_score("Drive.", "drive") == 1.0
    assert exact_match_score("Walk", "Drive") == 0.0
    assert exact_match_score("", "test") == 0.0


def test_compute_kug_metrics():
    a_mem = [0.2, 0.5, 0.8, 0.9]
    a_gen = [0.0, 0.01, 0.05, 0.1]

    metrics = compute_kug_metrics(a_mem, a_gen)
    assert abs(metrics["peak_a_mem"] - 0.9) < 1e-5
    assert abs(metrics["final_a_gen"] - 0.1) < 1e-5
    assert abs(metrics["kug_ratio"] - 9.0) < 1e-5
    assert metrics["auc_gen"] > 0.0


def test_plot_kug_diagnostics_ux_figures():
    results_data = {
        "Baseline SFT": {"a_mem": [0.3, 0.7, 0.9], "a_gen": [0.0, 0.02, 0.05]},
        "Faster-SFT (Hybrid)": {"a_mem": [0.3, 0.8, 0.95], "a_gen": [0.1, 0.5, 0.85]},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "test_plot.png")
        saved_path = plot_kug_diagnostics(results_data, output_path=out_png)
        assert os.path.exists(out_png)
        svg_path = os.path.join(tmpdir, "test_plot.svg")
        assert os.path.exists(svg_path)


def test_api_evaluator_sota_models():
    evaluator = APIModelEvaluator(model_name="gemini-3.6-flash", mock_mode=True)
    ans = evaluator.generate_answer("Query: Should I walk or drive to the car wash?\nAnswer: Drive")
    assert ans == "Drive"

    evaluator_gpt = APIModelEvaluator(model_name="gpt-5.6-sol", mock_mode=True)
    ans_gpt = evaluator_gpt.generate_answer("Query: walk or drive?")
    assert ans_gpt in ["Drive", "Walk", "Model Output Answer"]

    evaluator_minimax = APIModelEvaluator(model_name="minimax-m3", mock_mode=True)
    ans_minimax = evaluator_minimax.generate_answer("Query: walk or drive?")
    assert ans_minimax in ["Drive", "Walk", "Model Output Answer"]

    evaluator_fireworks = APIModelEvaluator(model_name="fireworks/deepseek-v3", mock_mode=True)
    ans_fireworks = evaluator_fireworks.generate_answer("Query: walk or drive?")
    assert ans_fireworks in ["Drive", "Walk", "Model Output Answer"]


def test_local_evaluator_and_evaluate_dataset():
    model = MockLocalModel()
    tokenizer = MockTokenizer()
    evaluator = LocalModelEvaluator(model, tokenizer)

    task_items = [
        GeneralizationTaskItem(id="1", document="Doc1", query="Q1", target_entity="Drive", category="car_wash"),
        GeneralizationTaskItem(id="2", document="Doc2", query="Q2", target_entity="Drive", category="car_wash"),
    ]

    res = evaluate_dataset(evaluator, task_items)
    assert res["total_count"] == 2
    assert res["overall_a_mem"] == 1.0
    assert res["overall_a_gen"] == 1.0
    assert "car_wash" in res["category_summary"]
