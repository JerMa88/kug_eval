"""
scripts/evaluate_sota.py
========================
CLI to evaluate a single SOTA frontier model on a kug_eval benchmark JSONL.

Usage examples
--------------
# Development / smoke test with mock (no API key needed)
python scripts/evaluate_sota.py \
    --data_path data/tasks/sota_generalization_benchmark.jsonl \
    --model_name gpt-5.6-sol --mock

# Live run — OpenAI (development model, test first)
python scripts/evaluate_sota.py \
    --data_path data/tasks/sota_generalization_benchmark.jsonl \
    --model_name gpt-5.6-sol \
    --out_dir outputs/eval_v2/gpt-5.6-sol \
    --max_workers 5

# Live run — Fireworks DeepSeek
python scripts/evaluate_sota.py \
    --data_path data/tasks/sota_generalization_benchmark.jsonl \
    --model_name deepseek-v4-flash \
    --out_dir outputs/eval_v2/deepseek-v4-flash \
    --max_workers 5
"""

import os
import json
import argparse
import logging
from datetime import datetime

from kug_eval.data.dataset import load_task_items_from_jsonl
from kug_eval.evaluation.evaluator import APIModelEvaluator, evaluate_dataset
from kug_eval.evaluation.metrics import plot_kug_diagnostics

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOTA frontier LLMs on kug_eval Generalization Benchmark."
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/tasks/sota_generalization_benchmark.jsonl",
        help="Path to input JSONL benchmark file.",
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt-5.6-sol",
        help=(
            "Frontier model identifier. Confirmed IDs:\n"
            "  OpenAI:    gpt-5.6-sol\n"
            "  Google:    gemini-3.6-flash\n"
            "  Anthropic: claude-fable-5\n"
            "  Fireworks: deepseek-v4-flash | kimi-k3 | qwen3.8-max | glm-5.2\n"
        ),
    )
    parser.add_argument(
        "--api_key", type=str, default="",
        help="API key override (reads from env if omitted).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in deterministic mock mode (no API key required, for CI testing).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="outputs/eval_v2",
        help="Directory to write results JSON and logs.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=5,
        help="Thread pool size for concurrent API calls.",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_model = args.model_name.replace("/", "_").replace(":", "_")

    # ── Full API call log path ────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, f"{safe_model}_api_calls_{run_ts}.jsonl")
    results_path = os.path.join(args.out_dir, f"{safe_model}_results.json")
    plot_path = os.path.join(args.out_dir, f"{safe_model}_kug_summary.png")

    logger.info(f"Model:       {args.model_name}")
    logger.info(f"Data:        {args.data_path}")
    logger.info(f"Output dir:  {args.out_dir}")
    logger.info(f"API log:     {log_path}")
    logger.info(f"Mock mode:   {args.mock}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading task items ...")
    task_items = load_task_items_from_jsonl(args.data_path, strict=False)
    logger.info(f"Loaded {len(task_items)} items.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = APIModelEvaluator(
        model_name=args.model_name,
        api_key=args.api_key or None,
        mock_mode=args.mock,
    )

    eval_results = evaluate_dataset(
        evaluator,
        task_items,
        max_workers=args.max_workers if not args.mock else 1,
        log_path=None if args.mock else log_path,
    )

    # ── Save results JSON ─────────────────────────────────────────────────────
    # Add run metadata
    eval_results["_meta"] = {
        "model_name":  args.model_name,
        "data_path":   args.data_path,
        "run_ts":      run_ts,
        "mock_mode":   args.mock,
        "log_path":    log_path if not args.mock else None,
        "total_items": len(task_items),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved → {results_path}")

    # ── Generate diagnostic plot ──────────────────────────────────────────────
    plot_data = {
        args.model_name: {
            "a_mem": [eval_results["overall_a_mem"]],
            "a_gen": [eval_results["overall_a_gen"]],
        }
    }
    plot_kug_diagnostics(
        plot_data, output_path=plot_path,
        title=f"KUG Evaluation: {args.model_name}",
    )
    logger.info(f"Diagnostic plot → {plot_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY  |  {args.model_name}")
    print("=" * 70)
    print(f"  Items evaluated:            {eval_results['total_count']}")
    print(f"  Strict EM  A_mem:           {eval_results['overall_a_mem']*100:.2f}%")
    print(f"  Strict EM  A_gen:           {eval_results['overall_a_gen']*100:.2f}%")
    print(f"  KUG Ratio  (A_mem/A_gen):   {eval_results['kug_ratio']:.4f}x")
    print(f"  Contains   A_mem (sec.):    {eval_results['overall_contains_mem']*100:.2f}%")
    print(f"  Contains   A_gen (sec.):    {eval_results['overall_contains_gen']*100:.2f}%")
    print("-" * 70)
    print("  Category Breakdown (Strict EM):")
    for cat, info in sorted(eval_results["category_summary"].items()):
        print(f"    {cat:<22} A_mem={info['a_mem']*100:.1f}%  "
              f"A_gen={info['a_gen']*100:.1f}%  "
              f"KUG={info['kug_ratio']:.2f}x  "
              f"n={info['count']}")
    print("=" * 70 + "\n")

    if not args.mock:
        print(f"  Full API call log (every payload+response): {log_path}\n")


if __name__ == "__main__":
    main()
