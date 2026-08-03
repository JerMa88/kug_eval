import os
import json
import argparse
import logging
from kug_eval.data.dataset import load_task_items_from_jsonl
from kug_eval.evaluation.evaluator import APIModelEvaluator, evaluate_dataset
from kug_eval.evaluation.metrics import plot_kug_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOTA LLMs on Generalization Tasks.")
    parser.add_argument("--data_path", type=str, default="data/tasks/sota_generalization_benchmark.jsonl", help="Path to JSONL task file.")
    parser.add_argument("--model_name", type=str, default="gemini-3.6-flash", help="SOTA Model identifier (e.g. gemini-3.6-flash, gpt-5.6-sol, claude-fable, deepseek-v4, kimi-k3, glm-5.2)")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for local testing")
    parser.add_argument("--out_dir", type=str, default="outputs/eval_sota", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Loading task dataset from {args.data_path}...")
    task_items = load_task_items_from_jsonl(args.data_path, strict=False)
    logger.info(f"Loaded {len(task_items)} task items across categories.")

    logger.info(f"Initializing APIModelEvaluator for model: {args.model_name} (mock_mode={args.mock})...")
    evaluator = APIModelEvaluator(model_name=args.model_name, api_key=args.api_key, mock_mode=args.mock)

    logger.info("Evaluating task items...")
    eval_results = evaluate_dataset(evaluator, task_items)

    out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Saved evaluation results to {out_file}")

    # Generate UX Plot
    plot_data = {
        args.model_name: {
            "a_mem": [eval_results["overall_a_mem"]],
            "a_gen": [eval_results["overall_a_gen"]],
        }
    }
    plot_path = os.path.join(args.out_dir, "kug_sota_summary.png")
    plot_kug_diagnostics(plot_data, output_path=plot_path, title=f"SOTA Generalization Evaluation: {args.model_name}")
    logger.info(f"Generated diagnostic plot at {plot_path}")

    print("\n" + "="*60)
    print(f" EVALUATION SUMMARY: {args.model_name}")
    print("="*60)
    print(f"Overall A_mem (Memorization):  {eval_results['overall_a_mem']*100:.1f}%")
    print(f"Overall A_gen (Generalization): {eval_results['overall_a_gen']*100:.1f}%")
    print(f"KUG Ratio (A_mem / A_gen):     {eval_results['kug_ratio']:.2f}x")
    print("-" * 60)
    print("Category Breakdown:")
    for cat, info in eval_results["category_summary"].items():
        print(f"  - {cat:20s}: A_mem={info['a_mem']*100:.1f}%, A_gen={info['a_gen']*100:.1f}%, KUG={info['kug_ratio']:.2f}x")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
