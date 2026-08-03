# `kug_eval`: LLM Knowing-Using Gap ($A_{\text{mem}}$ vs $A_{\text{gen}}$) Evaluation Framework

> **Dataset-Agnostic Diagnostic & Benchmarking Framework for LLM Generalization**

`kug_eval` is a modular, dataset-agnostic Python package and diagnostic evaluation harness designed to measure and analyze the **Knowing-Using Gap (KUG)** across SOTA Large Language Models.

---

## Key Features

1. **Dataset-Agnostic Data Contract**: Standardized `.jsonl` schema (`GeneralizationTaskItem`) decoupling factual memory retention ($A_{\text{mem}}$) from downstream generalization ($A_{\text{gen}}$).
2. **Layer-Wise Representation Routing Tracer**: Computes Linear Centered Kernel Alignment (CKA) and Cosine Similarity matrices across storage ($l_s$) and reasoning ($l_t$) layers.
3. **Multi-Model Evaluator Engine**: Evaluates local PyTorch models (with GPU KV-cache acceleration) and SOTA Frontier API models (Gemini 3.6 Flash, GPT 5.6 sol, Claude Fable, DeepSeek v4, Kimi K3, GLM 5.2).
4. **Classic Generalization Task Suite**: Pre-packaged benchmark task collection (`data/tasks/sota_generalization_benchmark.jsonl`) spanning 5 classic task families:
   - *Implicit Physical & Pragmatic Constraints* ("car wash" walk vs. drive problem)
   - *Inverse Knowledge / Reversal Curse* ($A \to B \implies B \to A$)
   - *Multi-Hop Relational Chaining* ($E_1 \to E_2 \to E_3$)
   - *Counterfactual Rule Overrides*
   - *Multi-Constraint Set Intersection*
5. **Publication-Quality UX Plotter**: Generates 300 DPI figures (`.png` and `.svg`) displaying learning trajectories, category breakdowns, and KUG ratios.

---

## Quickstart

### Installation
```bash
pip install -e .
```

### Run Evaluation CLI
Evaluate SOTA frontier models or local models on the generalization task suite:
```bash
# SOTA API Model (Mock mode / Offline verification)
python scripts/evaluate_sota.py --data_path data/tasks/sota_generalization_benchmark.jsonl --model_name gemini-3.6-flash --mock

# SOTA API Model with API Key
python scripts/evaluate_sota.py --data_path data/tasks/sota_generalization_benchmark.jsonl --model_name gpt-5.6-sol --api_key YOUR_API_KEY
```

### Run Analysis & Generate LaTeX / Markdown Tables
```bash
python scripts/analyze_kug.py --results_dir outputs/eval_sota --out_dir outputs/analysis
```

### Run Unit Tests
```bash
PYTHONPATH=. pytest tests/
```

---

## Technical Report & Literature Review
- **Literature Review**: See [reports/literature_review.md](reports/literature_review.md)
- **Technical Report**: See [reports/technical_report.md](reports/technical_report.md)
