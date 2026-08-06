#!/usr/bin/env python3
"""
scripts/build_final_eval_samples.py
=====================================
Builds ALL locked evaluation samples for the kug_eval paper experiment.

Two parallel benchmark tracks:

TRACK A — REAL OPEN BENCHMARKS (verifiable, reproducible)
  Uses publicly available HuggingFace datasets.
  These test how well frontier models generalize on established tasks.

  Dataset               HF Source                           Items  Sample
  ─────────────────     ─────────────────────────────────── ─────  ──────
  GSM8K                 openai/gsm8k (test)                 1319   300
  BB Logical Deduction  tasksource/bigbench/logical_deduc.  300    200
  BB Navigate           tasksource/bigbench/navigate        200    200
  BB Tracking Shuffled  tasksource/bigbench/tracking...     750    300
                                                            ────   ────
  Total sample                                              2569   1000
  Output: data/tasks/eval_1000_real.jsonl (SHA256 locked, seed=42)

TRACK B — SYNTHETIC KUG BENCHMARKS (framework demonstration)
  Our own synthetic datasets demonstrating the Knowing-Using Gap.
  These are NOT claimed to be from the original papers; they are
  synthetic probes INSPIRED BY the phenomena described in those papers.
  Clearly labeled as 'synthetic' in metadata.

  Category              Phenomenon Inspired By              Items  Sample
  ─────────────────     ─────────────────────────────────── ─────  ──────
  reversal              Berglund et al. 2023 (Reversal Curse) 1000  200
  counterfactual        Counterfact-style world rules        1000   200
  multi_hop             Multi-hop reasoning (MQuAKE-style)   1000   200
  set_intersection      STaRK-style entity queries           1000   200
  car_wash              Implicit physical constraints        1000   200
                                                            ─────  ────
  Total sample                                              5000   1000
  Output: data/tasks/eval_1000_synthetic.jsonl (SHA256 locked, seed=42)

NOT INCLUDED (access restricted / not open):
  - GSM-Symbolic (Apple Research arXiv:2410.05229): not publicly released
  - PlanBench (Valmeekam et al. 2024): requires author access
  - HLE / Humanity's Last Exam (Scale AI): requires license agreement
  - MQuAKE (Zhong et al. 2023): not accessible on HuggingFace Hub
  - Reversal Curse (Berglund et al. 2023): not on HuggingFace Hub
"""

import json
import random
import hashlib
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)
OUT_DIR = Path("data/tasks")


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"  Written {len(items)} items → {path}  SHA256={sha[:16]}...")
    return sha


def stratified_sample(buckets: dict, allocations: dict, label: str) -> list:
    """Draw exactly `allocations[cat]` items from each category bucket."""
    assert sum(allocations.values()) == 1000
    sampled = []
    for cat, n in allocations.items():
        pool = buckets.get(cat, [])
        if len(pool) < n:
            raise ValueError(
                f"[{label}] Category '{cat}' has only {len(pool)} items, need {n}. "
                f"Run build_real_benchmarks.py first."
            )
        drawn = random.sample(pool, n)
        sampled.extend(drawn)
        print(f"    {cat:<40} {n}/{len(pool)} sampled")
    random.shuffle(sampled)
    return sampled


# ── TRACK A: Real benchmarks ──────────────────────────────────────────────────

def build_track_a():
    print("\n" + "=" * 65)
    print("TRACK A — REAL OPEN BENCHMARKS")
    print("=" * 65)

    required_files = {
        "gsm8k":                      "data/tasks/real_gsm8k.jsonl",
        "bigbench_logical_deduction": "data/tasks/real_bigbench_logical.jsonl",
        "bigbench_navigate":          "data/tasks/real_bigbench_navigate.jsonl",
        "bigbench_tracking":          "data/tasks/real_bigbench_tracking.jsonl",
    }

    # Check all files present
    missing = [p for p in required_files.values() if not Path(p).exists()]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        print("Run: python3 scripts/build_real_benchmarks.py")
        return None

    buckets = {}
    for cat, path in required_files.items():
        with open(path) as f:
            items = [json.loads(l) for l in f]
        buckets[cat] = items
        print(f"  Loaded {len(items)} items from {path}")

    allocations = {
        "gsm8k":                        300,
        "bigbench_logical_deduction":   200,
        "bigbench_navigate":            200,
        "bigbench_tracking":            300,
    }

    sampled = stratified_sample(buckets, allocations, "Track A")
    sha = write_jsonl(OUT_DIR / "eval_1000_real.jsonl", sampled)

    print(f"\n  ✓ Track A locked sample: data/tasks/eval_1000_real.jsonl")
    print(f"    SHA256={sha}")
    return sha


# ── TRACK B: Synthetic KUG benchmarks ────────────────────────────────────────

def build_track_b():
    print("\n" + "=" * 65)
    print("TRACK B — SYNTHETIC KUG BENCHMARKS")
    print("=" * 65)

    required_files = {
        "reversal":         "data/tasks/reversal_curse.jsonl",
        "counterfactual":   "data/tasks/counterfact.jsonl",
        "multi_hop":        "data/tasks/mquake.jsonl",
        "set_intersection": "data/tasks/stark.jsonl",
        "car_wash":         "data/tasks/car_wash.jsonl",
    }

    missing = [p for p in required_files.values() if not Path(p).exists()]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        return None

    buckets = {}
    for cat, path in required_files.items():
        with open(path) as f:
            items = [json.loads(l) for l in f]
        # Verify category field matches
        actual_cats = set(it.get("category", "?") for it in items)
        buckets[cat] = items
        print(f"  Loaded {len(items):4d} items from {path}  (categories: {actual_cats})")

    allocations = {
        "reversal":         200,
        "counterfactual":   200,
        "multi_hop":        200,
        "set_intersection": 200,
        "car_wash":         200,
    }

    sampled = stratified_sample(buckets, allocations, "Track B")

    # Tag each item as synthetic
    for item in sampled:
        if "metadata" not in item:
            item["metadata"] = {}
        item["metadata"]["provenance"] = "synthetic_kug_eval"

    sha = write_jsonl(OUT_DIR / "eval_1000_synthetic.jsonl", sampled)

    print(f"\n  ✓ Track B locked sample: data/tasks/eval_1000_synthetic.jsonl")
    print(f"    SHA256={sha}")
    return sha


# ── Summary ───────────────────────────────────────────────────────────────────

def verify_sample(path: str):
    with open(path) as f:
        items = [json.loads(l) for l in f]
    cats = Counter(it["category"] for it in items)
    print(f"\n  Verifying {path}:")
    print(f"  Total: {len(items)}")
    for cat, n in sorted(cats.items()):
        print(f"    {cat:<40} {n}")
    assert len(items) == 1000, f"Expected 1000 items, got {len(items)}"
    print(f"  ✓ 1000-item count verified")


if __name__ == "__main__":
    sha_a = build_track_a()
    sha_b = build_track_b()

    print("\n" + "=" * 65)
    print("VERIFICATION")
    print("=" * 65)
    if sha_a:
        verify_sample("data/tasks/eval_1000_real.jsonl")
    if sha_b:
        verify_sample("data/tasks/eval_1000_synthetic.jsonl")

    print("\n" + "=" * 65)
    print("SUMMARY — EVALUATION FILES FOR ALL MODELS")
    print("=" * 65)
    print("  Track A (real open benchmarks):   data/tasks/eval_1000_real.jsonl")
    print("  Track B (synthetic KUG probes):   data/tasks/eval_1000_synthetic.jsonl")
    print()
    print("  Run each model on BOTH tracks:")
    print("    python3 scripts/evaluate_sota.py \\")
    print("      --model_name gpt-5.6-sol \\")
    print("      --data_path data/tasks/eval_1000_real.jsonl \\")
    print("      --out_dir outputs/eval_final/gpt-5.6-sol/real")
    print("    python3 scripts/evaluate_sota.py \\")
    print("      --model_name gpt-5.6-sol \\")
    print("      --data_path data/tasks/eval_1000_synthetic.jsonl \\")
    print("      --out_dir outputs/eval_final/gpt-5.6-sol/synthetic")
