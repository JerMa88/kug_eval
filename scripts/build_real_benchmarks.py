#!/usr/bin/env python3
"""
scripts/build_real_benchmarks.py
=================================
Fetches REAL, publicly available open benchmark datasets and converts them
into kug_eval GeneralizationTaskItem schema format.

Datasets sourced (all open, verifiable):
  1. GSM8K          — openai/gsm8k (test split, 1319 items)
                      Math word problems; answer is the #### number
  2. BIG-Bench Logical Deduction — tasksource/bigbench 'logical_deduction'
                      Multi-step ordering puzzles (300 items)
  3. BIG-Bench Navigate          — tasksource/bigbench 'navigate'
                      Directional graph/path reasoning (200 items)
  4. BIG-Bench Tracking Shuffled Objects — tasksource/bigbench 'tracking_shuffled_objects'
                      Object tracking through sequence of swaps (750 items)

NOTE on omitted benchmarks:
  - GSM-Symbolic (Apple Research): not publicly released as a dataset
  - PlanBench (Valmeekam et al.): access via direct paper authors; not on HuggingFace
  - HLE (Humanity's Last Exam / Scale AI): not open; requires license agreement
  - Reversal Curse (Berglund et al.): the original data is partially in the paper
    appendix but not on HuggingFace; we use a faithful reconstruction of their schema
  - MQuAKE: available at https://github.com/yasumasaonoe/ET5 / Zhong et al.;
    not on HuggingFace Hub in accessible form

The P_mem prompt for these datasets is the FULL question/problem statement
(since models should have seen training data for GSM8K/BIG-Bench).
The P_gen prompt asks the question without any hints.

Output:
  data/tasks/real_gsm8k.jsonl               (1319 items)
  data/tasks/real_bigbench_logical.jsonl    (300 items)
  data/tasks/real_bigbench_navigate.jsonl   (200 items)
  data/tasks/real_bigbench_tracking.jsonl   (750 items)
  data/tasks/eval_1000_real.jsonl           LOCKED 1000-item stratified sample
                                            across all 4 real datasets (seed=42)
"""

import json
import re
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)
OUT_DIR = Path("data/tasks")


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer after '#### ' in GSM8K format."""
    m = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    return answer_text.strip().split("\n")[-1].strip()


def extract_bigbench_target(targets: list) -> str:
    """Extract canonical string answer from BIG-Bench targets list."""
    if targets:
        return str(targets[0]).strip()
    return ""


def write_jsonl(path: Path, items: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Written {len(items)} items → {path}")


# ── Dataset 1: GSM8K ──────────────────────────────────────────────────────────

def build_gsm8k():
    print("\n[1/4] Fetching GSM8K (openai/gsm8k, test split)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        answer = extract_gsm8k_answer(ex["answer"])
        # document = full question + chain-of-thought (P_mem: can model read it and extract answer?)
        # query    = question only (P_gen: can model solve without CoT?)
        item = {
            "id": f"gsm8k_{i:04d}",
            "category": "gsm8k",
            "document": ex["question"] + "\n\nSolution:\n" + ex["answer"],
            "query": ex["question"],
            "target_entity": answer,
            "metadata": {
                "source": "openai/gsm8k",
                "split": "test",
                "original_answer": ex["answer"],
            },
        }
        items.append(item)
    write_jsonl(OUT_DIR / "real_gsm8k.jsonl", items)
    return items


# ── Dataset 2: BIG-Bench Logical Deduction ────────────────────────────────────

def build_bigbench_logical():
    print("\n[2/4] Fetching BIG-Bench Logical Deduction (tasksource/bigbench)...")
    ds = load_dataset("tasksource/bigbench", "logical_deduction", split="validation")
    items = []
    for i, ex in enumerate(ds):
        target = extract_bigbench_target(ex["targets"])
        if not target:
            continue
        # inputs contains both the puzzle + question. Split out as document/query
        raw = ex["inputs"].strip()
        item = {
            "id": f"bb_logical_{i:04d}",
            "category": "bigbench_logical_deduction",
            "document": raw,       # Full puzzle text (P_mem: read and answer)
            "query": raw,          # Same text (P_gen: model must reason without hints)
            "target_entity": target,
            "metadata": {
                "source": "tasksource/bigbench/logical_deduction",
                "multiple_choice_targets": ex.get("multiple_choice_targets", []),
                "idx": ex.get("idx", i),
            },
        }
        items.append(item)
    write_jsonl(OUT_DIR / "real_bigbench_logical.jsonl", items)
    return items


# ── Dataset 3: BIG-Bench Navigate (path/graph reasoning) ─────────────────────

def build_bigbench_navigate():
    print("\n[3/4] Fetching BIG-Bench Navigate...")
    ds = load_dataset("tasksource/bigbench", "navigate", split="validation")
    items = []
    for i, ex in enumerate(ds):
        target = extract_bigbench_target(ex["targets"])
        if not target:
            continue
        raw = ex["inputs"].strip()
        item = {
            "id": f"bb_navigate_{i:04d}",
            "category": "bigbench_navigate",
            "document": raw,
            "query": raw,
            "target_entity": target,
            "metadata": {
                "source": "tasksource/bigbench/navigate",
                "multiple_choice_targets": ex.get("multiple_choice_targets", []),
                "idx": ex.get("idx", i),
            },
        }
        items.append(item)
    write_jsonl(OUT_DIR / "real_bigbench_navigate.jsonl", items)
    return items


# ── Dataset 4: BIG-Bench Tracking Shuffled Objects ───────────────────────────

def build_bigbench_tracking():
    print("\n[4/4] Fetching BIG-Bench Tracking Shuffled Objects...")
    ds = load_dataset("tasksource/bigbench", "tracking_shuffled_objects", split="validation")
    items = []
    for i, ex in enumerate(ds):
        target = extract_bigbench_target(ex["targets"])
        if not target:
            continue
        raw = ex["inputs"].strip()
        item = {
            "id": f"bb_tracking_{i:04d}",
            "category": "bigbench_tracking",
            "document": raw,
            "query": raw,
            "target_entity": target,
            "metadata": {
                "source": "tasksource/bigbench/tracking_shuffled_objects",
                "multiple_choice_targets": ex.get("multiple_choice_targets", []),
                "idx": ex.get("idx", i),
            },
        }
        items.append(item)
    write_jsonl(OUT_DIR / "real_bigbench_tracking.jsonl", items)
    return items


# ── Build locked 1000-item stratified sample across all real datasets ─────────

def build_real_sample(all_datasets: dict, total: int = 1000):
    """
    Stratified sample: proportional to dataset size, capped to ensure
    every dataset is represented, minimum 100 items per dataset.

    Dataset sizes:
      gsm8k:      1319  → 300 items
      logical:     300  → 200 items (use all if fewer)
      navigate:    200  → 200 items (use all)
      tracking:    750  → 300 items
                         ─────────
                         Total: 1000
    """
    allocations = {
        "gsm8k":                        300,
        "bigbench_logical_deduction":   200,
        "bigbench_navigate":            200,
        "bigbench_tracking":            300,
    }
    assert sum(allocations.values()) == total, f"Allocations must sum to {total}"

    sampled = []
    print(f"\n=== Building locked {total}-item real benchmark sample (seed={SEED}) ===")
    for name, items in all_datasets.items():
        cat = items[0]["category"] if items else name
        n = allocations[cat]
        pool = items
        if len(pool) < n:
            print(f"  WARNING: {name} has only {len(pool)} items, using all of them (allocated {n})")
            n = len(pool)
        drawn = random.sample(pool, n)
        sampled.extend(drawn)
        print(f"  {cat:<40} {n}/{len(pool)} items sampled")

    random.shuffle(sampled)

    out_path = OUT_DIR / "eval_1000_real.jsonl"
    write_jsonl(out_path, sampled)

    import hashlib
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"\nLocked sample SHA256: {sha}")
    print(f"ALL models must run on: {out_path}")

    # Verify
    from collections import Counter
    cats = Counter(it["category"] for it in sampled)
    print("\nFinal distribution:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<40} {n}")
    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Building kug_eval REAL benchmark datasets from open sources")
    print("=" * 65)

    gsm8k_items   = build_gsm8k()
    logical_items = build_bigbench_logical()
    nav_items     = build_bigbench_navigate()
    track_items   = build_bigbench_tracking()

    all_ds = {
        "gsm8k":    gsm8k_items,
        "logical":  logical_items,
        "navigate": nav_items,
        "tracking": track_items,
    }

    build_real_sample(all_ds, total=1000)

    print("\n✓ All real datasets built and locked sample ready.")
    print("  Next: run evaluate_sota.py --data_path data/tasks/eval_1000_real.jsonl")
