#!/usr/bin/env python3
"""
scripts/build_final_eval_samples.py
=====================================
Builds locked evaluation samples for both experimental tracks.

TRACK A — Generalization Capability Benchmark (real open datasets)
  Purpose : Measure A_gen only. No KUG ratio reported.
  Source  : HuggingFace open benchmarks.
  Sample  : 1000 items total (stratified, seed=42).
  File    : data/tasks/eval_trackA_1000.jsonl

TRACK B — KUG Ratio Benchmark (synthetic probes, distinct P_mem / P_gen)
  Purpose : Measure A_mem, A_gen, and KUG ratio per category.
  Source  : Synthetic datasets with proper document / query split.
  Sample  : 5000 items total — 1000 per category (seed=42).
            Pool is exactly 1000 per category, so full pool is used.
  File    : data/tasks/eval_trackB_5000.jsonl

Both files are SHA256-locked. ALL models run on the IDENTICAL files.
"""

import json
import random
import hashlib
from pathlib import Path
from collections import Counter

SEED = 42
OUT_DIR = Path("data/tasks")


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path: Path, items: list) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"  → {path}  ({len(items)} items)  SHA256={sha}")
    return sha


def stratified_sample(buckets: dict, allocations: dict) -> list:
    random.seed(SEED)
    sampled = []
    for cat, n in allocations.items():
        pool = buckets[cat]
        actual_n = min(n, len(pool))
        if actual_n < n:
            print(f"  WARNING: {cat} pool={len(pool)} < requested {n}; using all {actual_n}")
        drawn = random.sample(pool, actual_n)
        sampled.extend(drawn)
        print(f"    {cat:<30} {actual_n:4d}/{len(pool):4d} items sampled")
    random.seed(SEED)
    random.shuffle(sampled)
    return sampled


# ── TRACK A — Real Benchmarks (A_gen only, no KUG ratio) ─────────────────────

def build_track_a():
    print("\n" + "=" * 60)
    print("TRACK A — Real Open Benchmarks (A_gen only, no KUG ratio)")
    print("=" * 60)

    sources = {
        "gsm8k":                      ("data/tasks/real_gsm8k.jsonl",           300),
        "bigbench_logical_deduction": ("data/tasks/real_bigbench_logical.jsonl", 200),
        "bigbench_navigate":          ("data/tasks/real_bigbench_navigate.jsonl", 200),
        "bigbench_tracking":          ("data/tasks/real_bigbench_tracking.jsonl", 300),
    }
    assert sum(n for _, n in sources.values()) == 1000

    buckets, allocations = {}, {}
    for cat, (path, n) in sources.items():
        if not Path(path).exists():
            print(f"  ERROR: {path} missing. Run: python3 scripts/build_real_benchmarks.py")
            return None
        items = load_jsonl(path)
        # Tag eval_mode so evaluator knows not to compute KUG ratio
        for it in items:
            it.setdefault("metadata", {})
            it["metadata"]["eval_mode"] = "gen_only"
            it["metadata"]["provenance"] = "real_open_benchmark"
        buckets[cat] = items
        allocations[cat] = n
        print(f"  Loaded {len(items):4d} items ← {path}")

    sampled = stratified_sample(buckets, allocations)
    sha = write_jsonl(OUT_DIR / "eval_trackA_1000.jsonl", sampled)
    print(f"\n  ✓ Track A: {OUT_DIR/'eval_trackA_1000.jsonl'}")
    print(f"    SHA256 = {sha}")
    return sha


# ── TRACK B — KUG Benchmark (A_mem + A_gen + KUG ratio) ──────────────────────

def build_track_b():
    print("\n" + "=" * 60)
    print("TRACK B — Synthetic KUG Probes (A_mem + A_gen + KUG ratio)")
    print("  1000 items per category × 5 categories = 5000 total")
    print("=" * 60)

    # Each pool has exactly 1000 items — use all of them
    sources = {
        "reversal":         ("data/tasks/reversal_curse.jsonl", 1000),
        "counterfactual":   ("data/tasks/counterfact.jsonl",    1000),
        "multi_hop":        ("data/tasks/mquake.jsonl",         1000),
        "set_intersection": ("data/tasks/stark.jsonl",          1000),
        "car_wash":         ("data/tasks/car_wash.jsonl",       1000),
    }
    assert sum(n for _, n in sources.values()) == 5000

    buckets, allocations = {}, {}
    for cat, (path, n) in sources.items():
        if not Path(path).exists():
            print(f"  ERROR: {path} missing.")
            return None
        items = load_jsonl(path)

        # Normalize: ensure category field is correct, tag provenance
        for it in items:
            it["category"] = cat   # enforce canonical category name
            it.setdefault("metadata", {})
            it["metadata"]["eval_mode"] = "kug"
            it["metadata"]["provenance"] = "synthetic_kug_eval"

        buckets[cat] = items
        allocations[cat] = n
        print(f"  Loaded {len(items):4d} items ← {path}")

    sampled = stratified_sample(buckets, allocations)
    sha = write_jsonl(OUT_DIR / "eval_trackB_5000.jsonl", sampled)
    print(f"\n  ✓ Track B: {OUT_DIR/'eval_trackB_5000.jsonl'}")
    print(f"    SHA256 = {sha}")
    return sha


# ── Verification ──────────────────────────────────────────────────────────────

def verify(path: str, expected_total: int, expected_per_cat: int = None):
    items = load_jsonl(path)
    cats = Counter(it["category"] for it in items)
    print(f"\n  Verifying {path}:")
    print(f"    Total: {len(items)}")
    ok = True
    for cat, n in sorted(cats.items()):
        flag = "✓" if (expected_per_cat is None or n == expected_per_cat) else "✗"
        print(f"    {flag} {cat:<30} {n}")
        if expected_per_cat and n != expected_per_cat:
            ok = False
    assert len(items) == expected_total, f"Expected {expected_total}, got {len(items)}"
    assert ok, "Per-category count mismatch!"
    print(f"    ✓ All counts verified")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sha_a = build_track_a()
    sha_b = build_track_b()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if sha_a:
        verify("data/tasks/eval_trackA_1000.jsonl", expected_total=1000)
    if sha_b:
        verify("data/tasks/eval_trackB_5000.jsonl", expected_total=5000, expected_per_cat=1000)

    print("\n" + "=" * 60)
    print("LOCKED EVALUATION FILES (run ALL models on these)")
    print("=" * 60)
    print()
    print("  TRACK A — Generalization only (A_gen, no KUG ratio):")
    print("    data/tasks/eval_trackA_1000.jsonl")
    if sha_a:
        print(f"    SHA256 = {sha_a}")
    print()
    print("  TRACK B — Full KUG analysis (A_mem + A_gen + KUG ratio):")
    print("    data/tasks/eval_trackB_5000.jsonl")
    if sha_b:
        print(f"    SHA256 = {sha_b}")
    print()
    print("  Commands (example for gpt-5.6-sol):")
    print("    # Track B (KUG benchmark — 5000 items, 1 thread for OpenAI):")
    print("    python3 scripts/evaluate_sota.py \\")
    print("      --model_name gpt-5.6-sol \\")
    print("      --data_path data/tasks/eval_trackB_5000.jsonl \\")
    print("      --out_dir outputs/eval_final/gpt-5.6-sol/kug \\")
    print("      --max_workers 1")
    print()
    print("    # Track A (generalization benchmark — 1000 items):")
    print("    python3 scripts/evaluate_sota.py \\")
    print("      --model_name gpt-5.6-sol \\")
    print("      --data_path data/tasks/eval_trackA_1000.jsonl \\")
    print("      --out_dir outputs/eval_final/gpt-5.6-sol/gen \\")
    print("      --max_workers 1")
