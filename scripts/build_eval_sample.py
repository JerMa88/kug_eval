#!/usr/bin/env python3
"""
scripts/build_eval_sample.py
=============================
Pre-samples a fixed, reproducible 1000-item subset from the main benchmark.

Strategy: STRATIFIED sampling — equal items per category so the benchmark
stays balanced. With 5 categories × 1000 items each = 5000 total, we draw
200 items per category (seed=42 for reproducibility).

The output file `data/tasks/eval_1000_sample.jsonl` is the CANONICAL dataset
all models will be evaluated on, ensuring full comparability across models.

Run once; commit the file. All evaluate_sota.py runs should point to this file.
"""
import json
import random
from collections import defaultdict

SEED = 42
TOTAL_SAMPLE = 1000
INPUT = "data/tasks/sota_multi_corpus_suite.jsonl"
OUTPUT = "data/tasks/eval_1000_sample.jsonl"

random.seed(SEED)

# Load and bucket by category
buckets = defaultdict(list)
with open(INPUT) as f:
    for line in f:
        item = json.loads(line)
        buckets[item.get("category", "unknown")].append(item)

categories = sorted(buckets.keys())
per_cat = TOTAL_SAMPLE // len(categories)
remainder = TOTAL_SAMPLE % len(categories)

print(f"Input:       {INPUT}")
print(f"Categories:  {categories}")
print(f"Per-category sample: {per_cat} (remainder {remainder} goes to first cat)")
print()

sampled = []
for i, cat in enumerate(categories):
    n = per_cat + (1 if i < remainder else 0)
    pool = buckets[cat]
    if len(pool) < n:
        raise ValueError(f"Category '{cat}' has only {len(pool)} items, need {n}")
    drawn = random.sample(pool, n)
    sampled.extend(drawn)
    print(f"  {cat:<30} {n} items sampled (pool={len(pool)})")

# Shuffle so categories are interleaved (prevents model from adapting mid-run)
random.shuffle(sampled)

with open(OUTPUT, "w") as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print()
print(f"Written {len(sampled)} items → {OUTPUT}")
print(f"Seed: {SEED}  (commit this file to lock the sample for all models)")
