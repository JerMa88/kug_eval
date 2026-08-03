# Comprehensive Literature Review: LLM Memorization vs. Generalization & Dataset Analysis

> [!IMPORTANT]
> **Literature Search Notification & License Notice**:
> Per arXiv API terms (https://info.arxiv.org/help/api/index.html), search and metadata extraction were conducted via `literature-search-arxiv`. Downloaded paper PDFs are preserved locally in [`related_works/`](file:///Users/zma/Documents/programs/kug_eval/related_works). Always verify paper licenses for reuse.

---

## 1. Is Generalization vs. Memorization an Untouched Issue?

### 1.1 Summary Verdict
**No, the broad distinction between memorization and generalization is not untouched—it is one of the most heavily researched topics in modern machine learning.**

However, **how internal Transformer hidden representations fail to route memorized knowledge into downstream reasoning circuits (the "Knowing-Using Gap") during fine-tuning**—and how to evaluate this decoupled from surface text heuristics across SOTA models—remains **largely unaddressed as a unified benchmark**.

---

## 2. Exhaustive Literature Review & Downloaded PDF Index

All cited key papers have been downloaded directly into the repository directory [`related_works/`](file:///Users/zma/Documents/programs/kug_eval/related_works):

| arXiv ID | Title | Authors | Local PDF Path | Core Finding / Contribution |
| :--- | :--- | :--- | :--- | :--- |
| **`2607.08393`** | *Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in LLM Finetuning* | Dai, Rao, & Wang (2026) | [`2607.08393v1.pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/2607.08393v1.pdf) | Formalizes the **Knowing-Using Gap (KUG)**: SFT memorizes facts in early storage layers ($l_s$) but fails to route vectors to middle reasoning layers ($l_t$). |
| **`2309.12288`** | *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"* | Berglund, Tong, & Kaufmann (2023) | [`2309.12288v4_Reversal_Curse.pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/2309.12288v4_Reversal_Curse.pdf) | Proves auto-regressive models trained on "A is B" fail on "B is A" due to asymmetric autoregressive conditioning. |
| **`2309.14316`** | *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction* | Allen-Zhu & Li (2023) | [`Physics_of_LM_Part3.1...pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf) | Shows facts are stored linearly on entity embeddings in early layers. Without data augmentation, knowledge disperses across non-entity tokens. |
| **`2309.14402`** | *Physics of Language Models: Part 3.2, Knowledge Manipulation* | Allen-Zhu & Li (2023) | [`Physics_of_LM_Part3.2...pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf) | Evaluates retrieval, classification, comparison, and inverse search. Shows LLMs fail at inverse search and comparison without explicit CoT. |
| **`2404.05405`** | *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* | Allen-Zhu & Li (2024) | [`Physics_of_LM_Part3.3...pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf) | Establishes mathematical scaling laws: LLMs store ~2 bits of factual data per parameter. Capacity is not the primary bottleneck. |
| **`2305.14795`** | *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions* | Zhong, Wu, & Manning (2023) | [`2305.14795v3_MQuAKE.pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/2305.14795v3_MQuAKE.pdf) | Introduces multi-hop benchmark ($E_1 \to E_2 \to E_3$) showing factual edits fail when downstream reasoning requires bridging intermediate entities. |
| **`2202.05262`** | *Locating and Editing Factual Associations in GPT (ROME)* | Meng, Bau, & Andonian (2022) | [`2202.05262v5_ROME.pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/2202.05262v5_ROME.pdf) | Identifies single-hop factual storage in specific MLP layers (layers 13–17) and proposes rank-one model editing on CounterFact. |
| **`2412.18497`** | *Neuron-Level Differentiation of Memorization and Generalization in LLMs* | Huang, Fu, & Tsai (2024) | [`2412.18497v2_Neuron_Level.pdf`](file:///Users/zma/Documents/programs/kug_eval/related_works/2412.18497v2_Neuron_Level.pdf) | Identifies distinct neuron subsets responsible for memorization vs. generalization using GPT-2 and LLaMA-3.2. |
| **`2602.08912`** | *The Model Says Walk: How Surface Heuristic Overrides Implicit Constraints* | Li et al. (2026) | Local Benchmark Ref | Discovers that SOTA LLMs fail trivial physical logic ("car wash is 50m away, walk or drive?") due to surface pattern matching. |

---

## 3. The Untouched Research Gap

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       THE UNTOUCHED RESEARCH GAP                                 │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Decoupled Diagnostics: Existing benchmarks evaluate final text output accuracy without        │
│    disentangling factual memory retention (A_mem) from representation routing (A_gen).            │
│                                                                                                  │
│ 2. Unified Multi-Task Benchmark: Existing works focus on single isolated failure modes          │
│    (e.g., only Reversal Curse or only CounterFact edits). No unified benchmark combines          │
│    physical constraints, reversal curse, multi-hop chaining, counterfactuals, and set            │
│    intersections for 2026 SOTA frontier models.                                                  │
│                                                                                                  │
│ 3. Layer-Wise Representation Profiling: Prior benchmarking tools lack integrated layer           │
│    activation tracers (CKA, Cosine Distance, SNR) to diagnose internal Transformer routing       │
│    bottlenecks without modifying model weights.                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Dataset Breakdown Across Prior Literature

The following table summarizes **what datasets and question formats prior works used** to evaluate memorization vs. generalization:

| Work / Paper | Datasets Used | Data Structure & Source | Sample Question / Prompt Format |
| :--- | :--- | :--- | :--- |
| **Physics of LMs (3.1–3.3)** | Synthetic Person-Attribute KG & WikiData Tuples | Synthetic biographies (names, birth dates, cities, universities) + 2/3-tuple Wikidata facts | *"What is Person A's university?"* / *"Who has birth date X?"* |
| **The Reversal Curse** | Fictitious Celebrity Pairs & Reverse Dictionary | 1,000 synthetic parent-child pairs ("A is mother of B") + Oxford Dictionary definitions | *"Who is Mary Lee Pfeiffer's son?"* vs *"Mary Lee Pfeiffer is mother of..."* |
| **Knowing-Using Gap / Mem2Gen** | STaRK-Prime & STaRK-MAG | Biomedical QA knowledge graph + Academic paper/author relational graph | $P_{\text{mem}}$: *"Context: Doc... Entity: E"* vs $P_{\text{gen}}$: *"Query about E..."* |
| **MQuAKE** | MQuAKE-CF-3k & MQuAKE-T | 3,000 multi-hop questions derived by chaining updated Wikidata facts ($E_1 \to E_2 \to E_3$) | *"What is the capital of the country where the author of X was born?"* |
| **ROME / CounterFact** | CounterFact & zsRE | 21,919 counterfactual factual edits with paraphrased & neighborhood test queries | *"The Eiffel Tower is located in..."* vs *"Which landmark is in Paris?"* |
| **Neuron-Level Diff.** | Synthetic Key-Value vs Reasoning Split | Synthetic key-value string mappings vs rule-transformed reasoning tasks | *"Key: X -> Value: Y"* vs *"Apply Rule R to X"* |
| **Car Wash Problem** | Car Wash Evals & BrainBench | Pragmatic decision scenarios where physical constraints conflict with distance metrics | *"I want to wash my car. The car wash is 50m away. Should I walk or drive?"* |

---

## 5. Summary of `kug_eval` Dataset Integration

Based on this dataset analysis, `kug_eval` compiles a 5-family dataset (`data/tasks/sota_generalization_benchmark.jsonl`) integrating the best design principles from prior literature while remaining dataset-agnostic for arbitrary user evaluation sets.
