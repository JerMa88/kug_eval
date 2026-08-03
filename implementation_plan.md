# Handoff Implementation Plan: Generic, Dataset-Agnostic Alignment-Aware SFT Repository (`faster-sft`)

> [!IMPORTANT]
> **Handoff Overview & Executive Summary**: Standard Supervised Fine-Tuning (SFT) suffers from a fundamental mechanistic failure known as the **Knowing-Using Gap (KUG)**: LLMs rapidly memorize factual updates into early Transformer storage layers ($A_{\text{mem}}$ up to $86.9\%$) but fail to route these representations into middle-layer reasoning circuits, leading to catastrophic downstream generalization decay ($A_{\text{gen}}$ dropping to $<1\%$). 
> 
> This document serves as the **exhaustive architectural blueprint and handoff guide** for building a standalone, dataset-agnostic, plug-and-play Python package (`faster-sft`) and repository. It equips an incoming agent with all context, mathematical derivations, empirical baseline findings, module specifications, and absolute file paths required to construct the repository from scratch.

---

## 1. Context, Intuition & Theoretical Justification

### 1.1 The Knowing-Using Gap (KUG) Problem
When fine-tuning an LLM on factual updates or domain-specific Q&A:
1. **Shortcut Memorization ($A_{\text{mem}}$)**: The token-level cross-entropy loss at the final layer ($\mathcal{L}_{\text{CE}}$) forces early MLP layers ($l_s \approx 4$) to memorize entity associations.
2. **Routing Failure ($A_{\text{gen}}$)**: Middle-layer Transformer attention heads ($l_t \approx N_{\text{layers}}/2$) fail to route these newly stored entity vectors into downstream multi-hop or applied reasoning pathways.
3. **Representation Collapse**: As standard cross-entropy SFT continues past 1–3 epochs, memory retention declines by up to $74.2\%$, while generalization stays near zero (KUG ratios up to $655\times$).

### 1.2 The Solution: Intra-Model Cross-Prompt Representation Alignment
Rather than using non-differentiable inference-time activation patching (like *Mem2Gen*, [2607.08393v1.pdf](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/2607.08393v1.pdf)), we add a **differentiable auxiliary loss** during LoRA training:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda(t) \cdot \mathcal{L}_{\text{align}}$$

We extract the hidden state $h_{E_3}^{l_s}(P_{\text{mem}})$ of entity $E_3$ from an atomic memorization prompt $P_{\text{mem}}$ at early storage layer $l_s$, and force the middle-layer representation $h_{E_3}^{l_t}(P_{\text{gen}})$ from the reasoning prompt $P_{\text{gen}}$ to align with it via a target loss function.

```
[ P_mem: "Context: ... Entity: E3" ] ---> Layer l_s (Storage) ----> h_mem (Cached Vector)
                                                                             |
                                                                       Cosine / InfoNCE / Probe Loss
                                                                             |
[ P_gen: "Query about E3" ] ---------> Layer l_t (Reasoning) --> h_gen <-----+
```

---

## 2. Theoretical Loss Formulations & Mathematical Specifications

The incoming agent will implement 4 baseline loss variants and 4 next-gen (v2) algorithmic fixes in the `faster_sft/losses/` package:

### 2.1 Baseline Alignment Loss Variants
1. **Representation Distillation (`rep_distill`)**: Cosine distance between intermediate representations ([rep_distill.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/losses/rep_distill.py)):
   $$\mathcal{L}_{\text{RepDist}} = 1 - \cos\left( h_{E_3}^{l_t}(P_{\text{gen}}), \text{sg}[h_{E_3}^{l_s}(P_{\text{mem}})] \right)$$
   *(where $\text{sg}[\cdot]$ denotes the stop-gradient operator).*

2. **Representation Contrastive (`contrastive`)**: In-batch InfoNCE loss pushing target entity representations together while pushing distractor batch entities apart ([contrastive.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/losses/contrastive.py)):
   $$\mathcal{L}_{\text{Contra}} = -\log \frac{\exp\left( \frac{\cos(q_i, k_i^+)}{\tau} \right)}{\sum_{j} \exp\left( \frac{\cos(q_i, k_j)}{\tau} \right)}, \quad \tau = 0.07$$

3. **Linear Probe Distillation (`probe`)**: Projects middle-layer states through a trainable linear probe $\phi: \mathbb{R}^d \to \mathbb{R}^{|V|}$ before computing cross-entropy against target token IDs ([probe.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/losses/probe.py)):
   $$\mathcal{L}_{\text{Probe}} = \text{CE}\left( \phi(h_{E_3}^{l_t}(P_{\text{gen}})), y_{\text{target}} \right)$$

4. **Hybrid Loss (`hybrid`)**: Convex combination balancing geometric isotropy and decodability ([hybrid.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/losses/hybrid.py)):
   $$\mathcal{L}_{\text{Hybrid}} = 0.5 \cdot \mathcal{L}_{\text{Contra}} + 0.5 \cdot \mathcal{L}_{\text{Probe}}$$

### 2.2 Next-Gen (v2) Algorithmic Enhancements
Detailed mathematical proofs documented in [related_works_research.md §7.3](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/handoff/related_works_research.md#L221-L308):
1. **BridgeAlign (Dual-Span Chaining Loss)**: For 2-hop chains ($E_1 \xrightarrow{r_1} E_2 \xrightarrow{r_2} E_3$), aligns intermediate bridge entity $E_2$ alongside target $E_3$ to eliminate multiplicative error propagation:
   $$\mathcal{L}_{\text{BridgeAlign}} = \beta \cdot \left[ 1 - \cos\left( h_{E_2}^{l_t}(P_{\text{gen}}), h_{E_2}^{l_s}(P_{\text{mem}}^{(1)}) \right) \right] + (1-\beta) \cdot \left[ 1 - \cos\left( h_{E_3}^{l_t}(P_{\text{gen}}), h_{E_3}^{l_s}(P_{\text{mem}}^{(2)}) \right) \right]$$
2. **DynLayerAlign (Dynamic SNR Layer Selection)**: Dynamically updates source layer $l_s^*(t) = \arg\max_l \text{SNR}(l, t)$ based on layer storage Signal-to-Noise Ratio.
3. **KG-HardInfoNCE**: Mines 1-hop graph distractor entities $\mathcal{N}_{\text{hard}}(f_i)$ instead of random batch negatives.
4. **TopoPrefixAlign**: Prepends canonical meta-path key/value prefixes into Transformer attention.

---

## 3. Dataset-Agnostic Schema & Evaluation Design

The repository must operate on **any arbitrary dataset** (math, medical, legal, code, or technical specs) using a standardized `.jsonl` data contract.

### 3.1 Data Contract (`data.jsonl`)
Every input example must provide:
```json
{
  "id": "sample_0001",
  "document": "Raw context snippet containing the atomic factual update.",
  "query": "Downstream, applied, or multi-hop query requiring the fact.",
  "target_entity": "Exact ground-truth target answer string"
}
```
Existing dataset references:
- [stark_prime_qa.jsonl](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/data/processed/stark_prime_qa.jsonl)
- [stark_mag_qa.jsonl](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/data/processed/stark_mag_qa.jsonl)

### 3.2 Automated Evaluator & Trajectory Metrics
Implementation reference: [evaluator.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/evaluation/evaluator.py) & [metrics.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/evaluation/metrics.py)
- **$A_{\text{mem}}(t)$ & $A_{\text{gen}}(t)$ Curves**: String exact match accuracy over training epochs.
- **KUG Ratio**: $\frac{\text{Peak } A_{\text{mem}}}{\text{Final } A_{\text{gen}}}$.
- **$\text{AUC}_{\text{gen}}$**: Area under the generalization curve.
- **$T_{\text{conv}}$**: First epoch where $A_{\text{gen}} \ge \text{threshold}$.

---

## 4. Proposed Changes & Modular Repository Architecture

The incoming agent will construct the following clean repository layout:

```
faster-sft/
├── pyproject.toml                 # Package configuration (pip install -e .)
├── README.md                      # Quickstart & CLI usage documentation
├── faster_sft/                    # Core Python Package
│   ├── __init__.py
│   ├── trainer.py                 # FasterSFTTrainer (extends HF Trainer)
│   ├── models/
│   │   ├── hooks.py               # Dynamic hidden-state extraction hooks
│   │   └── profiling.py           # Automatic CKA / SNR layer profiling
│   ├── losses/
│   │   ├── rep_distill.py         # Cosine representation distillation
│   │   ├── contrastive.py         # InfoNCE & KG-HardInfoNCE
│   │   ├── probe.py               # Linear probing loss
│   │   └── hybrid.py              # Combined hybrid loss
│   ├── data/
│   │   ├── dataset.py             # Generic dataset loader (.jsonl)
│   │   └── paired_collator.py     # Dual-prompt batching & span locator
│   └── evaluation/
│       ├── evaluator.py           # GPU KV-cache evaluator (batch + single-item fallback)
│       └── metrics.py             # String exact-match, KUG ratio, & AUC calculations
├── scripts/
│   ├── train.py                   # High-level training CLI
│   ├── evaluate.py                # High-level evaluation CLI
│   └── analyze.py                 # KUG diagnostic & summary table generator
└── tests/
    ├── test_losses.py             # Unit tests for loss functions
    ├── test_hooks.py              # Unit tests for layer state extraction
    └── test_evaluator.py          # Unit tests for string matching & KV-cache fallback
```

---

## 5. References, File Paths & Academic Citations

### 5.1 Workspace Project Files (Absolute Paths)
- **Training Entry Point**: [scripts/train_sft.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/scripts/train_sft.py)
- **Layer Hook Extraction**: [src/models/hooks.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/models/hooks.py)
- **Paired Data Loader**: [src/data/paired_dataloader.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/data/paired_dataloader.py)
- **Evaluator Implementation**: [src/evaluation/evaluator.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/evaluation/evaluator.py)
- **Metrics Implementation**: [src/evaluation/metrics.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/src/evaluation/metrics.py)
- **Batch Evaluation Driver**: [scripts/evaluate_all.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/scripts/evaluate_all.py)
- **KUG Alignment Analysis**: [scripts/analyze_alignment.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/scripts/analyze_alignment.py)
- **Full Literature & Proof Document**: [handoff/related_works_research.md](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/handoff/related_works_research.md)
- **Empirical Results Artifact**: [preliminary_results.md](file:///users/jerryma/.gemini/antigravity-ide/brain/49e3c0f2-963a-47c7-9833-8e145d8986c1/preliminary_results.md)
- **Walkthrough Artifact**: [walkthrough.md](file:///users/jerryma/.gemini/antigravity-ide/brain/49e3c0f2-963a-47c7-9833-8e145d8986c1/walkthrough.md)

### 5.2 Local PDF Papers & Internal Documentation (Absolute Paths)
- **Mem2Gen Paper**: [2607.08393v1.pdf](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/2607.08393v1.pdf)
- **Physics of Language Models 3.1 (Knowledge Storage/Extraction)**: [Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf)
- **Physics of Language Models 3.2 (Knowledge Manipulation)**: [Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf)
- **Physics of Language Models 3.3 (Knowledge Capacity Scaling)**: [Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf)
- **Initial Idea Notes**: [idea.txt](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/related_works/idea.txt)

### 5.3 Key Academic Citations & arXiv Identifiers
- **Mem2Gen (Knowing-Using Gap)**: arXiv:2607.08393 (*"Knowing-Using Gap in Large Language Models"*)
- **STaRK Benchmark**: arXiv:2404.13207 (*"STaRK: Benchmarking LLM Retrieval over Semi-Structured Knowledge Bases"*)
- **On-Policy Representation Distillation (OPRD)**: arXiv:2606.06021
- **Rank-One Model Editing Multi-hop Limitations**: arXiv:2601.04600 (*"On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions"*)
- **Attribution-Controlled Knowledge Editing (ACE)**: arXiv:2510.07896
- **Targeted Lexical Injection**: arXiv:2506.15415
- **TinyBERT**: arXiv:1909.10351 (*"TinyBERT: Distilling BERT for Natural Language Processing"*)

---

## 6. Step-by-Step Handoff Roadmap for Next Agent

### Step 1: Package Foundation & Core Data Module
- Create `pyproject.toml` with dependencies (`torch>=2.0`, `transformers>=4.40`, `peft>=0.10`, `datasets`).
- Implement `faster_sft/data/dataset.py` to read the standardized `.jsonl` schema.
- Implement `faster_sft/data/paired_collator.py` to tokenise $P_{\text{mem}}$ and $P_{\text{gen}}$ pairs and dynamically locate entity span token indices.

### Step 2: Layer Extraction & Loss Module
- Build `faster_sft/models/hooks.py` using PyTorch `register_forward_hook` to extract hidden states at specified layers ($l_s^{\text{early}}, l_s^{\text{late}}, l_t$).
- Implement loss classes in `faster_sft/losses/`: `RepDistillLoss`, `ContrastiveLoss`, `ProbeLoss`, and `HybridLoss`.
- Implement next-gen enhancements: `BridgeAlign` and `KG-HardInfoNCE`.

### Step 3: Custom Trainer (`FasterSFTTrainer`)
- Inherit from HuggingFace `Trainer` in `faster_sft/trainer.py`.
- Override `compute_loss()` to extract dual-prompt hidden states and compute $\mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{align}}$.

### Step 4: Robust Evaluator & Metrics
- Implement `faster_sft/evaluation/evaluator.py` featuring:
  - Standard `model.generate()` batched evaluation.
  - Robust `_manual_generate_item()` fallback for custom models (e.g. Nanbeige) using zero-padding per-item KV-caching.
- Implement `faster_sft/evaluation/metrics.py` for exact-match string normalization, $A_{\text{mem}}$, $A_{\text{gen}}$, $\text{AUC}$, and KUG ratios.

### Step 5: CLI Scripts & Verification
- Create `scripts/train.py`, `scripts/evaluate.py`, and `scripts/analyze.py`.
- Write unit tests in `tests/` covering loss gradient computation, hook detachment, and evaluation metrics.

---

## 7. Verification Plan

### Automated Unit Tests
Run pytest over core package components:
```bash
pytest tests/test_losses.py
pytest tests/test_hooks.py
pytest tests/test_evaluator.py
```

### End-to-End Synthetic Pipeline Verification
Train a lightweight model (`Qwen/Qwen3.5-2B` or `fdtn-ai/antares-1b`) on a synthetic test dataset for 3 epochs:
```bash
# 1. Train baseline vs hybrid
python scripts/train.py --data_path data/processed/synthetic_qa.jsonl --model_id Qwen/Qwen3.5-2B --loss_variant hybrid --out_dir outputs/test_run

# 2. Evaluate checkpoints
python scripts/evaluate.py --run_dir outputs/test_run --data_path data/processed/synthetic_qa.jsonl

# 3. Generate KUG analysis
python scripts/analyze.py --runs_dir outputs/test_run
```
Confirm that `eval_results.json` is generated, $A_{\text{mem}}$ and $A_{\text{gen}}$ are non-zero, and the summary table prints cleanly.
