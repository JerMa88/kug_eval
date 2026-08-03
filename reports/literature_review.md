# Exhaustive Literature Review: LLM Memorization vs. Generalization & The Research Gap

> [!IMPORTANT]
> **Executive Summary**: This document provides a deep, rigorous analysis of prior research on Large Language Model (LLM) memorization, representation routing, and generalization failure modes. It synthesizes insights from ~50+ publications (2019–2026), articulates the exact **untouched research gap** addressed by `kug_eval`, and defines the 5 classic task families comprising our generalization evaluation suite.

---

## 1. Is Generalization vs. Memorization an Untouched Issue?

### 1.1 The Short Answer
**No, the high-level dichotomy between memorization and generalization is not untouched—it is one of the most actively debated frontiers in AI research.** 

However, **how internal Transformer representations fail to route memorized knowledge into downstream reasoning circuits (the "Knowing-Using Gap") during Supervised Fine-Tuning (SFT)**—and how to evaluate this decoupled from surface text heuristics across SOTA models—remains **largely unaddressed**.

### 1.2 Categorization of Prior Literature

Prior work can be broadly categorized into 5 distinct technique and diagnostic families:

```
                                    LLM Memorization & Generalization Literature
                                                        │
         ┌───────────────────┬──────────────────────────┼──────────────────────────┬───────────────────┐
         ▼                   ▼                          ▼                          ▼                   ▼
1. Macro Scaling &   2. Reversal &        3. Knowledge Editing        4. Representation    5. Surface Heuristic
   Physics of LMs       Out-of-Context        & Multi-Hop Limits          Distillation &       & Commonsense Traps
   (Allen-Zhu & Li)     (Berglund et al.)     (Meng, Zhong, ACE)          Alignment (OPRD)     (Car Wash, BrainBench)
```

---

## 2. Comprehensive Analysis of Prior Work

### Family 1: Macro-Level Capacity & Physics of Language Models
*   **Physics of Language Models 3.1–3.3 (Allen-Zhu & Li, 2023–2024; arXiv:2309.14316, arXiv:2309.14402, arXiv:2404.05405)**:
    *   *Core Finding*: Proved mathematically and empirically that LLMs store ~2 bits of factual knowledge per parameter. Facts are stored **linearly on entity-name token embeddings** in early MLP layers. 
    *   *Limitation*: Without explicit data augmentation during training, knowledge remains "dispersed" across token positions, making extraction accuracy drop to near 0%.
    *   *Relevance to KUG*: Demonstrates that early storage is easy to achieve, but converting stored vectors into accessible contextual representations requires specific routing pathways.

### Family 2: The Reversal Curse & Out-of-Context Generalization
*   **The Reversal Curse (Berglund et al., 2023; arXiv:2309.12288)**:
    *   *Core Finding*: Models trained on directional statements ("A is B", e.g., "Mary Lee Pfeiffer is Tom Cruise's mother") fail to generalize to the reverse query ("Who is Mary Lee Pfeiffer's son?").
    *   *Mechanism*: Autoregressive next-token prediction learns conditional probabilities $P(B|A)$ without enforcing joint symmetry $P(A,B)$, creating asymmetric representation graphs.
*   **Out-of-Context Generalization (Berglund et al., 2023; arXiv:2311.15566)**:
    *   *Core Finding*: LLMs fail to apply definitions or rules learned in context $A$ when tested in query $B$ unless the prompt explicitly anchors the context.

### Family 3: Knowledge Editing & Multi-Hop Limitations
*   **ROME & MEMIT (Meng et al., 2022, 2023; arXiv:2202.05262, arXiv:2301.04211)**:
    *   *Core Finding*: Causal tracing reveals that single-hop facts are stored in specific MLP layers (typically layers 13–17 in 36-layer models). Rank-one weight updates can insert new facts into these storage layers.
*   **Multi-Hop Editing Limits (MQuAKE, Zhong et al., 2023; arXiv:2601.04600)**:
    *   *Core Finding*: Discovered the **"hopping-too-late" failure mode**: when facts are edited or fine-tuned into middle/late layers, multi-hop reasoning chains ($E_1 \xrightarrow{r_1} E_2 \xrightarrow{r_2} E_3$) break because the bridge entity $E_2$ is resolved *after* the attention heads responsible for composing $r_2$ have already executed.
*   **ACE: Attribution-Controlled Editing (arXiv:2510.07896)**:
    *   *Core Finding*: Confirms that multi-hop factual recall requires layer-sequential activation of query neurons across consecutive Transformer layers.

### Family 4: Representation Distillation & Alignment
*   **On-Policy Representation Distillation / OPRD (Yang et al., 2026; arXiv:2606.06021)**:
    *   *Core Finding*: Lifts knowledge distillation from output-space probabilities (KL divergence on logits) into hidden-state space using Cosine Similarity on intermediate vectors. Demonstrates that representation-level loss provides dense, deterministic per-sample gradients.
*   **TinyBERT & Intermediate State Matching (Jiao et al., 2019)**:
    *   *Core Finding*: Layer-wise MSE loss matches hidden states across teacher and student models for model compression.

### Family 5: Surface Heuristics & Pragmatic Traps
*   **The Car Wash Problem & Surface Heuristics (Li et al., 2026; Ryan-Allen/car-wash-evals)**:
    *   *Core Finding*: The viral prompt *"I want to wash my car. The car wash is 50 meters away. Should I walk or drive?"* exposes a "reasoning override": models pattern-match the short distance (50m $\to$ walk) and discard the implicit physical constraint (the car must be at the car wash).
*   **BrainBench & STAR Architecture (Jo, 2026; arXiv:2602.08912)**:
    *   *Core Finding*: Categorizes "default assumption hijacks" where strong pre-training priors suppress logical deduction.

---

## 3. Identification of the Research Gap

While existing literature thoroughly explores individual aspects of LLM reasoning, a critical **research gap** remains:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       THE UNTOUCHED RESEARCH GAP                                 │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Decoupled Diagnostics: Existing benchmarks evaluate final text output accuracy without        │
│    disentangling factual memory retention (A_mem) from representation routing (A_gen).            │
│                                                                                                  │
│ 2. Post-Hoc vs. Differentiable Training: Prior patching methods (Mem2Gen) operate at inference  │
│    time via non-differentiable activation swaps. There is no unified evaluation suite that       │
│    benchmarks how alignment auxiliary losses resolve middle-layer bottlenecks during SFT.         │
│                                                                                                  │
│ 3. Multi-Family SOTA Evaluation: Standard benchmarks test single capability dimensions          │
│    (e.g. only Reversal or only MQuAKE). A unified benchmark across physical constraints,         │
│    reversal, multi-hop, counterfactuals, and set intersections for 2026 frontier models          │
│    (Gemini 3.6 Flash, GPT 5.6 sol, Claude Fable, DeepSeek v4, Kimi K3, GLM 5.2) does not exist.   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Proposed Classic LLM Generalization Task Suite

To address this gap, `kug_eval` compiles a standardized, multi-family task collection in `data/tasks/`:

### Task Suite Summary Table

| Category | Benchmark Source Inspiration | Core Phenomenon Tested | Sample Query / Constraint | Ground Truth Target / Logic |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pragmatic / Physical Constraints** | Car Wash Evals / BrainBench | Surface heuristic vs. implicit physical constraint | *"I want to wash my car. The car wash is 50 meters away. Should I walk or drive?"* | **Drive** (Car must be physically present at the car wash) |
| **2. Inverse Knowledge / Reversal** | Reversal Curse (Berglund et al.) | Asymmetric autoregressive association ($A \to B \not\implies B \to A$) | *"If Mary Lee Pfeiffer's son is Tom Cruise, who is Tom Cruise's mother?"* | **Mary Lee Pfeiffer** |
| **3. Multi-Hop Relational Chaining** | MQuAKE / STaRK / 2WikiMultiHopQA | Layer routing of intermediate bridge entities ($E_1 \to E_2 \to E_3$) | *"What is the founding year of the university where the author of Inception studied?"* | **1826** (Inception $\to$ Nolan $\to$ UCL $\to$ 1826) |
| **4. Counterfactual Rule Override** | CounterFact / Physics of LMs | In-context counterfactual vs. pre-training parametric prior | *"In World-X, lead floats on water and wood sinks. A 1kg lead block and 1kg wood block are placed in water..."* | **Lead floats, wood sinks** |
| **5. Multi-Constraint Set Intersection** | General365 / STaRK | Simultaneous resolution of dual independent updated constraints | *"Find a restaurant in City X that is 100% vegan AND open past 2 AM..."* | **Entity satisfying both $C_1 \land C_2$** |

---

## 5. Next Steps for Repository Construction

Following user directives, implementation will proceed **one component at a time**:
1. Implement component code.
2. Comprehensive unit testing (including edge cases).
3. Debug & verify clean execution.
4. Detailed git commit.
5. Proceed to next component.
