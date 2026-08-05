import re
import string
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_answer(s: str) -> str:
    """
    Standard text normalization for Exact Match evaluation:
    - Lowercase
    - Remove punctuation
    - Remove English articles (a, an, the)
    - Fix extra whitespace
    """
    if not isinstance(s, str):
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """
    Strict Exact Match (EM): Returns 1.0 only if the normalized prediction
    is identical to the normalized ground truth string.

    NOTE: This is the PRIMARY metric for all reported results.
    Strict EM requires the model to produce the target entity and nothing else
    (after normalization: lowercasing, punctuation removal, article stripping).
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_pred == norm_gt else 0.0


def contains_match_score(prediction: str, ground_truth: str) -> float:
    """
    Substring / Contains Match: Returns 1.0 if normalized ground_truth is a
    substring of the normalized prediction.

    NOTE: This is a SECONDARY metric used for the LLM-as-judge analysis only.
    It captures cases where a verbose model wraps the correct answer in prose.
    It is NOT used as the primary A_mem / A_gen accuracy metric.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt in norm_pred else 0.0


def compute_kug_metrics(
    a_mem_history: List[float],
    a_gen_history: List[float],
) -> Dict[str, float]:
    """
    Computes diagnostic trajectory metrics:
    - Peak A_mem: Maximum memorization accuracy achieved.
    - Final A_gen: Downstream generalization accuracy at final epoch.
    - KUG Ratio: Peak A_mem / max(Final A_gen, 1e-5).
    - AUC_gen: Area under the generalization curve.
    """
    if not a_mem_history or not a_gen_history:
        return {
            "peak_a_mem": 0.0,
            "final_a_gen": 0.0,
            "kug_ratio": 0.0,
            "auc_gen": 0.0,
        }

    peak_a_mem = max(a_mem_history)
    final_a_gen = a_gen_history[-1]
    kug_ratio = peak_a_mem / max(final_a_gen, 1e-5)
    
    # NumPy 2.0 compatible trapezoidal integration
    arr = np.array(a_gen_history, dtype=float)
    if len(arr) < 2:
        auc_gen = float(arr[0])
    elif hasattr(np, "trapezoid"):
        auc_gen = float(np.trapezoid(arr))
    else:
        auc_gen = float(np.sum((arr[:-1] + arr[1:]) / 2.0))

    return {
        "peak_a_mem": float(peak_a_mem),
        "final_a_gen": float(final_a_gen),
        "kug_ratio": float(kug_ratio),
        "auc_gen": float(auc_gen),
    }


def plot_kug_diagnostics(
    results_dict: Dict[str, Dict[str, List[float]]],
    output_path: str = "outputs/kug_diagnostics.png",
    title: str = "Knowing-Using Gap (KUG) Trajectory & Layer Routing",
):
    """
    Generates a high-resolution, publication-quality diagnostic figure for UX presentation.
    Uses curated dark/light modern styling, smooth confidence intervals, and clean annotations.
    """
    # Modern publication style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    colors = sns.color_palette("muted")

    # Panel 1: Learning Curves (A_mem vs A_gen)
    for idx, (run_name, data) in enumerate(results_dict.items()):
        epochs = list(range(len(data.get("a_mem", []))))
        c = colors[idx % len(colors)]
        
        if "a_mem" in data:
            ax1.plot(epochs, data["a_mem"], label=f"{run_name} ($A_{{mem}}$)", color=c, linestyle="--", linewidth=2, alpha=0.85)
        if "a_gen" in data:
            ax1.plot(epochs, data["a_gen"], label=f"{run_name} ($A_{{gen}}$)", color=c, linestyle="-", linewidth=2.5)

    ax1.set_title("Factual Memorization ($A_{mem}$) vs Generalization ($A_{gen}$)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch / Training Step", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: KUG Ratio Summary Comparison
    run_names = list(results_dict.keys())
    kug_ratios = []
    for rname in run_names:
        data = results_dict[rname]
        mem = data.get("a_mem", [0.0])
        gen = data.get("a_gen", [0.0])
        m = compute_kug_metrics(mem, gen)
        kug_ratios.append(m["kug_ratio"])

    bars = ax2.bar(run_names, kug_ratios, color=colors[:len(run_names)], alpha=0.85, width=0.5)
    ax2.set_title("Knowing-Using Gap Ratio ($A_{mem} / A_{gen}$)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Model / Condition", fontsize=10, fontweight="bold")
    ax2.set_ylabel("KUG Ratio (Lower is Better)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    
    svg_path = output_path.rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
