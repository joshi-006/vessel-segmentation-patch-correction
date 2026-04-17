"""
evaluation/metrics.py
=====================
Pure evaluation logic — NO correction, NO model inference.

Public API
----------
compute_dice()
compute_iou()
compute_metrics()
failure_analysis()
build_ablation_table()
plot_dice_distribution()
plot_ablation_bar()
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pixel-level metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Dice / F1 coefficient between two binary maps.

    Parameters
    ----------
    pred, gt : np.ndarray  – any dtype; binarised internally at 0.5
    eps      : float       – smoothing constant

    Returns
    -------
    float in [0, 1]
    """
    p  = (pred > 0).astype(np.float32)
    g  = (gt   > 0).astype(np.float32)
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def compute_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Intersection-over-Union between two binary maps."""
    p  = (pred > 0).astype(np.float32)
    g  = (gt   > 0).astype(np.float32)
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return float((tp + eps) / (tp + fp + fn + eps))


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute Dice, IoU, Precision, and Recall in one pass.

    Parameters
    ----------
    pred, gt : np.ndarray  – binary, any numeric dtype
    eps      : float

    Returns
    -------
    dict with keys: "dice", "iou", "precision", "recall"
    """
    p  = (pred > 0).astype(np.float32)
    g  = (gt   > 0).astype(np.float32)
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return {
        "dice":      float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou":       float((tp + eps)     / (tp + fp + fn + eps)),
        "precision": float((tp + eps)     / (tp + fp + eps)),
        "recall":    float((tp + eps)     / (tp + fn + eps)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Failure / success analysis
# ─────────────────────────────────────────────────────────────────────────────

def failure_analysis(
    dice_before: list[float],
    dice_after: list[float],
    delta_margin: float = 0.001,
) -> dict:
    """
    Categorise images into improved / neutral / degraded after correction.

    Parameters
    ----------
    dice_before  : list of float  – Dice before correction, one per image
    dice_after   : list of float  – Dice after  correction, one per image
    delta_margin : float           – |ΔDice| < margin → neutral

    Returns
    -------
    dict with keys:
        "delta_arr"     – np.ndarray of ΔDice values
        "improved_idx"  – indices where ΔDice > +margin
        "neutral_idx"   – indices where |ΔDice| ≤ margin
        "degraded_idx"  – indices where ΔDice < −margin
        "n_improved"    – int
        "n_neutral"     – int
        "n_degraded"    – int
        "mean_delta"    – float
    """
    before_arr = np.clip(np.array(dice_before), 0, 1)
    after_arr  = np.clip(np.array(dice_after),  0, 1)
    delta_arr  = after_arr - before_arr

    improved_idx = np.where(delta_arr >  delta_margin)[0]
    degraded_idx = np.where(delta_arr < -delta_margin)[0]
    neutral_idx  = np.where(np.abs(delta_arr) <= delta_margin)[0]

    return {
        "delta_arr":    delta_arr,
        "improved_idx": improved_idx,
        "neutral_idx":  neutral_idx,
        "degraded_idx": degraded_idx,
        "n_improved":   int(len(improved_idx)),
        "n_neutral":    int(len(neutral_idx)),
        "n_degraded":   int(len(degraded_idx)),
        "mean_delta":   float(np.mean(delta_arr)),
    }


def print_failure_summary(
    dice_before: list[float],
    dice_after: list[float],
    acceptance_rate: float | None = None,
) -> None:
    """Print a concise correction summary to stdout."""
    fa = failure_analysis(dice_before, dice_after)
    print(f"\nMean Dice Before: {np.mean(dice_before):.7f}")
    print(f"Mean Dice After:  {np.mean(dice_after):.7f}")
    print(f"Mean Improvement: {fa['mean_delta']:+.7f}")
    print(
        f"Improved (Δ > 0.001): {fa['n_improved']}  "
        f"Neutral: {fa['n_neutral']}  "
        f"Degraded (Δ < -0.001): {fa['n_degraded']}"
    )
    if acceptance_rate is not None:
        print(f"Acceptance rate (pass 2): {acceptance_rate:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ablation table builder
# ─────────────────────────────────────────────────────────────────────────────

def _row(
    name: str,
    df: pd.DataFrame,
    ref_dice: float | None = None,
) -> dict:
    """Format one ablation row."""
    d     = float(np.clip(df["dice"].mean(), 0, 1))
    delta = f"{d - ref_dice:+.6f}" if ref_dice is not None else "—"
    return {
        "Method":    name,
        "Dice":      round(d, 6),
        "IoU":       round(df["iou"].mean(),       6),
        "Precision": round(df["precision"].mean(), 6),
        "Recall":    round(df["recall"].mean(),    6),
        "ΔDice":     delta,
    }


def build_ablation_table(
    baseline_df:  pd.DataFrame,
    tta_df:       pd.DataFrame,
    entropy_df:   pd.DataFrame,
    variance_df:  pd.DataFrame,
    random_df:    pd.DataFrame,
    mi_only_df:   pd.DataFrame,
    final_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the 8-row ablation study table (methods A–H).

    Parameters
    ----------
    All DataFrames must have columns: dice, iou, precision, recall.

    Returns
    -------
    pd.DataFrame
    """
    base_dice = float(baseline_df["dice"].mean())
    tta_dice  = float(tta_df["dice"].mean())

    rows = [
        _row("A  Baseline  (ensemble, thresh=0.45, no TTA)",    baseline_df, None),
        _row("B  + TTA + val-threshold + morphology",           tta_df,      base_dice),
        _row("C  + MC Dropout uncertainty maps  (mask = B)",    tta_df,      tta_dice),
        _row("D  + Entropy-based patch correction  [compare]",  entropy_df,  tta_dice),
        _row("E  + Variance-based patch correction [compare]",  variance_df, tta_dice),
        _row("F  + Random patch correction        [compare]",   random_df,   tta_dice),
        _row("G  + MI-only gate correction        [ablation]",  mi_only_df,  tta_dice),
        _row("H  + MI-only + Safe update rule    [PROPOSED]",   final_df,    tta_dice),
    ]
    return pd.DataFrame(rows)


def print_ablation_table(ablation: pd.DataFrame) -> None:
    """Pretty-print the ablation table."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY")
    print("=" * 90)
    print(ablation.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_dice_distribution(
    dice_before: list[float],
    dice_after: list[float],
    output_dir: str,
    filename: str = "failure_case_distribution.png",
) -> None:
    """
    Histogram of ΔDice (after − before) across the test set.
    Saved to ``output_dir / filename``.
    """
    delta_arr = np.clip(np.array(dice_after), 0, 1) - np.clip(np.array(dice_before), 0, 1)

    plt.figure(figsize=(8, 3))
    plt.hist(delta_arr, bins=30, color="steelblue", edgecolor="white")
    plt.axvline(0, color="red", linestyle="--", label="No change")
    plt.title("Dice change distribution (after − before correction)")
    plt.xlabel("ΔDice")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close()


def plot_ablation_bar(
    ablation: pd.DataFrame,
    output_dir: str,
    filename: str = "ablation_bar_chart.png",
) -> None:
    """
    Grouped bar chart of mean Dice per ablation method (A–H).
    Saved to ``output_dir / filename``.
    """
    methods_short = [
        "A\nBaseline", "B\nTTA", "C\nMC\nDrop",
        "D\nEntropy", "E\nVar", "F\nRandom", "G\nMI-only", "H\nProposed",
    ]
    dice_vals = ablation["Dice"].values
    colors    = [
        "#d9534f" if i < 3 else "#f0ad4e" if i < 6 else "#5bc0de" if i == 6 else "#5cb85c"
        for i in range(len(dice_vals))
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods_short, dice_vals, color=colors, edgecolor="white", width=0.6)
    ax.set_ylim(min(dice_vals) - 0.002, max(dice_vals) + 0.002)

    for bar, val in zip(bars, dice_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0001,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title("Ablation Study — Dice per method")
    ax.set_ylabel("Mean Dice")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close()
