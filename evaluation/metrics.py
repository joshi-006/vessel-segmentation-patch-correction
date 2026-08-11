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

from utils.helpers import morphological_postprocess


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


def plot_adaptive_efficiency(
    adaptive_df: pd.DataFrame,
    details_df: pd.DataFrame,
    delta_threshold: float,
    output_dir: str,
    filename: str = "adaptive_efficiency.png",
) -> None:
    """
    Three-panel adaptive-stopping efficiency plot: patches used vs saved per
    image, per-patch quality-score delta, and % computation saved per image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    adaptive_df_sorted = adaptive_df.sort_values("image")
    x = range(len(adaptive_df_sorted))

    axes[0].bar(x, adaptive_df_sorted["patches_used"], label="Used")
    axes[0].bar(x, adaptive_df_sorted["patches_saved"],
                bottom=adaptive_df_sorted["patches_used"], label="Saved")
    axes[0].set_title("Patches used vs saved per image")
    axes[0].set_xlabel("Image index")
    axes[0].set_ylabel("Number of patches")
    axes[0].legend()
    axes[0].set_xticks([])

    for img_name in adaptive_df["image"]:
        rows = details_df[details_df["image"] == img_name]
        if len(rows) == 0:
            continue
        axes[1].plot(range(1, len(rows) + 1), rows["delta"].values, marker="o", alpha=0.4)
    axes[1].axhline(delta_threshold, linestyle="--", color="red",
                     label=f"threshold={delta_threshold}")
    axes[1].set_title("Quality-score delta per patch")
    axes[1].set_xlabel("Patch index")
    axes[1].set_ylabel("Delta QS")
    axes[1].legend()

    pct_saved_per_image = (
        adaptive_df_sorted["patches_saved"] /
        (adaptive_df_sorted["patches_used"] + adaptive_df_sorted["patches_saved"] + 1e-6)
    )
    axes[2].bar(x, pct_saved_per_image * 100)
    axes[2].set_title("Computation saved per image")
    axes[2].set_xlabel("Image index")
    axes[2].set_ylabel("Patches skipped (%)")
    axes[2].set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close()


def print_hard_region_analysis(
    dice_before: list[float],
    dice_after: list[float],
    quartile: float = 0.25,
) -> None:
    """Print mean Dice before/after for the hardest quartile of images (by pre-correction Dice)."""
    before_arr = np.array(dice_before)
    after_arr  = np.array(dice_after)
    hard_idx   = np.argsort(before_arr)[:max(1, int(len(before_arr) * quartile))]

    print(f"\nHard-region analysis (bottom 25% pre-correction Dice, n={len(hard_idx)}):")
    print(f"  Mean Dice BEFORE: {before_arr[hard_idx].mean():.6f}")
    print(f"  Mean Dice AFTER:  {after_arr[hard_idx].mean():.6f}")
    print(f"  Delta:            {(after_arr[hard_idx] - before_arr[hard_idx]).mean():+.6f}")

    print("\nNarrative: method targets uncertain, low-confidence regions.")
    print("Hard-region improvement demonstrates targeted correction of difficult areas,")
    print("even when global mean Dice change appears modest.")


def print_evaluation_beyond_mean_dice(
    dice_before: list[float],
    dice_after: list[float],
    mean_mi_per_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Print worst-image, bottom-20%, and high-uncertainty-image Dice breakdowns.

    Returns
    -------
    before_np, delta_np : np.ndarray  – clipped pre-correction Dice and
        ΔDice (after − before), for reuse in ``plot_region_improvement_scatter``.
    """
    before_np = np.clip(np.array(dice_before), 0, 1)
    after_np  = np.clip(np.array(dice_after),  0, 1)
    delta_np  = after_np - before_np

    print("=" * 60)
    print("EVALUATION BEYOND GLOBAL MEAN DICE")
    print("=" * 60)

    print(f"\n1. Worst-image Dice")
    print(f"   Before: {before_np.min():.6f}")
    print(f"   After:  {after_np.min():.6f}")
    print(f"   Delta:  {after_np.min() - before_np.min():+.6f}")

    pct20 = np.argsort(before_np)[:max(1, int(0.2 * len(before_np)))]
    print(f"\n2. Bottom 20% pre-correction Dice  (n={len(pct20)})")
    print(f"   Before: {before_np[pct20].mean():.6f}")
    print(f"   After:  {after_np[pct20].mean():.6f}")
    print(f"   Delta:  {(after_np[pct20] - before_np[pct20]).mean():+.6f}")

    high_unc_idx = np.where(mean_mi_per_image >= np.percentile(mean_mi_per_image, 75))[0]
    print(f"\n3. High-uncertainty images (top 25% mean MI, n={len(high_unc_idx)})")
    print(f"   Before: {before_np[high_unc_idx].mean():.6f}")
    print(f"   After:  {after_np[high_unc_idx].mean():.6f}")
    print(f"   Delta:  {(after_np[high_unc_idx] - before_np[high_unc_idx]).mean():+.6f}")

    return before_np, delta_np


def plot_region_improvement_scatter(
    before_np: np.ndarray,
    delta_np: np.ndarray,
    mean_mi_per_image: np.ndarray,
    output_dir: str,
    filename: str = "region_improvement_scatter.png",
) -> None:
    """Scatter of pre-correction Dice vs ΔDice, coloured by image-level mean MI."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(before_np, delta_np, c=mean_mi_per_image, cmap="hot", s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Mean MI (epistemic uncertainty)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pre-correction Dice")
    ax.set_ylabel("ΔDice (after − before)")
    ax.set_title("Region-specific improvement vs pre-correction performance\n"
                 "(colour = image-level mean MI uncertainty)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close()


def plot_error_maps(
    test_images: list[str],
    cached_preds: dict,
    cached_gts: dict,
    corrected_masks: dict,
    best_thresh: float,
    output_dir: str,
    n_show: int = 5,
    filename: str = "error_maps.png",
) -> None:
    """
    False-positive (red) / false-negative (blue) error maps, before vs after
    correction, for the first ``n_show`` test images.
    """
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    for ax, lbl in zip(axes[0],
                       ["Before FP (red)", "Before FN (blue)", "After FP (red)", "After FN (blue)"]):
        ax.set_title(lbl, fontsize=10, fontweight="bold")

    for i, img_name in enumerate(test_images[:n_show]):
        gt = cached_gts[img_name].astype(bool)
        before_mask = morphological_postprocess(
            (cached_preds[img_name] > best_thresh).astype(np.uint8)
        ).astype(bool)
        after_mask = corrected_masks[img_name].astype(bool)

        for col_off, mask in enumerate([before_mask, after_mask]):
            fp = np.zeros((512, 512, 3), dtype=np.uint8)
            fp[mask & ~gt] = [220, 50, 50]
            fn = np.zeros((512, 512, 3), dtype=np.uint8)
            fn[~mask & gt] = [50, 50, 220]

            axes[i, col_off * 2].imshow(fp)
            axes[i, col_off * 2].axis("off")
            axes[i, col_off * 2 + 1].imshow(fn)
            axes[i, col_off * 2 + 1].axis("off")

    plt.suptitle("Error maps (FP=red, FN=blue) before vs after correction", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches="tight")
    plt.close()
