"""
correction/patch_correction.py
===============================
Core correction method.  Contains ONLY correction / patch-selection logic —
no training, no evaluation metrics, no plotting.

Public API
----------
corrected_with_guard()
apply_corrected_patches()
select_patches_mi_only()
select_top_patches_non_overlap()

Internal helpers
----------------
patch_mi_analysis()
patch_uncertainty_analysis()
apply_correction_to_mi_patch()
adaptive_refinement_stopping()
"""

from __future__ import annotations

import os

import numpy as np
import torch

from utils.helpers import compute_image_quality_score


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (imported / overridden from config in main.py)
# ─────────────────────────────────────────────────────────────────────────────

SAFE_UPDATE_MARGIN = 0.015
MI_HIGH_THRESHOLD  = 0.025
MIN_CONF_CHANGE    = 0.012


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core correction with safety guard  ← THE heart of the method
# ─────────────────────────────────────────────────────────────────────────────

def corrected_with_guard(
    correction_model,
    img_rgb: np.ndarray,
    seg_patch: np.ndarray,
    mi_patch: np.ndarray,
    mean_prob_patch: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    safe_update_margin: float = SAFE_UPDATE_MARGIN,
    min_conf_change: float = MIN_CONF_CHANGE,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Apply the correction model to a single patch and decide whether to accept
    the update using the safe-update rule.

    Parameters
    ----------
    correction_model : nn.Module
        Trained PatchCorrectionUNet (eval mode).
    img_rgb : np.ndarray
        Raw image crop, shape (H, W, 3), uint8 or float.
    seg_patch : np.ndarray
        Current binary segmentation crop, shape (H, W).
    mi_patch : np.ndarray
        Mutual-information uncertainty crop, shape (H, W), values in [0, 1].
    mean_prob_patch : np.ndarray
        Mean MC-Dropout probability crop, shape (H, W).
    device : torch.device
    threshold : float
        Binarisation threshold for the corrected probability map.
    safe_update_margin : float
        Minimum improvement in quality score required to accept the update.
    min_conf_change : float
        Minimum absolute change in confidence required before checking score.

    Returns
    -------
    new_mask  : np.ndarray  uint8  H×W  – corrected binary mask
    new_prob  : np.ndarray  float  H×W  – corrected probability map
    accepted  : bool        – True if the safe-update gate passed
    """
    # ── Build input tensor ────────────────────────────────────────────────────
    img_p = ((img_rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
    x = torch.from_numpy(
        np.concatenate([img_p, seg_patch[np.newaxis], mi_patch[np.newaxis]], axis=0)
    ).float().unsqueeze(0).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits   = correction_model(x)
        new_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    new_mask   = (new_prob > threshold).astype(np.uint8)
    eps        = 1e-6
    old_mask   = (seg_patch > 0.5).astype(np.float32)
    new_mask_f = (new_prob  > threshold).astype(np.float32)

    # ── Gate 1: minimum confidence change ────────────────────────────────────
    old_conf = float((mean_prob_patch * old_mask).sum()   / (old_mask.sum()   + eps))
    new_conf = float((new_prob        * new_mask_f).sum() / (new_mask_f.sum() + eps))

    if abs(new_conf - old_conf) < min_conf_change:
        return new_mask, new_prob, False

    # ── Gate 2: uncertainty must decrease ────────────────────────────────────
    old_unc = float(np.mean(np.clip(mi_patch, 0, 1)))
    new_unc = float(np.mean(
        -(new_prob * np.log(np.clip(new_prob, 1e-7, 1))
          + (1 - new_prob) * np.log(np.clip(1 - new_prob, 1e-7, 1)))
    ))

    # ── Gate 3: flip-rate cap (avoid catastrophic re-labelling) ──────────────
    flip_rate = float(np.mean(np.abs(new_mask_f - old_mask)))
    if flip_rate > 0.30:
        return new_mask, new_prob, False

    # ── Safe-update rule: composite quality score must improve ────────────────
    old_score = (old_conf ** 0.5) * ((1 - old_unc) ** 0.5)
    new_score = (new_conf ** 0.5) * ((1 - new_unc) ** 0.5)
    accepted  = new_score > old_score + safe_update_margin

    return new_mask, new_prob, accepted


# ─────────────────────────────────────────────────────────────────────────────
# 2. Apply saved corrected patches to a full mask
# ─────────────────────────────────────────────────────────────────────────────

def apply_corrected_patches(
    mask: np.ndarray,
    base_name: str,
    corrected_dir: str,
    mi_map: np.ndarray,
    mean_map: np.ndarray,
    top_k: int,
    best_thresh: float = 0.5,
    mi_floor: float = 0.02,
) -> np.ndarray:
    """
    Load pre-saved corrected patch files for one image and splice the top-K
    (by MI score) back into the full segmentation mask.

    Parameters
    ----------
    mask          : np.ndarray  uint8  H×W  – current segmentation
    base_name     : str  – image filename stem (no extension)
    corrected_dir : str  – directory with ``*_corrected.npy`` files
    mi_map        : np.ndarray  H×W  – MI uncertainty map
    mean_map      : np.ndarray  H×W  – mean MC-Dropout probability (unused here,
                                        reserved for future weighting)
    top_k         : int  – maximum patches to apply
    best_thresh   : float – binarisation threshold
    mi_floor      : float – minimum patch-level MI score; patches below are skipped

    Returns
    -------
    new_mask : np.ndarray  uint8  H×W
    """
    all_files = [
        f for f in os.listdir(corrected_dir)
        if f.startswith(base_name + "_r") and f.endswith("_corrected.npy")
    ]

    patch_info = []
    for fname in all_files:
        parts    = fname.replace("_corrected.npy", "").split("_")
        r        = int(next(p[1:] for p in parts if p.startswith("r")))
        c        = int(next(p[1:] for p in parts if p.startswith("c")))
        size     = int(next(p[1:] for p in parts if p.startswith("s")))
        mi_score = float(
            np.percentile(np.clip(mi_map[r:r+size, c:c+size], 0, 1), 90)
        )
        patch_info.append((fname, r, c, size, mi_score))

    # Rank by MI score (highest first)
    patch_info.sort(key=lambda x: x[4], reverse=True)

    new_mask = mask.copy()
    for fname, r, c, size, mi_score in patch_info[:top_k]:
        if mi_score < mi_floor:
            continue
        corr = np.load(os.path.join(corrected_dir, fname))
        new_mask[r:r+size, c:c+size] = (corr > best_thresh).astype(np.uint8)

    return new_mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. Patch selection
# ─────────────────────────────────────────────────────────────────────────────

def select_patches_mi_only(
    mi_map: np.ndarray,
    patch_size: int,
    top_k: int,
    mi_floor: float = 0.015,
) -> tuple[list, list]:
    """
    Slide a window over ``mi_map`` and collect patches whose 90th-percentile
    MI exceeds ``mi_floor``.

    Parameters
    ----------
    mi_map     : np.ndarray  H×W
    patch_size : int
    top_k      : int
    mi_floor   : float  – minimum 90th-percentile MI to include a patch

    Returns
    -------
    all_patches : list of (r, c, patch_size, mi_score)  – all qualifying patches
    top_patches : list of (r, c, patch_size, mi_score)  – top-K subset
    """
    H, W    = mi_map.shape
    patches = []
    stride  = patch_size // 2

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            mi_score = float(
                np.percentile(
                    np.clip(mi_map[r:r+patch_size, c:c+patch_size], 0, 1), 90
                )
            )
            if mi_score >= mi_floor:
                patches.append((r, c, patch_size, mi_score))

    patches.sort(key=lambda x: x[3], reverse=True)
    return patches, patches[:top_k]


def select_top_patches_non_overlap(
    patches: list,
    top_k: int,
) -> list:
    """
    Greedily select up to ``top_k`` non-overlapping patches from a
    pre-sorted (score descending) candidate list.

    Overlap test: centres of two patches are closer than ``size // 2`` in
    both dimensions → overlapping.

    Parameters
    ----------
    patches : list of (r, c, size, score)
    top_k   : int

    Returns
    -------
    selected : list of (r, c, size, score)
    """
    selected: list = []
    for r, c, size, score in patches:
        overlaps = any(
            abs(r - sr) < size // 2 and abs(c - sc) < size // 2
            for sr, sc, ss, _ in selected
        )
        if not overlaps:
            selected.append((r, c, size, score))
        if len(selected) >= top_k:
            break
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 4. Internal / ablation helpers
# ─────────────────────────────────────────────────────────────────────────────

def patch_mi_analysis(
    mi_map: np.ndarray,
    patch_size: int,
    top_k: int,
    mi_floor: float = 0.010,
) -> tuple[list, list]:
    """
    Slide a window and score each patch by the mean of its top-10 MI pixels.
    (Used for adaptive-stopping analysis.)
    """
    H, W    = mi_map.shape
    patches = []
    stride  = patch_size // 2

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            flat  = mi_map[r:r+patch_size, c:c+patch_size].flatten()
            score = float(np.mean(np.sort(flat)[-10:]))
            if score > mi_floor:
                patches.append((r, c, patch_size, score))

    patches.sort(key=lambda x: x[3], reverse=True)
    return patches, patches[:top_k]


def patch_uncertainty_analysis(
    unc_map: np.ndarray,
    patch_size: int,
    top_k: int,
    threshold: float = 0.010,
) -> tuple[list, list]:
    """
    Generic sliding-window scorer for any uncertainty map (entropy, variance,
    MI).  Scores by mean of top-10 pixels per patch.
    """
    H, W    = unc_map.shape
    patches = []
    stride  = patch_size // 2

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            flat  = unc_map[r:r+patch_size, c:c+patch_size].flatten()
            score = float(np.mean(np.sort(flat)[-10:]))
            if score > threshold:
                patches.append((r, c, patch_size, score))

    patches.sort(key=lambda x: x[3], reverse=True)
    return patches, patches[:top_k]


def apply_correction_to_mi_patch(
    mi_map: np.ndarray,
    r: int,
    c: int,
    patch_size: int,
    damping: float = 0.3,
) -> np.ndarray:
    """
    Simulate the MI reduction after a patch has been corrected (used in
    adaptive-stopping quality-score calculations).

    Returns a copy of ``mi_map`` with the target region dampened.
    """
    mi_new = mi_map.copy()
    mi_new[r:r+patch_size, c:c+patch_size] *= damping
    return mi_new


def adaptive_refinement_stopping(
    mi_map: np.ndarray,
    all_patches: list,
    patch_size: int = 81,
    delta_threshold: float = 0.001,
    min_patches: int = 1,
) -> tuple[list, int, str]:
    """
    Simulate applying patches in rank order and stop early when the marginal
    quality improvement falls below ``delta_threshold``.

    Parameters
    ----------
    mi_map          : np.ndarray  H×W  – MI uncertainty map
    all_patches     : list of (r, c, size, avg_mi) – ranked candidates
    patch_size      : int
    delta_threshold : float  – stop if ΔQS < this value
    min_patches     : int   – always apply at least this many patches

    Returns
    -------
    results     : list of dict  – per-patch statistics
    n_used      : int           – number of patches applied
    stop_reason : str           – "threshold" or "exhausted"
    """
    mi_map      = np.clip(mi_map, 0, 1)
    current_map = mi_map.copy()
    prev_qs     = compute_image_quality_score(current_map)
    results: list = []

    for i, (r, c, patch_sz, avg_mi) in enumerate(all_patches):
        new_map = apply_correction_to_mi_patch(current_map, r, c, patch_sz)
        new_qs  = compute_image_quality_score(new_map)
        delta   = new_qs - prev_qs

        results.append({
            "rank":      i + 1,
            "row":       r,
            "col":       c,
            "patch_mi":  float(avg_mi),
            "qs_before": float(prev_qs),
            "qs_after":  float(new_qs),
            "delta":     float(delta),
        })

        if i >= min_patches and delta < delta_threshold:
            return results, i + 1, "threshold"

        prev_qs     = new_qs
        current_map = new_map

    return results, len(all_patches), "exhausted"
