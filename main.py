"""
main.py
=======
Orchestrator for the Uncertainty-Guided Patch Correction pipeline.

Imports only from the four sub-packages:
    correction/   evaluation/   models/   utils/

SETUP (run once):
-----------------
1.  pip install torch torchvision timm==0.6.13 opencv-python albumentations \\
                tqdm scikit-image numpy pandas matplotlib

2.  git clone https://github.com/berenslab/MIDL24-segmentation_quality_control.git

3.  Update the CONFIG section below for your local paths.

Run:
    python main.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import glob
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  —  update these paths for your local environment
# ─────────────────────────────────────────────────────────────────────────────

MIDL_REPO_PATH = "./MIDL24-segmentation_quality_control"
sys.path.append(MIDL_REPO_PATH)          # must precede local module imports

FIVES_BASE     = "./data/fundus-image-dataset-for-vessel-segmentation"
TRAIN_IMG_DIR  = os.path.join(FIVES_BASE, "train/Original")
TRAIN_MASK_DIR = os.path.join(FIVES_BASE, "train/Ground truth")
TEST_IMG_DIR   = os.path.join(FIVES_BASE, "test/Original")
TEST_MASK_DIR  = os.path.join(FIVES_BASE, "test/Ground truth")

MODEL_DIR        = "./models"
OUTPUT_DIR       = "./results"
PATCH_EXPORT_DIR = os.path.join(OUTPUT_DIR, "patches")
CORRECTED_DIR    = os.path.join(OUTPUT_DIR, "corrected_patches")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE            = 81
TOP_K                 = 16
N_PASSES              = 30
MAX_IMAGES            = 200
N_CORR_PASSES         = 2
UNCERTAINTY_THRESHOLD = 0.014
VAL_SPLIT_RATIO       = 0.15

SAFE_UPDATE_MARGIN    = 0.015
MI_HIGH_THRESHOLD     = 0.025
MIN_CONF_CHANGE       = 0.012

DELTA_THRESHOLD       = 0.002
MIN_PATCHES           = 1

EPOCHS                = 50

# ─────────────────────────────────────────────────────────────────────────────
# Local package imports  (after sys.path.append above)
# ─────────────────────────────────────────────────────────────────────────────
from models import (
    PatchCorrectionUNet,
    PatchCorrectionDataset,
    correction_loss,
    load_models,
)
from correction import (
    corrected_with_guard,
    apply_corrected_patches,
    select_patches_mi_only,
    select_top_patches_non_overlap,
    patch_mi_analysis,
    patch_uncertainty_analysis,
    adaptive_refinement_stopping,
)
from evaluation import (
    compute_metrics,
    failure_analysis,
    print_failure_summary,
    build_ablation_table,
    print_ablation_table,
    plot_dice_distribution,
    plot_ablation_bar,
)
from utils import (
    preprocess_image,
    load_ground_truth,
    morphological_postprocess,
    tta_predict,
    mc_dropout_predict,
    compute_mutual_information,
    compute_entropy_map,
    compute_variance_map,
    compute_image_quality_score,
)

# Override correction module defaults with config values
import correction.patch_correction as _pc
_pc.SAFE_UPDATE_MARGIN = SAFE_UPDATE_MARGIN
_pc.MIN_CONF_CHANGE    = MIN_CONF_CHANGE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt(img_name: str, img_dir: str, mask_dir: str) -> np.ndarray:
    """Load and binarise a GT mask matching the image size (512×512)."""
    return load_ground_truth(os.path.join(mask_dir, img_name))


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"TOP_K={TOP_K}  N_PASSES={N_PASSES}  SAFE_UPDATE_MARGIN={SAFE_UPDATE_MARGIN}  "
          f"MI_HIGH_THRESHOLD={MI_HIGH_THRESHOLD}  MIN_CONF_CHANGE={MIN_CONF_CHANGE}")
    print("Pass 1: UNGATED (aggressive).  Pass 2: GATED by corrected_with_guard().")

    for d in [OUTPUT_DIR, PATCH_EXPORT_DIR, CORRECTED_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: Load FR-UNet ensemble ────────────────────────────────────────
    models = load_models(MODEL_DIR, device)

    # ── Step 2: Baseline evaluation (no TTA) ─────────────────────────────────
    test_images      = sorted(os.listdir(TEST_IMG_DIR))[:MAX_IMAGES]
    baseline_results = []

    for img_name in tqdm(test_images, desc="Baseline eval"):
        _, img_tensor = preprocess_image(os.path.join(TEST_IMG_DIR, img_name), device=device)
        gt = _load_gt(img_name, TEST_IMG_DIR, TEST_MASK_DIR)

        with torch.no_grad():
            preds = [torch.sigmoid(m(img_tensor)) for m in models]
            pred  = torch.mean(torch.stack(preds), dim=0).cpu().numpy().squeeze()

        mask = (pred > 0.35).astype(np.uint8)
        mask = morphological_postprocess(mask)
        baseline_results.append(compute_metrics(mask, gt))

    baseline_df = pd.DataFrame(baseline_results)
    print("\nBaseline (no TTA, thresh=0.35):")
    print(baseline_df.mean())

    # ── Step 3: Val-set threshold sweep ──────────────────────────────────────
    random.seed(42)
    all_train_images  = sorted(os.listdir(TRAIN_IMG_DIR))
    n_val_thresh      = max(1, int(VAL_SPLIT_RATIO * len(all_train_images)))
    val_thresh_images = random.sample(all_train_images, n_val_thresh)

    thresholds      = np.arange(0.30, 0.65, 0.025)
    thresh_dice_val = {t: [] for t in thresholds}

    for img_name in tqdm(val_thresh_images, desc="Val threshold sweep"):
        _, img_tensor = preprocess_image(os.path.join(TRAIN_IMG_DIR, img_name), device=device)
        pred_prob     = tta_predict(models, img_tensor)

        gt_val = load_ground_truth(os.path.join(TRAIN_MASK_DIR, img_name))
        for t in thresholds:
            mask = morphological_postprocess((pred_prob > t).astype(np.uint8))
            thresh_dice_val[t].append(compute_metrics(mask, gt_val)["dice"])

    mean_dice_per_thresh = {t: np.mean(v) for t, v in thresh_dice_val.items()}
    BEST_THRESH = max(mean_dice_per_thresh, key=mean_dice_per_thresh.get)

    print("\nThreshold sweep (val split of train set):")
    for t, d in sorted(mean_dice_per_thresh.items()):
        marker = "  <-- BEST" if t == BEST_THRESH else ""
        print(f"  thresh={t:.3f}  dice={d:.6f}{marker}")
    print(f"\nBEST_THRESH = {BEST_THRESH:.3f}")

    # ── Step 4: Cache TTA predictions on test set ─────────────────────────────
    cached_preds, cached_gts = {}, {}
    for img_name in tqdm(test_images, desc="TTA predictions (test set)"):
        _, img_tensor          = preprocess_image(os.path.join(TEST_IMG_DIR, img_name), device=device)
        cached_preds[img_name] = tta_predict(models, img_tensor)
        cached_gts[img_name]   = _load_gt(img_name, TEST_IMG_DIR, TEST_MASK_DIR)

    # ── Step 5: TTA baseline eval ─────────────────────────────────────────────
    tta_results = []
    for img_name in tqdm(test_images, desc="TTA baseline eval"):
        mask = morphological_postprocess(
            (cached_preds[img_name] > BEST_THRESH).astype(np.uint8)
        )
        tta_results.append(compute_metrics(mask, cached_gts[img_name]))

    tta_df = pd.DataFrame(tta_results)
    print("\nTTA Baseline (optimal threshold + morphology):")
    print(tta_df.mean())

    # ── Step 6: MC Dropout — cache uncertainty maps ───────────────────────────
    mc_cache = {}
    for img_name in tqdm(test_images, desc="MC Dropout uncertainty"):
        _, img_tensor = preprocess_image(os.path.join(TEST_IMG_DIR, img_name), device=device)
        preds_mc      = mc_dropout_predict(models, img_tensor, N_PASSES)
        mean_map, mi_map, _, _ = compute_mutual_information(preds_mc)
        mc_cache[img_name] = {
            "mean":     mean_map,
            "mi":       np.clip(mi_map, 0, None),
            "entropy":  compute_entropy_map(preds_mc),
            "variance": compute_variance_map(preds_mc),
        }
    print("MC Dropout done.")

    # ── Step 7: Export patches ────────────────────────────────────────────────
    shutil.rmtree(PATCH_EXPORT_DIR); shutil.rmtree(CORRECTED_DIR)
    os.makedirs(PATCH_EXPORT_DIR, exist_ok=True)
    os.makedirs(CORRECTED_DIR,    exist_ok=True)

    for img_name in tqdm(test_images, desc="Exporting patches"):
        img_rgb, _ = preprocess_image(os.path.join(TEST_IMG_DIR, img_name), device=device)
        mi_map     = mc_cache[img_name]["mi"]
        binary_mask = morphological_postprocess(
            (cached_preds[img_name] > BEST_THRESH).astype(np.uint8)
        )
        gt = _load_gt(img_name, TEST_IMG_DIR, TEST_MASK_DIR)

        _, top_candidates = select_patches_mi_only(mi_map, PATCH_SIZE, TOP_K,
                                                   mi_floor=UNCERTAINTY_THRESHOLD)
        top_patches = select_top_patches_non_overlap(top_candidates, TOP_K)

        base = os.path.splitext(img_name)[0]
        for rank, (r, c, size, _) in enumerate(top_patches):
            pname = f"{base}_r{r}_c{c}_s{size}_k{rank}"
            np.save(os.path.join(PATCH_EXPORT_DIR, pname + "_img.npy"), img_rgb[r:r+size, c:c+size])
            np.save(os.path.join(PATCH_EXPORT_DIR, pname + "_seg.npy"), binary_mask[r:r+size, c:c+size])
            np.save(os.path.join(PATCH_EXPORT_DIR, pname + "_mi.npy"),  mi_map[r:r+size, c:c+size])
            np.save(os.path.join(PATCH_EXPORT_DIR, pname + "_gt.npy"),  gt[r:r+size, c:c+size])

    n_exported = len(glob.glob(os.path.join(PATCH_EXPORT_DIR, "*_img.npy")))
    print("Exported patches:", n_exported)

    # ── Step 8: Load patches into arrays ─────────────────────────────────────
    img_patches, seg_patches, mi_patches, gt_patches = [], [], [], []
    for img_path in sorted(glob.glob(os.path.join(PATCH_EXPORT_DIR, "*_img.npy"))):
        base = img_path.replace("_img.npy", "")
        img_patches.append(np.load(img_path))
        seg_patches.append(np.load(base + "_seg.npy"))
        mi_patches.append(np.clip(np.load(base + "_mi.npy"), 0, 1))
        gt_patches.append(np.load(base + "_gt.npy"))

    img_patches = np.array(img_patches, dtype=np.float32)
    seg_patches = np.array(seg_patches, dtype=np.float32)
    mi_patches  = np.array(mi_patches,  dtype=np.float32)
    gt_patches  = np.array(gt_patches,  dtype=np.float32)
    print("Loaded patches:", len(img_patches))

    # ── Step 9: Build dataset / data-loaders ─────────────────────────────────
    dataset   = PatchCorrectionDataset(img_patches, seg_patches, mi_patches, gt_patches)
    n_val     = int(0.2 * len(dataset))
    n_train   = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2,
                              pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    # ── Step 10: Train patch correction model ─────────────────────────────────
    CKPT_PATH        = os.path.join(OUTPUT_DIR, "patch_correction_best.pth")
    correction_model = PatchCorrectionUNet().to(device)
    optimizer        = torch.optim.AdamW(correction_model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler        = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    best_val = float("inf")

    for epoch in range(EPOCHS):
        correction_model.train()
        train_loss = 0.0
        for x, gt in train_loader:
            x, gt = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            seg_inp = x[:, 3:4, :, :]
            optimizer.zero_grad()
            loss = correction_loss(correction_model(x), gt, seg_inp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(correction_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        correction_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, gt in val_loader:
                xd, gd = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
                val_loss += correction_loss(correction_model(xd), gd, xd[:, 3:4, :, :]).item()
        val_loss /= len(val_loader)

        flag = "  [BEST]" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(correction_model.state_dict(), CKPT_PATH)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}{flag}")

    correction_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    correction_model.eval()
    print("Best patch correction model loaded.")

    # ── Step 11: Run correction model on all exported patches ─────────────────
    for img_path in tqdm(sorted(glob.glob(os.path.join(PATCH_EXPORT_DIR, "*_img.npy"))),
                         desc="Correcting patches"):
        base    = img_path.replace("_img.npy", "")
        gt_path = base + "_gt.npy"
        if not os.path.exists(gt_path):
            continue

        img_rgb_p = np.load(img_path)
        seg       = np.load(base + "_seg.npy")
        mi        = np.clip(np.load(base + "_mi.npy"), 0, 1)
        img_p     = img_rgb_p.transpose(2, 0, 1)

        x = torch.from_numpy(
            np.concatenate([img_p, seg[np.newaxis], mi[np.newaxis]], axis=0)
        ).float().unsqueeze(0).to(device)

        with torch.no_grad():
            new_prob = torch.sigmoid(correction_model(x)).squeeze().cpu().numpy()

        corrected = morphological_postprocess((new_prob > BEST_THRESH).astype(np.uint8))
        out_base  = os.path.join(CORRECTED_DIR, os.path.basename(base))
        np.save(out_base + "_corrected.npy",      corrected)
        np.save(out_base + "_corrected_prob.npy", new_prob)

    print("Patch correction inference done.")

    # ── Step 12: Adaptive stopping analysis ───────────────────────────────────
    adaptive_results, adaptive_details = [], []
    total_possible_patches = 0

    for img_name in tqdm(test_images, desc="Adaptive stopping analysis"):
        mi_map = np.clip(mc_cache[img_name]["mi"], 0, 1)
        all_patches, _ = patch_mi_analysis(mi_map, PATCH_SIZE, TOP_K)
        patch_ranked   = sorted(
            [(r, c, sz, np.mean(np.sort(mi_map[r:r+sz, c:c+sz].flatten())[-10:]))
             for r, c, sz, _ in all_patches],
            key=lambda x: x[3], reverse=True,
        )
        top_patches = patch_ranked[:TOP_K]
        total_possible_patches += len(top_patches)

        baseline_qs = float(np.clip(compute_image_quality_score(mi_map), 0, 1))
        patch_results, n_used, stop_reason = adaptive_refinement_stopping(
            mi_map, top_patches,
            patch_size=PATCH_SIZE,
            delta_threshold=DELTA_THRESHOLD,
            min_patches=MIN_PATCHES,
        )
        final_qs = float(np.clip(
            patch_results[-1]["qs_after"] if patch_results else baseline_qs, 0, 1
        ))
        adaptive_results.append({
            "image":         img_name,
            "baseline_qs":   baseline_qs,
            "final_qs":      final_qs,
            "patches_used":  n_used,
            "patches_saved": len(top_patches) - n_used,
            "reason":        stop_reason,
        })
        for pr in patch_results:
            pr["image"] = img_name
            adaptive_details.append(pr)

    adaptive_df = pd.DataFrame(adaptive_results)
    details_df  = pd.DataFrame(adaptive_details)
    adaptive_df.to_csv(os.path.join(OUTPUT_DIR, "adaptive_results.csv"), index=False)
    details_df.to_csv(os.path.join(OUTPUT_DIR,  "adaptive_details.csv"), index=False)

    total_used  = adaptive_df["patches_used"].sum()
    total_saved = adaptive_df["patches_saved"].sum()
    pct_saved   = 100 * total_saved / max(total_possible_patches, 1)
    print(f"\nAdaptive Stopping: {total_used} applied, {total_saved} skipped "
          f"({pct_saved:.1f}% saved)")
    print(adaptive_df["reason"].value_counts().to_string())

    # ── Step 13: Final multi-pass evaluation ──────────────────────────────────
    random.seed(0)
    success_count        = 0
    total_patch_attempts = 0
    dice_before_list, dice_after_list = [], []
    full_results, random_results      = [], []

    for img_name in tqdm(test_images, desc="Final evaluation"):
        img_rgb, img_tensor = preprocess_image(os.path.join(TEST_IMG_DIR, img_name), device=device)
        cache     = mc_cache[img_name]
        mi_map    = np.clip(cache["mi"], 0, 1)
        mean_map  = cache["mean"]
        pred_prob = cached_preds[img_name]
        gt        = cached_gts[img_name]
        base_name = os.path.splitext(img_name)[0]

        current_mask = morphological_postprocess(
            (pred_prob > BEST_THRESH).astype(np.uint8)
        )
        dice_before_list.append(float(np.clip(compute_metrics(current_mask, gt)["dice"], 0, 1)))

        # Random baseline
        rng_mask       = current_mask.copy()
        all_corr_files = [f for f in os.listdir(CORRECTED_DIR)
                          if f.startswith(base_name + "_r") and f.endswith("_corrected.npy")]
        if all_corr_files:
            chosen = random.sample(all_corr_files,
                                   min(max(1, len(all_corr_files) // 2), len(all_corr_files)))
            for fname in chosen:
                parts = fname.replace("_corrected.npy", "").split("_")
                r    = int(next(p[1:] for p in parts if p.startswith("r")))
                c    = int(next(p[1:] for p in parts if p.startswith("c")))
                size = int(next(p[1:] for p in parts if p.startswith("s")))
                rng_mask[r:r+size, c:c+size] = np.load(
                    os.path.join(CORRECTED_DIR, fname)).astype(np.uint8)
        random_results.append(compute_metrics(morphological_postprocess(rng_mask), gt))

        # Pass 1: MI-ranked ungated correction
        current_mask = apply_corrected_patches(
            current_mask, base_name, CORRECTED_DIR, mi_map, mean_map, TOP_K,
            best_thresh=BEST_THRESH,
        )

        # Pass 2: residual uncertain regions with safety guard
        for _ in range(N_CORR_PASSES - 1):
            preds_mc2        = mc_dropout_predict(models, img_tensor, n_passes=10)
            mean2, mi2, _, _ = compute_mutual_information(preds_mc2)
            mi2              = np.clip(mi2, 0, 1)

            _, top_cands2 = select_patches_mi_only(mi2, PATCH_SIZE, TOP_K // 2,
                                                    mi_floor=UNCERTAINTY_THRESHOLD)
            top_p2 = select_top_patches_non_overlap(top_cands2, TOP_K // 2)

            for r, c, size, patch_mi in top_p2:
                seg_p = current_mask[r:r+size, c:c+size]
                mi_p  = np.clip(mi2[r:r+size, c:c+size], 0, 1)
                corr, _, accepted = corrected_with_guard(
                    correction_model,
                    img_rgb[r:r+size, c:c+size],
                    seg_p, mi_p,
                    mean2[r:r+size, c:c+size],
                    device,
                    threshold=BEST_THRESH,
                )
                total_patch_attempts += 1
                if accepted:
                    current_mask[r:r+size, c:c+size] = corr
                    success_count += 1

        current_mask = morphological_postprocess(current_mask)
        m_after = compute_metrics(current_mask, gt)
        dice_after_list.append(float(np.clip(m_after["dice"], 0, 1)))
        full_results.append(m_after)

    final_df  = pd.DataFrame(full_results)
    random_df = pd.DataFrame(random_results)

    print("\nFinal segmentation metrics (test set, post-correction):")
    print(final_df.mean())

    acceptance_rate = (success_count / total_patch_attempts
                       if total_patch_attempts > 0 else None)
    print_failure_summary(dice_before_list, dice_after_list, acceptance_rate)
    plot_dice_distribution(dice_before_list, dice_after_list, OUTPUT_DIR)

    # ── Step 14: Ablation study (entropy / variance / MI-only baselines) ──────
    entropy_results, variance_results = [], []

    for img_name in tqdm(test_images, desc="Entropy/Variance baselines"):
        cache     = mc_cache[img_name]
        pred_prob = cached_preds[img_name]
        gt        = cached_gts[img_name]
        base_name = os.path.splitext(img_name)[0]

        for unc_key, res_list in [("entropy", entropy_results), ("variance", variance_results)]:
            unc_map = cache[unc_key].copy()
            if unc_key == "variance":
                unc_map = unc_map / (unc_map.max() + 1e-9)

            cur = morphological_postprocess(
                (pred_prob > BEST_THRESH).astype(np.uint8)
            )
            patches, _ = patch_uncertainty_analysis(unc_map, PATCH_SIZE, TOP_K,
                                                     threshold=UNCERTAINTY_THRESHOLD)
            for r, c, size, _ in select_top_patches_non_overlap(patches, TOP_K):
                pname      = f"{base_name}_r{r}_c{c}_s{size}"
                candidates = [f for f in os.listdir(CORRECTED_DIR)
                              if f.startswith(pname) and f.endswith("_corrected.npy")]
                if candidates:
                    cur[r:r+size, c:c+size] = np.load(
                        os.path.join(CORRECTED_DIR, candidates[0])).astype(np.uint8)

            res_list.append(compute_metrics(morphological_postprocess(cur), gt))

    mi_only_results = []
    for img_name in tqdm(test_images, desc="MI-only gate ablation"):
        cache     = mc_cache[img_name]
        mi_map    = np.clip(cache["mi"], 0, 1)
        pred_prob = cached_preds[img_name]
        gt        = cached_gts[img_name]
        base_name = os.path.splitext(img_name)[0]

        cur = morphological_postprocess(
            (pred_prob > BEST_THRESH).astype(np.uint8)
        )
        all_files = [f for f in os.listdir(CORRECTED_DIR) if f.startswith(base_name + "_r")]
        pi = []
        for fname in all_files:
            parts  = fname.replace("_corrected.npy", "").split("_")
            r      = int(next(p[1:] for p in parts if p.startswith("r")))
            c      = int(next(p[1:] for p in parts if p.startswith("c")))
            size   = int(next(p[1:] for p in parts if p.startswith("s")))
            avg_mi = np.mean(np.sort(mi_map[r:r+size, c:c+size].flatten())[-10:])
            pi.append((fname, r, c, size, avg_mi))

        pi.sort(key=lambda x: x[4], reverse=True)
        for fname, r, c, size, avg_mi in pi[:TOP_K]:
            if avg_mi >= UNCERTAINTY_THRESHOLD:
                cur[r:r+size, c:c+size] = np.load(
                    os.path.join(CORRECTED_DIR, fname)).astype(np.uint8)

        mi_only_results.append(compute_metrics(morphological_postprocess(cur), gt))

    ablation = build_ablation_table(
        baseline_df  = baseline_df,
        tta_df       = tta_df,
        entropy_df   = pd.DataFrame(entropy_results),
        variance_df  = pd.DataFrame(variance_results),
        random_df    = random_df,
        mi_only_df   = pd.DataFrame(mi_only_results),
        final_df     = final_df,
    )
    print_ablation_table(ablation)
    ablation.to_csv(os.path.join(OUTPUT_DIR, "ablation_study.csv"), index=False)
    plot_ablation_bar(ablation, OUTPUT_DIR)

    # ── Step 15: Qualitative visualisation ────────────────────────────────────
    n_show = 5
    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
    col_labels = ["Original", "Ground Truth", "Before Correction",
                  "After Correction", "MI Uncertainty"]
    for ax, lbl in zip(axes[0], col_labels):
        ax.set_title(lbl, fontsize=11, fontweight="bold")

    for i, img_name in enumerate(test_images[:n_show]):
        raw = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMG_DIR, img_name)), cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (512, 512))

        mi_map    = mc_cache[img_name]["mi"]
        mean_map  = mc_cache[img_name]["mean"]
        gt        = cached_gts[img_name]
        base_name = os.path.splitext(img_name)[0]

        before_mask = morphological_postprocess(
            (cached_preds[img_name] > BEST_THRESH).astype(np.uint8)
        )
        after_mask = morphological_postprocess(
            apply_corrected_patches(
                before_mask.copy(), base_name, CORRECTED_DIR,
                mi_map, mean_map, TOP_K, best_thresh=BEST_THRESH,
            )
        )

        d_b = compute_metrics(before_mask, gt)["dice"]
        d_a = compute_metrics(after_mask,  gt)["dice"]

        axes[i, 0].imshow(raw);                       axes[i, 0].axis("off")
        axes[i, 1].imshow(gt,           cmap="gray"); axes[i, 1].axis("off")
        axes[i, 2].imshow(before_mask,  cmap="gray")
        axes[i, 2].set_xlabel(f"Dice={d_b:.4f}", fontsize=9); axes[i, 2].axis("off")
        axes[i, 3].imshow(after_mask,   cmap="gray")
        axes[i, 3].set_xlabel(f"Dice={d_a:.4f}  Δ={d_a-d_b:+.4f}", fontsize=9)
        axes[i, 3].axis("off")
        axes[i, 4].imshow(np.clip(mi_map, 0, 1), cmap="hot", vmin=0, vmax=1)
        axes[i, 4].axis("off")

    plt.suptitle("Before vs after uncertainty-driven patch correction",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qualitative_results.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    print("Saved all output files to:", OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
