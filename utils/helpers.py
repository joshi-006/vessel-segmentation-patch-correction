"""
utils/helpers.py
================
Small reusable helpers:
  - image preprocessing / normalisation
  - TTA & MC-Dropout inference wrappers
  - uncertainty map computation
  - morphological post-processing
  - quality-score utilities
"""

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & normalisation
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_path: str, size: int = 512, device=None):
    """
    Load a fundus image, apply CLAHE + Gaussian blur on the green channel,
    resize, normalise to [-1, 1], and return (numpy_array, torch_tensor).

    Parameters
    ----------
    img_path : str
        Path to the image file.
    size : int
        Target spatial resolution (square).
    device : torch.device | None
        Device to place the tensor on.  Defaults to CPU.

    Returns
    -------
    img : np.ndarray  shape (size, size, 3)  float32 in [-1, 1]
    tensor : torch.Tensor  shape (1, 3, size, size)
    """
    if device is None:
        device = torch.device("cpu")

    img   = cv2.imread(img_path)
    green = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = clahe.apply(green)
    green = cv2.GaussianBlur(green, (3, 3), 0)

    img  = np.stack([green, green, green], axis=-1)
    img  = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img  = img.astype(np.float32) / 255.0
    img  = (img - 0.5) / 0.5

    tensor = (
        torch.tensor(img, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    return img, tensor


def load_ground_truth(mask_path: str, size: int = 512) -> np.ndarray:
    """Load and binarise a ground-truth segmentation mask."""
    gt = cv2.imread(mask_path, 0)
    gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    return (gt > 127).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Morphological post-processing
# ─────────────────────────────────────────────────────────────────────────────

def morphological_postprocess(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    Remove small connected components (< min_area pixels) from a binary mask.

    Parameters
    ----------
    mask     : np.ndarray  uint8, values in {0, 1}
    min_area : int         Minimum component area to keep.

    Returns
    -------
    cleaned : np.ndarray  uint8
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return cleaned.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# TTA & MC-Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def tta_predict(models: list, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Test-time augmentation over 5 geometric transforms across all ensemble
    members.  Returns the mean probability map.

    Parameters
    ----------
    models     : list of nn.Module  (eval mode set internally)
    img_tensor : torch.Tensor  shape (1, C, H, W)

    Returns
    -------
    np.ndarray  shape (H, W)  float32 in [0, 1]
    """
    augmented = [
        img_tensor,
        torch.flip(img_tensor, dims=[3]),
        torch.flip(img_tensor, dims=[2]),
        torch.rot90(img_tensor, k=1, dims=[2, 3]),
        torch.rot90(img_tensor, k=3, dims=[2, 3]),
    ]
    deaugs = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
    ]
    preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            for aug_inp, deaug_fn in zip(augmented, deaugs):
                p = torch.sigmoid(model(aug_inp))
                preds.append(deaug_fn(p).squeeze().cpu().numpy())
    return np.mean(np.stack(preds), axis=0)


def mc_dropout_predict(
    models: list, img_tensor: torch.Tensor, n_passes: int = 30
) -> np.ndarray:
    """
    Monte-Carlo Dropout inference: run each model in *train* mode to activate
    dropout, collect `n_passes` stochastic predictions per model.

    Parameters
    ----------
    models     : list of nn.Module
    img_tensor : torch.Tensor  shape (1, C, H, W)
    n_passes   : int

    Returns
    -------
    np.ndarray  shape (len(models) * n_passes, H, W)  float32 in [0, 1]
    """
    preds = []
    with torch.no_grad():
        for model in models:
            model.train()          # activates Dropout layers
            for _ in range(n_passes):
                preds.append(
                    torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                )
    return np.stack(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty map helpers
# ─────────────────────────────────────────────────────────────────────────────

def bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """Pixel-wise Shannon entropy of a Bernoulli distribution."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_mutual_information(
    pred_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose predictive uncertainty into epistemic (MI) and aleatoric parts.

    Parameters
    ----------
    pred_stack : np.ndarray  shape (N, H, W)

    Returns
    -------
    mean_map   : np.ndarray  H×W  – mean prediction
    mi_map     : np.ndarray  H×W  – mutual information (epistemic uncertainty)
    total_unc  : np.ndarray  H×W  – total entropy
    aleatoric  : np.ndarray  H×W  – mean aleatoric entropy
    """
    mean_map  = np.mean(pred_stack, axis=0)
    total_unc = bernoulli_entropy(mean_map)
    aleatoric = np.mean(bernoulli_entropy(pred_stack), axis=0)
    mi_map    = np.clip(total_unc - aleatoric, 0, None)
    return mean_map, mi_map, total_unc, aleatoric


def compute_entropy_map(pred_stack: np.ndarray) -> np.ndarray:
    """Entropy of the mean prediction across MC samples."""
    return bernoulli_entropy(np.mean(pred_stack, axis=0))


def compute_variance_map(pred_stack: np.ndarray) -> np.ndarray:
    """Pixel-wise variance across MC samples."""
    return np.var(pred_stack, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Quality-score utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_score(mi_map: np.ndarray) -> float:
    """Simple quality score: 1 − mean(MI)."""
    return float(1 - np.mean(np.clip(mi_map, 0, 1)))


def compute_image_quality_score(mi_map: np.ndarray) -> float:
    """Smoothed quality score using a 0.7 power on MI before averaging."""
    return float(1 - np.mean(np.clip(mi_map, 0, 1) ** 0.7))
