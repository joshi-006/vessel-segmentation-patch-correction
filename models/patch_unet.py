"""
models/patch_unet.py
====================
Contains:
  - AttentionGate
  - PatchConvBlock
  - PatchCorrectionUNet   ← Attention U-Net for patch-level correction
  - PatchCorrectionDataset
  - Loss functions: dice_loss, focal_loss, focal_dice_loss,
                    preservation_loss, correction_loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """Soft attention gate used in skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear",
                               align_corners=False)
        return x * self.psi(F.relu(g1 + x1))


class PatchConvBlock(nn.Module):
    """Double conv → BN → ReLU block with optional spatial dropout."""

    def __init__(self, in_c: int, out_c: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────

class PatchCorrectionUNet(nn.Module):
    """
    Attention U-Net that refines a segmentation patch given the raw image
    patch, the current segmentation, and the MI uncertainty map.

    Input channels (default 5):
        [img_R, img_G, img_B, seg_patch, mi_patch]
    Output: single-channel logit map (same spatial size as input).
    """

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = PatchConvBlock(in_channels, 32)
        self.enc2 = PatchConvBlock(32,  64)
        self.enc3 = PatchConvBlock(64,  128)
        self.enc4 = PatchConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = PatchConvBlock(256, 512)

        # Decoder with attention gates
        self.up4  = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = PatchConvBlock(512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = PatchConvBlock(256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = PatchConvBlock(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = PatchConvBlock(64, 32)

        self.out  = nn.Conv2d(32, 1, 1)

    def _up_cat(
        self,
        upsampled: torch.Tensor,
        skip: torch.Tensor,
        att_gate: AttentionGate,
    ) -> torch.Tensor:
        u = upsampled
        if u.shape[-2:] != skip.shape[-2:]:
            u = F.interpolate(u, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return torch.cat([u, att_gate(u, skip)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._up_cat(self.up4(b),  e4, self.att4))
        d3 = self.dec3(self._up_cat(self.up3(d4), e3, self.att3))
        d2 = self.dec2(self._up_cat(self.up2(d3), e2, self.att2))
        d1 = self.dec1(self._up_cat(self.up1(d2), e1, self.att1))
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PatchCorrectionDataset(Dataset):
    """
    Holds pre-extracted image patches, segmentation patches, MI patches, and
    ground-truth patches.  Returns (input_tensor, gt_tensor) pairs.

    input_tensor channels: [img_R, img_G, img_B, seg, mi]
    """

    def __init__(
        self,
        imgs: np.ndarray,   # (N, H, W, 3)
        segs: np.ndarray,   # (N, H, W)
        mis:  np.ndarray,   # (N, H, W)
        gts:  np.ndarray,   # (N, H, W)
    ):
        self.imgs = imgs
        self.segs = segs
        self.mis  = mis
        self.gts  = gts

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.imgs[idx].copy()).float().permute(2, 0, 1)
        seg = torch.from_numpy(self.segs[idx].copy()).float().unsqueeze(0)
        mi  = torch.from_numpy(self.mis[idx].copy()).float().unsqueeze(0)
        gt  = torch.from_numpy(self.gts[idx].copy()).float().unsqueeze(0)
        return torch.cat([img, seg, mi], dim=0), gt


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss computed over spatial dims."""
    pred  = torch.clamp(torch.sigmoid(pred), 1e-6, 1 - 1e-6)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for class-imbalanced binary segmentation."""
    prob    = torch.sigmoid(pred)
    bce     = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t     = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * bce * (1 - p_t) ** gamma).mean()


def focal_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined focal + Dice loss (40 / 60 weighting)."""
    return 0.4 * focal_loss(pred, target) + 0.6 * dice_loss(pred, target)


def preservation_loss(pred: torch.Tensor, seg_input: torch.Tensor) -> torch.Tensor:
    """Penalise large deviations from the existing segmentation."""
    return torch.mean(torch.abs(torch.sigmoid(pred) - seg_input))


def correction_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seg_input: torch.Tensor,
    preservation_weight: float = 0.08,
) -> torch.Tensor:
    """
    Primary training objective.

    L = focal_dice(pred, gt) + preservation_weight × preservation(pred, seg)
    """
    base = focal_dice_loss(pred, gt)
    pres = preservation_loss(pred, seg_input)
    return base + preservation_weight * pres
