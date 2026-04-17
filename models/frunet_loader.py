"""
models/frunet_loader.py
=======================
Loads the pre-trained FR-UNet MC-Dropout ensemble from disk.

Requires the MIDL24 repo to be cloned and its path added to sys.path
*before* importing this module (see main.py / config.py).
"""

import os
import torch


def load_models(model_dir: str, device: torch.device, n_models: int = 5) -> list:
    """
    Load N FR-UNet MC-Dropout ensemble members from ``model_dir``.

    Expected filenames: ``FRUNet_MC_0.pth``, ``FRUNet_MC_1.pth``, …

    Parameters
    ----------
    model_dir : str
        Directory containing the ``.pth`` weight files.
    device : torch.device
    n_models : int
        Number of ensemble members to load.

    Returns
    -------
    list of nn.Module  (each set to eval mode, moved to ``device``)
    """
    # Imported here so callers can inject the MIDL path before importing.
    from models.frunet import FR_UNet  # noqa: E402

    loaded = []
    for i in range(n_models):
        path = os.path.join(model_dir, f"FRUNet_MC_{i}.pth")
        m = FR_UNet(num_classes=1, num_channels=3, dropout=0.3)
        m.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)
        )
        m.to(device).eval()
        loaded.append(m)

    print(f"Loaded {len(loaded)} FR-UNet ensemble members.")
    return loaded
