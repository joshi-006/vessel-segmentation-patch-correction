"""
models/frunet_loader.py
=======================
Loads the pre-trained FR-UNet MC-Dropout ensemble from disk.

Requires the MIDL24 repo to be cloned locally (see main.py / config.py for
the expected path). The FR_UNet class is imported directly from the MIDL24
repo's own ``models/frunet.py`` at call time, working around the fact that
this project also has a top-level ``models`` package of its own.
"""

import os
import sys
import torch


def _import_midl24_frunet(midl_repo_path: str):
    """
    Import ``FR_UNet`` from the MIDL24 repo's own ``models`` package.

    This project also defines a top-level ``models`` package (this one), so
    a plain ``from models.frunet import FR_UNet`` resolves to *this* package
    instead of the MIDL24 repo as soon as ``models`` is cached in
    sys.modules or the project root precedes the MIDL24 path on sys.path.
    Temporarily put the MIDL24 path first and evict any cached ``models``
    entries so the import machinery is forced to resolve ``models.frunet``
    against the MIDL24 repo, then restore everything afterward so the rest
    of the codebase keeps using this project's own ``models`` package.
    """
    saved_path    = list(sys.path)
    saved_modules = {name: mod for name, mod in sys.modules.items()
                      if name == "models" or name.startswith("models.")}
    for name in saved_modules:
        del sys.modules[name]

    sys.path.insert(0, os.path.abspath(midl_repo_path))
    try:
        from models.frunet import FR_UNet
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        sys.modules.update(saved_modules)

    return FR_UNet


def load_models(
    model_dir: str,
    device: torch.device,
    n_models: int = 5,
    midl_repo_path: str = "./MIDL24-segmentation_quality_control",
) -> list:
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
    midl_repo_path : str
        Path to the cloned MIDL24-segmentation_quality_control repo, whose
        ``models/frunet.py`` defines the FR_UNet architecture.

    Returns
    -------
    list of nn.Module  (each set to eval mode, moved to ``device``)
    """
    FR_UNet = _import_midl24_frunet(midl_repo_path)

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
