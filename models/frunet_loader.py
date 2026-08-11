"""
models/frunet_loader.py
=======================
Loads the pre-trained FR-UNet MC-Dropout ensemble from disk.

Requires the MIDL24 repo to be cloned locally (see main.py / config.py for
the expected path). The FR_UNet class is imported directly from the cloned
MIDL24 repo at call time.
"""

import os
import sys
import torch


# The MIDL24 repo has been restructured upstream since this project was
# first built against it: current clones expose the model at
# segmentation_quality_control/models/frunet.py (a proper installable
# package, per its pyproject.toml), but this project's docs/history assumed
# an older top-level models/frunet.py layout. Try the current layout first,
# fall back to the old one, so this keeps working whichever the user has
# cloned.
_MIDL24_FRUNET_CANDIDATES = [
    "segmentation_quality_control.models.frunet",  # current upstream layout
    "models.frunet",                               # older upstream layout
]


def _import_midl24_frunet(midl_repo_path: str):
    """
    Import ``FR_UNet`` from the cloned MIDL24 repo, trying each known layout.

    This project also defines a top-level ``models`` package (this one), so
    a plain ``from models.frunet import FR_UNet`` (the older MIDL24 layout)
    would resolve to *this* package instead of the MIDL24 repo as soon as
    ``models`` is cached in sys.modules or the project root precedes the
    MIDL24 path on sys.path. Temporarily put the MIDL24 path first and evict
    any cached modules with a colliding top-level name so resolution is
    forced against the MIDL24 repo, then restore everything afterward so the
    rest of the codebase keeps using this project's own packages.
    """
    top_level_names = {path.split(".")[0] for path in _MIDL24_FRUNET_CANDIDATES}

    saved_path    = list(sys.path)
    saved_modules = {name: mod for name, mod in sys.modules.items()
                      if name.split(".")[0] in top_level_names}
    for name in saved_modules:
        del sys.modules[name]

    sys.path.insert(0, os.path.abspath(midl_repo_path))
    try:
        errors = []
        for module_path in _MIDL24_FRUNET_CANDIDATES:
            try:
                mod = __import__(module_path, fromlist=["FR_UNet"])
                return mod.FR_UNet
            except ImportError as e:
                errors.append(f"  {module_path}: {e}")
        raise ImportError(
            f"Could not import FR_UNet from the MIDL24 repo at {midl_repo_path!r}. "
            "Tried:\n" + "\n".join(errors) +
            "\nCheck that the repo cloned correctly and its layout hasn't changed again."
        )
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name.split(".")[0] in top_level_names:
                del sys.modules[name]
        sys.modules.update(saved_modules)


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
        Path to the cloned MIDL24-segmentation_quality_control repo,
        somewhere under which frunet.py defines the FR_UNet architecture
        (see _MIDL24_FRUNET_CANDIDATES for the layouts tried).

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
