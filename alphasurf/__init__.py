import os

from omegaconf import OmegaConf

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

OmegaConf.register_new_resolver(
    "alphasurf_dir",
    lambda *parts: os.path.join(_PACKAGE_DIR, *parts),
    replace=True,
)
