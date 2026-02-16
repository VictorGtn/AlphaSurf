"""
Shared configuration utilities for alphasurf tasks.
"""

from torch_geometric.data import Data


def merge_surface_config(base_cfg, override_cfg=None):
    """
    Merge a base surface/graph config with on-fly overrides.

    Copies all public attributes from base_cfg into a new Data object,
    then selectively overrides with on-fly specific keys.

    Args:
        base_cfg: Base OmegaConf config (e.g., cfg.cfg_surface or cfg.cfg_graph).
        override_cfg: Optional on-fly override config.

    Returns:
        Data object with merged configuration.
    """
    merged = Data()
    for key in dir(base_cfg):
        if not key.startswith("_"):
            try:
                setattr(merged, key, getattr(base_cfg, key))
            except Exception:
                pass

    if override_cfg is not None:
        for key in [
            "surface_method",
            "alpha_value",
            "face_reduction_rate",
            "max_vert_number",
            "min_vert_number",
            "use_pymesh",
            "use_whole_surfaces",
        ]:
            if hasattr(override_cfg, key):
                setattr(merged, key, getattr(override_cfg, key))

    return merged
