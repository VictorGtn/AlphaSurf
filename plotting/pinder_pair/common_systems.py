"""The system set every PINDER figure is computed on.

1835 holo, 309 apo and 1582 af2 complexes: scored by every method in the
benchmark and built from single-connected-component structures. The ids are
frozen on disk so each figure reports the same systems instead of re-deriving
an intersection from whichever dumps it happens to read.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "alphasurf" / "tasks" / "pinder_pair"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "pinder_pair"
ID_DIR = TASK_DIR / "repaired_common_ids_20260907"
SETTINGS = ("holo", "apo", "af2")
SEEDS = (2024, 2025, 2026)


@lru_cache(maxsize=None)
def common_ids(setting):
    return frozenset(pd.read_csv(ID_DIR / f"common_systems_{setting}.csv")["id"])


def restrict(values, setting, source=""):
    """Keep the common systems, rejecting a run that does not cover them all."""
    ids = common_ids(setting)
    selected = values[values["system_id"].isin(ids)]
    if len(selected) != len(ids):
        raise ValueError(
            f"{source or setting} covers {len(selected)} of {len(ids)} "
            f"common {setting} systems"
        )
    return selected


def read_results(path, setting, columns=("system_id", "auroc", "is_homodimer")):
    """Load one per-system dump, restricted to the common systems."""
    return restrict(pd.read_csv(path, usecols=list(columns)), setting, path.name)
