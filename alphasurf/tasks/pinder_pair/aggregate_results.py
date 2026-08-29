"""Aggregate per-system test results across methods."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ATOM_CSV = REPO_ROOT / "data" / "pdb_atom_components.csv"
DEFAULT_PINDER_DIR = REPO_ROOT / "data" / "pinder-pair"
SETTINGS = ["holo", "apo", "af2"]


def _is_homodimer(system_id):
    parts = system_id.split("--")
    if len(parts) != 2:
        return False
    return parts[0].split("_")[-1] == parts[1].split("_")[-1]


def load_pdb_component_map(atom_csv, restrict_to=None):
    df = pd.read_csv(atom_csv)
    df = df[df["error"].isna()]
    if restrict_to is not None:
        df = df[df["pdb_name"].isin(restrict_to)]
    return dict(zip(df["pdb_name"], df["n_components"]))


def candidate_pdb_names_for(system_ids, setting):
    names = set()
    for sid in system_ids:
        for side in ("L", "R"):
            names.add(f"{sid}_{side}_{setting}.pdb")
            names.add(f"{sid}_{side}.pdb")
    return names


def component_count(system_id, side, setting, pdb_to_nc):
    for cand in (f"{system_id}_{side}_{setting}.pdb", f"{system_id}_{side}.pdb"):
        if cand in pdb_to_nc:
            return pdb_to_nc[cand]
    return None


def system_is_single_component(system_id, setting, pdb_to_nc):
    for side in ("L", "R"):
        if component_count(system_id, side, setting, pdb_to_nc) != 1:
            return False
    return True


def load_test_ids(pinder_dir, setting):
    p = Path(pinder_dir) / f"systems_test_{setting}.csv"
    if not p.exists():
        return None
    return set(pd.read_csv(p)["id"])


def print_header():
    cols = [
        ("Method", 30),
        ("AUROC mean", 10),
        ("Homo", 10),
        ("Hetero", 10),
        ("BACC", 10),
        ("N", 5),
    ]
    print("  " + "".join(f"{name:>{width}}" for name, width in cols))
    print("  " + "-" * sum(w for _, w in cols))


def print_row(name, df):
    is_homo = df["system_id"].map(_is_homodimer)
    homo = df.loc[is_homo, "auroc"].mean()
    hetero = df.loc[~is_homo, "auroc"].mean()
    print(
        f"  {name:<30} {df['auroc'].mean():>10.4f} "
        f"{homo:>10.4f} {hetero:>10.4f} {df['bacc'].mean():>10.4f} {len(df):>5}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", default="per_system_results")
    p.add_argument("--atom-csv", default=DEFAULT_ATOM_CSV)
    p.add_argument("--pinder-dir", default=DEFAULT_PINDER_DIR)
    p.add_argument(
        "--test-mode",
        action="store_true",
        help="Impute 0.5 AUROC for test PDBs that should work but lack a method result.",
    )
    p.add_argument(
        "--impute-dir",
        default=None,
        help="Directory with {method}_imputed_systems_{setting}.csv files "
        "(one column 'system_id'); those systems are forced to 0.5 AUROC "
        "for the matching method in test mode. Defaults to --dump-dir.",
    )
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    if not args.test_mode:
        pdb_to_nc = load_pdb_component_map(args.atom_csv)
        print(f"Loaded {len(pdb_to_nc)} single-component PDBs from {args.atom_csv}")

    for setting in SETTINGS:
        print(f"\n{'=' * 60}")
        print(f"Setting: {setting}")
        print(f"{'=' * 60}")

        dfs = {}
        for csv_file in sorted(dump_dir.glob(f"*_{setting}.csv")):
            name = csv_file.stem.rsplit("_", 1)[0]
            dfs[name] = pd.read_csv(csv_file)
            print(f"  {name}: {len(dfs[name])} systems")

        if args.test_mode:
            test_ids = load_test_ids(args.pinder_dir, setting)
            if test_ids is None:
                print(f"  No systems_test_{setting}.csv found; skipping.")
                continue
            setting_names = candidate_pdb_names_for(test_ids, setting)
            pdb_to_nc = load_pdb_component_map(args.atom_csv, restrict_to=setting_names)
            should_work = {
                sid
                for sid in test_ids
                if system_is_single_component(sid, setting, pdb_to_nc)
            }
            print(
                f"\n  [{setting}] atom CSV PDBs: {len(pdb_to_nc)} | "
                f"test: {len(test_ids)} -> should-work: {len(should_work)}"
            )
            print_header()
            impute_dir = Path(args.impute_dir) if args.impute_dir else dump_dir
            impute_map = {}
            for csv_file in impute_dir.glob(f"*_imputed_systems_{setting}.csv"):
                key = csv_file.stem.replace(f"_imputed_systems_{setting}", "")
                impute_map[key] = set(pd.read_csv(csv_file)["system_id"])
            for name, df in dfs.items():
                df_method = df.copy()
                n_forced = 0
                for key, force_ids in impute_map.items():
                    if key in name:
                        mask = df_method["system_id"].isin(force_ids & should_work)
                        n_forced += int(mask.sum())
                        df_method.loc[mask, ["auroc", "bacc"]] = np.nan
                full = pd.DataFrame({"system_id": list(should_work)})
                full = full.merge(
                    df_method[["system_id", "auroc", "bacc"]],
                    on="system_id",
                    how="left",
                )
                n_imputed = int(full["auroc"].isna().sum())
                full[["auroc", "bacc"]] = (
                    full[["auroc", "bacc"]].astype(float).fillna(0.5)
                )
                tag_parts = []
                if n_imputed:
                    tag_parts.append(f"{n_imputed} imputed")
                if n_forced:
                    tag_parts.append(f"{n_forced} forced")
                tag = f"  [{', '.join(tag_parts)}]" if tag_parts else ""
                print_row(f"{name}{tag}", full)
        else:
            if len(dfs) < 2:
                print("  Need at least 2 methods for intersection")
                continue
            common_ids = set(dfs[next(iter(dfs))]["system_id"])
            for df in dfs.values():
                common_ids &= set(df["system_id"])
            before = len(common_ids)
            common_ids = {
                sid
                for sid in common_ids
                if system_is_single_component(sid, setting, pdb_to_nc)
            }
            print(
                f"\n  Common systems: {before} -> {len(common_ids)} after single-comp filter"
            )
            print_header()
            for name, df in dfs.items():
                print_row(name, df[df["system_id"].isin(common_ids)])


if __name__ == "__main__":
    main()
