"""
Preprocessing script for PINDER clustered dataset (~42k structures).

PINDER provides:
- Clustered subset (~42k systems) split by interface similarity
- Test systems in 3 settings:
  - Bound (holo): ground truth complex structures
  - Apo (unbound): experimental unbound structures
  - AF2: AlphaFold2 predicted structures

Usage:
    # Download clustered dataset
    python preprocess.py --output_dir /path/to/pinder_data

    # Specific split
    python preprocess.py --output_dir /path/to/pinder_data --split train

    # Prepare test set with specific setting
    python preprocess.py --output_dir /path/to/pinder_data --split test --test_setting apo
"""

import argparse
import functools
import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Configure logging to suppress verbose pinder/gsutil logs
logging.getLogger("pinder").setLevel(logging.WARNING)
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastpdb")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")


def _count_atoms_in_pdb(pdb_path: Path) -> int:
    if not pdb_path.exists():
        return 0
    count = 0
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                count += 1
    return count


def process_id_chunk(
    ids: List[str],
    output_dir: str,
    split: str,
    test_setting: str,
    overwrite: bool = False,
    cluster_id_by_system: Dict[str, str] = None,
) -> Tuple[int, List[Dict]]:
    """Process a chunk of systems using PinderSystem properties directly."""
    try:
        from pinder.core.loader.loader import PinderLoader
    except ImportError:
        return len(ids), []

    output_dir = Path(output_dir)
    pdb_dir = output_dir / "pdb"
    systems_info = []
    loader = PinderLoader(ids=ids)

    for i in range(len(ids)):
        try:
            item = loader[i]
            if isinstance(item, tuple):
                system = item[0]
            else:
                system = item

            system_id = system.entry.id
            setting = test_setting if split == "test" else "holo"

            if setting == "holo":
                # native_R/L extracted from the dimer — avoids corrupted
                # test_set_pdbs for test systems, works for train/val too.
                r_struct = system.native_R
                l_struct = system.native_L
            elif setting == "apo":
                r_struct = system.apo_receptor
                l_struct = system.apo_ligand
            elif setting == "af2":
                r_struct = system.pred_receptor
                l_struct = system.pred_ligand
            else:
                continue

            if r_struct is None or l_struct is None:
                continue

            suffix = f"_{setting}" if split == "test" else ""
            receptor_dest = pdb_dir / f"{system_id}_R{suffix}.pdb"
            ligand_dest = pdb_dir / f"{system_id}_L{suffix}.pdb"

            if overwrite or not receptor_dest.exists():
                r_struct.to_pdb(receptor_dest)
            if overwrite or not ligand_dest.exists():
                l_struct.to_pdb(ligand_dest)

            entry = {
                "id": system_id,
                "receptor_id": f"{system_id}_R",
                "ligand_id": f"{system_id}_L",
                "setting": setting,
                "n_atoms_R": _count_atoms_in_pdb(receptor_dest),
                "n_atoms_L": _count_atoms_in_pdb(ligand_dest),
            }
            if cluster_id_by_system is not None:
                cid = cluster_id_by_system.get(system_id)
                if cid is not None:
                    entry["cluster_id"] = cid
            systems_info.append(entry)

        except Exception as e:
            sys_id_str = ids[i] if i < len(ids) else f"index_{i}"
            print(f"Error processing system {sys_id_str}: {e}")
            continue

    return len(ids), systems_info


def download_pinder_clustered(
    output_dir: str,
    split: str = "train",
    test_setting: str = "apo",
    num_workers: int = 8,
    limit: int = None,
    overwrite: bool = False,
    all_members: bool = False,
) -> List[Dict]:
    """Download PINDER split using multiprocessing.

    all_members=True keeps every system in the split (no cluster dedup) and
    attaches cluster_id to each row so the dataset can group and sample at
    draw time.
    """
    try:
        from pinder.core import get_index
    except ImportError:
        raise ImportError(
            "pinder package not installed. Install with: pip install pinder"
        )

    output_dir = Path(output_dir)
    pdb_dir = output_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PINDER {split} split...")
    index = get_index()
    split_index = index[index["split"] == split].copy()

    cluster_id_by_system = None
    if all_members:
        print(f"Keeping ALL members per cluster ({len(split_index)} systems)...")
        cluster_id_by_system = dict(
            zip(split_index["id"].tolist(), split_index["cluster_id"].tolist())
        )
    else:
        print("Sampling 1 representative per cluster (seed=42)...")
        split_index = split_index.sample(frac=1, random_state=42).drop_duplicates(
            subset=["cluster_id"]
        )

    if limit:
        print(f"Limiting to first {limit} systems for testing...")
        split_index = split_index.head(limit)

    print(f"Processing {len(split_index)} systems with {num_workers} workers...")

    all_ids = split_index["id"].tolist()

    chunk_size = 50
    chunks = [all_ids[i : i + chunk_size] for i in range(0, len(all_ids), chunk_size)]

    process_func = functools.partial(
        process_id_chunk,
        output_dir=output_dir,
        split=split,
        test_setting=test_setting,
        overwrite=overwrite,
        cluster_id_by_system=cluster_id_by_system,
    )

    systems_info = []

    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(all_ids), desc=f"Processing {split}") as pbar:
                for count, result in pool.imap_unordered(process_func, chunks):
                    systems_info.extend(result)
                    pbar.update(count)
    else:
        with tqdm(total=len(all_ids), desc=f"Processing {split}") as pbar:
            for chunk in chunks:
                count, result = process_id_chunk(
                    chunk,
                    output_dir,
                    split,
                    test_setting,
                    overwrite,
                    cluster_id_by_system,
                )
                systems_info.extend(result)
                pbar.update(count)

    print(f"\nTotal processed: {len(systems_info)}")
    return systems_info


def save_split(
    systems_info: List[Dict], output_dir: Path, split: str, setting: str = None
):
    """Save systems info to CSV."""
    import pandas as pd

    df = pd.DataFrame(systems_info)
    df["split"] = split

    if setting and split == "test":
        filename = f"systems_{split}_{setting}.csv"
    else:
        filename = f"systems_{split}.csv"

    df.to_csv(output_dir / filename, index=False)
    print(f"Saved {len(df)} systems to {filename}")


def create_index_parquet(output_dir: str):
    """Combine split CSVs into index.parquet."""
    import pandas as pd

    output_dir = Path(output_dir)
    dfs = []

    # Load all split files
    for csv_file in output_dir.glob("systems_*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output_dir / "index.parquet")
        print(f"Created index.parquet with {len(combined)} total systems")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PINDER clustered dataset (~42k structures) for PinderPair task"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to process: train, val, test, or 'all' (default: all)",
    )
    parser.add_argument(
        "--test_setting",
        type=str,
        default="all",
        choices=["holo", "apo", "af2", "all"],
        help="Test structure setting: holo (bound), apo (unbound), af2 (predicted), or all (default)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of systems for testing (default: None)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDB files (default: False)",
    )
    parser.add_argument(
        "--all_members",
        action="store_true",
        help=(
            "Keep every system in the split (no 1-per-cluster dedup). "
            "Attaches cluster_id to each CSV row so the dataset can sample "
            "members per draw. Typically used with --split train."
        ),
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_to_process = (
        ["train", "val", "test"] if args.split in (None, "all") else [args.split]
    )

    for split in splits_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print(f"{'=' * 60}")

        if split == "test" and args.test_setting == "all":
            settings_to_process = ["holo", "apo", "af2"]
        elif split == "test":
            settings_to_process = [args.test_setting]
        else:
            settings_to_process = ["holo"]

        for setting in settings_to_process:
            print(f"  > Setting: {setting}")
            systems = download_pinder_clustered(
                output_dir,
                split,
                setting,
                args.num_workers,
                args.limit,
                args.overwrite,
                all_members=args.all_members,
            )
            save_split(systems, output_dir, split, setting if split == "test" else None)

    print("\nCreating combined index...")
    create_index_parquet(output_dir)

    print(f"""
{"=" * 60}
PINDER clustered dataset prepared!
{"=" * 60}

Directory structure:
{output_dir}/
├── pdb/
│   ├── {{system_id}}_R.pdb   # Receptor
│   ├── {{system_id}}_L.pdb   # Ligand
│   ...
├── index.parquet
├── systems_train.csv
├── systems_val.csv
└── systems_test_{args.test_setting}.csv

Dataset info:
- ~42k clustered systems (split by interface similarity)
- Train/Val: holo (bound) structures
- Test: {args.test_setting} structures

To train:
    python train.py data_dir={output_dir}

To evaluate different test settings:
    # Prepare apo test set
    python preprocess.py --output_dir {output_dir} --split test --test_setting apo

    # Prepare AF2 test set
    python preprocess.py --output_dir {output_dir} --split test --test_setting af2
""")


if __name__ == "__main__":
    main()
