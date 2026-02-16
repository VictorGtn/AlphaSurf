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
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Configure logging to suppress verbose pinder/gsutil logs
logging.getLogger("pinder").setLevel(logging.WARNING)
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastpdb")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")


def process_id_chunk(
    ids: List[str],
    output_dir: str,
    split: str,
    test_setting: str,
    overwrite: bool = False,
) -> Tuple[int, List[Dict]]:
    """Process a chunk of systems."""
    try:
        from pinder.core.loader.loader import PinderLoader
    except ImportError:
        return len(ids), []

    output_dir = Path(output_dir)
    pdb_dir = output_dir / "pdb"
    systems_info = []

    # Initialize loader for this chunk
    loader = PinderLoader(ids=ids)

    # Use index-based iteration to handle errors in __getitem__ or loading
    for i in range(len(ids)):
        try:
            # Access by index to isolate errors to specific systems
            # Handle loader output format (can be tuple or system)
            item = loader[i]
            if isinstance(item, tuple):
                system = item[0]
            else:
                system = item

            system_id = system.entry.id

            # Get appropriate structures based on setting
            if split == "test":
                receptor_path, ligand_path = _get_test_structures(system, test_setting)
            else:
                # For train/val, use holo (bound) structures
                receptor_path = _get_structure_path(system, "receptor", "holo")
                ligand_path = _get_structure_path(system, "ligand", "holo")

            if receptor_path is None or ligand_path is None:
                # DEBUG: print why we skipped
                # print(
                #     f"SKIP {system_id}: Paths are None (R={receptor_path}, L={ligand_path})"
                # )
                continue

            # Skip check for cloud paths
            if not (
                str(receptor_path).startswith("gs://")
                or str(receptor_path).startswith("s3://")
            ):
                if not os.path.exists(receptor_path) or not os.path.exists(ligand_path):
                    # float check
                    # print(
                    #     f"SKIP {system_id}: Local paths don't exist (R={receptor_path}, L={ligand_path})"
                    # )
                    continue

            # Copy to standardized location
            # Use setting-specific names to allow coexistence of Holo, Apo, and AF2
            suffix = ""
            if split == "test":
                suffix = f"_{test_setting}"

            receptor_dest = pdb_dir / f"{system_id}_R{suffix}.pdb"
            ligand_dest = pdb_dir / f"{system_id}_L{suffix}.pdb"

            # Only copy if files don't exist or overwrite is enabled
            if overwrite or not receptor_dest.exists():
                _copy_file(receptor_path, receptor_dest)

            if overwrite or not ligand_dest.exists():
                _copy_file(ligand_path, ligand_dest)

            systems_info.append(
                {
                    "id": system_id,
                    "receptor_id": f"{system_id}_R",
                    "ligand_id": f"{system_id}_L",
                    "setting": test_setting if split == "test" else "holo",
                }
            )

        except Exception as e:
            # Try to get system ID if possible, otherwise use index
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
) -> List[Dict]:
    """
    Download PINDER clustered dataset (~42k structures) using multiprocessing.
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

    print("Sampling 1 representative per cluster (seed=42)...")
    split_index = split_index.sample(frac=1, random_state=42).drop_duplicates(
        subset=["cluster_id"]
    )

    if limit:
        print(f"Limiting to first {limit} systems for testing...")
        split_index = split_index.head(limit)

    print(f"Processing {len(split_index)} systems with {num_workers} workers...")

    # Get list of IDs
    all_ids = split_index["id"].tolist()

    # Split IDs into chunks
    # Split IDs into small chunks for smooth progress bar
    chunk_size = 50
    chunks = [all_ids[i : i + chunk_size] for i in range(0, len(all_ids), chunk_size)]

    # Prepare arguments for each chunk
    process_func = functools.partial(
        process_id_chunk,
        output_dir=output_dir,
        split=split,
        test_setting=test_setting,
        overwrite=overwrite,
    )

    systems_info = []

    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for smoother updates
            with tqdm(total=len(all_ids), desc=f"Processing {split}") as pbar:
                for count, result in pool.imap_unordered(process_func, chunks):
                    systems_info.extend(result)
                    pbar.update(count)
    else:
        # Serial processing
        with tqdm(total=len(all_ids), desc=f"Processing {split}") as pbar:
            for chunk in chunks:
                count, result = process_id_chunk(
                    chunk, output_dir, split, test_setting, overwrite
                )
                systems_info.extend(result)
                pbar.update(count)

    print(f"\nTotal processed: {len(systems_info)}")
    return systems_info


def _get_structure_path(system, chain_type: str, setting: str) -> Optional[str]:
    """
    Get path to structure file for a chain.

    Args:
        system: PinderSystem object
        chain_type: 'receptor' or 'ligand'
        setting: 'holo' (bound), 'apo' (unbound), 'af2' (predicted)
    """
    try:
        if setting == "holo":
            # Bound (from complex)
            if chain_type == "receptor":
                return (
                    str(system.holo_receptor.filepath) if system.holo_receptor else None
                )
            else:
                return str(system.holo_ligand.filepath) if system.holo_ligand else None

        elif setting == "apo":
            # Unbound experimental structure
            if chain_type == "receptor":
                return (
                    str(system.apo_receptor.filepath) if system.apo_receptor else None
                )
            else:
                return str(system.apo_ligand.filepath) if system.apo_ligand else None

        elif setting == "af2":
            # AlphaFold2 predicted
            if chain_type == "receptor":
                return (
                    str(system.pred_receptor.filepath) if system.pred_receptor else None
                )
            else:
                return str(system.pred_ligand.filepath) if system.pred_ligand else None

    except AttributeError:
        pass

    return None


def _get_test_structures(system, setting: str):
    """Get receptor and ligand paths for test setting."""
    receptor_path = _get_structure_path(system, "receptor", setting)
    ligand_path = _get_structure_path(system, "ligand", setting)

    # Fallback to holo if specific setting not available
    if receptor_path is None:
        receptor_path = _get_structure_path(system, "receptor", "holo")
    if ligand_path is None:
        ligand_path = _get_structure_path(system, "ligand", "holo")

    return receptor_path, ligand_path


def _copy_file(src: str, dst: Path):
    """Copy file, handling both local and remote paths."""
    import shutil
    import subprocess

    if str(src).startswith("gs://") or str(src).startswith("s3://"):
        # Cloud storage — try gsutil, fallback to aws s3
        result = subprocess.run(
            ["gsutil", "cp", str(src), str(dst)],
            capture_output=True,
        )
        if result.returncode != 0:
            subprocess.run(
                ["aws", "s3", "cp", str(src), str(dst)],
                capture_output=True,
            )
    else:
        shutil.copy2(src, dst)


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
                output_dir, split, setting, args.num_workers, args.limit, args.overwrite
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
