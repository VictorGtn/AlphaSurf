"""Summarize MISATO protein sizes from the preprocessed binding-site cache."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch


QUANTILES = (0, 1, 5, 25, 50, 75, 90, 95, 99, 100)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/misato")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
    )
    return parser.parse_args()


def load_record(path):
    path = Path(path)
    item = torch.load(path, map_location="cpu", weights_only=False)
    labels = item["y"].view(-1)
    source_index = item.get("protein_source_index", item.get("atom_pos"))
    if source_index is None:
        raise KeyError("missing both protein_source_index and atom_pos")
    return {
        "pdb_id": path.stem,
        "residues": int(labels.numel()),
        "heavy_atoms": int(len(source_index)),
        "positive_residues": int(labels.sum().item()),
    }


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    quantile_values = np.percentile(values, QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(percentile): float(value)
            for percentile, value in zip(QUANTILES, quantile_values)
        },
    }


def summarize(records):
    residues = [record["residues"] for record in records]
    heavy_atoms = [record["heavy_atoms"] for record in records]
    positive = sum(record["positive_residues"] for record in records)
    total_residues = sum(residues)
    return {
        "complexes": len(records),
        "residues": describe(residues),
        "heavy_atoms": describe(heavy_atoms),
        "heavy_atoms_per_residue": describe(
            [record["heavy_atoms"] / record["residues"] for record in records]
        ),
        "positive_residues": positive,
        "total_residues": total_residues,
        "positive_residue_rate": positive / total_residues,
        "zero_positive_complexes": sum(
            record["positive_residues"] == 0 for record in records
        ),
    }


def load_split_map(split_dir):
    split_for_id = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.txt"
        for pdb_id in path.read_text().splitlines():
            if pdb_id.strip():
                split_for_id[pdb_id.strip().lower()] = split
    return split_for_id


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "binding_site"
    split_dir = Path(args.split_dir) if args.split_dir else data_dir / "splits"
    output_dir = (
        Path(args.output_dir) if args.output_dir else data_dir / "size_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cache_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No preprocessed .pt files found in {cache_dir}")
    split_for_id = load_split_map(split_dir)

    workers = max(1, min(args.num_workers, len(paths)))
    failures = []
    records = []
    if workers == 1:
        iterator = map(load_record, paths)
        for path, result in zip(paths, iterator):
            records.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [(path, executor.submit(load_record, path)) for path in paths]
            for index, (path, future) in enumerate(futures, 1):
                try:
                    records.append(future.result())
                except Exception as exc:
                    failures.append({"pdb_id": path.stem, "error": repr(exc)})
                if index % 1000 == 0:
                    print(f"Loaded {index}/{len(paths)} complexes", flush=True)

    for record in records:
        record["split"] = split_for_id.get(record["pdb_id"], "unknown")
        record["positive_fraction"] = record["positive_residues"] / record["residues"]

    records.sort(key=lambda record: record["pdb_id"])
    csv_path = output_dir / "misato_protein_sizes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    groups = {"all": records}
    groups.update(
        {
            split: [record for record in records if record["split"] == split]
            for split in ("train", "val", "test", "unknown")
        }
    )
    summary = {
        "source_files": len(paths),
        "loaded_complexes": len(records),
        "failures": failures,
        "groups": {name: summarize(group) for name, group in groups.items() if group},
        "largest_by_residues": sorted(
            records, key=lambda record: record["residues"], reverse=True
        )[:20],
        "largest_by_heavy_atoms": sorted(
            records, key=lambda record: record["heavy_atoms"], reverse=True
        )[:20],
        "residue_thresholds": {
            "lt_16": sum(record["residues"] < 16 for record in records),
            "lt_32": sum(record["residues"] < 32 for record in records),
            "ge_500": sum(record["residues"] >= 500 for record in records),
            "ge_1000": sum(record["residues"] >= 1000 for record in records),
        },
    }
    json_path = output_dir / "misato_size_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
