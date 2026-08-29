#!/usr/bin/env python3
"""Combine per-method PINDER preprocessing benchmarks and render the table."""

import argparse
import csv
import shutil
import statistics
from pathlib import Path

from benchmark_pinder_preprocessing import Method, summarize, write_outputs


METHODS = (
    Method("alpha_complex", "Alpha-complex"),
    Method("edtsurf", "EDTSurf", 0.3),
    Method("edtsurf", "EDTSurf", 0.4),
    Method("edtsurf", "EDTSurf", 0.5),
    Method("nanoshaper", "NanoShaper", 0.4),
    Method("nanoshaper", "NanoShaper", 0.5),
    Method(
        "msms_simplified", "MSMS Simplified", face_reduction_rate=0.1, engine="msms"
    ),
)
METHOD_DIRS = ("alpha", "edt03", "edt04", "edt05", "nano04", "nano05", "msms")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--surface-timing-summary", type=Path, required=True)
    parser.add_argument("--msms-timing-raw", type=Path, required=True)
    return parser.parse_args()


def apply_surface_timings(summary, surface_timing_summary, msms_timing_raw):
    with surface_timing_summary.open(newline="") as handle:
        timing_rows = list(csv.DictReader(handle))
    timings = {(row["method"], row["grid_scale"]): row for row in timing_rows}

    for row in summary:
        grid_scale = "" if row["grid_scale"] == "" else f"{float(row['grid_scale']):g}"
        method = "msms" if row["method"] == "msms_simplified" else row["method"]
        timing = timings[(method, grid_scale)]
        row["surface_mean_ms"] = float(timing["time_ms"])
        row["surface_std_ms"] = float(timing["time_std"])

    with msms_timing_raw.open(newline="") as handle:
        simplification_times = [
            float(row["t_simplify"]) * 1000
            for row in csv.DictReader(handle)
            if row["crash"] == "False"
        ]
    msms = next(row for row in summary if row["method"] == "msms_simplified")
    msms["mesh_mean_ms"] = statistics.fmean(simplification_times)
    msms["mesh_std_ms"] = statistics.pstdev(simplification_times)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.benchmark_dir / "combined"
    rows = []
    manifests = []

    for method_dir in METHOD_DIRS:
        directory = args.benchmark_dir / method_dir
        raw_path = directory / "pinder_preprocessing_raw.csv"
        manifest_path = directory / "pdb_sample_manifest.csv"
        with raw_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
        manifests.append(manifest_path)

    reference_manifest = manifests[0].read_bytes()
    if any(path.read_bytes() != reference_manifest for path in manifests[1:]):
        raise SystemExit("Per-method jobs did not use identical PDB manifests")

    summary = summarize(rows, METHODS)
    apply_surface_timings(
        summary,
        args.surface_timing_summary,
        args.msms_timing_raw,
    )
    raw_path, summary_path, table_path = write_outputs(rows, summary, output_dir)
    shutil.copyfile(manifests[0], output_dir / "pdb_sample_manifest.csv")
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"LaTeX table: {table_path}")


if __name__ == "__main__":
    main()
