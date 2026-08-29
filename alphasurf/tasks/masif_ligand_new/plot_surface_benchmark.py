#!/usr/bin/env python3
"""Create timing and mesh-size bar plots from a benchmark summary."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    ("alpha_complex", "Alpha-complex"),
    ("edtsurf", "EDTSurf"),
    ("nanoshaper", "NanoShaper"),
    ("msms", "MSMS (0.1 reduction)"),
    ("edtsurf_default", "EDTSurf 4.0"),
    ("nanoshaper_default", "NanoShaper 2.0"),
    ("msms_full", "MSMS (full)"),
)


def load_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_plot(labels, means, errors, ylabel, title, output):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=errors, capsize=3, color="#4878a8")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=250)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.summary)
    selected = []
    for method, label in ORDER:
        matches = [row for row in rows if row["method"] == method]
        if method in {"edtsurf", "nanoshaper"}:
            matches = [
                row for row in matches if row["grid_scale"] in {"0.3", "0.4", "0.5"}
            ]
        for row in matches:
            grid = row["grid_scale"]
            current_label = (
                label
                if method not in {"edtsurf", "nanoshaper"}
                else f"{row['label']} {grid}"
            )
            selected.append((current_label, row))

    labels = [item[0] for item in selected]
    timing = [float(item[1]["mean_surface_ms"]) for item in selected]
    timing_std = [float(item[1]["std_surface_ms"]) for item in selected]
    vertices = [float(item[1]["mean_vertices"]) for item in selected]
    vertices_std = [float(item[1]["std_vertices"]) for item in selected]
    output_dir = args.summary.parent
    make_plot(
        labels,
        timing,
        timing_std,
        "Surface generation time (ms/protein)",
        "MaSIF-Ligand surface-generation speed",
        output_dir / "surface_generation_speed.png",
    )
    make_plot(
        labels,
        vertices,
        vertices_std,
        "Mean vertices per protein",
        "MaSIF-Ligand mesh size",
        output_dir / "surface_mean_vertices.png",
    )


if __name__ == "__main__":
    main()
