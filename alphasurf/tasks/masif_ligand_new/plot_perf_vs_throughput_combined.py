#!/usr/bin/env python3
"""Plot MaSIF-Ligand and Pinder performance against input time."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


COLORS = {
    "Alpha Complex": "#E41A1C",
    "EDTsurf gs=0.3": "#9ECAE1",
    "EDTsurf gs=0.4": "#4292C6",
    "EDTsurf gs=0.5": "#08519C",
    "NanoShaper gs=0.4": "#41AB5D",
    "NanoShaper gs=0.5": "#238B45",
    "MSMS": "#6A3D9A",
}
MARKERS = {
    "Alpha Complex": "o",
    "EDTsurf gs=0.3": "o",
    "EDTsurf gs=0.4": "o",
    "EDTsurf gs=0.5": "o",
    "NanoShaper gs=0.4": "o",
    "NanoShaper gs=0.5": "o",
    "MSMS": "o",
}
FAMILY_COLORS = {
    "Alpha Complex": COLORS["Alpha Complex"],
    "EDTsurf": COLORS["EDTsurf gs=0.4"],
    "NanoShaper": COLORS["NanoShaper gs=0.5"],
    "MSMS": COLORS["MSMS"],
}
MASIF = {
    "Alpha Complex": (51.9 / 2220, 0.8407, 0.0257),
    "EDTsurf gs=0.5": (66.5 / 2243, 0.792537, 0.019214),
    "NanoShaper gs=0.5": (101.6 / 2243, 0.841914, 0.028190),
    "MSMS": (131.2 / 2107, 0.8304, 0.0301),
}

PINDER_TIME = {
    "Alpha Complex": 1 / 38.01,
    "EDTsurf gs=0.3": 1 / 52.97,
    "EDTsurf gs=0.4": 1 / 39.26,
    "EDTsurf gs=0.5": 1 / 25.86,
    "NanoShaper gs=0.4": 1 / 19.87,
    "NanoShaper gs=0.5": 1 / 12.68,
    "MSMS": 0.057,
}
PINDER_THRESHOLD = 0.0342

PINDER_DIR = Path(__file__).resolve().parents[1] / "pinder_pair"
PINDER_SUMMARY = PINDER_DIR / "perf_vs_throughput_repaired_common_all_summary.csv"
PINDER_COMPLEX_TYPE_SUMMARY = (
    PINDER_DIR / "perf_vs_throughput_repaired_common_homo_hetero_summary.csv"
)
PLOTTED_METHODS = frozenset(COLORS)


def load_pinder(path=PINDER_SUMMARY):
    panels = {"Holo": {}, "Apo": {}, "AF2": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}[row["setting"]]
            method = row["method"]
            if method not in PLOTTED_METHODS:
                continue
            if method in panels[split]:
                raise ValueError(f"Duplicate {method}/{split} row in {path}")
            panels[split][method] = (
                float(row["auroc_mean"]),
                float(row["auroc_std"]),
            )
    return panels


def load_pinder_complex_types(path=PINDER_COMPLEX_TYPE_SUMMARY):
    panels = {"Holo": {}, "Apo": {}, "AF2": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}[row["setting"]]
            method = row["method"]
            if method not in PLOTTED_METHODS:
                continue
            panels[split][method, row["complex_type"]] = (
                float(row["auroc_mean"]),
                float(row["auroc_std"]),
            )
    return panels


def fit_axes(ax, xs, values):
    x_span = max(xs) - min(xs)
    x_pad = max(0.004, 0.08 * x_span)
    lower = min(mean - error for mean, error in values)
    upper = max(mean + error for mean, error in values)
    y_pad = max(0.004, 0.10 * (upper - lower))
    ax.set_xlim(max(0, min(xs) - x_pad), max(xs) + x_pad)
    ax.set_ylim(max(0, lower - y_pad), min(1, upper + y_pad))


def style_axes(ax, title, threshold):
    xmin, xmax = ax.get_xlim()
    if threshold < xmax:
        ax.axvspan(max(threshold, xmin), xmax, color="#f5f5f5", zorder=0)
    if xmin < threshold < xmax:
        ax.axvline(
            threshold, color="grey", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1
        )
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_xlabel("Time per protein (sec/prot)", fontsize=14)
    ax.set_ylabel("Mean per-system AUROC", fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(linestyle="--", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_point(ax, method, x, y, error=0.0, facecolor=None, marker=None):
    ax.errorbar(
        x,
        y,
        yerr=error,
        fmt=marker or MARKERS[method],
        color=COLORS[method],
        markerfacecolor=COLORS[method] if facecolor is None else facecolor,
        markeredgecolor=COLORS[method],
        markeredgewidth=1.4,
        markersize=11,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )


def connect_families(ax, values, value_index=0):
    for family in ("EDTsurf", "NanoShaper"):
        methods = [method for method in values if method.startswith(family)]
        ax.plot(
            [PINDER_TIME[method] for method in methods],
            [values[method][value_index] for method in methods],
            color=FAMILY_COLORS[family],
            linewidth=1.4,
            alpha=0.25,
            zorder=2,
        )


def save_figure(fig, output):
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main_panel(title, values, output, masif=False):
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    xs = []
    bounds = []
    if not masif:
        connect_families(ax, values)
    for method, value in values.items():
        if masif:
            x, mean, error = value
        else:
            mean, error = value
            x = PINDER_TIME[method]
        xs.append(x)
        bounds.append((mean, error))
        plot_point(ax, method, x, mean, error)
    fit_axes(ax, xs, bounds)
    style_axes(ax, title, 0.03 if masif else PINDER_THRESHOLD)
    fig.tight_layout(pad=0.3)
    save_figure(fig, output)


def plot_complex_type_panel(title, values, complex_type, output):
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    selected = {method: values[method, complex_type] for method in COLORS}
    connect_families(ax, selected)
    xs = []
    bounds = []
    for method in COLORS:
        result = values[method, complex_type]
        x = PINDER_TIME[method]
        xs.append(x)
        bounds.append(result)
        plot_point(ax, method, x, result[0], result[1])
    fit_axes(ax, xs, bounds)
    style_axes(ax, title, PINDER_THRESHOLD)
    fig.tight_layout(pad=0.3)
    save_figure(fig, output)


def plot_overview(pinder, output):
    panels = [("MaSIF-Ligand", MASIF, True)] + [
        (f"PINDER {split}", pinder[split], False) for split in ("Holo", "Apo", "AF2")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    all_bounds = []

    for ax, (title, values, masif) in zip(axes, panels):
        xs = []
        bounds = []
        if not masif:
            connect_families(ax, values)
        for method, value in values.items():
            if masif:
                x, mean, error = value
            else:
                mean, error = value
                x = PINDER_TIME[method]
            xs.append(x)
            bounds.append((mean, error))
            all_bounds.append((mean, error))
            plot_point(ax, method, x, mean, error)
        fit_axes(ax, xs, bounds)
        style_axes(ax, title, 0.03 if masif else PINDER_THRESHOLD)

    lower = min(mean - error for mean, error in all_bounds)
    upper = max(mean + error for mean, error in all_bounds)
    y_pad = max(0.004, 0.10 * (upper - lower))
    for ax in axes:
        ax.set_ylim(max(0, lower - y_pad), min(1, upper + y_pad))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor=COLORS[method],
            markersize=8,
            label=method,
        )
        for method in COLORS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, output)


def export_independent(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pinder = load_pinder()
    complex_types = load_pinder_complex_types()
    plot_overview(pinder, output_dir.parent / "perf_vs_throughput_masif_pinder")
    plot_main_panel("MaSIF-Ligand", MASIF, output_dir / "masif_ligand", masif=True)
    for split in ("Holo", "Apo", "AF2"):
        stem = split.lower()
        plot_main_panel(f"PINDER {split}", pinder[split], output_dir / f"pinder_{stem}")
        for complex_type in ("Homo", "Hetero"):
            plot_complex_type_panel(
                f"PINDER {split} {complex_type}",
                complex_types[split],
                complex_type,
                output_dir / f"pinder_{stem}_{complex_type.lower()}",
            )
    print(f"Saved ten PDF and PNG panels in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("perf_vs_throughput_independent_panels"),
    )
    args = parser.parse_args()
    export_independent(args.output_dir)


if __name__ == "__main__":
    main()
