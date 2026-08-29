#!/usr/bin/env python3
"""Plot surface-generation speed and mesh size from the benchmark summary."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "alpha_complex": "#E41A1C",
    "edtsurf_0.3": "#9ECAE1",
    "edtsurf_0.4": "#4292C6",
    "edtsurf_0.5": "#08519C",
    "edtsurf_2.0": "#243B53",
    "nanoshaper_0.3": "#A1D99B",
    "nanoshaper_0.4": "#41AB5D",
    "nanoshaper_0.5": "#238B45",
    "nanoshaper_0.6": "#005A32",
    "nanoshaper_2.0": "#145A32",
    "msms": "#6A3D9A",
    "msms_simplified": "#B07CC6",
}
DISPLAY_FAMILY = {"edtsurf": "EDTSurf", "nanoshaper": "NanoShaper"}
FAMILY = {
    "alpha_complex": "alpha_complex",
    "edtsurf": "edtsurf",
    "nanoshaper": "nanoshaper",
    "msms": "msms",
    "msms_simplified": "msms",
}
BAR_WIDTH = 0.5
FAMILY_GAP = 0.8
METRIC_FIGURE_SIZE = (10.0, 6.0)
METRIC_RECT = (0.16, 0.10, 0.98, 0.80)
AXIS_FONT_SIZE = 22
TICK_FONT_SIZE = 18
VALUE_FONT_SIZE = 16


def _x_positions(rows):
    positions = []
    previous_family = None
    position = 0.0
    for row in rows:
        family = FAMILY.get(row["method"], row["method"])
        if previous_family is not None and family != previous_family:
            position += FAMILY_GAP
        positions.append(position)
        position += BAR_WIDTH
        previous_family = family
    return np.asarray(positions)


def _family_boundaries(rows, positions):
    return [
        (positions[i - 1] + positions[i]) / 2.0
        for i in range(1, len(rows))
        if FAMILY.get(rows[i - 1]["method"], rows[i - 1]["method"])
        != FAMILY.get(rows[i]["method"], rows[i]["method"])
    ]


def _load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["name"] = DISPLAY_FAMILY.get(row["method"], row["label"])
        if row["grid_scale"]:
            row["name"] += f"  gs={row['grid_scale']}"
        row["axis_name"] = (
            row["name"]
            .replace("Alpha Complex", "Alpha\nComplex")
            .replace("  gs=", "\ngs=")
        )
        row["time_ms"] = float(row["time_ms"])
        row["time_std"] = float(row["time_std"])
        row["vertices"] = float(row["vertices"])
        row["vertices_std"] = float(row["vertices_std"])
        row["color"] = COLORS[
            row["method"] + (f"_{row['grid_scale']}" if row["grid_scale"] else "")
        ]
    return rows


def _save(fig, path, tight=False):
    save_kwargs = {"dpi": 300}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **save_kwargs)
    fig.savefig(path.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
    print(f"wrote {path}")


def _add_break_marks(ax_top, ax_bottom):
    delta = 0.015
    for xpos in (0, 1):
        ax_top.plot(
            (xpos - delta, xpos + delta),
            (-delta, delta),
            transform=ax_top.transAxes,
            color="#333333",
            clip_on=False,
            linewidth=1.0,
        )
        ax_bottom.plot(
            (xpos - delta, xpos + delta),
            (1 - delta, 1 + delta),
            transform=ax_bottom.transAxes,
            color="#333333",
            clip_on=False,
            linewidth=1.0,
        )


def _style_axis(ax):
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_metric(
    rows,
    value_key,
    error_key,
    ylabel,
    output_path,
    value_format,
    y_break=None,
    y_scale="linear",
    y_max=None,
    show_errorbars=True,
):
    if y_max is not None and y_scale != "linear":
        raise ValueError("A finite y-axis maximum is only supported for linear plots")
    if y_max is not None and y_break is not None:
        raise ValueError("Use either y_max or y_break, not both")
    x = _x_positions(rows)
    boundaries = _family_boundaries(rows, x)
    values = np.array([r[value_key] for r in rows])
    errors = np.array([r[error_key] for r in rows])
    displayed_errors = errors if show_errorbars else np.zeros_like(errors)
    upper_values = values + displayed_errors
    label_values = np.minimum(values, y_max) if y_max is not None else values
    if y_break is None:
        label_indices = list(range(len(values)))
        if y_scale == "log":

            def close_labels(i, j):
                return abs(np.log10(upper_values[i]) - np.log10(upper_values[j])) < 0.08
        else:
            min_label_gap = max(1.0, 0.12 * label_values.max())

            def close_labels(i, j):
                return abs(label_values[i] - label_values[j]) < min_label_gap
    else:
        low_max, high_min = y_break
        label_indices = [i for i, upper in enumerate(upper_values) if upper < high_min]
        min_label_gap = max(1.0, 0.12 * low_max)

        def close_labels(i, j):
            return abs(upper_values[i] - upper_values[j]) < min_label_gap

    label_levels = {}
    for i in label_indices:
        level = 0
        for j in label_indices:
            if j >= i:
                break
            if abs(x[i] - x[j]) <= 1.1 and close_labels(i, j):
                level = max(level, label_levels[j] + 1)
        label_levels[i] = level

    if y_break is None:
        fig, ax = plt.subplots(figsize=METRIC_FIGURE_SIZE)
        axes = (ax,)
    else:
        low_max, high_min = y_break
        high_max = upper_values.max() * 1.1
        fig, (ax_high, ax_low) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=METRIC_FIGURE_SIZE,
            gridspec_kw={"height_ratios": (1, 2), "hspace": 0.06},
        )
        ax_high.set_ylim(high_min, high_max)
        ax_low.set_ylim(0, low_max)
        ax_high.spines["bottom"].set_visible(False)
        ax_low.spines["top"].set_visible(False)
        _add_break_marks(ax_high, ax_low)
        axes = (ax_high, ax_low)

    for axis in axes:
        axis.set_xlim(x[0] - 0.7, x[-1] + 0.7)
        for boundary in boundaries:
            axis.axvline(boundary, color="#D9D9D9", linewidth=1.0, zorder=0)

    if y_break is None:
        ax = axes[0]
        bar_bottom = 1.0 if y_scale == "log" else 0.0
        bar_kwargs = {
            "width": BAR_WIDTH,
            "color": [row["color"] for row in rows],
            "edgecolor": "none",
            "zorder": 3,
        }
        if show_errorbars:
            bar_kwargs.update(
                yerr=errors,
                error_kw={
                    "ecolor": "#333333",
                    "elinewidth": 1.1,
                    "capsize": 3,
                    "capthick": 1.0,
                },
            )
        ax.bar(x, values - bar_bottom, bottom=bar_bottom, **bar_kwargs)
        for i in range(len(rows)):
            label_y = label_values[i]
            label_offset = 7 + 19 * label_levels.get(i, 0)
            ax.annotate(
                value_format.format(value=values[i], error=errors[i]),
                (x[i], label_y),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=VALUE_FONT_SIZE,
                color="#333333",
                zorder=6,
                clip_on=False,
            )
    else:
        low_max, high_min = y_break
        axis_ranges = ((axes[1], 0.0, low_max), (axes[0], high_min, high_max))
        for i, row in enumerate(rows):
            value = values[i]
            error = errors[i]
            lower = value - displayed_errors[i]
            upper = value + displayed_errors[i]
            for ax, axis_low, axis_high in axis_ranges:
                bar_low = max(axis_low, 0.0)
                bar_high = min(axis_high, value)
                if bar_high > bar_low:
                    ax.bar(
                        x[i],
                        bar_high - bar_low,
                        bottom=bar_low,
                        width=BAR_WIDTH,
                        color=row["color"],
                        edgecolor="none",
                        zorder=3,
                    )

                error_low = max(axis_low, lower)
                error_high = min(axis_high, upper)
                if show_errorbars and error_high >= error_low:
                    center = (error_low + error_high) / 2.0
                    ax.errorbar(
                        x[i],
                        center,
                        yerr=[[center - error_low], [error_high - center]],
                        fmt="none",
                        ecolor="#333333",
                        elinewidth=1.1,
                        capsize=3,
                        capthick=1.0,
                        zorder=4,
                    )

            label_ax = axes[0] if upper >= high_min else axes[1]
            label_offset = 7 + 19 * label_levels.get(i, 0)
            label_ax.annotate(
                value_format.format(value=value, error=error),
                (x[i], upper),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=VALUE_FONT_SIZE,
                color="#333333",
                zorder=6,
                clip_on=False,
            )

    ax_bottom = axes[-1]
    ax_bottom.set_xticks([])
    for ax in axes:
        _style_axis(ax)
    if y_break is None and y_scale == "linear":
        upper = y_max if y_max is not None else upper_values.max() * 1.18
        ax.set_ylim(0, upper)
    elif y_break is None:
        ax.set_yscale("log")
        positive_lower = values[values > 0]
        lower = max(1e-6, np.min(positive_lower) * 0.8)
        upper = upper_values.max() * 1.18
        ax.set_ylim(lower, upper)
    axes[-1].set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
    fig.subplots_adjust(
        left=METRIC_RECT[0],
        bottom=METRIC_RECT[1],
        right=METRIC_RECT[2],
        top=METRIC_RECT[3],
    )
    _save(fig, output_path)


def _plot_tradeoff(rows, output_path):
    fig, ax = plt.subplots(figsize=(max(6.0, len(rows) * 0.68), 4.8))
    label_offsets = {
        "alpha_complex": (-58, -15),
        "edtsurf_0.3": (8, -15),
        "edtsurf_0.4": (10, 10),
        "edtsurf_0.5": (10, 8),
        "nanoshaper_0.4": (10, 10),
        "nanoshaper_0.5": (10, -14),
        "nanoshaper_0.3": (10, 10),
        "nanoshaper_0.6": (10, -14),
        "edtsurf_2.0": (10, 10),
        "nanoshaper_2.0": (10, -14),
        "msms": (10, 8),
        "msms_simplified": (10, -14),
    }
    for row in rows:
        ax.errorbar(
            row["vertices"],
            row["time_ms"],
            xerr=row["vertices_std"],
            yerr=row["time_std"],
            fmt="o",
            color=row["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7,
            capsize=3,
            capthick=1.0,
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            row["name"],
            (row["vertices"], row["time_ms"]),
            xytext=label_offsets[
                row["method"] + (f"_{row['grid_scale']}" if row["grid_scale"] else "")
            ],
            textcoords="offset points",
            fontsize=7.0,
            color="#222222",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean surface vertices")
    ax.set_ylabel("Mean surface processing time (ms/protein)")
    ax.grid(which="major", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, tight=True)


def _parse_axis_break(value):
    if value.lower() == "none":
        return None
    low, high = (float(part) for part in value.split(","))
    return low, high


def main():
    parser = argparse.ArgumentParser()
    default_csv = (
        Path(__file__).with_name("cc_sweep_output") / "surface_speed_vertices.csv"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--output-dir", type=Path, default=default_csv.parent)
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    parser.add_argument(
        "--speed-break",
        default="275,900",
        help="Speed y-axis break as low,high; use 'none' for a continuous axis",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=None,
        help="Upper limit for the linear speed axis; values above it are clipped",
    )
    parser.add_argument(
        "--vertices-break",
        default="none",
        help="Vertex y-axis break as low,high; use 'none' for a continuous axis",
    )
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    speed_break = None if args.y_scale == "log" else _parse_axis_break(args.speed_break)
    if args.speed_max is not None:
        speed_break = None
    vertices_break = (
        None if args.y_scale == "log" else _parse_axis_break(args.vertices_break)
    )
    if args.output_stem:
        speed_output = args.output_dir / f"{args.output_stem}_speed.png"
        vertices_output = args.output_dir / f"{args.output_stem}_vertices.png"
        tradeoff_output = args.output_dir / f"{args.output_stem}_tradeoff.png"
    else:
        speed_output = args.output_dir / "surface_generation_speed_updated.png"
        vertices_output = args.output_dir / "surface_mean_vertices_updated.png"
        tradeoff_output = args.output_dir / "surface_generation_tradeoff_updated.png"
    _plot_metric(
        rows,
        "time_ms",
        "time_std",
        "Mean time (ms/protein)",
        speed_output,
        "{value:.1f}",
        y_break=speed_break,
        y_scale=args.y_scale,
        y_max=args.speed_max,
        show_errorbars=False,
    )
    _plot_metric(
        rows,
        "vertices",
        "vertices_std",
        "Mean vertices/protein",
        vertices_output,
        "{value:,.0f}",
        y_break=vertices_break,
        y_scale="log",
        show_errorbars=False,
    )
    _plot_tradeoff(rows, tradeoff_output)


if __name__ == "__main__":
    main()
