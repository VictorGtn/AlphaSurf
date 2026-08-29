#!/usr/bin/env python3
"""Create the shared surface-method legend used with the benchmark plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_surface_speed import COLORS


GROUPS = (
    (
        "",
        (
            ("Alpha Complex", "alpha_complex"),
            ("MSMS", "msms"),
            ("MSMS simplified", "msms_simplified"),
        ),
    ),
    (
        "",
        tuple(
            (f"EDTSurf  gs={scale}", f"edtsurf_{scale}")
            for scale in ("0.3", "0.4", "0.5", "2.0")
        ),
    ),
    (
        "",
        tuple(
            (f"NanoShaper  gs={scale}", f"nanoshaper_{scale}")
            for scale in ("0.3", "0.4", "0.5", "0.6", "2.0")
        ),
    ),
)


def _handle(color):
    return Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=8,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )


def main():
    output = (
        Path(__file__).with_name("pinder_surface_benchmark_serial")
        / "surface_generation_legend"
    )
    all_entries = [entry for _, group in GROUPS for entry in group]
    fig = plt.figure(figsize=(8.0, 2.0))
    for index, (_, group_entries) in enumerate(GROUPS):
        handles = [_handle(COLORS[key]) for _, key in group_entries]
        labels = [label for label, _ in group_entries]
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02 + index * 0.325, 0.96),
            frameon=False,
            fontsize=10,
            markerfirst=False,
            handlelength=1.0,
            handletextpad=0.5,
            labelspacing=0.75,
            borderaxespad=0,
        )

    save_kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(output.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(output.with_suffix(".png"), **save_kwargs)
    plt.close(fig)
    print(f"wrote {output}.pdf and {output}.png")

    horizontal = output.with_name("surface_generation_legend_hor")
    handles = [_handle(COLORS[key]) for _, key in all_entries]
    labels = [label for label, _ in all_entries]
    fig = plt.figure(figsize=(30.0, 1.1))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(all_entries),
        frameon=False,
        fontsize=9.5,
        markerfirst=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.savefig(horizontal.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(horizontal.with_suffix(".png"), **save_kwargs)
    plt.close(fig)
    print(f"wrote {horizontal}.pdf and {horizontal}.png")


if __name__ == "__main__":
    main()
