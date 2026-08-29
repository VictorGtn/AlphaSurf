#!/usr/bin/env python3
"""Export standalone legends for the performance figures."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_perf_vs_throughput_combined import COLORS

plt.rcParams.update({"font.size": 14})


def horizontal_legend(output):
    fig, ax = plt.subplots(figsize=(15, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def marker(x, method):
        ax.plot(
            x,
            0.5,
            marker="o",
            color=COLORS[method],
            markerfacecolor=COLORS[method],
            markersize=11,
            linestyle="none",
        )

    ax.text(0.015, 0.5, "Alpha Complex", va="center")
    marker(0.145, "Alpha Complex")

    ax.text(0.185, 0.5, "EDTsurf:", va="center")
    for x, scale in ((0.285, "0.3"), (0.37, "0.4"), (0.455, "0.5")):
        ax.text(x, 0.5, scale, ha="right", va="center")
        marker(x + 0.012, f"EDTsurf gs={scale}")

    ax.text(0.505, 0.5, "NanoShaper:", va="center")
    for x, scale in ((0.67, "0.4"), (0.755, "0.5")):
        ax.text(x, 0.5, scale, ha="right", va="center")
        marker(x + 0.014, f"NanoShaper gs={scale}")

    ax.text(0.85, 0.5, "MSMS", va="center")
    marker(0.91, "MSMS")

    fig.tight_layout(pad=0.05)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        output.with_suffix(".png"), dpi=250, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def detailed_legend(output):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def marker(x, y, method):
        ax.plot(
            x,
            y,
            marker="o",
            color=COLORS[method],
            markerfacecolor=COLORS[method],
            markersize=11,
            linestyle="none",
        )

    ax.text(0.05, 3.5, "Alpha Complex", va="center")
    marker(0.35, 3.5, "Alpha Complex")

    ax.text(0.05, 2.5, "EDTsurf:", va="center")
    for x, scale in ((0.43, "0.3"), (0.63, "0.4"), (0.83, "0.5")):
        ax.text(x, 2.5, scale, ha="right", va="center")
        marker(x + 0.035, 2.5, f"EDTsurf gs={scale}")

    ax.text(0.05, 1.5, "NanoShaper:", va="center")
    for x, scale in ((0.58, "0.4"), (0.81, "0.5")):
        ax.text(x, 1.5, scale, ha="right", va="center")
        marker(x + 0.04, 1.5, f"NanoShaper gs={scale}")

    ax.text(0.05, 0.5, "MSMS", va="center")
    marker(0.2, 0.5, "MSMS")

    fig.tight_layout(pad=0.1)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        output.with_suffix(".png"), dpi=250, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def main():
    output_dir = Path(__file__).with_name("perf_vs_throughput_independent_panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    horizontal_legend(output_dir / "legend_horizontal")
    detailed_legend(output_dir / "legend_detailed_vertical")
    print(f"Saved standalone legends in {output_dir}")


if __name__ == "__main__":
    main()
