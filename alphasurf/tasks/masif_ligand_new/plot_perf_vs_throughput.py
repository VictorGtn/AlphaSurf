from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = [
    ("Alpha Complex", 0.8407, 0.0257, 51.9, 2220, "#E41A1C", r"$\alpha=0$"),
    ("EDTSurf", 0.815878, 0.027797, 52.7, 2243, "#9ECAE1", "gs=0.3"),
    ("EDTSurf", 0.810197, 0.008999, 56.8, 2243, "#4292C6", "gs=0.4"),
    ("EDTSurf", 0.792537, 0.019214, 66.5, 2243, "#08519C", "gs=0.5"),
    ("NanoShaper", 0.821492, 0.020135, 91.3, 2242, "#41AB5D", "gs=0.4"),
    ("NanoShaper", 0.841914, 0.028190, 101.6, 2243, "#238B45", "gs=0.5"),
    ("MSMS", 0.8304, 0.0301, 131.2, 2107, "#6A3D9A", "fr=0.1"),
]

FAMILY_COLORS = {
    "Alpha Complex": "#E41A1C",
    "EDTSurf": "#4292C6",
    "NanoShaper": "#238B45",
    "MSMS": "#6A3D9A",
}


def main():
    fig, ax = plt.subplots(figsize=(5.4, 4.5))

    for family in ("EDTSurf", "NanoShaper"):
        values = [item for item in METHODS if item[0] == family]
        ax.plot(
            [item[3] / item[4] for item in values],
            [item[1] for item in values],
            color=FAMILY_COLORS[family],
            linewidth=1.4,
            alpha=0.25,
            zorder=2,
        )

    xs = []
    bounds = []
    for family, mean, std, wall_time, successes, color, _setting in METHODS:
        seconds_per_protein = wall_time / successes
        xs.append(seconds_per_protein)
        bounds.append((mean, std))
        ax.errorbar(
            seconds_per_protein,
            mean,
            yerr=std,
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.4,
            markersize=11,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )

    x_span = max(xs) - min(xs)
    x_pad = max(0.004, 0.08 * x_span)
    lower = min(mean - error for mean, error in bounds)
    upper = max(mean + error for mean, error in bounds)
    y_pad = max(0.004, 0.10 * (upper - lower))
    ax.set_xlim(max(0, min(xs) - x_pad), max(xs) + x_pad)
    ax.set_ylim(max(0, lower - y_pad), min(1, upper + y_pad))

    xmin, xmax = ax.get_xlim()
    ax.axvspan(max(0.03, xmin), xmax, color="#f5f5f5", zorder=0)
    ax.axvline(0.03, color="grey", linestyle="--", linewidth=1, alpha=0.8, zorder=1)

    ax.set_xlabel("Time per protein (sec/prot)", fontsize=14)
    ax.set_ylabel("Balanced accuracy", fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_title("MaSIF-Ligand", fontsize=17, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    output = Path(__file__).with_name("perf_vs_throughput_masif_ligand")
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {output}.png and {output}.pdf")


if __name__ == "__main__":
    main()
