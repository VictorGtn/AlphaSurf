"""Overlay the joint graph+mesh noise sweep and the alpha noise sweep.

Both sweeps are summarised by their own module so the curves stay identical to
`noise_high_common_only_std` and `alpha_noise_common_only_std`. The two noise
levels are different quantities, so they get their own x-axis: sigma below,
max sampled alpha above.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common_systems import FIG_DIR, SETTINGS
from plot_alpha_noise_common import (
    ALPHA_LEVELS,
    Y_MIN_FLOOR,
    axis_limits,
    summarize as summarize_alpha,
)
from plot_noise_high_common import NOISE_LEVELS, summarize as summarize_sigma

SIGMA_COLOR = "#E66101"
ALPHA_COLOR = "#DE77AE"
MAX_Y_INTERVALS = 4


def panel_points(summary, setting, conditions):
    panel = summary[
        (summary["setting"] == setting) & (summary["subset"] == "all")
    ].set_index("condition")
    return panel.loc["none"], panel.loc[conditions]


def plot(sigma_summary, alpha_summary, output_name="noise_combined_common_only_std"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))

    # Both x-axis titles apply to every panel, so only the middle one carries them.
    labelled = axes[len(axes) // 2]

    for ax, setting in zip(axes, SETTINGS):
        baseline, sigma_points = panel_points(
            sigma_summary, setting, [f"sigma_{level}" for level in NOISE_LEVELS]
        )
        _, alpha_points = panel_points(
            alpha_summary, setting, [f"alpha_max_{level}" for level in ALPHA_LEVELS]
        )

        ax.axhline(baseline.auroc_mean, color="#444444", linewidth=1.4)
        ax.axhspan(
            baseline.auroc_mean - baseline.auroc_std,
            baseline.auroc_mean + baseline.auroc_std,
            color="#777777",
            alpha=0.12,
        )
        ax.errorbar(
            NOISE_LEVELS,
            sigma_points["auroc_mean"],
            yerr=sigma_points["auroc_std"],
            color=SIGMA_COLOR,
            marker="o",
            markersize=9,
            linewidth=1.7,
            capsize=4,
        )
        ax.set_xlim(0.04, 1.06)
        ax.set_xticks(NOISE_LEVELS)
        if ax is labelled:
            ax.set_xlabel("Graph+mesh noise sigma")

        top = ax.twiny()
        top.errorbar(
            ALPHA_LEVELS,
            alpha_points["auroc_mean"],
            yerr=alpha_points["auroc_std"],
            color=ALPHA_COLOR,
            marker="o",
            markersize=9,
            linewidth=1.7,
            capsize=4,
        )
        top.set_xlim(0.8, 5.2)
        top.set_xticks(ALPHA_LEVELS)
        if ax is labelled:
            top.set_xlabel("Max sampled alpha")

        floor = Y_MIN_FLOOR.get(setting)
        ymin, ymax, step = axis_limits(baseline, sigma_points, floor)
        amin, amax, astep = axis_limits(baseline, alpha_points, floor)
        ymin, ymax, step = min(ymin, amin), max(ymax, amax), max(step, astep)
        # Overlaying both sweeps widens the range, so coarsen the ticks to match.
        while (ymax - ymin) / step > MAX_Y_INTERVALS:
            step *= 2
        ax.set_ylim(ymin, ymax)
        top.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        # Fixed height, so the middle panel's extra top label cannot shift its title.
        top.set_title(setting.upper(), y=1.30)

    axes[0].set_ylabel("Mean per-system AUROC")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.78, wspace=0.25)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIG_DIR / output_name
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    _, sigma_summary = summarize_sigma()
    _, alpha_summary = summarize_alpha()
    for setting in SETTINGS:
        sigma_base, _ = panel_points(sigma_summary, setting, [])
        alpha_base, _ = panel_points(alpha_summary, setting, [])
        print(
            f"{setting}: baseline sigma-sweep {sigma_base.auroc_mean:.4f} "
            f"({sigma_base.n_systems} systems), alpha-sweep "
            f"{alpha_base.auroc_mean:.4f} ({alpha_base.n_systems} systems)"
        )
    print(plot(sigma_summary, alpha_summary))


if __name__ == "__main__":
    main()
