from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter


HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE / "per_system_results_perf_rebuilt_explicit"
NOISE_DIR = HERE / "per_system_results_noise_repaired"
NOISE_01310_DIR = HERE / "per_system_results_noise_01310"
TUFT_DIR = HERE / "per_system_results_alpha_tuft_repaired"
ATOM_CSV = HERE.parents[2] / "data" / "pdb_atom_components.csv"
SETTINGS = ("holo", "apo", "af2")
SEEDS = (2024, 2025, 2026)
Y_TICK_STEP = 0.005
Y_SPAN = 0.020
NOISE_LEVELS = (0.1, 0.3, 0.5, 0.75, 1.0)


def single_component_ids(system_ids, setting):
    atoms = pd.read_csv(ATOM_CSV, usecols=["pdb_name", "n_components", "error"])
    atoms = atoms[atoms["error"].isna()]
    component_map = dict(zip(atoms["pdb_name"], atoms["n_components"]))

    def count(system_id, side):
        for name in (
            f"{system_id}_{side}_{setting}.pdb",
            f"{system_id}_{side}.pdb",
        ):
            if name in component_map:
                return component_map[name]
        return None

    return {
        system_id
        for system_id in system_ids
        if count(system_id, "L") == 1 and count(system_id, "R") == 1
    }


def common_method_ids(setting):
    paths = sorted(METHOD_DIR.glob(f"*_{setting}.csv"))
    common = set(pd.read_csv(paths[0], usecols=["system_id"])["system_id"])
    for path in paths[1:]:
        common.intersection_update(
            pd.read_csv(path, usecols=["system_id"])["system_id"]
        )
    return single_component_ids(common, setting)


def result_path(condition, seed, setting):
    if condition == "none":
        return TUFT_DIR / f"alpha_tuft_s{seed}_provided_{setting}.csv"
    sigma = condition.removeprefix("sigma_")
    noise_dir = NOISE_01310_DIR if sigma in {"0.1", "0.3", "1.0"} else NOISE_DIR
    return noise_dir / f"joint_mesh_g{sigma}_m{sigma}_s{seed}_provided_{setting}.csv"


def summarize():
    rows = []
    for setting in SETTINGS:
        common = common_method_ids(setting)
        print(f"{setting}: {len(common)} common systems")
        conditions = ("none", *(f"sigma_{level}" for level in NOISE_LEVELS))
        for condition in conditions:
            for seed in SEEDS:
                path = result_path(condition, seed, setting)
                values = pd.read_csv(
                    path, usecols=["system_id", "auroc", "is_homodimer"]
                )
                values = values[values["system_id"].isin(common)]
                if len(values) != len(common):
                    missing = len(common) - len(values)
                    raise ValueError(f"{path.name} lacks {missing} common systems")
                for subset, selected in (
                    ("all", values),
                    ("homo", values[values["is_homodimer"]]),
                    ("hetero", values[~values["is_homodimer"]]),
                ):
                    rows.append(
                        {
                            "condition": condition,
                            "setting": setting,
                            "subset": subset,
                            "seed": seed,
                            "auroc": selected["auroc"].mean(),
                            "n_systems": len(selected),
                        }
                    )

    per_run = pd.DataFrame(rows)
    summary = per_run.groupby(["condition", "setting", "subset"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        n_seeds=("seed", "nunique"),
        n_systems=("n_systems", "min"),
    )
    return per_run, summary


def plot(summary, include_legend=True, output_name="noise_high_common_only_std"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))
    noised_color = "#E66101"

    for ax, setting in zip(axes, SETTINGS):
        panel = summary[
            (summary["setting"] == setting) & (summary["subset"] == "all")
        ].set_index("condition")
        baseline = panel.loc["none"]
        ax.axhline(
            baseline.auroc_mean,
            color="#444444",
            linewidth=1.4,
            label="Baseline",
        )
        ax.axhspan(
            baseline.auroc_mean - baseline.auroc_std,
            baseline.auroc_mean + baseline.auroc_std,
            color="#777777",
            alpha=0.12,
        )

        point_names = [f"sigma_{level}" for level in NOISE_LEVELS]
        points = panel.loc[point_names]
        ax.errorbar(
            NOISE_LEVELS,
            points["auroc_mean"],
            yerr=points["auroc_std"],
            color=noised_color,
            marker="o",
            markersize=11,
            linewidth=1.7,
            capsize=4,
            label="Noised",
        )
        y_values = np.concatenate(
            [
                [
                    baseline.auroc_mean - baseline.auroc_std,
                    baseline.auroc_mean + baseline.auroc_std,
                ],
                points["auroc_mean"].to_numpy() - points["auroc_std"].to_numpy(),
                points["auroc_mean"].to_numpy() + points["auroc_std"].to_numpy(),
            ]
        )
        ymin = np.floor(y_values.min() / Y_TICK_STEP) * Y_TICK_STEP
        ymax = np.ceil(y_values.max() / Y_TICK_STEP) * Y_TICK_STEP
        missing_steps = max(
            0,
            round((Y_SPAN - (ymax - ymin)) / Y_TICK_STEP),
        )
        ymin -= (missing_steps // 2) * Y_TICK_STEP
        ymax += (missing_steps - missing_steps // 2) * Y_TICK_STEP
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0.04, 1.06)
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xlabel("Noise level")
        ax.set_title(setting.upper())
        ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
        ax.grid(axis="y", linestyle="--", alpha=0.2)

    axes[0].set_ylabel("Mean per-system AUROC")
    if include_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
        )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.31, top=0.88, wspace=0.25)

    output = HERE / output_name
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_complex_type_panels(summary):
    output_dir = HERE / "noise_high_common_homo_hetero_panels"
    output_dir.mkdir(exist_ok=True)
    noised_color = "#E66101"

    for setting in SETTINGS:
        for subset in ("homo", "hetero"):
            panel = summary[
                (summary["setting"] == setting) & (summary["subset"] == subset)
            ].set_index("condition")
            baseline = panel.loc["none"]
            points = panel.loc[[f"sigma_{level}" for level in NOISE_LEVELS]]

            fig, ax = plt.subplots(figsize=(5.4, 4.5))
            ax.axhline(
                baseline.auroc_mean,
                color="#444444",
                linewidth=1.4,
                label="Baseline",
            )
            ax.axhspan(
                baseline.auroc_mean - baseline.auroc_std,
                baseline.auroc_mean + baseline.auroc_std,
                color="#777777",
                alpha=0.12,
            )
            ax.errorbar(
                NOISE_LEVELS,
                points["auroc_mean"],
                yerr=points["auroc_std"],
                color=noised_color,
                marker="o",
                markersize=11,
                linewidth=1.7,
                capsize=4,
                label="Noised",
            )

            y_values = np.concatenate(
                [
                    [
                        baseline.auroc_mean - baseline.auroc_std,
                        baseline.auroc_mean + baseline.auroc_std,
                    ],
                    points["auroc_mean"].to_numpy() - points["auroc_std"].to_numpy(),
                    points["auroc_mean"].to_numpy() + points["auroc_std"].to_numpy(),
                ]
            )
            ymin = np.floor(y_values.min() / Y_TICK_STEP) * Y_TICK_STEP
            ymax = np.ceil(y_values.max() / Y_TICK_STEP) * Y_TICK_STEP
            missing_steps = max(0, round((Y_SPAN - (ymax - ymin)) / Y_TICK_STEP))
            ymin -= (missing_steps // 2) * Y_TICK_STEP
            ymax += (missing_steps - missing_steps // 2) * Y_TICK_STEP
            ymin -= Y_TICK_STEP
            ymax += Y_TICK_STEP

            ax.set_xlim(0.04, 1.06)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(NOISE_LEVELS)
            ax.set_xlabel("Noise level", fontsize=14)
            ax.set_ylabel("Mean per-system AUROC", fontsize=14)
            ax.set_title(
                f"PINDER {setting.upper()} {subset.capitalize()}",
                fontsize=17,
                fontweight="bold",
            )
            ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="lower left", frameon=False, fontsize=11)
            fig.tight_layout(pad=0.3)

            output = output_dir / f"noise_high_{setting}_{subset}"
            fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
            fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

    return output_dir


if __name__ == "__main__":
    per_run, summary = summarize()
    per_run.to_csv(HERE / "noise_high_common_only_per_run.csv", index=False)
    summary.to_csv(HERE / "noise_high_common_only_summary.csv", index=False)
    output = plot(summary)
    output_dir = plot_complex_type_panels(summary)
    print(summary.to_string(index=False))
    print(f"Saved {output}.png and {output}.pdf")
    print(f"Saved homo/hetero panels in {output_dir}")
