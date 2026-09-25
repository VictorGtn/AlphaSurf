import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common_systems import FIG_DIR, SETTINGS, TASK_DIR, common_ids, read_results

REPAIRED_DIR = TASK_DIR / "per_system_results_perf_rebuilt_explicit"
METHODS = {
    "Alpha Complex": {
        "seconds": 1 / 38.01,
        "color": "#E53935",
        "marker": "o",
        "grid_scale": "",
        "runs": {
            2024: "alpha_complex_s2024_provided_{setting}.csv",
            2025: "alpha_complex_s2025_provided_{setting}.csv",
            2026: "alpha_complex_s2026_provided_{setting}.csv",
        },
    },
    "EDTsurf gs=0.3": {
        "seconds": 1 / 52.97,
        "color": "#9ecae1",
        "marker": "o",
        "grid_scale": "0.3",
        "runs": {
            2024: "edtsurf_gs0.3_s2024_provided_{setting}.csv",
            2025: "edtsurf_gs0.3_s2025_provided_{setting}.csv",
            2026: "edtsurf_gs0.3_s2026_provided_{setting}.csv",
        },
    },
    "EDTsurf gs=0.4": {
        "seconds": 1 / 39.26,
        "color": "#3182bd",
        "marker": "o",
        "grid_scale": "0.4",
        "runs": {
            2024: "edtsurf_gs0.4_s2024_provided_{setting}.csv",
            2025: "edtsurf_gs0.4_s2025_provided_{setting}.csv",
            2026: "edtsurf_gs0.4_s2026_provided_{setting}.csv",
        },
    },
    "EDTsurf gs=0.5": {
        "seconds": 1 / 25.86,
        "color": "#08306b",
        "marker": "o",
        "grid_scale": "0.5",
        "runs": {
            2024: "edtsurf_gs0.5_s2024_provided_{setting}.csv",
            2025: "edtsurf_gs0.5_s2025_provided_{setting}.csv",
            2026: "edtsurf_gs0.5_s2026_provided_{setting}.csv",
        },
    },
    "NanoShaper gs=0.4": {
        "seconds": 1 / 19.87,
        "color": "#41AB5D",
        "marker": "o",
        "grid_scale": "0.4",
        "runs": {
            2024: "nanoshaper_gs0.4_s2024_provided_{setting}.csv",
            2025: "nanoshaper_gs0.4_s2025_provided_{setting}.csv",
            2026: "nanoshaper_gs0.4_s2026_provided_{setting}.csv",
        },
    },
    "NanoShaper gs=0.5": {
        "seconds": 1 / 12.68,
        "color": "#238B45",
        "marker": "o",
        "grid_scale": "0.5",
        "runs": {
            2024: "nanoshaper_gs0.5_s2024_provided_{setting}.csv",
            2025: "nanoshaper_gs0.5_s2025_provided_{setting}.csv",
            2026: "nanoshaper_gs0.5_s2026_provided_{setting}.csv",
        },
    },
    "MSMS": {
        "seconds": 0.057,
        "color": "#6D4C41",
        "marker": "o",
        "grid_scale": "",
        "runs": {
            2024: "msms_0.1_dist2.0_s2024_provided_{setting}.csv",
            2025: "msms_0.1_dist2.0_s2025_provided_{setting}.csv",
            2026: "msms_0.1_dist2.0_s2026_provided_{setting}.csv",
        },
    },
}


def summarize():
    per_run = []
    for setting in SETTINGS:
        print(f"{setting}: {len(common_ids(setting))} common systems")
        for method, config in METHODS.items():
            for seed, template in config["runs"].items():
                path = REPAIRED_DIR / template.format(setting=setting)
                values = read_results(path, setting, ("system_id", "auroc"))["auroc"]
                per_run.append(
                    {
                        "method": method,
                        "setting": setting,
                        "seed": seed,
                        "auroc": values.mean(),
                        "n_systems": len(values),
                        "file": path.name,
                    }
                )

    per_run = pd.DataFrame(per_run)
    aggregate = per_run.groupby(["method", "setting"], as_index=False, sort=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        n_seeds=("seed", "nunique"),
        n_systems=("n_systems", "min"),
    )
    return per_run, aggregate


def summarize_by_complex_type():
    per_run = []
    for setting in SETTINGS:
        loaded = {}
        for method, config in METHODS.items():
            for seed, template in config["runs"].items():
                path = REPAIRED_DIR / template.format(setting=setting)
                loaded[method, seed] = (path, read_results(path, setting))

        reference = None
        for path, results in loaded.values():
            labels = results.set_index("system_id")["is_homodimer"].sort_index()
            if reference is None:
                reference = labels
            elif not labels.equals(reference):
                raise ValueError(f"Inconsistent homo/hetero labels in {path}")

        counts = reference.value_counts()
        print(
            f"{setting}: {len(reference)} common systems "
            f"({int(counts.get(True, 0))} homo, {int(counts.get(False, 0))} hetero)"
        )
        for method, config in METHODS.items():
            for seed in config["runs"]:
                path, results = loaded[method, seed]
                for is_homodimer, complex_type in ((True, "Homo"), (False, "Hetero")):
                    values = results.loc[
                        results["is_homodimer"] == is_homodimer, "auroc"
                    ]
                    per_run.append(
                        {
                            "method": method,
                            "setting": setting,
                            "complex_type": complex_type,
                            "seed": seed,
                            "auroc": values.mean(),
                            "n_systems": len(values),
                            "file": path.name,
                        }
                    )

    per_run = pd.DataFrame(per_run)
    aggregate = per_run.groupby(
        ["method", "setting", "complex_type"], as_index=False, sort=False
    ).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        n_seeds=("seed", "nunique"),
        n_systems=("n_systems", "min"),
    )
    return per_run, aggregate


def plot(aggregate, output, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    split_names = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}

    for ax, setting in zip(axes, SETTINGS):
        panel = aggregate[aggregate["setting"] == setting].set_index("method")
        for family, color in (("EDTsurf", "#3182bd"), ("NanoShaper", "#238B45")):
            family_methods = [name for name in METHODS if name.startswith(family)]
            ax.plot(
                [METHODS[name]["seconds"] for name in family_methods],
                [panel.loc[name, "auroc_mean"] for name in family_methods],
                color=color,
                linewidth=1.4,
                alpha=0.25,
                zorder=2,
            )

        for method, config in METHODS.items():
            result = panel.loc[method]
            ax.errorbar(
                config["seconds"],
                result.auroc_mean,
                yerr=result.auroc_std,
                fmt=config["marker"],
                markersize=11,
                color=config["color"],
                markeredgecolor="white",
                markeredgewidth=1.2,
                capsize=4,
                linewidth=1.5,
                zorder=4,
            )
            if config["grid_scale"]:
                ax.annotate(
                    config["grid_scale"],
                    (config["seconds"], result.auroc_mean),
                    textcoords="offset points",
                    xytext=(7, -6),
                    fontsize=8,
                    color=config["color"],
                )

        ax.axvspan(0.0342, 0.1, color="grey", alpha=0.10, zorder=0)
        ax.axvline(0.0342, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0.011, 0.102)
        ax.set_xlabel("Time per protein (sec/prot)")
        ax.set_ylabel("Mean per-system AUROC")
        ax.set_title(f"{split_names[setting]} split", fontsize=13, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.margins(y=0.08)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for label, color in (
            ("Alpha Complex", "#E53935"),
            ("EDTsurf", "#3182bd"),
            ("NanoShaper", "#238B45"),
            ("MSMS", "#6D4C41"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    if title is None:
        title = (
            "Performance vs Throughput — three-seed mean ± STD\n"
            "Exact common method-comparison systems"
        )
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.16,
        top=0.83 if title else 0.94,
        wspace=0.25,
    )
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_by_complex_type(aggregate, output):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    split_names = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}
    x_offset = {"Homo": -0.0006, "Hetero": 0.0006}

    for ax, setting in zip(axes, SETTINGS):
        panel = aggregate[aggregate["setting"] == setting].set_index(
            ["method", "complex_type"]
        )
        for method, config in METHODS.items():
            homo = panel.loc[method, "Homo"]
            hetero = panel.loc[method, "Hetero"]
            x = config["seconds"]
            ax.plot(
                [x + x_offset["Homo"], x + x_offset["Hetero"]],
                [homo.auroc_mean, hetero.auroc_mean],
                color=config["color"],
                linewidth=1,
                alpha=0.45,
                zorder=2,
            )
            for complex_type, result in (("Homo", homo), ("Hetero", hetero)):
                marker = "o"
                ax.errorbar(
                    x + x_offset[complex_type],
                    result.auroc_mean,
                    yerr=result.auroc_std,
                    fmt=marker,
                    markersize=10,
                    color=config["color"],
                    markerfacecolor=(
                        config["color"] if complex_type == "Homo" else "white"
                    ),
                    markeredgecolor=config["color"],
                    markeredgewidth=1.7,
                    capsize=3,
                    linewidth=1.3,
                    zorder=4,
                )
            if config["grid_scale"]:
                ax.annotate(
                    config["grid_scale"],
                    (x, max(homo.auroc_mean, hetero.auroc_mean)),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=config["color"],
                )

        ax.axvspan(0.0342, 0.1, color="grey", alpha=0.10, zorder=0)
        ax.axvline(0.0342, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0.011, 0.102)
        ax.set_xlabel("Time per protein (sec/prot)")
        ax.set_ylabel("Mean per-system AUROC")
        ax.set_title(f"{split_names[setting]} split", fontsize=13, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.margins(y=0.10)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=9,
            label=label,
        )
        for label, color in (
            ("Alpha Complex", "#E53935"),
            ("EDTsurf", "#3182bd"),
            ("NanoShaper", "#238B45"),
            ("MSMS", "#6D4C41"),
        )
    ]
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            markerfacecolor=facecolor,
            markeredgecolor="#555555",
            markeredgewidth=1.7,
            linewidth=0,
            markersize=9,
            label=label,
        )
        for label, facecolor in (("Homodimer", "#555555"), ("Heterodimer", "white"))
    ]
    fig.legend(
        handles=method_handles + type_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Performance vs Throughput — Homo vs Hetero, three-seed mean ± STD\n"
        "Exact common method-comparison systems",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.83, wspace=0.25)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homo-hetero", action="store_true")
    parser.add_argument("--separate-complex-types", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=FIG_DIR / "perf_vs_throughput_seed_mean_std"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.homo_hetero or args.separate_complex_types:
        per_run, aggregate = summarize_by_complex_type()
    else:
        per_run, aggregate = summarize()
    per_run.to_csv(
        args.output.with_name(f"{args.output.name}_per_run.csv"), index=False
    )
    aggregate.to_csv(
        args.output.with_name(f"{args.output.name}_summary.csv"), index=False
    )
    if args.separate_complex_types:
        for complex_type in ("Homo", "Hetero"):
            subset = aggregate[aggregate["complex_type"] == complex_type]
            output = args.output.with_name(f"{args.output.name}_{complex_type.lower()}")
            plot(
                subset,
                output,
                title=(
                    f"Performance vs Throughput — {complex_type}dimers, "
                    "three-seed mean ± STD\n"
                    "Exact common method-comparison systems"
                ),
            )
    elif args.homo_hetero:
        plot_by_complex_type(aggregate, args.output)
    else:
        plot(aggregate, args.output)
    if args.separate_complex_types:
        print(f"Saved {args.output}_homo and {args.output}_hetero as PNG and PDF")
    else:
        print(f"Saved {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()
