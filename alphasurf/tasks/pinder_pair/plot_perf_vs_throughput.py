import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


NOISE_FILE_RE = re.compile(
    r"^(?P<mode>joint_mesh|joint|indep(?:endent)?|none)"
    r"_g(?P<sigma_graph>[0-9.eE+-]+)_m(?P<sigma_mesh>[0-9.eE+-]+)"
    r"(?:_s(?P<seed>[0-9]+))?(?:_(?P<checkpoint>best|last))?"
    r"_(?P<setting>holo|apo|af2)$"
)

# Keep these identical to the raw MSMS values in the throughput comparison.
MSMS_REFERENCE_AUROC = {"holo": 0.9345, "apo": 0.8658, "af2": 0.8779}
NO_NOISE_SEED_2024_RUN = "exp_h100_463477"


def _is_homodimer(system_id):
    parts = str(system_id).split("--")
    return len(parts) == 2 and parts[0].split("_")[-1] == parts[1].split("_")[-1]


def _noise_run_from_path(path):
    """Parse both old and new noising run names.

    Old canonical files have no ``_sSEED`` suffix and are the seed-2024 runs.
    ``indep`` and ``independent`` are normalized to one mode.  Opaque
    ``exp_h100_*`` aliases are intentionally ignored so an old run cannot be
    counted twice when its canonical, nicely named copy is present.
    """
    match = NOISE_FILE_RE.match(path.stem)
    if match is None:
        return None
    run = match.groupdict()
    run["mode"] = "independent" if run["mode"].startswith("indep") else run["mode"]
    run["sigma_graph"] = float(run["sigma_graph"])
    run["sigma_mesh"] = float(run["sigma_mesh"])
    run["seed"] = int(run["seed"] or 2024)
    # Before checkpoint-labelled dumps were introduced, train.py evaluated
    # best and then overwrote it with last under the same filename.
    run["checkpoint"] = run["checkpoint"] or "last"
    run["path"] = path
    return run


def _single_component_ids(atom_csv, system_ids, setting):
    if atom_csv is None or not atom_csv.exists():
        return set(system_ids)
    atoms = pd.read_csv(atom_csv, usecols=["pdb_name", "n_components", "error"])
    atoms = atoms[atoms["error"].isna()]
    component_map = dict(zip(atoms["pdb_name"], atoms["n_components"]))

    def count(sid, side):
        for name in (f"{sid}_{side}_{setting}.pdb", f"{sid}_{side}.pdb"):
            if name in component_map:
                return component_map[name]
        return None

    return {sid for sid in system_ids if count(sid, "L") == 1 and count(sid, "R") == 1}


def _discover_noise_runs(noise_dir, settings, checkpoint):
    runs = []
    for path in sorted(noise_dir.glob("*.csv")):
        run = _noise_run_from_path(path)
        if (
            run is not None
            and run["setting"] in settings
            and run["checkpoint"] == checkpoint
        ):
            runs.append(run)
    return runs


def _method_comparison_ids(method_dir, settings, atom_csv):
    """Return the exact per-setting intersections used by the method plot."""
    benchmark_ids = {}
    for setting in settings:
        # Match aggregate_results.py exactly: its method-comparison table uses
        # every per-system result CSV in the original result directory.
        files = sorted(method_dir.glob(f"*_{setting}.csv"))
        if len(files) < 2:
            raise SystemExit(f"Need at least two {setting} method CSVs in {method_dir}")
        ids = set(pd.read_csv(files[0], usecols=["system_id"])["system_id"])
        for path in files[1:]:
            ids.intersection_update(
                pd.read_csv(path, usecols=["system_id"])["system_id"]
            )
        ids = _single_component_ids(atom_csv, ids, setting)
        benchmark_ids[setting] = ids
        print(f"{setting}: {len(ids)} method-comparison benchmark systems")
    return benchmark_ids


def _summarize_noise_runs(runs, atom_csv, benchmark_ids=None):
    """Return one row per run/subset on a common system intersection."""
    rows = []
    for setting in sorted({run["setting"] for run in runs}):
        setting_runs = [run for run in runs if run["setting"] == setting]
        loaded = []
        for run in setting_runs:
            df = pd.read_csv(run["path"])
            if not {"system_id", "auroc"}.issubset(df.columns):
                print(
                    f"Skipping {run['path'].name}: requires system_id and auroc columns"
                )
                continue
            loaded.append((run, df))
        if not loaded:
            continue

        if benchmark_ids is None:
            common_ids = set(loaded[0][1]["system_id"])
            for _, df in loaded[1:]:
                common_ids.intersection_update(df["system_id"])
            common_ids = _single_component_ids(atom_csv, common_ids, setting)
        else:
            common_ids = set(benchmark_ids[setting])
            for run, df in loaded:
                missing = common_ids.difference(df["system_id"])
                if missing:
                    raise SystemExit(
                        f"{run['path'].name} lacks {len(missing)} systems from "
                        f"the method-comparison {setting} benchmark"
                    )
        print(
            f"{setting}: {len(loaded)} runs, {len(common_ids)} common single-component systems"
        )

        for run, df in loaded:
            df = df[df["system_id"].isin(common_ids)].copy()
            is_homo = df["system_id"].map(_is_homodimer)
            for subset, mask in (
                ("all", np.ones(len(df), dtype=bool)),
                ("homo", is_homo.to_numpy()),
                ("hetero", (~is_homo).to_numpy()),
            ):
                values = df.loc[mask, "auroc"].astype(float)
                rows.append(
                    {
                        "mode": run["mode"],
                        "sigma_graph": run["sigma_graph"],
                        "sigma_mesh": run["sigma_mesh"],
                        "seed": run["seed"],
                        "setting": setting,
                        "checkpoint": run["checkpoint"],
                        "subset": subset,
                        "auroc": values.mean(),
                        "n_systems": len(values),
                        "file": run["path"].name,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_seed_summary(per_run):
    keys = ["mode", "sigma_graph", "sigma_mesh", "setting", "subset"]
    grouped = per_run.groupby(keys, as_index=False, sort=True)
    summary = grouped.agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        n_seeds=("seed", "nunique"),
        n_systems=("n_systems", "min"),
    )
    seed_lists = grouped["seed"].agg(lambda x: ",".join(map(str, sorted(set(x)))))
    summary = summary.merge(seed_lists.rename(columns={"seed": "seeds"}), on=keys)
    summary["auroc_std"] = summary["auroc_std"].fillna(0.0)
    summary["auroc_sem"] = summary["auroc_std"] / np.sqrt(summary["n_seeds"])
    return summary


def _load_log_summaries(paths, settings, checkpoint):
    """Convert compact best/last checkpoint log summaries to long format."""
    frames = [pd.read_csv(path) for path in paths]
    wide = pd.concat(frames, ignore_index=True)
    wide = wide[
        wide["setting"].isin(settings) & (wide["checkpoint"] == checkpoint)
    ].drop_duplicates(
        ["mode", "sigma_graph", "sigma_mesh", "seed", "setting", "checkpoint"],
        keep="last",
    )

    rows = []
    for _, run in wide.iterrows():
        for subset, column, n_systems in (
            ("all", "auroc_mean", int(run.n_homo + run.n_hetero)),
            ("homo", "auroc_homo", int(run.n_homo)),
            ("hetero", "auroc_hetero", int(run.n_hetero)),
        ):
            rows.append(
                {
                    "mode": run["mode"],
                    "sigma_graph": run.sigma_graph,
                    "sigma_mesh": run.sigma_mesh,
                    "seed": int(run.seed),
                    "setting": run.setting,
                    "checkpoint": checkpoint,
                    "subset": subset,
                    "auroc": run[column],
                    "n_systems": n_systems,
                    "file": run.get("log_file", ""),
                }
            )
    return pd.DataFrame(rows)


def _noise_curves(data, mode, min_seeds):
    """Return readable curves for one noising family."""
    usable = data[(data["mode"] == mode) & (data["n_seeds"] >= min_seeds)]
    curves = []

    if mode == "joint":
        return [("joint", usable.sort_values("sigma_graph"), "#d62728", "o", "-")]
    if mode == "joint_mesh":
        return [
            (
                r"joint + mesh ($\sigma_m=\sigma_g$)",
                usable.sort_values("sigma_graph"),
                "#9467bd",
                "*",
                "-.",
            )
        ]

    indep = usable
    low_diag = indep[
        np.isclose(indep["sigma_graph"], indep["sigma_mesh"])
        & (indep["sigma_graph"] <= 0.005)
    ].sort_values("sigma_graph")
    if not low_diag.empty:
        curves.append((r"$\sigma_m=\sigma_g\leq0.005$", low_diag, "#17becf", "D", ":"))

    indep_colors = {0.05: "#1f77b4"}
    for sigma_mesh, color in indep_colors.items():
        curve = indep[np.isclose(indep["sigma_mesh"], sigma_mesh)].sort_values(
            "sigma_graph"
        )
        if not curve.empty:
            curves.append((rf"$\sigma_m={sigma_mesh:g}$", curve, color, "s", "--"))
    return curves


def _plot_noise_summary(summary, output_dir, error_kind, min_seeds, show_msms=True):
    error_col = f"auroc_{error_kind}"
    modes = ("independent", "joint_mesh")
    settings = ("holo", "apo", "af2")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    legend_items = {}
    y_lows = []
    y_highs = []

    for ax, setting in zip(axes, settings):
        data = summary[(summary["setting"] == setting) & (summary["subset"] == "all")]
        if data.empty:
            ax.set_visible(False)
            continue

        if show_msms:
            msms_line = ax.axhline(
                MSMS_REFERENCE_AUROC[setting],
                color="#6D4C41",
                linestyle="--",
                lw=1.4,
                alpha=0.9,
                zorder=1,
            )
            legend_items.setdefault("MSMS", msms_line)
            y_lows.append(MSMS_REFERENCE_AUROC[setting])
            y_highs.append(MSMS_REFERENCE_AUROC[setting])

        baseline = data[data["mode"] == "none"]
        if not baseline.empty:
            base = baseline.iloc[0]
            baseline_line = ax.axhline(
                base.auroc_mean,
                color="#444444",
                lw=1.2,
                zorder=1,
            )
            ax.axhspan(
                base.auroc_mean - base[error_col],
                base.auroc_mean + base[error_col],
                color="#777777",
                alpha=0.09,
                zorder=0,
            )
            legend_items.setdefault("No noise", baseline_line)
            y_lows.append(base.auroc_mean - base[error_col])
            y_highs.append(base.auroc_mean + base[error_col])

        for mode in modes:
            for label, curve, color, marker, linestyle in _noise_curves(
                data, mode, min_seeds
            ):
                if curve.empty:
                    continue
                container = ax.errorbar(
                    curve["sigma_graph"],
                    curve["auroc_mean"],
                    yerr=curve[error_col],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    marker=None,
                    capsize=3,
                    alpha=0.82,
                    zorder=2,
                )
                legend_items.setdefault(label, container)
                y_lows.extend((curve["auroc_mean"] - curve[error_col]).tolist())
                y_highs.extend((curve["auroc_mean"] + curve[error_col]).tolist())
                for _, point in curve.iterrows():
                    ax.plot(
                        point.sigma_graph,
                        point.auroc_mean,
                        marker=marker,
                        markersize=10 if marker == "*" else 7,
                        linestyle="none",
                        markerfacecolor=color if point.n_seeds >= 3 else "white",
                        markeredgecolor=color,
                        markeredgewidth=1.5,
                        zorder=4,
                    )

        ax.set_xscale("log")
        ax.set_xlabel(r"Graph noise $\sigma_g$ ($\AA$)")
        n_systems = int(data["n_systems"].min())
        ax.set_title(
            f"{setting.upper()} (N={n_systems:,})", fontsize=12, fontweight="bold"
        )
        ax.grid(ls="--", alpha=0.18)
        ax.margins(x=0.10)

    finite_lows = np.asarray(y_lows, dtype=float)
    finite_highs = np.asarray(y_highs, dtype=float)
    finite_lows = finite_lows[np.isfinite(finite_lows)]
    finite_highs = finite_highs[np.isfinite(finite_highs)]
    if finite_lows.size and finite_highs.size:
        y_min = finite_lows.min()
        y_max = finite_highs.max()
        margin = max(0.002, 0.04 * (y_max - y_min))
        axes[0].set_ylim(0.81, y_max + margin)

    axes[0].set_ylabel("All systems\nMean per-system AUROC", fontsize=10)
    legend_items["Open marker: 2 seeds"] = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#555555",
        markeredgewidth=1.5,
    )
    fig.legend(
        legend_items.values(),
        legend_items.keys(),
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"Noise augmentation — mean across seeds ± {error_kind.upper()}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.25,
        top=0.84,
        wspace=0.20,
    )
    out = output_dir / f"noise_seed_aggregate_all_{error_kind}"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png / .pdf")


def _plot_alpha_summary(summary, output_dir, error_kind, min_seeds):
    """Plot random-alpha augmentation separately from coordinate-noise sigmas."""
    error_col = f"auroc_{error_kind}"
    data = summary[
        summary["mode"].isin(["none", "alpha"])
        & (summary["subset"] == "all")
        & (summary["n_seeds"] >= min_seeds)
    ]
    if "alpha" not in set(data["mode"]):
        return

    settings = ("holo", "apo", "af2")
    styles = {
        "none": ("No noise", "#555555", "o"),
        "alpha": (r"$\alpha\sim U(0,5)$", "#e377c2", "D"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=False)
    for ax, setting in zip(axes, settings):
        panel = data[data["setting"] == setting].set_index("mode")
        present = [mode for mode in ("none", "alpha") if mode in panel.index]
        for x, mode in enumerate(present):
            point = panel.loc[mode]
            label, color, marker = styles[mode]
            ax.errorbar(
                x,
                point.auroc_mean,
                yerr=point[error_col],
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markersize=8,
                capsize=5,
                linewidth=1.8,
                zorder=3,
            )
        ax.set_xticks(range(len(present)), [styles[m][0] for m in present])
        ax.set_xlim(-0.55, max(0.55, len(present) - 0.45))
        ax.set_title(setting.upper(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.22)
        ax.set_ylabel("Mean per-system AUROC")

    fig.suptitle(
        rf"Random $\alpha\sim U(0,5)$ augmentation — mean across seeds ± {error_kind.upper()}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.84, wspace=0.28)
    out = output_dir / f"noise_alpha_seed_aggregate_all_{error_kind}"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png / .pdf")


def plot_noising(args):
    settings = (
        ("holo", "apo", "af2") if args.noise_setting == "all" else (args.noise_setting,)
    )
    default_output = Path(args.noise_dir) if args.noise_dir else Path(".")
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.noise_summary_csv:
        paths = [Path(path) for path in args.noise_summary_csv]
        per_run = _load_log_summaries(paths, settings, args.noise_checkpoint)
    else:
        noise_dir = Path(args.noise_dir)
        atom_csv = Path(args.atom_csv) if args.atom_csv else None
        runs = _discover_noise_runs(noise_dir, settings, args.noise_checkpoint)
        if not runs:
            raise SystemExit(
                f"No canonical {args.noise_checkpoint!r}-checkpoint noising CSVs "
                f"found in {noise_dir}"
            )
        if args.noise_baseline_dir:
            baseline_dir = Path(args.noise_baseline_dir)
            for run in runs:
                if run["mode"] != "none":
                    continue
                path = baseline_dir / (
                    f"alpha_complex_s{run['seed']}_provided_{run['setting']}.csv"
                )
                if not path.exists():
                    raise SystemExit(f"Missing repaired no-noise baseline: {path}")
                run["path"] = path
        benchmark_ids = None
        if args.method_benchmark_dir:
            method_dir = Path(args.method_benchmark_dir)
            benchmark_ids = _method_comparison_ids(method_dir, settings, atom_csv)
            # The original method/noise comparison defines the seed-2024
            # no-noise baseline as the 30-epoch no-tufting run 463477.
            for run in runs:
                if run["mode"] == "none" and run["seed"] == 2024:
                    reference = (
                        method_dir / f"{NO_NOISE_SEED_2024_RUN}_{run['setting']}.csv"
                    )
                    if not reference.exists():
                        raise SystemExit(
                            f"Missing seed-2024 no-noise baseline: {reference}"
                        )
                    run["path"] = reference
        per_run = _summarize_noise_runs(runs, atom_csv, benchmark_ids)
    summary = _aggregate_seed_summary(per_run)
    per_run.to_csv(output_dir / "noise_per_run_summary.csv", index=False)
    summary.to_csv(output_dir / "noise_seed_aggregate.csv", index=False)

    print("\nSeed coverage by combination:")
    coverage = summary[summary["subset"] == "all"][
        ["mode", "sigma_graph", "sigma_mesh", "setting", "n_seeds", "seeds"]
    ]
    print(coverage.to_string(index=False))
    _plot_noise_summary(
        summary, output_dir, args.error, args.min_seeds, not args.hide_msms
    )
    _plot_alpha_summary(summary, output_dir, args.error, args.min_seeds)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--adjusted",
    action="store_true",
    help="Use imputation-adjusted AUROCs (missing systems forced to 0.5)",
)
parser.add_argument(
    "--show-validity", action="store_true", help="Show validity on right axis"
)
parser.add_argument(
    "--inverse-x",
    action="store_true",
    help="Plot 1/throughput (sec/prot) instead of throughput (prot/sec)",
)
parser.add_argument(
    "--noise-dir",
    default=None,
    help="Plot only noising runs discovered from per-system CSV filenames",
)
parser.add_argument(
    "--noise-summary-csv",
    nargs="+",
    default=None,
    help="Seed-level checkpoint summaries parsed from training logs",
)
parser.add_argument(
    "--noise-baseline-dir",
    default=None,
    help="Directory containing repaired Alpha Complex per-system baseline CSVs",
)
parser.add_argument(
    "--noise-setting", choices=["all", "holo", "apo", "af2"], default="all"
)
parser.add_argument(
    "--noise-checkpoint",
    choices=["best", "last"],
    default="last",
    help="Checkpoint family to aggregate; old unlabeled dumps are last",
)
parser.add_argument(
    "--error",
    choices=["std", "sem"],
    default="std",
    help="Error bar across seed-level AUROC means",
)
parser.add_argument(
    "--min-seeds",
    type=int,
    default=2,
    help="Only plot combinations having at least this many seeds",
)
parser.add_argument("--hide-msms", action="store_true")
parser.add_argument(
    "--atom-csv",
    default=str(
        Path(__file__).resolve().parents[2] / "data" / "pdb_atom_components.csv"
    ),
    help="Optional PDB component CSV; pass an empty string to disable filtering",
)
parser.add_argument(
    "--method-benchmark-dir",
    default=None,
    help=(
        "Restrict noising results to the exact per-system intersection used by "
        "the original performance-vs-throughput method comparison"
    ),
)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()

if args.noise_dir or args.noise_summary_csv:
    plot_noising(args)
    raise SystemExit(0)


# Per-method visual identity, shared across both modes.
# EDTsurf uses a blue gradient by grid scale and NanoShaper a green gradient.
METHOD_STYLE = {
    "Alpha Complex": {
        "color": "#E53935",
        "marker": "o",
        "family": "Alpha Complex",
        "gs": "",
    },
    "EDTsurf gs=0.3": {
        "color": "#9ecae1",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.3",
    },
    "EDTsurf gs=0.4": {
        "color": "#3182bd",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.4",
    },
    "EDTsurf gs=0.5": {
        "color": "#08306b",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.5",
    },
    "NanoShaper gs=0.4": {
        "color": "#41AB5D",
        "marker": "o",
        "family": "NanoShaper",
        "gs": "0.4",
    },
    "NanoShaper gs=0.5": {
        "color": "#238B45",
        "marker": "o",
        "family": "NanoShaper",
        "gs": "0.5",
    },
    "MSMS": {"color": "#6D4C41", "marker": "o", "family": "MSMS", "gs": ""},
}

THROUGHPUT = {
    "Alpha Complex": 38.01,
    "EDTsurf gs=0.3": 52.97,
    "EDTsurf gs=0.4": 39.26,
    "EDTsurf gs=0.5": 25.86,
    "NanoShaper gs=0.4": 19.87,
    "NanoShaper gs=0.5": 12.68,
    "MSMS": 1 / 0.057,
}

VALIDITY = {
    "Alpha Complex": 94.50,
    "EDTsurf gs=0.3": 89.50,
    "EDTsurf gs=0.4": 94.00,
    "EDTsurf gs=0.5": 92.00,
    "NanoShaper gs=0.4": 90.30,
    "NanoShaper gs=0.5": 92.47,
    "MSMS": 99.00,
}

# Raw (un-adjusted) AUROC means.
RAW = {
    "Alpha Complex": {
        "holo": 0.9209,
        "apo": 0.8593,
        "af2": 0.8718,
        "holo_homo": 0.9260,
        "holo_hetero": 0.8923,
        "apo_homo": 0.8607,
        "apo_hetero": 0.8477,
        "af2_homo": 0.8757,
        "af2_hetero": 0.8408,
    },
    "EDTsurf gs=0.3": {
        "holo": 0.9022,
        "apo": 0.8448,
        "af2": 0.8363,
        "holo_homo": 0.9094,
        "holo_hetero": 0.8620,
        "apo_homo": 0.8492,
        "apo_hetero": 0.8084,
        "af2_homo": 0.8442,
        "af2_hetero": 0.7719,
    },
    "EDTsurf gs=0.4": {
        "holo": 0.9184,
        "apo": 0.8577,
        "af2": 0.8547,
        "holo_homo": 0.9250,
        "holo_hetero": 0.8817,
        "apo_homo": 0.8617,
        "apo_hetero": 0.8242,
        "af2_homo": 0.8629,
        "af2_hetero": 0.7890,
    },
    "EDTsurf gs=0.5": {
        "holo": 0.9341,
        "apo": 0.8640,
        "af2": 0.8750,
        "holo_homo": 0.9394,
        "holo_hetero": 0.9049,
        "apo_homo": 0.8665,
        "apo_hetero": 0.8434,
        "af2_homo": 0.8810,
        "af2_hetero": 0.8270,
    },
    "NanoShaper gs=0.4": {
        "holo": 0.9464,
        "apo": 0.7507,
        "af2": 0.7325,
        "holo_homo": 0.9506,
        "holo_hetero": 0.9234,
        "apo_homo": 0.7441,
        "apo_hetero": 0.8055,
        "af2_homo": 0.7279,
        "af2_hetero": 0.7696,
    },
    "NanoShaper gs=0.5": {
        "holo": 0.9468,
        "apo": 0.7796,
        "af2": 0.7666,
        "holo_homo": 0.9503,
        "holo_hetero": 0.9275,
        "apo_homo": 0.7708,
        "apo_hetero": 0.8522,
        "af2_homo": 0.7649,
        "af2_hetero": 0.7803,
    },
    "MSMS": {
        "holo": 0.9345,
        "apo": 0.8658,
        "af2": 0.8779,
        "holo_homo": 0.9395,
        "holo_hetero": 0.9070,
        "apo_homo": 0.8676,
        "apo_hetero": 0.8511,
        "af2_homo": 0.8838,
        "af2_hetero": 0.8306,
    },
}

# Imputation-adjusted AUROC means (missing systems forced to 0.5 before averaging).
ADJUSTED = {
    "Alpha Complex": {
        "holo": 0.8738,
        "apo": 0.8267,
        "af2": 0.8132,
        "holo_homo": 0.8774,
        "holo_hetero": 0.8533,
        "apo_homo": 0.8270,
        "apo_hetero": 0.8239,
        "af2_homo": 0.8306,
        "af2_hetero": 0.7108,
    },
    "EDTsurf gs=0.3": {
        "holo": 0.8572,
        "apo": 0.8114,
        "af2": 0.7815,
        "holo_homo": 0.8627,
        "holo_hetero": 0.8266,
        "apo_homo": 0.8139,
        "apo_hetero": 0.7902,
        "af2_homo": 0.8004,
        "af2_hetero": 0.6701,
    },
    "EDTsurf gs=0.4": {
        "holo": 0.8716,
        "apo": 0.8228,
        "af2": 0.7947,
        "holo_homo": 0.8765,
        "holo_hetero": 0.8442,
        "apo_homo": 0.8251,
        "apo_hetero": 0.8027,
        "af2_homo": 0.8145,
        "af2_hetero": 0.6783,
    },
    "EDTsurf gs=0.5": {
        "holo": 0.8855,
        "apo": 0.8263,
        "af2": 0.8075,
        "holo_homo": 0.8892,
        "holo_hetero": 0.8649,
        "apo_homo": 0.8269,
        "apo_hetero": 0.8212,
        "af2_homo": 0.8255,
        "af2_hetero": 0.7014,
    },
    "NanoShaper gs=0.4": {
        "holo": 0.8965,
        "apo": 0.7173,
        "af2": 0.6884,
        "holo_homo": 0.8992,
        "holo_hetero": 0.8817,
        "apo_homo": 0.7113,
        "apo_hetero": 0.7706,
        "af2_homo": 0.6929,
        "af2_hetero": 0.6624,
    },
    "NanoShaper gs=0.5": {
        "holo": 0.8968,
        "apo": 0.7479,
        "af2": 0.7197,
        "holo_homo": 0.8989,
        "holo_hetero": 0.8852,
        "apo_homo": 0.7394,
        "apo_hetero": 0.8231,
        "af2_homo": 0.7282,
        "af2_hetero": 0.6700,
    },
    "MSMS": {
        "holo": 0.8813,
        "apo": 0.8282,
        "af2": 0.8177,
        "holo_homo": 0.8854,
        "holo_hetero": 0.8581,
        "apo_homo": 0.8280,
        "apo_hetero": 0.8302,
        "af2_homo": 0.8371,
        "af2_hetero": 0.7040,
    },
}

# Adjusted MSMS comes from disk_msms_0.1_66185 (best run after msms_ext 1%-test
# imputation). Adjusted validity for MSMS reflects the same filtering.
ADJUSTED_VALIDITY = dict(VALIDITY)
ADJUSTED_VALIDITY["MSMS"] = 49.52


def build_data(adjusted):
    perf = ADJUSTED if adjusted else RAW
    validity = ADJUSTED_VALIDITY if adjusted else VALIDITY
    data = {}
    for name, style in METHOD_STYLE.items():
        d = dict(style)
        d["throughput"] = THROUGHPUT[name]
        d["validity"] = validity[name]
        d.update(perf[name])
        data[name] = d
    return data


data = build_data(args.adjusted)

splits = ["holo", "apo", "af2"]
split_labels = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}

kinds = [
    ("all", "all", "All systems"),
    ("homo", "homo", "Homodimers"),
    ("hetero", "hetero", "Heterodimers"),
]

x_vline = 0.0342 if args.inverse_x else 1 / 0.0342
x_label = "Time per protein (sec/prot)" if args.inverse_x else "Throughput (prot/sec)"
y_label = "AUROC (missing systems imputed at 0.5)" if args.adjusted else "AUROC"


def plot_x(throughput):
    return 1.0 / throughput if args.inverse_x else throughput


def value_for(d, split, key):
    return d[split] if key == "all" else d[f"{split}_{key}"]


families = [
    ("Alpha Complex", "#E53935"),
    ("EDTsurf", "#3182bd"),
    ("NanoShaper", "#238B45"),
    ("MSMS", "#6D4C41"),
]
family_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=c,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1,
        label=name,
    )
    for name, c in families
]

out_prefix = "perf_vs_throughput_adjusted" if args.adjusted else "perf_vs_throughput"
title_tag = "adjusted AUROC" if args.adjusted else "raw AUROC"

for key, suffix, title_qualifier in kinds:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, split in zip(axes, splits):
        if args.show_validity:
            ax2 = ax.twinx()
            for name, d in data.items():
                ax2.scatter(
                    plot_x(d["throughput"]),
                    d["validity"],
                    s=110,
                    facecolors="none",
                    edgecolors=d["color"],
                    marker=d["marker"],
                    linewidths=1.5,
                    alpha=0.45,
                    zorder=3,
                )

        for prefix, line_color in [("EDTsurf", "#3182bd"), ("NanoShaper", "#238B45")]:
            variants = sorted(
                (plot_x(d["throughput"]), value_for(d, split, key))
                for n, d in data.items()
                if n.startswith(prefix)
            )
            if len(variants) > 1:
                xs, ys = zip(*variants)
                ax.plot(xs, ys, lw=1.4, color=line_color, alpha=0.25, zorder=2)

        for name, d in data.items():
            x_val = plot_x(d["throughput"])
            y_val = value_for(d, split, key)
            if d.get("filled", True):
                ax.scatter(
                    x_val,
                    y_val,
                    s=200,
                    c=d["color"],
                    marker=d["marker"],
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=5,
                )
            else:
                ax.scatter(
                    x_val,
                    y_val,
                    s=200,
                    facecolors="none",
                    edgecolors=d["color"],
                    marker=d["marker"],
                    linewidths=2.0,
                    zorder=5,
                )
            if d["gs"]:
                ax.annotate(
                    d["gs"],
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(6, -6),
                    fontsize=8,
                    color=d["color"],
                    zorder=6,
                )

        x_max_data = max(plot_x(d["throughput"]) for d in data.values())
        x_min_data = min(plot_x(d["throughput"]) for d in data.values())
        if args.inverse_x:
            ax.axvspan(x_vline, x_max_data * 1.2, color="grey", alpha=0.10, zorder=0)
        else:
            ax.axvspan(
                max(0, x_min_data * 0.8), x_vline, color="grey", alpha=0.10, zorder=0
            )
        ax.axvline(x=x_vline, color="grey", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10, color="#333")
        if args.show_validity:
            ax2.set_ylabel("Validity (%)", fontsize=10, color="#999")
            ax2.set_ylim(86, 101)
            ax2.tick_params(axis="y", colors="#999")
            ax2.spines["right"].set_color("#CCCCCC")
        ax.tick_params(axis="y", colors="#333")
        ax.set_title(f"{split_labels[split]} split", fontsize=13, fontweight="bold")
        ax.spines[["top"]].set_visible(False)
        ax.grid(ls="--", alpha=0.15)
        ax.margins(x=0.10, y=0.08)

    handles = list(family_handles)
    if args.show_validity:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="gray",
                markersize=8,
                markeredgewidth=1.5,
                label="Validity (%) — right axis",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        framealpha=0.9,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Performance vs Throughput ({title_tag}, {title_qualifier})",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.88, wspace=0.35)
    out = f"{out_prefix}_{suffix}"
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png / .pdf")
