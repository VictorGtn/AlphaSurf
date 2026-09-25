#!/usr/bin/env python3
"""
Per-protein distributions of the spectral agreement metrics.

Reads the spectral_comparison.py outputs and draws, for the surface comparison
and for the discretisation control, the correlation of the HKS and of the heat
kernel H_t(i, j) on pairs in a window that moves out with t, 0-10, 5-15 and 10-20 A apart at t = 5, 10
and 20 (Pearson over all pairs):
  spectral_distributions.pdf - one violin per metric and diffusion time
  spectral_worst_tail.pdf    - mean correlation of the worst N% proteins without tufting,
                               and of the same proteins with tufting
  spectral_by_distance.pdf   - kernel correlation per geodesic distance window
  spectral_worst_tail_by_distance_pearson.pdf
                             - the worst-tail view of the kernel correlation per distance window
  spectral_hks_scatter.pdf   - per-sample HKS on one representative protein
Metrics absent from the csv are skipped.

Each protein contributes one point to the first two figures.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
OUTPUTS = os.path.join(script_dir, "outputs")
DEFAULT_DIRS = (os.path.join(OUTPUTS, "spectral_heat"),)

# Per reference surface: pair, legend label, colour, and the scatter's axis labels.
BASELINES = {
    "msms_dec": (
        ("alpha:msms_dec", "alpha (no tufting) vs MSMS", "#E41A1C", ("alpha, no tufting", "MSMS")),
        ("alpha_tuft:msms_dec", "alpha (tufting) vs MSMS", "#377EB8", ("alpha, tufting", "MSMS")),
        ("msms_full:msms_dec", "control (discretisation)", "#7A7A7A", ("MSMS (full)", "MSMS (simplified)")),
    ),
    "msms_full": (
        ("alpha:msms_full", "alpha (no tufting) vs MSMS", "#E41A1C", ("alpha, no tufting", "MSMS")),
        ("alpha_tuft:msms_full", "alpha (tufting) vs MSMS", "#377EB8", ("alpha, tufting", "MSMS")),
        ("msms_dec:msms_full", "control: MSMS (simplified) vs MSMS", "#7A7A7A", ("MSMS (simplified)", "MSMS")),
    ),
    # Colours of plot_surface_speed.py; untufted alpha takes a darker red to stay apart from tufted alpha.
    "msms_full_grid": (
        ("alpha:msms_full", "alpha (no tufting) vs MSMS", "#99000D", ("alpha, no tufting", "MSMS")),
        ("alpha_tuft:msms_full", "alpha (tufting) vs MSMS", "#E41A1C", ("alpha, tufting", "MSMS")),
        ("nanoshaper@0.3:msms_full", "NanoShaper (gs = 0.3) vs MSMS", "#A1D99B", ("NanoShaper (gs = 0.3)", "MSMS")),
        ("nanoshaper@0.4:msms_full", "NanoShaper (gs = 0.4) vs MSMS", "#41AB5D", ("NanoShaper (gs = 0.4)", "MSMS")),
        ("nanoshaper@0.5:msms_full", "NanoShaper (gs = 0.5) vs MSMS", "#238B45", ("NanoShaper (gs = 0.5)", "MSMS")),
        ("edtsurf@0.3:msms_full", "EDTSurf (gs = 0.3) vs MSMS", "#9ECAE1", ("EDTSurf (gs = 0.3)", "MSMS")),
        ("edtsurf@0.4:msms_full", "EDTSurf (gs = 0.4) vs MSMS", "#4292C6", ("EDTSurf (gs = 0.4)", "MSMS")),
        ("edtsurf@0.5:msms_full", "EDTSurf (gs = 0.5) vs MSMS", "#08519C", ("EDTSurf (gs = 0.5)", "MSMS")),
        ("msms_dec:msms_full", "control: MSMS (simplified) vs MSMS", "#B07CC6", ("MSMS (simplified)", "MSMS")),
    ),
}
PAIRS, COLORS, AXIS_LABELS = {}, {}, {}


def use_baseline(name):
    for d in (PAIRS, COLORS, AXIS_LABELS):
        d.clear()
    for pair, label, color, axes in BASELINES[name]:
        PAIRS[pair], COLORS[pair], AXIS_LABELS[pair] = label, color, axes


use_baseline("msms_dec")
TIME_AXIS = r"diffusion time $t$ (s)"
PLOT_TIMES = ("5", "10", "20")

# Geodesic distances are in diffusion lengths sqrt(4t) of each time.
DISTANCE_UNIT = r"$\sqrt{4t}$"
# Column names in the csv are prefix + time + suffix; a window that moves with t maps each time to (suffix, label).
METRICS = (
    ("hks_corr_t", "", "HKS\n(Pearson)"),
    ("heat_pearson_t", {"5": ("_g0-10", "0–10 Å"), "10": ("_g5-15", "5–15 Å"), "20": ("_g10-20", "10–20 Å")},
     "Heat kernel, all pairs\n(Pearson)"),
)
DISTANCE_WINDOWS = (("_g0-1L", "0–1"), ("_g1-2L", "1–2"), ("_g2-3L", "2–3"), ("_g3-4L", "3–4"))
DISTANCE_METRICS = (("heat_pearson_t", "Pearson"),)


def load(input_dirs, keep=None):
    """Gather every wanted pair, taking each from the first directory that has it."""
    out = {}
    for directory in input_dirs:
        path = os.path.join(directory, "spectral_comparison.csv")
        if not os.path.exists(path):
            continue
        rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok" and r["pair"]]
        if keep is not None:
            rows = [r for r in rows if r["pdb_id"] in keep]
        for pair in PAIRS:
            if pair in out:
                continue
            got = [r for r in rows if r["pair"] == pair]
            if got:
                out[pair] = got
    missing = [p for p in PAIRS if p not in out]
    if len(missing) == len(PAIRS):
        raise SystemExit(f"none of the pairs is in the given csv files: {missing}")
    if missing:
        print(f"pairs missing from the given csv files, left out: {missing}")
        for d in (PAIRS, COLORS, AXIS_LABELS):
            for p in missing:
                d.pop(p)
    return out


def times_in(rows):
    """Diffusion times present in the csv, read off the hks_corr columns."""
    tags = [k[len("hks_corr_t"):] for k in rows[0] if k.startswith("hks_corr_t")]
    return sorted(tags, key=float)


def find_npz(input_dirs, pdb, pair):
    name = f"{pdb}__{pair.replace(':', '_vs_')}.npz"
    for directory in input_dirs:
        path = os.path.join(directory, "per_protein", name)
        if os.path.exists(path):
            return path
    return None


def column(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] not in ("", "nan")])


def metric_column(prefix, t, suffix):
    return prefix + t + (suffix[t][0] if isinstance(suffix, dict) else suffix)


def _violins(ax, data, keys, x):
    step = 0.78 / len(PAIRS)
    for i, pair in enumerate(PAIRS):
        offset = (i - (len(PAIRS) - 1) / 2) * step
        values = [column(data[pair], key) for key in keys]
        parts = ax.violinplot(values, positions=x + offset, widths=step * 0.9,
                              showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[pair])
            body.set_alpha(0.65)
            body.set_edgecolor("none")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)


def _style(ax):
    ax.grid(axis="y", ls=":", color="0.85", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _legend_below(fig, handles, labels, ncol=3, top=0.0):
    """Legend under the panels, ncol entries per row, with the panels laid out above it and top inches free."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8.5, frameon=False)
    rows = -(-len(labels) // ncol)
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, (0.22 * rows + 0.12) / height, 1, 1 - top / height))


def _pair_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[p], alpha=0.65) for p in PAIRS]
    _legend_below(fig, handles, list(PAIRS.values()))


def plot_distributions(data, out_path, times, metrics):
    ncols = min(3, len(metrics))
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows + 0.6), squeeze=False)
    x = np.arange(len(times))
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    for ax, (corr_key, suffix, title) in zip(axes.flat, metrics):
        _violins(ax, data, [metric_column(corr_key, t, suffix) for t in times], x)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n{suffix[t][1]}" if isinstance(suffix, dict) else t for t in times])
        ax.set_xlabel(TIME_AXIS)
        ax.set_ylim(top=1.02)
        ax.set_ylabel("correlation")
        _style(ax)
    _pair_legend(fig)
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_distance(data, out_path, times):
    """Kernel correlation over all pairs, per geodesic distance window, one panel per metric and time."""
    fig, axes = plt.subplots(len(DISTANCE_METRICS), len(times), figsize=(3.6 * len(times), 3.4 * len(DISTANCE_METRICS) + 0.6),
                             sharey="row", squeeze=False)
    x = np.arange(len(DISTANCE_WINDOWS))
    for row, (prefix, name) in enumerate(DISTANCE_METRICS):
        for col, t in enumerate(times):
            ax = axes[row][col]
            _violins(ax, data, [prefix + t + suffix for suffix, _ in DISTANCE_WINDOWS], x)
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in DISTANCE_WINDOWS])
            ax.set_ylim(top=1.02)
            _style(ax)
            if row == 0:
                ax.set_title(f"$t={t}$ s", fontsize=10)
            if row == len(DISTANCE_METRICS) - 1:
                ax.set_xlabel(f"geodesic distance (in {DISTANCE_UNIT})")
        axes[row][0].set_ylabel(f"heat kernel, all pairs\n({name})")
    _pair_legend(fig)
    fig.savefig(out_path)
    plt.close(fig)


TAIL_PERCENTS = (100, 50, 20, 10, 5, 2, 1)


def tail_means(ranking, values):
    """Mean of values over the proteins with the lowest ranking values, for each fraction of TAIL_PERCENTS."""
    order = np.argsort(ranking)
    return [values[order[:max(1, int(np.ceil(pct / 100 * len(order))))]].mean() for pct in TAIL_PERCENTS]


def plot_worst_tail(data, out_path, times, metrics, title=None):
    """Mean correlation of the worst N% proteins without tufting, and of the same proteins with tufting.

    Every other pair is drawn dashed over its own worst N%.
    """
    untufted, tufted, *others = PAIRS
    ncol = min(2, len(times))
    legend_height = 0.22 * -(-len(PAIRS) // ncol) + 0.12
    fig, axes = plt.subplots(len(metrics), len(times),
                             figsize=(max(3.3 * len(times), 4.2), 3.1 * len(metrics) + legend_height),
                             sharex=True, squeeze=False)
    for row, (corr_key, suffix, metric_title) in enumerate(metrics):
        for col, t in enumerate(times):
            ax = axes[row][col]
            key = metric_column(corr_key, t, suffix)
            valid = lambda r: r[key] not in ("", "nan")
            tufted_rows = {r["pdb_id"]: float(r[key]) for r in data[tufted] if valid(r)}
            both = [(float(r[key]), tufted_rows[r["pdb_id"]]) for r in data[untufted]
                    if valid(r) and r["pdb_id"] in tufted_rows]
            before, after = np.array(both).T
            ax.plot(TAIL_PERCENTS, tail_means(before, before), "o-", color=COLORS[untufted], lw=1.8, ms=4.5,
                    label=f"{PAIRS[untufted]}, worst N%")
            ax.plot(TAIL_PERCENTS, tail_means(before, after), "o-", color=COLORS[tufted], lw=1.8, ms=4.5,
                    label=f"{PAIRS[tufted]}, same proteins")
            for other in others:
                own = column(data[other], key)
                ax.plot(TAIL_PERCENTS, tail_means(own, own), "o--", color=COLORS[other], lw=1.2, ms=3.5, alpha=0.7,
                        zorder=1, label=f"{PAIRS[other]}, its own worst N%")
            ax.set_xscale("log")
            ax.set_xlim(1.3 * max(TAIL_PERCENTS), min(TAIL_PERCENTS) / 1.3)
            ax.set_xticks(TAIL_PERCENTS)
            ax.set_xticklabels([f"{p}" for p in TAIL_PERCENTS])
            ax.minorticks_off()
            titles = [f"$t={t}$ s"] if row == 0 else []
            if isinstance(suffix, dict):
                titles.append(f"pairs {suffix[t][1]} apart")
            if titles:
                ax.set_title("\n".join(titles), fontsize=10)
            ax.grid(ls=":", color="0.85", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == len(metrics) - 1:
                ax.set_xlabel(f"mean over the worst N%\n({len(before)} proteins)")
        axes[row][0].set_ylabel(metric_title, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    _legend_below(fig, *axes[0][0].get_legend_handles_labels(), ncol=ncol,
                  top=0.3 * len(title.splitlines()) if title else 0.0)
    fig.savefig(out_path)
    plt.close(fig)


def representative_pdb(data, input_dirs, ref):
    """Protein whose HKS agreement at the reference time is closest to the median."""
    target = {p: np.median(column(data[p], "hks_corr_t" + ref)) for p in PAIRS}
    by_pdb = {p: {r["pdb_id"]: r for r in data[p]} for p in PAIRS}
    best, best_score = None, np.inf
    for pdb in by_pdb[next(iter(PAIRS))]:
        if any(pdb not in by_pdb[p] or find_npz(input_dirs, pdb, p) is None for p in PAIRS):
            continue
        score = sum(abs(float(by_pdb[p][pdb]["hks_corr_t" + ref]) - target[p]) for p in PAIRS)
        if score < best_score:
            best, best_score = pdb, score
    return best


def plot_hks_scatter(pdb, input_dirs, out_path, ref, time_index):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.3 * len(PAIRS), 4.2))
    for ax, pair in zip(axes, PAIRS):
        d = np.load(find_npz(input_dirs, pdb, pair))
        a, b = d["hks_a"][time_index], d["hks_b"][time_index]
        ax.scatter(a, b, s=9, alpha=0.4, linewidths=0, color=COLORS[pair])
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1.0, ls="--", label="$y=x$")
        slope = float(np.polyfit(a, b, 1)[0])
        ax.plot([lo, hi], np.polyval(np.polyfit(a, b, 1), [lo, hi]), color="black", lw=1.2,
                label=f"regression (slope {slope:.2f})")
        r = float(np.corrcoef(a, b)[0, 1])
        ax.set_xlabel(f"HKS, {AXIS_LABELS[pair][0]}")
        ax.set_ylabel(f"HKS, {AXIS_LABELS[pair][1]}")
        ax.set_title(f"{PAIRS[pair]}\n$r={r:.3f}$, $n={len(a)}$", fontsize=9.5)
        ax.legend(fontsize=8, framealpha=0.95)
        ax.grid(ls=":", color="0.85", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{pdb}   —   per-sample HKS at $t={ref}$ s", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", action="append", default=None,
                        help="repeatable; each pair is taken from the first directory that has it")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pdb-id", default=None, help="protein for the scatter; default picks a median one")
    parser.add_argument("--pdb-ids", default=None, help="file of pdb ids, one per line, to restrict every figure to")
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="msms_dec",
                        help="reference surface every pair is compared against")
    parser.add_argument("--title", default=None, help="suptitle of the worst-tail figure, e.g. the reference surface")
    parser.add_argument("--times", default=",".join(PLOT_TIMES), help="comma-separated diffusion times to plot")
    args = parser.parse_args()
    use_baseline(args.baseline)
    input_dirs = args.input_dir or list(DEFAULT_DIRS)
    output_dir = args.output_dir or input_dirs[0]
    os.makedirs(output_dir, exist_ok=True)

    keep = set(open(args.pdb_ids).read().split()) if args.pdb_ids else None
    data = load(input_dirs, keep)
    first = next(iter(PAIRS))
    times = [t for t in times_in(data[first]) if t in args.times.split(",")]
    # The learned diffusion times have median 2.8 s, so the closest grid point
    # is the one the scatters should be read at.
    ref = min(times, key=lambda t: abs(float(t) - 2.8))
    print(f"proteins per pair: { {p: len(v) for p, v in data.items()} }")
    print(f"times: {times}   reference: {ref}")

    columns = data[first][0]
    metrics = [m for m in METRICS if all(metric_column(m[0], t, m[1]) in columns for t in times)]
    print(f"metrics: {[[metric_column(m[0], t, m[1]) for t in times] for m in metrics]}")
    plot_distributions(data, os.path.join(output_dir, "spectral_distributions.pdf"), times, metrics)
    plot_worst_tail(data, os.path.join(output_dir, "spectral_worst_tail.pdf"), times, metrics, args.title)
    if all(f"{prefix}{times[0]}{suffix}" in columns for prefix, _ in DISTANCE_METRICS for suffix, _ in DISTANCE_WINDOWS):
        plot_by_distance(data, os.path.join(output_dir, "spectral_by_distance.pdf"), times)
        for prefix, name in DISTANCE_METRICS:
            window_metrics = [(prefix, suffix, f"Heat kernel {label} {DISTANCE_UNIT}, all pairs\n({name})")
                              for suffix, label in DISTANCE_WINDOWS]
            plot_worst_tail(data, os.path.join(output_dir, f"spectral_worst_tail_by_distance_{name.lower()}.pdf"),
                            times, window_metrics, args.title)

    pdb = args.pdb_id or representative_pdb(data, input_dirs, ref)
    plot_hks_scatter(pdb, input_dirs, os.path.join(output_dir, "spectral_hks_scatter.pdf"),
                     ref, times.index(ref))
    print(f"representative protein: {pdb}")
    print(f"figures written to {output_dir}")


if __name__ == "__main__":
    main()
