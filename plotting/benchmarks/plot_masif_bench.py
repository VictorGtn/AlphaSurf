"""Plot masif benchmark results from the raw CSV.

Produces 3 plots:
  1. Surface generation time vs vertex count (all methods overlaid)
  2. Spectral operator time vs vertex count (all methods overlaid)
  3. Bar chart of mean times per method
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
BENCH_DIR = SCRIPTS_DIR / "outputs" / "masif_benchmark"
RAW_CSV = BENCH_DIR / "masif_benchmark_raw.csv"
SUMMARY_CSV = BENCH_DIR / "masif_benchmark_summary.csv"


def _label(method, param):
    if method == "nanoshaper":
        return f"NanoShaper gs={param}"
    if method == "alpha_algo2":
        return r"$\alpha$-complex (algo2)"
    return method


def _color_for(method, param):
    cmap_gs = plt.colormaps.get_cmap("Oranges").resampled(6)
    colors = {
        "alpha_algo2": "tab:blue",
        "edtsurf": "tab:green",
        "msms": "tab:red",
    }
    if method == "nanoshaper":
        gs_order = [0.3, 0.4, 0.5, 0.6, 0.8]
        idx = gs_order.index(float(param)) if float(param) in gs_order else 0
        return cmap_gs(idx + 1)
    return colors.get(method, "tab:gray")


def _scatter_metric(ax, df, y_col, y_label, log_y=True):
    groups = df.groupby(["method", "param"])
    for (method, param), sub in sorted(groups, key=lambda t: t[0]):
        c = _color_for(method, param)
        lbl = _label(method, param)
        ax.scatter(
            sub["n_verts"], sub[y_col], s=5, alpha=0.3, color=c, label=lbl, zorder=2
        )

        if len(sub) > 10:
            xs = sub["n_verts"].values
            n_bins = min(20, max(5, len(sub) // 20))
            bins = pd.cut(xs, bins=n_bins)
            grouped = (
                sub.groupby(bins, observed=True)
                .agg(
                    n_v=("n_verts", "mean"),
                    y_mean=(y_col, "mean"),
                    y_std=(y_col, "std"),
                )
                .dropna()
            )
            if len(grouped) > 1:
                ax.plot(grouped["n_v"], grouped["y_mean"], color=c, lw=2, zorder=4)
                ax.fill_between(
                    grouped["n_v"],
                    grouped["y_mean"] - grouped["y_std"],
                    grouped["y_mean"] + grouped["y_std"],
                    color=c,
                    alpha=0.1,
                    zorder=1,
                )

    ax.set_xlabel("Vertices")
    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, markerscale=2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    df = df[~df["crash"] & (df["t_spectral"] > 0)]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    _scatter_metric(ax1, df, "t_gen", "Surface Generation Time (s)")
    ax1.set_title("Surface Generation Time vs Mesh Size")
    fig1.tight_layout()
    out1 = FIG_DIR / "masif_benchmark_gen.png"
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"Saved {out1}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    _scatter_metric(ax2, df, "t_spectral", "Spectral Operator Time (s)", log_y=False)

    # Fit power law t = a * n^p and overlay reference lines
    n_all = df["n_verts"].values
    t_all = df["t_spectral"].values
    log_n = np.log(n_all)
    log_t = np.log(t_all)
    p, log_a = np.polyfit(log_n, log_t, 1)
    a = np.exp(log_a)
    n_ref = np.linspace(n_all.min(), n_all.max(), 200)
    ax2.plot(
        n_ref, a * n_ref**p, "k--", lw=1.5, alpha=0.6, label=rf"Fit: $O(n^{{{p:.2f}}})$"
    )
    ax2.plot(
        n_ref,
        n_ref / n_ref[0] * t_all[n_all.argmin()],
        "k:",
        lw=1,
        alpha=0.4,
        label=r"Reference $O(n)$",
    )

    ax2.set_title("Spectral Operator Time vs Mesh Size")
    ax2.legend(fontsize=7, markerscale=2)
    fig2.tight_layout()
    out2 = FIG_DIR / "masif_benchmark_spectral.png"
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"Saved {out2}  (fitted exponent p={p:.2f})")

    summary = pd.read_csv(SUMMARY_CSV)
    if len(summary) == 0:
        return

    labels = []
    for _, row in summary.iterrows():
        m, p = row["Method"], row["Param"]
        if m == "nanoshaper":
            labels.append(f"NanoShaper\ngs={p}")
        elif m == "alpha_algo2":
            labels.append(r"$\alpha$-complex" + f"\n(a={p})")
        else:
            labels.append(m.capitalize())

    fig3, ax3 = plt.subplots(figsize=(max(14, len(summary) * 1.5), 6))
    x = np.arange(len(labels))
    w = 0.25

    ax3.bar(x - w, summary["Avg Gen (s)"], w, label="Generation", color="tab:blue")
    ax3.bar(x, summary["Avg Spectral (s)"], w, label="Spectral Ops", color="tab:orange")
    ax3.bar(
        x + w,
        summary["Throughput (s/protein)"],
        w,
        label="Throughput (wall/N)",
        color="tab:green",
        alpha=0.7,
    )

    for i, row in summary.iterrows():
        ax3.text(
            i,
            0.02,
            f"{row['Avg Verts']:.0f}v",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    ax3.set_ylabel("Time (seconds)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_title("MaSIF Benchmark: Surface Gen + Spectral Ops")
    fig3.tight_layout()
    out3 = FIG_DIR / "masif_benchmark_bar.png"
    fig3.savefig(out3, dpi=300)
    plt.close(fig3)
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()
