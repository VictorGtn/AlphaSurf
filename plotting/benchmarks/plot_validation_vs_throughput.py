"""Plot validation rate vs 1/Throughput from a benchmark summary.

Usage:
    # Auto-read validation rates from summary CSV:
    python plot_validation_vs_throughput.py masif_bench_5139707.log

    # Override specific rates manually:
    python plot_validation_vs_throughput.py masif_bench_5139707.log \
        --rates alpha_algo2=0.95 nanoshaper_0.3=0.99

    # With goal thresholds (shades region below each):
    python plot_validation_vs_throughput.py masif_bench_5139707.log \
        --goal-vr 0.90 --goal-tp 5.0

Parses the summary table from the log file AND reads Valid Rate from the
summary CSV (masif_benchmark_summary.csv in the same directory).
If --rates are given, they override the CSV values.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"


def parse_summary(log_path):
    """Parse the summary table from a benchmark log file."""
    methods = []
    in_table = False
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r"^-{20,}", line):
                in_table = True
                continue
            if re.match(r"^={20,}", line):
                if in_table:
                    break
                continue
            if in_table and line:
                parts = line.split("|")
                if len(parts) < 9:
                    continue
                method = parts[0].strip()
                param = parts[1].strip()
                throughput = float(parts[8].strip())
                key = f"{method}_{param}" if param not in ("0.0", "0") else method
                methods.append(
                    {
                        "method": method,
                        "param": param,
                        "key": key,
                        "throughput": throughput,
                    }
                )
    return methods


def load_valid_rates_from_csv(csv_path):
    """Load Valid Rate from summary CSV, keyed by method_param."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "Valid Rate" not in df.columns:
        return {}
    rates = {}
    for _, row in df.iterrows():
        method = row["Method"]
        param = row["Param"]
        key = f"{method}_{param}" if param not in (0.0, 0) else method
        rates[key] = row["Valid Rate"]
    return rates


def _label(method, param):
    if method == "nanoshaper":
        return f"NanoShaper gs={param}"
    if method == "alpha_algo2":
        return r"$\alpha$-complex"
    return method.capitalize()


def _color_for(method, param):
    gs_cmap = plt.colormaps.get_cmap("Oranges").resampled(6)
    colors = {
        "alpha_algo2": "tab:blue",
        "edtsurf": "tab:green",
        "msms": "tab:red",
    }
    if method == "nanoshaper":
        gs_order = [0.3, 0.4, 0.5, 0.6, 0.8]
        try:
            idx = gs_order.index(float(param))
        except ValueError:
            idx = 0
        return gs_cmap(idx + 1)
    return colors.get(method, "tab:gray")


def main():
    parser = argparse.ArgumentParser(description="Plot validation rate vs 1/Throughput")
    parser.add_argument("log_file", help="Benchmark log file with summary table")
    parser.add_argument(
        "--rates",
        nargs="+",
        default=None,
        help="Override validation rates as key=val pairs",
    )
    parser.add_argument(
        "--goal-vr",
        type=float,
        default=None,
        help="Target validation rate; shades region below",
    )
    parser.add_argument(
        "--goal-tp",
        type=float,
        default=None,
        help="Target 1/Throughput (proteins/s); shades region below",
    )
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    summary_csv = os.path.join(log_dir, "masif_benchmark_summary.csv")
    rates = load_valid_rates_from_csv(summary_csv)

    if args.rates:
        for r in args.rates:
            k, v = r.split("=")
            rates[k] = float(v)

    if not rates:
        print(
            "No validation rates found. Provide --rates or run benchmark with "
            "cluster validation to generate Valid Rate in the summary CSV."
        )
        sys.exit(1)

    methods = parse_summary(args.log_file)

    xs, ys, labels, colors = [], [], [], []
    for m in methods:
        if m["key"] not in rates:
            print(f"Warning: no validation rate for {m['key']}, skipping")
            continue
        vr = rates[m["key"]]
        inv_tp = 1.0 / m["throughput"]
        xs.append(vr)
        ys.append(inv_tp)
        labels.append(_label(m["method"], m["param"]))
        colors.append(_color_for(m["method"], m["param"]))

    if not xs:
        print("No matching methods found.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 6))
    for x, y, lbl, c in zip(xs, ys, labels, colors):
        ax.scatter(x, y, s=80, color=c, zorder=3)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Validation Rate")
    ax.set_ylabel("1 / Throughput (proteins/s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Validation Rate vs Effective Throughput")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if args.goal_vr is not None:
        ax.axvline(args.goal_vr, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_betweenx(
            [ylim[0], ylim[1]],
            xlim[0],
            args.goal_vr,
            color="red",
            alpha=0.06,
            zorder=0,
        )
        ax.text(
            args.goal_vr,
            ylim[1],
            f"  vr={args.goal_vr}",
            va="top",
            fontsize=8,
            color="red",
        )

    if args.goal_tp is not None:
        ax.axhline(args.goal_tp, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_between(
            [xlim[0], xlim[1]],
            ylim[0],
            args.goal_tp,
            color="red",
            alpha=0.06,
            zorder=0,
        )
        ax.text(
            xlim[1],
            args.goal_tp,
            f" tp={args.goal_tp} ",
            va="bottom",
            ha="right",
            fontsize=8,
            color="red",
        )

    ax.annotate(
        "faster & more reliable",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        color="gray",
        ha="right",
        style="italic",
    )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or FIG_DIR / "validation_vs_throughput.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
