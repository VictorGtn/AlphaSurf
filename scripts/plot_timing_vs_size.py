"""
Plot timing results vs protein size (n_atoms).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
df = pd.read_csv("timing_results_20260130_134759.csv")

# Filter out rows with errors
df = df[df["error"].isna() | (df["error"] == "")]

print(f"Total proteins: {len(df)}")
print(f"Atom range: {df['n_atoms'].min()} - {df['n_atoms'].max()}")

# Create bins of size 1000 up to the max atom count
max_atoms = df["n_atoms"].max()
upper_bound = int(np.ceil(max_atoms / 1000)) * 1000
bins = list(range(0, upper_bound + 1000, 1000))
labels = [f"{bins[i] // 1000}k-{bins[i + 1] // 1000}k" for i in range(len(bins) - 1)]

df["size_bin"] = pd.cut(df["n_atoms"], bins=bins, labels=labels)

# Calculate binned stats for plotting
bin_centers = []
bin_medians = []
for i in range(len(bins) - 1):
    label = labels[i]
    subset = df[df["size_bin"] == label]
    if len(subset) > 0:
        center = (bins[i] + bins[i + 1]) / 2
        median_speedup = (subset["t_msms"] / subset["t_alpha"]).median()
        bin_centers.append(center)
        bin_medians.append(median_speedup)

# Create figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Combined Scatter: Alpha & MSMS timing vs n_atoms
ax1 = axes[0]
# Alpha
ax1.scatter(
    df["n_atoms"], df["t_alpha"], alpha=0.5, s=10, c="tab:blue", label="Alpha Complex"
)
z_alpha = np.polyfit(df["n_atoms"], df["t_alpha"], 1)
p_alpha = np.poly1d(z_alpha)
x_line = np.linspace(df["n_atoms"].min(), df["n_atoms"].max(), 100)
ax1.plot(
    x_line,
    p_alpha(x_line),
    "b--",
    alpha=0.8,
    label=f"Alpha Fit: {z_alpha[0] * 1000:.3f} ms/atom",
)

# MSMS
ax1.scatter(df["n_atoms"], df["t_msms"], alpha=0.5, s=10, c="tab:orange", label="MSMS")
z_msms = np.polyfit(df["n_atoms"], df["t_msms"], 1)
p_msms = np.poly1d(z_msms)
ax1.plot(
    x_line,
    p_msms(x_line),
    color="darkorange",
    linestyle="--",
    alpha=0.8,
    label=f"MSMS Fit: {z_msms[0] * 1000:.3f} ms/atom",
)

ax1.set_xlabel("Number of Atoms")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Generation Time vs Protein Size")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Speedup ratio (MSMS / Alpha)
ax2 = axes[1]
speedup = df["t_msms"] / df["t_alpha"]
ax2.scatter(df["n_atoms"], speedup, alpha=0.5, s=10, c="tab:green", label="Data Points")
ax2.axhline(
    y=speedup.median(),
    color="r",
    linestyle="--",
    label=f"Median speedup: {speedup.median():.1f}x",
)
# Add binned median line
ax2.plot(bin_centers, bin_medians, "k-o", linewidth=2, label="Binned Median (1k)")
ax2.set_xlabel("Number of Atoms")
ax2.set_ylabel("Speedup (MSMS time / Alpha time)")
ax2.set_title("Alpha Complex Speedup over MSMS")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Histogram of speedups
ax3 = axes[2]
ax3.hist(speedup, bins=50, edgecolor="black", alpha=0.7, color="tab:green")
ax3.axvline(
    x=speedup.median(),
    color="r",
    linestyle="--",
    label=f"Median: {speedup.median():.1f}x",
)
ax3.axvline(
    x=speedup.mean(),
    color="orange",
    linestyle="--",
    label=f"Mean: {speedup.mean():.1f}x",
)
ax3.set_xlabel("Speedup (MSMS / Alpha)")
ax3.set_ylabel("Count")
ax3.set_title("Distribution of Speedup")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("timing_vs_size.png", dpi=150)
print("Saved timing_vs_size.png")

# Print summary stats
print("\n=== Summary Statistics ===")
print("\nAlpha Complex:")
print(f"  Mean time: {df['t_alpha'].mean() * 1000:.1f} ms")
print(f"  Median time: {df['t_alpha'].median() * 1000:.1f} ms")
print(f"  Min time: {df['t_alpha'].min() * 1000:.1f} ms")
print(f"  Max time: {df['t_alpha'].max() * 1000:.1f} ms")

print("\nMSMS:")
print(f"  Mean time: {df['t_msms'].mean() * 1000:.1f} ms")
print(f"  Median time: {df['t_msms'].median() * 1000:.1f} ms")
print(f"  Min time: {df['t_msms'].min() * 1000:.1f} ms")
print(f"  Max time: {df['t_msms'].max() * 1000:.1f} ms")

print("\nSpeedup (MSMS/Alpha):")
print(f"  Mean: {speedup.mean():.1f}x faster")
print(f"  Median: {speedup.median():.1f}x faster")
print(f"  Min: {speedup.min():.1f}x")
print(f"  Max: {speedup.max():.1f}x")

# Binned analysis
print("\n=== Timing by Protein Size Bins ===")

# (Bins already created above)

print(f"{'Bin':<12} {'Count':>6} {'Alpha (ms)':>12} {'MSMS (ms)':>12} {'Speedup':>10}")
print("-" * 56)
for label in labels:
    subset = df[df["size_bin"] == label]
    if len(subset) > 0:
        alpha_med = subset["t_alpha"].median() * 1000
        msms_med = subset["t_msms"].median() * 1000
        speedup = (subset["t_msms"] / subset["t_alpha"]).median()
        print(
            f"{label:<12} {len(subset):>6} {alpha_med:>12.1f} {msms_med:>12.1f} {speedup:>10.1f}x"
        )
