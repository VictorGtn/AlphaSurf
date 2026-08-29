import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = {
    "Alpha Complex": {
        "validity": 94.50,
        "throughput": 38.01,
        "color": "#E53935",
        "marker": "s",
        "family": "Alpha Complex",
        "gs": "",
    },
    "EDTsurf gs=0.3": {
        "validity": 89.50,
        "throughput": 52.97,
        "color": "#1E88E5",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.3",
    },
    "EDTsurf gs=0.4": {
        "validity": 94.00,
        "throughput": 39.26,
        "color": "#1E88E5",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.4",
    },
    "EDTsurf gs=0.5": {
        "validity": 92.00,
        "throughput": 25.86,
        "color": "#1E88E5",
        "marker": "o",
        "family": "EDTsurf",
        "gs": "0.5",
    },
    "NanoShaper gs=0.4": {
        "validity": 90.30,
        "throughput": 19.87,
        "color": "#8E24AA",
        "marker": "D",
        "family": "NanoShaper",
        "gs": "0.4",
    },
    "NanoShaper gs=0.5": {
        "validity": 92.47,
        "throughput": 12.68,
        "color": "#8E24AA",
        "marker": "D",
        "family": "NanoShaper",
        "gs": "0.5",
    },
    "NanoShaper gs=0.6": {
        "validity": 93.15,
        "throughput": 8.13,
        "color": "#8E24AA",
        "marker": "D",
        "family": "NanoShaper",
        "gs": "0.6",
    },
    "MSMS": {
        "validity": 99.00,
        "throughput": 15.23,
        "color": "#6D4C41",
        "marker": "X",
        "family": "MSMS",
        "gs": "",
    },
}

threshold_time = 1.0 / 27.0

fig, ax = plt.subplots(figsize=(9, 7))

ax.axhspan(threshold_time, 0.14, color="black", alpha=0.06, zorder=0)
ax.axhline(
    y=threshold_time, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1
)
ax.text(
    100,
    threshold_time,
    f"  Throughput threshold ({threshold_time:.4f} s/prot)",
    fontsize=8,
    color="gray",
    va="bottom",
    ha="right",
)

for prefix, line_color in [("EDTsurf", "#1E88E5"), ("NanoShaper", "#8E24AA")]:
    variants = sorted(
        [
            (d["validity"], 1.0 / d["throughput"])
            for n, d in data.items()
            if n.startswith(prefix)
        ],
    )
    if len(variants) > 1:
        xs, ys = zip(*variants)
        ax.plot(xs, ys, lw=1.4, color=line_color, alpha=0.25, zorder=2)

for name, d in data.items():
    t = 1.0 / d["throughput"]
    ax.scatter(
        d["validity"],
        t,
        s=200,
        c=d["color"],
        marker=d["marker"],
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )
    if d["gs"]:
        ax.annotate(
            d["gs"],
            (d["validity"], t),
            textcoords="offset points",
            xytext=(6, -6),
            fontsize=8,
            color=d["color"],
            zorder=6,
        )

ax.set_xlim(89, 100)
ax.set_ylim(0, 0.14)
ax.set_xlabel("Validity (%)", fontsize=11)
ax.set_ylabel("Execution Time (sec/prot)", fontsize=11)
ax.set_title("Validity vs Execution Time", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.2)

families = [
    ("Alpha Complex", "#E53935", "s"),
    ("EDTsurf", "#1E88E5", "o"),
    ("NanoShaper", "#8E24AA", "D"),
    ("MSMS", "#6D4C41", "X"),
]
handles = [
    Line2D(
        [0],
        [0],
        marker=m,
        color="w",
        markerfacecolor=c,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1,
        label=name,
    )
    for name, c, m in families
]
handles.append(
    Line2D(
        [0],
        [0],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Throughput threshold",
    )
)
ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("validity_vs_time.png", dpi=200, bbox_inches="tight")
plt.savefig("validity_vs_time.pdf", bbox_inches="tight")
print("Saved validity_vs_time.png / .pdf")
