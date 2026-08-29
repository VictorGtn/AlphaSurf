import matplotlib.pyplot as plt

# Best run per surface method (selected by best_holo_auroc_mean)
# MSMS has no apo/af2 data; EDTsurf gs=0.3/0.4 and NanoShaper gs=0.4 also have no apo/af2
data = {
    "Alpha Complex": {
        "sec_per_prot": 0.02631,
        "holo": 0.9205,
        "apo": 0.8635,
        "af2": 0.8574,
        "color": "#E53935",
        "marker": "o",
    },
    "EDTsurf gs=0.3": {
        "sec_per_prot": 0.01888,
        "holo": 0.9211,
        "color": "#9ecae1",
        "marker": "o",
    },
    "EDTsurf gs=0.4": {
        "sec_per_prot": 0.02547,
        "holo": 0.9303,
        "color": "#3182bd",
        "marker": "o",
    },
    "EDTsurf gs=0.5": {
        "sec_per_prot": 0.03867,
        "holo": 0.9323,
        "apo": 0.8631,
        "af2": 0.8556,
        "color": "#08306b",
        "marker": "o",
    },
    "NanoShaper gs=0.4": {
        "sec_per_prot": 0.05033,
        "holo": 0.9510,
        "color": "#41AB5D",
        "marker": "o",
    },
    "NanoShaper gs=0.5": {
        "sec_per_prot": 0.07886,
        "holo": 0.9512,
        "apo": 0.7618,
        "af2": 0.7328,
        "color": "#238B45",
        "marker": "o",
    },
    "MSMS": {
        "sec_per_prot": 0.057,
        "holo": 0.9536,
        "color": "#6D4C41",
        "marker": "o",
    },
}

splits = ["holo", "apo", "af2"]
split_titles = {"holo": "Holo", "apo": "Apo", "af2": "AF2"}

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)

for ax, split in zip(axes, splits):
    for name, d in data.items():
        if split not in d:
            continue
        ax.scatter(
            d["sec_per_prot"],
            d[split],
            s=160,
            c=d["color"],
            marker=d["marker"],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=name,
        )
        ax.annotate(
            name,
            (d["sec_per_prot"], d[split]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8.5,
            color=d["color"],
            fontweight="bold",
        )

    ax.set_xlabel("sec / protein", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(f"{split_titles[split]} (test)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")

fig.suptitle(
    "Surface Method: Performance vs Speed (sec/prot)",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("perf_vs_secpprot.png", dpi=200, bbox_inches="tight")
plt.savefig("perf_vs_secpprot.pdf", bbox_inches="tight")
print("Saved perf_vs_secpprot.png / .pdf")
