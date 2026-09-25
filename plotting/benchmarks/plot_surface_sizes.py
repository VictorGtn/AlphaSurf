from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
FIG_DIR.mkdir(parents=True, exist_ok=True)

methods = [
    ("alpha complex (tufting)",    "#E53935", "s"),
    ("alpha complex (no tufting)", "#E53935", "s"),
    ("edtsurf 0.3",        "#1E88E5", "o"),
    ("edtsurf 0.4",        "#1E88E5", "o"),
    ("edtsurf 0.5",        "#1E88E5", "o"),
    ("nanoshaper 0.3",     "#8E24AA", "D"),
    ("nanoshaper 0.4",     "#8E24AA", "D"),
    ("nanoshaper 0.5",     "#8E24AA", "D"),
    ("nanoshaper 0.6",     "#8E24AA", "D"),
    ("msms (coarsened)",   "#6D4C41", "X"),
]

mean_verts = [1370, 1232, 924, 1614, 2438, 1420, 2697, 4375, 6439, 1450]
mean_faces = [2746, 2470, 1844, 3223, 4865, 2835, 5415, 8778, 12909, 2909]

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(methods))
bar_width = 0.35

for i, (name, color, marker) in enumerate(methods):
    if "no tufting" in name:
        ax.bar(x[i] - bar_width/2, mean_verts[i], bar_width,
               facecolor="white", edgecolor=color, linewidth=2.0)
    else:
        ax.bar(x[i] - bar_width/2, mean_verts[i], bar_width, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.8)

handles = []
seen = set()
for name, color, marker in methods:
    family = name.split(" ")[0]
    if family not in seen:
        seen.add(family)
        handles.append(plt.Rectangle((0,0), 1, 1, color=color, label=family))

ax.set_xticks(x)
ax.set_xticklabels([m[0].replace("_", " ") for m in methods], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean Vertices", fontsize=11)
ax.set_title("Mean Surface Size per Method (100k proteins)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.2)

ax.legend(handles=handles, loc="upper left", fontsize=9, title="Method", title_fontsize=10)

for i, v in enumerate(mean_verts):
    ax.text(x[i] - bar_width/2, v + 200, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color="#333")

plt.tight_layout()
plt.savefig(FIG_DIR / "surface_sizes.png", dpi=200, bbox_inches="tight")
plt.savefig(FIG_DIR / "surface_sizes.pdf", bbox_inches="tight")
print("Saved surface_sizes.png / .pdf")
