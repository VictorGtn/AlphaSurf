#!/usr/bin/env python
"""
Plot noising sweep results: one figure per setting (holo/apo/af2),
each with homo/hetero subplots side-by-side.

X-axis: sigma_graph noise level.
Lines:
  - joint mode (sg values, sm=0)  -- includes g=1.0
  - joint_mesh mode (joint + vertex-normal mesh noise)
  - independent mode grouped by sigma_mesh (3 sub-lines: sm=0.05/0.10/0.30)
  - horizontal reference: 30-ep baseline
"""

import matplotlib.pyplot as plt

# (sg, sm, holo_homo, holo_hetero, apo_homo, apo_hetero, af2_homo, af2_hetero)
JOINT = [
    (0.02, 0.00, 0.9252, 0.8932, 0.8611, 0.8382, 0.8725, 0.8388),
    (0.05, 0.00, 0.9275, 0.8973, 0.8667, 0.8472, 0.8739, 0.8400),
    (0.10, 0.00, 0.9256, 0.8928, 0.8668, 0.8504, 0.8742, 0.8381),
    (0.30, 0.00, 0.9281, 0.8961, 0.8675, 0.8555, 0.8752, 0.8415),
    (1.00, 0.00, 0.9197, 0.8819, 0.8659, 0.8477, 0.8563, 0.8070),
]

# joint_mesh: joint atom noise + vertex-normal mesh noise (single point at sg=0.3, sm=0.3)
JOINT_MESH = [
    (0.30, 0.30, 0.9303, 0.8978, 0.8688, 0.8609, 0.8753, 0.8374),
]

INDEP = [
    (0.02, 0.05, 0.9240, 0.8933, 0.8618, 0.8487, 0.8776, 0.8479),
    (0.05, 0.05, 0.9257, 0.8930, 0.8645, 0.8503, 0.8767, 0.8469),
    (0.10, 0.05, 0.9246, 0.8917, 0.8635, 0.8493, 0.8773, 0.8465),
    (0.10, 0.10, 0.9263, 0.8946, 0.8634, 0.8519, 0.8775, 0.8466),
    (0.30, 0.05, 0.9268, 0.8945, 0.8680, 0.8580, 0.8779, 0.8467),
    (0.30, 0.10, 0.9281, 0.8967, 0.8664, 0.8535, 0.8770, 0.8442),
    (0.30, 0.30, 0.9276, 0.8975, 0.8662, 0.8532, 0.8775, 0.8465),
]

# 30-ep baseline (exp_h100_463477): (holo_homo, holo_hetero, apo_homo, apo_hetero, af2_homo, af2_hetero)
BASELINE_30 = (0.9361, 0.9082, 0.8701, 0.8591, 0.8855, 0.8512)

SETTINGS = [
    ("holo", 0, 1, 1731),
    ("apo", 2, 3, 287),
    ("af2", 4, 5, 1332),
]

INDEP_COLORS = {0.05: "#1f77b4", 0.10: "#ff7f0e", 0.30: "#2ca02c"}


def plot_setting(name, homo_idx, hetero_idx, n_systems):
    fig, (ax_homo, ax_hetero) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{name.upper()}  (N={n_systems} common systems)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, idx, label in [
        (ax_homo, homo_idx, "homo"),
        (ax_hetero, hetero_idx, "hetero"),
    ]:
        # Joint line (now includes g=1.0)
        xs = [r[0] for r in JOINT]
        ys = [r[2 + idx] for r in JOINT]
        ax.plot(
            xs,
            ys,
            "o-",
            color="#d62728",
            linewidth=2,
            markersize=8,
            label="joint (sm=0)",
            zorder=5,
        )

        # joint_mesh: single point, distinct marker (star)
        xs = [r[0] for r in JOINT_MESH]
        ys = [r[2 + idx] for r in JOINT_MESH]
        ax.plot(
            xs,
            ys,
            "*",
            color="#9467bd",
            markersize=18,
            label="joint_mesh (sg=0.3, sm=0.3)",
            zorder=6,
        )

        # Independent lines grouped by sm
        for sm in sorted(INDEP_COLORS.keys()):
            rows = [r for r in INDEP if r[1] == sm]
            xs = [r[0] for r in rows]
            ys = [r[2 + idx] for r in rows]
            ax.plot(
                xs,
                ys,
                "s--",
                color=INDEP_COLORS[sm],
                linewidth=1.5,
                markersize=7,
                label=f"indep sm={sm}",
                alpha=0.85,
                zorder=4,
            )

        # Baseline (30-ep only)
        ax.axhline(
            BASELINE_30[idx],
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="30-ep baseline",
            zorder=2,
        )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_g$ (Å, log scale)")
        ax.set_ylabel("AUROC")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    out = f"noise_sweep_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    for name, h_idx, he_idx, n in SETTINGS:
        plot_setting(name, h_idx, he_idx, n)
