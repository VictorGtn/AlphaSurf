#!/usr/bin/env python3
"""
Fall-off of the heat kernel with geodesic distance on the MSMS surface.

For a random subset of the proteins of a spectral_comparison.py run, rebuilds the full MSMS mesh,
draws its farthest-point samples as spectral_comparison.py does, and averages
K_t(x_i, x_j) / sqrt(K_t(x_i, x_i) K_t(x_j, x_j)) per bin of heat-method geodesic distance, with the
network's Laplacian modes. The normalisation is the kernel's own value at the source at the same time,
its maximum by Cauchy-Schwarz. Prints the distance at which the mean profile falls to each of LEVELS,
and draws the mean profile over proteins:
  <output_dir>/kernel_profile.pdf
  <output_dir>/kernel_profile.npz
"""

import argparse
import csv
import os
import sys
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(script_dir))
# spectral_comparison.py does the computation these figures read.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import spectral_comparison as sc  # noqa: E402

OUTPUTS = os.path.join(REPO_ROOT, "scripts", "outputs")
FIG_DIR = os.path.join(REPO_ROOT, "plotting", "figures", "spectral")
EDGES = np.arange(0.0, 30.5, 0.5)
LEVELS = (0.5, 0.25, 0.1, 0.05, 0.01)
MSMS_COLOR = "#6A3D9A"


def profile(pdb_path, mesh_args, times, k_eig):
    """Mean normalised kernel per distance bin, one row per time."""
    meshes, _ = sc.build_mesh_set(pdb_path, mesh_args)
    full = meshes["msms_full"]
    spec = sc.spectra_for(full, k_eig)
    G = sc.geodesic_matrix(full, full["sample_vertex"])
    iu = np.triu_indices(len(G), 1)
    which = np.digitize(G[iu], EDGES) - 1
    ok = (which >= 0) & (which < len(EDGES) - 1)
    out = np.full((len(times), len(EDGES) - 1), np.nan)
    for i, t in enumerate(times):
        K = sc.point_kernel(spec["evals"], spec["evecs"][full["sample_vertex"]], t)
        scale = np.sqrt(np.diag(K))
        k = (K / np.outer(scale, scale))[iu]
        sums = np.bincount(which[ok], weights=k[ok], minlength=len(EDGES) - 1)
        counts = np.bincount(which[ok], minlength=len(EDGES) - 1)
        out[i] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def crossing(centres, values, level):
    """First distance at which values fall below level, linearly interpolated between bins."""
    below = np.flatnonzero(values < level)
    if len(below) == 0:
        return np.nan
    j = below[0]
    if j == 0:
        return centres[0]
    x0, x1, y0, y1 = centres[j - 1], centres[j], values[j - 1], values[j]
    return x0 + (y0 - level) * (x1 - x0) / (y0 - y1)


def plot(centres, profiles, times, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 3.2), sharey=True, squeeze=False)
    for ax, i, t in zip(axes[0], range(len(times)), times):
        ax.plot(centres, np.nanmean(profiles[:, i], axis=0), color=MSMS_COLOR, lw=1.8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for level in (0.5, 0.1):
            ax.axhline(level, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"$t={t:g}$ s", fontsize=10)
        ax.set_xlabel("geodesic distance $d$ (Å)")
        ax.grid(ls=":", color="0.85", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("heat kernel at distance $d$\n(% of its value at the source)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=os.path.join(OUTPUTS, "spectral_heat_sampled_0_10"),
                        help="spectral_comparison.py run whose proteins are sampled")
    parser.add_argument("--pdb-dir", default=os.path.join(REPO_ROOT, "data", "pinder-pair", "pdb"))
    parser.add_argument("--output-dir", default=FIG_DIR)
    parser.add_argument("--n-proteins", type=int, default=None, help="random subset size; default every protein")
    parser.add_argument("--times", default="5,10,20", help="diffusion times")
    parser.add_argument("--k-eig", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed of the protein subset")
    parser.add_argument("--allow-multiple-components", action="store_true",
                        help="as in spectral_comparison.py; set it when the run was made with it")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    times = [float(t) for t in args.times.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [r for r in csv.DictReader(open(os.path.join(args.input_dir, "spectral_comparison.csv")))
            if r["status"] == "ok" and r["pair"] == "alpha:msms_full"]
    pdbs = [r["pdb_id"] for r in rows]
    if args.n_proteins is not None and args.n_proteins < len(pdbs):
        picked = np.random.default_rng(args.seed).choice(len(pdbs), size=args.n_proteins, replace=False)
        pdbs = [pdbs[i] for i in picked]
    mesh_args = SimpleNamespace(
        pairs=["msms_full:msms_full"], alpha_value=0.0, msms_density=1.0, msms_reduction=0.1, min_vert_number=16,
        max_vert_number=1000000, support="sampled", msms_radius_offset=0.0, msms_probe=None,
        grid_scale=0.5, n_samples=args.n_samples, seed=2024,
        allow_multiple_components=args.allow_multiple_components,
    )
    work = partial(profile, mesh_args=mesh_args, times=times, k_eig=args.k_eig)
    with Pool(args.workers) as pool:
        profiles = np.array(pool.map(work, [os.path.join(args.pdb_dir, f"{pdb}.pdb") for pdb in pdbs]))

    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    np.savez(os.path.join(args.output_dir, "kernel_profile.npz"), profiles=profiles, centres=centres,
             times=np.array(times), pdbs=np.array(pdbs))
    print(f"{len(pdbs)} proteins; distance (A) at which the mean profile falls to each level, "
          f"10-90% of proteins in brackets")
    for i, t in enumerate(times):
        mean = np.nanmean(profiles[:, i], axis=0)
        cells = []
        for level in LEVELS:
            per = np.array([crossing(centres, p, level) for p in profiles[:, i]])
            cells.append(f"{level:.0%} {crossing(centres, mean, level):.1f} "
                         f"({np.nanquantile(per, 0.1):.1f}-{np.nanquantile(per, 0.9):.1f})")
        print(f"t={t:g}: " + " | ".join(cells))
    out_path = os.path.join(args.output_dir, "kernel_profile.pdf")
    plot(centres, profiles, times, out_path)
    print(f"figure written to {out_path}")


if __name__ == "__main__":
    main()
