#!/usr/bin/env python3
"""Render a mesh-only comparison of the available surface representations."""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


METHODS = {
    "alpha": {"key": "algo2", "color": "#E41A1C", "label": "Alpha Complex"},
    "msms": {"key": "msms", "color": "#6A3D9A", "label": "MSMS"},
    "edtsurf_0.3": {
        "key": "edtsurf_0.3",
        "color": "#9ECAE1",
        "label": "EDTsurf\ngs = 0.3",
    },
    "edtsurf_0.4": {
        "key": "edtsurf_0.4",
        "color": "#4292C6",
        "label": "EDTsurf\ngs = 0.4",
    },
    "edtsurf_0.5": {
        "key": "edtsurf_0.5",
        "color": "#08519C",
        "label": "EDTsurf\ngs = 0.5",
    },
    "nanoshaper_0.3": {
        "key": "nanoshaper_0.3",
        "color": "#A1D99B",
        "label": "NanoShaper\ngs = 0.3",
    },
    "nanoshaper_0.4": {
        "key": "nanoshaper_0.4",
        "color": "#41AB5D",
        "label": "NanoShaper\ngs = 0.4",
    },
    "nanoshaper_0.5": {
        "key": "nanoshaper_0.5",
        "color": "#238B45",
        "label": "NanoShaper\ngs = 0.5",
    },
    "nanoshaper_0.6": {
        "key": "nanoshaper_0.6",
        "color": "#005A32",
        "label": "NanoShaper\ngs = 0.6",
    },
}

PANEL_ORDER = (
    "alpha",
    "edtsurf_0.3",
    "edtsurf_0.4",
    "edtsurf_0.5",
    "nanoshaper_0.3",
    "nanoshaper_0.4",
    "nanoshaper_0.5",
    "nanoshaper_0.6",
    "msms",
)


def mesh_limits(vertices):
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2.0
    half_width = 0.5 * np.max(upper - lower) * 1.06
    return center, half_width


def mesh_collection(vertices, faces, color, alpha=1.0):
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals /= lengths[:, None]

    light_direction = np.array([0.45, -0.35, 0.82])
    light_direction /= np.linalg.norm(light_direction)
    illumination = np.clip(normals @ light_direction, 0.0, 1.0)
    illumination = 0.42 + 0.58 * illumination

    rgb = np.asarray(mcolors.to_rgb(color))
    face_colors = np.column_stack(
        (rgb[None, :] * illumination[:, None], np.full(len(faces), alpha))
    )
    return Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=(0.08, 0.08, 0.08, 0.24),
        linewidths=0.16,
        antialiased=True,
    )


def plot_mesh(ax, vertices, faces, color, label):
    if len(vertices) == 0 or len(faces) == 0:
        return

    center, half_width = mesh_limits(vertices)
    ax.add_collection3d(mesh_collection(vertices, faces, color))
    for axis, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(coordinate - half_width, coordinate + half_width)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=38)
    ax.set_axis_off()
    ax.text2D(
        0.5,
        0.035,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#222222",
        linespacing=1.1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", help="PDB stem, or the corresponding NPZ stem")
    parser.add_argument(
        "--npz-dir",
        default=Path(__file__).parent / "cc_sweep_output" / "surfaces",
        type=Path,
        help="Directory containing the surface-comparison NPZ files",
    )
    parser.add_argument("--output", type=Path, help="PNG output path")
    parser.add_argument("--pdf", type=Path, help="Optional vector PDF output path")
    parser.add_argument("--dpi", type=int, default=400, help="Raster output resolution")
    args = parser.parse_args()

    stem = Path(args.pdb).stem
    npz_path = args.npz_dir / f"{stem}.npz"
    if not npz_path.exists():
        parser.error(f"NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        fig = plt.figure(figsize=(10, 10), facecolor="white")
        grid = GridSpec(
            3,
            3,
            figure=fig,
            left=0.005,
            right=0.995,
            bottom=0.005,
            top=0.995,
            wspace=0.005,
            hspace=0.005,
        )

        for index, name in enumerate(PANEL_ORDER):
            row, column = divmod(index, 3)
            spec = METHODS[name]
            ax = fig.add_subplot(grid[row, column], projection="3d")
            vertices = np.asarray(data[f"{spec['key']}_verts"])
            faces = np.asarray(data[f"{spec['key']}_faces"], dtype=np.int32)
            plot_mesh(ax, vertices, faces, spec["color"], spec["label"])

    output = args.output or Path(f"surface_grid_{stem}_publication.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    print(f"Saved: {output}")

    if args.pdf is not None:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf, facecolor="white")
        print(f"Saved: {args.pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
