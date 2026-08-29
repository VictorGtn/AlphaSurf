#!/usr/bin/env python3
"""Plot one representative fragmented surface for each Pinder method."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402
from extract_interface_cc import count_components_all_atom  # noqa: E402
from visualize_all_methods import mesh_collection, mesh_limits  # noqa: E402


METHODS = (
    ("algo2", "Alpha Complex", "#E41A1C"),
    ("msms_ext", "MSMS", "#6A3D9A"),
    ("edtsurf_0.3", "EDTsurf\ngs = 0.3", "#9ECAE1"),
    ("edtsurf_0.4", "EDTsurf\ngs = 0.4", "#4292C6"),
    ("edtsurf_0.5", "EDTsurf\ngs = 0.5", "#08519C"),
    ("nanoshaper_0.3", "NanoShaper\ngs = 0.3", "#A1D99B"),
    ("nanoshaper_0.4", "NanoShaper\ngs = 0.4", "#41AB5D"),
    ("nanoshaper_0.5", "NanoShaper\ngs = 0.5", "#238B45"),
    ("nanoshaper_0.6", "NanoShaper\ngs = 0.6", "#005A32"),
)

DEFAULT_CSV = Path(__file__).parent / "cc_sweep_output" / "cc_threshold_sweep.csv"
DEFAULT_SURFACE_DIR = Path(__file__).parent / "cc_sweep_output" / "surfaces"
DEFAULT_EDTSURF_CSV = (
    Path(__file__).parent / "cc_sweep_output_edtsurf_outer" / "cc_threshold_sweep.csv"
)
DEFAULT_EDTSURF_SURFACE_DIR = (
    Path(__file__).parent / "cc_sweep_output_edtsurf_outer" / "surfaces"
)
DEFAULT_CLASSIFICATION_CSV = (
    Path(__file__).parent / "cc_sweep_output" / "pinder_pair_ca_classification.csv"
)
DEFAULT_PDB_DIR = Path(__file__).parents[1] / "data" / "pinder-pair" / "pdb"
FRAGMENT_COLOR = "#BDBDBD"
EDTSURF_METHODS = frozenset({"edtsurf_0.3", "edtsurf_0.4", "edtsurf_0.5"})
MIN_EDTSURF_PLOT_FACES = 2000
MAX_EDT_GEOMETRY_CANDIDATES = 8
EDT_REJECT_PDB_IDS = frozenset({"8q6t"})
PREFERRED_EDT_FAILURES = {
    "edtsurf_0.3": "8h8c__A1_Q87PI5--8h8c__C1_Q87PI5_L.pdb",
    "edtsurf_0.4": "8cp6__B1_Q9I0F4--8cp6__C1_Q9I0F4_R.pdb",
    "edtsurf_0.5": "6emw__L1_Q2YSD6--6emw__FA1_Q2G1U5_R.pdb",
}


def parse_sizes(value):
    if not value:
        return []
    return sorted((int(size) for size in str(value).split(";")), reverse=True)


def load_rows(csv_path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_edtsurf_rows(rows, edtsurf_rows):
    edtsurf_by_name = {row["pdb_name"]: row for row in edtsurf_rows}
    fields = ("status", "n_verts", "n_faces", "cc_sizes")
    for row in rows:
        replacement = edtsurf_by_name.get(row["pdb_name"])
        if replacement is None:
            continue
        for method in EDTSURF_METHODS:
            for field in fields:
                key = f"{method}_{field}"
                row[key] = replacement[key]
    return rows


def filter_full_atom(rows, classification_csv):
    with classification_csv.open(newline="") as handle:
        full_atom = {
            row["pdb_name"]
            for row in csv.DictReader(handle)
            if row["category"] == "full_atom"
        }
    return [row for row in rows if row["pdb_name"] in full_atom]


def atom_graph_is_connected(pdb_path):
    parsed = parse_pdb_path(pdb_path.resolve(), use_pqr=False)
    atom_positions = np.asarray(parsed[5], dtype=np.float64)
    n_components, _ = count_components_all_atom(atom_positions, cutoff=5.0)
    return n_components == 1


def choose_failures(rows, surface_dirs, pdb_dir):
    candidates_by_method = {method: [] for method, _, _ in METHODS}
    atom_graph_cache = {}
    for row in rows:
        pdb_name = row["pdb_name"]
        for method, _, _ in METHODS:
            sizes = parse_sizes(row.get(f"{method}_cc_sizes", ""))
            status = row.get(f"{method}_status", "")
            if not (status.startswith("ok") and len(sizes) > 1 and sizes[0] > sizes[1]):
                continue

            n_faces = int(row.get(f"{method}_n_faces", 0) or 0)
            if method in EDTSURF_METHODS and n_faces < MIN_EDTSURF_PLOT_FACES:
                continue
            fragmentation = sizes[1] / sum(sizes)
            candidate = (fragmentation, n_faces, -len(sizes), pdb_name, row, sizes)
            candidates_by_method[method].append(candidate)

    selected = {}
    used_edt_pdb_ids = set()
    for method, _, _ in METHODS:
        candidates = sorted(
            candidates_by_method[method], key=lambda item: item[:4], reverse=True
        )
        if method in EDTSURF_METHODS:
            preferred_name = PREFERRED_EDT_FAILURES[method]
            preferred = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate[3] == preferred_name
                ),
                None,
            )
            if preferred is not None:
                pdb_path = pdb_dir / preferred_name
                atom_graph_cache[preferred_name] = atom_graph_is_connected(pdb_path)
                if atom_graph_cache[preferred_name]:
                    selected[method] = preferred
                    used_edt_pdb_ids.add(Path(preferred_name).stem.split("__", 1)[0])
                    continue

            evaluated = []
            for candidate in candidates:
                pdb_name = candidate[3]
                pdb_id = Path(pdb_name).stem.split("__", 1)[0]
                if pdb_id in used_edt_pdb_ids or pdb_id in EDT_REJECT_PDB_IDS:
                    continue
                if pdb_name not in atom_graph_cache:
                    atom_graph_cache[pdb_name] = atom_graph_is_connected(
                        pdb_dir / pdb_name
                    )
                if not atom_graph_cache[pdb_name]:
                    continue
                npz_path = surface_dirs[method] / f"{Path(pdb_name).stem}.npz"
                with np.load(npz_path, allow_pickle=True) as data:
                    vertices = np.asarray(data[f"{method}_verts"], dtype=np.float32)
                    faces = np.asarray(data[f"{method}_faces"], dtype=np.int32)
                visibility, projected_area = component_visibility_metrics(
                    vertices, faces
                )
                evaluated.append(
                    (visibility * projected_area * candidate[0], candidate)
                )
                if len(evaluated) == MAX_EDT_GEOMETRY_CANDIDATES:
                    break
            if not evaluated:
                continue
            _, candidate = max(
                evaluated, key=lambda item: (item[0], item[1][0], item[1][1])
            )
            selected[method] = candidate
            used_edt_pdb_ids.add(Path(candidate[3]).stem.split("__", 1)[0])
            continue

        for candidate in candidates:
            pdb_name = candidate[3]
            if pdb_name not in atom_graph_cache:
                atom_graph_cache[pdb_name] = atom_graph_is_connected(pdb_dir / pdb_name)
            if not atom_graph_cache[pdb_name]:
                continue
            selected[method] = candidate
            break

    missing = [method for method, _, _ in METHODS if method not in selected]
    if missing:
        raise RuntimeError(
            "No saved Pinder connected-component failure for: " + ", ".join(missing)
        )
    missing_pdbs = [
        candidate[3]
        for candidate in selected.values()
        if not (pdb_dir / candidate[3]).is_file()
    ]
    if missing_pdbs:
        raise RuntimeError("Missing Pinder PDB files: " + ", ".join(missing_pdbs))
    return selected


def select_single_pdb(rows, pdb_name, pdb_dir):
    row = next((row for row in rows if row["pdb_name"] == pdb_name), None)
    if row is None:
        raise RuntimeError(f"PDB is absent from the filtered CC results: {pdb_name}")
    if not (pdb_dir / pdb_name).is_file():
        raise RuntimeError(f"Missing Pinder PDB file: {pdb_name}")

    selected = {}
    for method, _, _ in METHODS:
        sizes = parse_sizes(row.get(f"{method}_cc_sizes", ""))
        n_faces = int(row.get(f"{method}_n_faces", 0) or 0)
        fragmentation = sizes[1] / sum(sizes) if len(sizes) > 1 else 0.0
        selected[method] = (
            fragmentation,
            n_faces,
            -len(sizes),
            pdb_name,
            row,
            sizes,
        )
    return selected


def component_masks(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    sizes = np.asarray(sizes)
    order = sorted(
        range(len(sizes)), key=lambda component: int(sizes[component]), reverse=True
    )
    return [labels == component for component in order]


def component_visibility_metrics(vertices, faces):
    masks = component_masks(vertices, faces)
    if len(masks) < 2:
        return 0.0, 0.0

    triangle_centers = vertices[faces].mean(axis=1)
    centers = [triangle_centers[mask].mean(axis=0) for mask in masks[:2]]
    delta = centers[1] - centers[0]
    view_direction = np.array(
        [
            np.cos(np.deg2rad(23)) * np.cos(np.deg2rad(38)),
            np.cos(np.deg2rad(23)) * np.sin(np.deg2rad(38)),
            np.sin(np.deg2rad(23)),
        ]
    )
    right_direction = np.array([np.sin(np.deg2rad(38)), -np.cos(np.deg2rad(38)), 0.0])
    up_direction = np.cross(view_direction, right_direction)
    projected_areas = []
    for mask in masks[:2]:
        projected_vertices = vertices[faces[mask]].reshape(-1, 3)
        screen_coordinates = (
            projected_vertices @ np.array([right_direction, up_direction]).T
        )
        projected_areas.append(float(np.prod(np.ptp(screen_coordinates, axis=0))))
    projected_distance = np.linalg.norm(
        delta - np.dot(delta, view_direction) * view_direction
    )
    extent = np.ptp(vertices, axis=0).max()
    visibility = float(projected_distance / extent) if extent else 0.0
    return visibility, min(projected_areas)


def component_separating_azimuth(vertices, faces, masks):
    triangle_centers = vertices[faces].mean(axis=1)
    centers = [triangle_centers[mask].mean(axis=0) for mask in masks[:2]]
    delta = centers[1] - centers[0]
    return np.rad2deg(np.arctan2(delta[1], delta[0])) + 90.0


def plot_failed_mesh(ax, vertices, faces, method_color, label):
    masks = component_masks(vertices, faces)
    used_vertices = vertices[np.unique(faces)]
    center, half_width = mesh_limits(used_vertices)

    for index, mask in enumerate(masks):
        color = method_color if index == 0 else FRAGMENT_COLOR
        ax.add_collection3d(mesh_collection(vertices, faces[mask], color))

    for axis, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(coordinate - half_width, coordinate + half_width)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(
        elev=15,
        azim=(
            component_separating_azimuth(vertices, faces, masks)
            if len(masks) > 1
            else 38
        ),
    )
    ax.set_axis_off()
    ax.text2D(
        0.5,
        0.035,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#222222",
        linespacing=1.1,
    )


def add_fragment_inset(fig, parent_ax, vertices, faces):
    masks = component_masks(vertices, faces)
    if len(masks) < 2:
        return
    fragment_faces = faces[masks[1]]
    fragment_vertices = vertices[np.unique(fragment_faces)]
    center, half_width = mesh_limits(fragment_vertices)

    parent_box = parent_ax.get_position()
    width = parent_box.width * 0.27
    height = parent_box.height * 0.27
    left = parent_box.x1 - width - parent_box.width * 0.03
    bottom = parent_box.y1 - height - parent_box.height * 0.03
    inset = fig.add_axes([left, bottom, width, height], projection="3d")
    inset.add_collection3d(mesh_collection(vertices, fragment_faces, FRAGMENT_COLOR))
    for axis, coordinate in zip(
        (inset.set_xlim, inset.set_ylim, inset.set_zlim), center
    ):
        axis(coordinate - half_width, coordinate + half_width)
    inset.set_box_aspect((1, 1, 1))
    inset.set_proj_type("ortho")
    inset.view_init(elev=23, azim=38)
    inset.set_axis_off()
    inset.patch.set_alpha(0)
    fig.add_artist(
        Rectangle(
            (left, bottom),
            width,
            height,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#BDBDBD",
            linewidth=0.7,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--edtsurf-csv", type=Path, default=DEFAULT_EDTSURF_CSV)
    parser.add_argument(
        "--edtsurf-surface-dir", type=Path, default=DEFAULT_EDTSURF_SURFACE_DIR
    )
    parser.add_argument(
        "--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV
    )
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("--pdb-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    rows = merge_edtsurf_rows(load_rows(args.csv), load_rows(args.edtsurf_csv))
    rows = filter_full_atom(rows, args.classification_csv)
    surface_dirs = {
        method: args.edtsurf_surface_dir
        if method in EDTSURF_METHODS
        else args.surface_dir
        for method, _, _ in METHODS
    }
    selected = (
        select_single_pdb(rows, args.pdb_name, args.pdb_dir)
        if args.pdb_name
        else choose_failures(rows, surface_dirs, args.pdb_dir)
    )
    print(
        "Selected PDB across all methods:"
        if args.pdb_name
        else "Selected independent connected-component failures:"
    )
    for method, _, _ in METHODS:
        _, _, _, pdb_name, _, sizes = selected[method]
        print(f"  {method}: {pdb_name} ({';'.join(map(str, sizes))})")

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

    for index, (method, label, color) in enumerate(METHODS):
        _, _, _, pdb_name, _, _ = selected[method]
        npz_path = surface_dirs[method] / f"{Path(pdb_name).stem}.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            vertices = np.asarray(data[f"{method}_verts"], dtype=np.float32)
            faces = np.asarray(data[f"{method}_faces"], dtype=np.int32)

        row_index, column = divmod(index, 3)
        ax = fig.add_subplot(grid[row_index, column], projection="3d")
        plot_failed_mesh(
            ax,
            vertices,
            faces,
            color,
            label,
        )
        if method in EDTSURF_METHODS:
            add_fragment_inset(fig, ax, vertices, faces)

    output = args.output or Path("cc_failure_meshes_independent.png")
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
