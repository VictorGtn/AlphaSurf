#!/usr/bin/env python3
"""
Mesh Statistics Analysis: Compare Multiple Surface Datasets

Computes and visualizes:
- Number of vertices per mesh (log scale)
- Edge lengths
- Face areas
- Dihedral angles between adjacent faces

Supports multiprocessing for faster analysis and multiple datasets.
"""

import argparse
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Force headless backend before importing pyplot
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import igl

    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("Warning: igl not available, falling back to manual normal computation")

# Default paths (used when no --dirs specified)
BASE_DIR = "/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/data/masif_ligand/msms_01"
DEFAULT_DIRS = [
    f"{BASE_DIR}/surfaces_full_alpha0_fr1.0",
    f"{BASE_DIR}/surfaces_full_alpha2_fr0.7",
    f"{BASE_DIR}/surfaces_full_alpha2_fr0.8",
    f"{BASE_DIR}/surfaces_full_alpha2_fr0.9",
    f"{BASE_DIR}/surfaces_full_alpha2_fr1.0",
    f"{BASE_DIR}/surfaces_full_alpha4_fr0.7",
    f"{BASE_DIR}/surfaces_full_alpha4_fr0.8",
    f"{BASE_DIR}/surfaces_full_alpha4_fr0.9",
    f"{BASE_DIR}/surfaces_full_alpha4_fr1.0",
    f"{BASE_DIR}/surfaces_full_alpha8_fr0.7",
    f"{BASE_DIR}/surfaces_full_alpha8_fr0.8",
    f"{BASE_DIR}/surfaces_full_alpha8_fr0.9",
    f"{BASE_DIR}/surfaces_full_alpha8_fr1.0",
]

DEFAULT_NAMES = [
    "alpha0_fr1.0",
    "alpha2_fr0.7",
    "alpha2_fr0.8",
    "alpha2_fr0.9",
    "alpha2_fr1.0",
    "alpha4_fr0.7",
    "alpha4_fr0.8",
    "alpha4_fr0.9",
    "alpha4_fr1.0",
    "alpha8_fr0.7",
    "alpha8_fr0.8",
    "alpha8_fr0.9",
    "alpha8_fr1.0",
]

OUTPUT_DIR = "/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/scripts/"


def compute_edge_lengths(verts, faces):
    """Compute edge lengths from vertices and faces."""
    edges = []
    for i in range(3):
        j = (i + 1) % 3
        edge_vecs = verts[faces[:, i]] - verts[faces[:, j]]
        edge_lengths = np.linalg.norm(edge_vecs, axis=1)
        edges.append(edge_lengths)
    return np.concatenate(edges)


def compute_face_areas(verts, faces):
    """Compute face areas using cross product."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def compute_face_angles(verts, faces):
    """Compute interior angles of each triangular face.
    Each triangle contributes 3 angles, returned as a flat array.
    Angles are returned in degrees.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Edge vectors from each vertex
    e01 = v1 - v0  # edge from v0 to v1
    e02 = v2 - v0  # edge from v0 to v2
    e10 = v0 - v1  # edge from v1 to v0
    e12 = v2 - v1  # edge from v1 to v2
    e20 = v0 - v2  # edge from v2 to v0
    e21 = v1 - v2  # edge from v2 to v1

    # Normalize edge vectors
    def normalize(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v / norms

    e01_n = normalize(e01)
    e02_n = normalize(e02)
    e10_n = normalize(e10)
    e12_n = normalize(e12)
    e20_n = normalize(e20)
    e21_n = normalize(e21)

    # Compute angles at each vertex using dot product
    # Angle at v0: between edges e01 and e02
    cos_angle0 = np.clip(np.sum(e01_n * e02_n, axis=1), -1, 1)
    angle0 = np.arccos(cos_angle0) * 180 / np.pi

    # Angle at v1: between edges e10 and e12
    cos_angle1 = np.clip(np.sum(e10_n * e12_n, axis=1), -1, 1)
    angle1 = np.arccos(cos_angle1) * 180 / np.pi

    # Angle at v2: between edges e20 and e21
    cos_angle2 = np.clip(np.sum(e20_n * e21_n, axis=1), -1, 1)
    angle2 = np.arccos(cos_angle2) * 180 / np.pi

    # Concatenate all angles
    return np.concatenate([angle0, angle1, angle2])


def orient_faces_consistently(faces):
    """Orient faces consistently using igl.bfs_orient."""
    if HAS_IGL:
        # bfs_orient returns (oriented_faces, C) where C is component IDs
        oriented_faces, _ = igl.bfs_orient(faces.astype(np.int32))
        return oriented_faces
    else:
        return faces


def compute_face_normals(verts, faces, orient=True):
    """Compute face normals using igl if available, otherwise use cross product.

    Args:
        verts: (N, 3) vertices
        faces: (F, 3) face indices
        orient: If True, orient faces consistently before computing normals
    """
    verts_np = verts.astype(np.float64)
    faces_np = faces.astype(np.int32)

    # Orient faces consistently if requested
    if orient and HAS_IGL:
        faces_np, _ = igl.bfs_orient(faces_np)

    if HAS_IGL:
        # Use igl for face normals
        face_normals = igl.per_face_normals(
            verts_np, faces_np, np.array([0.0, 0.0, 0.0])
        )
        return face_normals, faces_np
    else:
        # Fallback to manual computation
        v0 = verts[faces_np[:, 0]]
        v1 = verts[faces_np[:, 1]]
        v2 = verts[faces_np[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8
        return cross / norms, faces_np


def compute_dihedral_angles(verts, faces):
    """Compute dihedral angles between adjacent faces with consistent orientation."""
    # First orient faces consistently, then compute normals
    normals, oriented_faces = compute_face_normals(verts, faces, orient=True)

    # Build edge-to-face adjacency using oriented faces
    edge_dict = {}
    for face_idx in range(len(oriented_faces)):
        face = oriented_faces[face_idx]
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edge_dict:
                edge_dict[edge] = []
            edge_dict[edge].append(face_idx)

    # Compute dihedral angles for edges with exactly 2 adjacent faces
    angles = []
    for edge, face_indices in edge_dict.items():
        if len(face_indices) == 2:
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi  # in degrees
            angles.append(angle)

    return np.array(angles) if angles else np.array([0.0])


def load_and_analyze_surface(filepath):
    """Load a surface file and compute statistics."""
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        # Extract vertices and faces
        verts = data.get("verts", None)
        faces = data.get("faces", None)

        if verts is None or faces is None:
            return None

        # Convert to numpy
        if isinstance(verts, torch.Tensor):
            verts = verts.numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.numpy()

        verts = verts.astype(np.float64)
        faces = faces.astype(np.int64)

        # Compute statistics
        n_verts = verts.shape[0]
        n_faces = faces.shape[0]
        n_edges = n_faces * 3 // 2  # Euler formula approximation

        edge_lengths = compute_edge_lengths(verts, faces)
        face_areas = compute_face_areas(verts, faces)
        face_angles = compute_face_angles(verts, faces)
        dihedral_angles = compute_dihedral_angles(verts, faces)

        return {
            "n_verts": n_verts,
            "n_faces": n_faces,
            "n_edges": n_edges,
            "edge_lengths": edge_lengths,
            "face_areas": face_areas,
            "face_angles": face_angles,
            "dihedral_angles": dihedral_angles,
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def analyze_dataset(data_dir, name, max_files=None, n_workers=1):
    """Analyze all surfaces in a directory using multiprocessing."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 60}")

    files = list(Path(data_dir).glob("*.pt"))
    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} files, using {n_workers} workers")

    all_stats = {
        "n_verts": [],
        "n_faces": [],
        "n_edges": [],
        "edge_lengths": [],
        "face_areas": [],
        "face_angles": [],
        "dihedral_angles": [],
    }

    if n_workers > 1:
        # Use multiprocessing
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(load_and_analyze_surface, files),
                    total=len(files),
                    desc=f"Processing {name}",
                )
            )
    else:
        # Single-threaded
        results = []
        for f in tqdm(files, desc=f"Processing {name}"):
            results.append(load_and_analyze_surface(f))

    # Aggregate results
    for stats in results:
        if stats:
            all_stats["n_verts"].append(stats["n_verts"])
            all_stats["n_faces"].append(stats["n_faces"])
            all_stats["n_edges"].append(stats["n_edges"])
            all_stats["edge_lengths"].extend(stats["edge_lengths"].tolist())
            all_stats["face_areas"].extend(stats["face_areas"].tolist())
            all_stats["face_angles"].extend(stats["face_angles"].tolist())
            all_stats["dihedral_angles"].extend(stats["dihedral_angles"].tolist())

    return all_stats


def print_summary_stats(stats, name):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"Summary Statistics: {name}")
    print(f"{'=' * 60}")

    metrics = [
        ("Vertices per mesh", stats["n_verts"]),
        ("Faces per mesh", stats["n_faces"]),
        ("Edges per mesh", stats["n_edges"]),
        ("Edge lengths (Å)", stats["edge_lengths"]),
        ("Face areas (Å²)", stats["face_areas"]),
        ("Face angles (°)", stats["face_angles"]),
        ("Dihedral angles (°)", stats["dihedral_angles"]),
    ]

    results = {}
    for metric_name, values in metrics:
        values = np.array(values)
        results[metric_name] = {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
        }
        print(f"\n{metric_name}:")
        print(f"  Min: {results[metric_name]['min']:.4f}")
        print(f"  Max: {results[metric_name]['max']:.4f}")
        print(f"  Mean: {results[metric_name]['mean']:.4f}")
        print(f"  Median: {results[metric_name]['median']:.4f}")
        print(f"  Std: {results[metric_name]['std']:.4f}")

    return results


def plot_per_metric(all_stats, all_results, names, output_dir):
    """Create one PNG per metric with N subplots (one per dataset) in a grid."""
    n_datasets = len(names)

    # Calculate grid dimensions
    n_cols = min(4, n_datasets)  # Max 4 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Generate colors from a colormap
    cmap = cm.get_cmap("tab10") if n_datasets <= 10 else cm.get_cmap("tab20")
    colors = [cmap(i) for i in range(n_datasets)]

    # (stats_key, display_title, results_key, filename, use_log_scale)
    metrics = [
        (
            "n_verts",
            "Vertices per Mesh",
            "Vertices per mesh",
            "vertices_per_mesh",
            True,
        ),
        ("edge_lengths", "Edge Lengths (Å)", "Edge lengths (Å)", "edge_lengths", False),
        ("face_areas", "Face Areas (Å²)", "Face areas (Å²)", "face_areas", False),
        ("face_angles", "Face Angles (°)", "Face angles (°)", "face_angles", False),
        (
            "dihedral_angles",
            "Dihedral Angles (°)",
            "Dihedral angles (°)",
            "dihedral_angles",
            False,
        ),
    ]

    output_paths = []

    for key, title, result_key, filename, use_log in metrics:
        # Create figure with grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_datasets == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Collect all values to determine common bin range and x-axis limits
        all_values = []
        for stats in all_stats:
            values = np.array(stats[key])
            if len(values) > 0:
                p99 = np.percentile(values, 99)
                filtered = values[values <= p99]
                all_values.extend(filtered.tolist())

        if not all_values:
            plt.close()
            continue

        # Compute common bins for all datasets
        all_values_arr = np.array(all_values)
        x_min, x_max = np.min(all_values_arr), np.max(all_values_arr)

        # Use log-spaced bins for vertices, linear for others
        if use_log:
            # Ensure positive values for log scale
            x_min = max(x_min, 1)
            bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        else:
            bins = np.linspace(x_min, x_max, 50)

        # Find global y-max for consistent y-axis
        y_max = 0
        for stats in all_stats:
            values = np.array(stats[key])
            if len(values) > 0:
                p99 = np.percentile(values, 99)
                filtered = values[values <= p99]
                if use_log:
                    filtered = filtered[filtered > 0]  # Remove zeros for log scale
                counts, _ = np.histogram(filtered, bins=bins, density=True)
                y_max = max(y_max, np.max(counts))
        y_max *= 1.1  # Add 10% padding

        # Plot each dataset in its own subplot
        for i, (name, stats, results) in enumerate(zip(names, all_stats, all_results)):
            ax = axes[i]
            values = np.array(stats[key])

            if len(values) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    name, fontsize=11, fontweight="bold", color=colors[i], pad=15
                )
                continue

            # Filter outliers
            p99 = np.percentile(values, 99)
            filtered = values[values <= p99]
            if use_log:
                filtered = filtered[filtered > 0]  # Remove zeros for log scale

            # Plot histogram
            ax.hist(
                filtered,
                bins=bins,
                alpha=0.7,
                color=colors[i],
                edgecolor="white",
                linewidth=0.5,
                density=True,
            )

            # Set log scale if needed
            if use_log:
                ax.set_xscale("log")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel(title, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(name, fontsize=11, fontweight="bold", color=colors[i], pad=15)
            ax.grid(True, alpha=0.3)

            # Add stats annotation
            r = results[result_key]
            stats_text = f"μ={r['mean']:.2f}\nσ={r['std']:.2f}\nmed={r['median']:.2f}"
            ax.text(
                0.97,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9),
            )

        # Hide unused subplots
        for j in range(n_datasets, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{title} Distribution", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    return output_paths


def plot_density_comparison_multi(
    all_stats,
    all_results,
    names,
    output_dir,
    output_name="mesh_statistics_comparison.png",
):
    """Create a summary table comparing multiple datasets (table only, no plots)."""
    n_datasets = len(names)

    # Adjust figure height based on number of datasets and rows (14 rows: 7 metrics x 2 for mean/max)
    n_metric_rows = 14  # 7 metrics x 2 (mean + max)
    fig_height = 2 + n_metric_rows * 0.4

    fig = plt.figure(figsize=(max(14, 3 + n_datasets * 2), fig_height))
    ax_table = fig.add_subplot(111)
    ax_table.axis("off")

    # Prepare table data with mean AND max rows for each metric
    header = ["Metric", "Stat"] + [name for name in names]
    table_data = [header]

    metric_names = [
        ("Vertices per mesh", "Vertices per mesh", "n_verts"),
        ("Faces per mesh", "Faces per mesh", "n_faces"),
        ("Edges per mesh", "Edges per mesh", "n_edges"),
        ("Edge lengths (Å)", "Edge lengths (Å)", "edge_lengths"),
        ("Face areas (Å²)", "Face areas (Å²)", "face_areas"),
        ("Face angles (°)", "Face angles (°)", "face_angles"),
        ("Dihedral angles (°)", "Dihedral angles (°)", "dihedral_angles"),
    ]

    for display_name, key, stats_key in metric_names:
        # Mean row
        mean_row = [display_name, "Mean"]
        for results in all_results:
            r = results[key]
            mean_row.append(f"{r['mean']:.2f} ± {r['std']:.2f}")
        table_data.append(mean_row)

        # Max row
        max_row = ["", "Max"]
        for results in all_results:
            r = results[key]
            max_row.append(f"{r['max']:.2f}")
        table_data.append(max_row)

    # Calculate column widths
    col_widths = [0.18, 0.08] + [0.74 / n_datasets] * n_datasets

    # Create table
    table = ax_table.table(
        cellText=table_data, cellLoc="center", loc="center", colWidths=col_widths
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Header row styling
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row styling: alternate colors for each metric pair (mean+max)
    for i in range(1, len(table_data)):
        metric_pair_idx = (i - 1) // 2  # Each metric takes 2 rows
        for j in range(len(table_data[0])):
            if metric_pair_idx % 2 == 0:
                table[(i, j)].set_facecolor("#D9E2F3")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")

    # Add title
    fig.suptitle(
        "Mesh Statistics Comparison: Multiple Surface Datasets",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved table to: {output_path}")
    plt.close()
    return output_path


def compute_extreme_value_analysis(all_stats, names, output_dir):
    """
    Compute 5 extreme value analyses:
    1. Cross-max comparison: % of MSMS values > max(Alpha_n) for each Alpha surface
    2. Percentile comparison: P95, P99, P99.9 for each dataset
    3. Tail ratio: % exceeding specific thresholds
    4. Conditional mean of extremes: Mean of top 1% values
    5. IQR-based outlier counts
    """
    # Metrics to analyze (focus on per-mesh metrics, not per-element)
    metrics = [
        ("n_verts", "Vertices per mesh"),
        ("n_faces", "Faces per mesh"),
        ("n_edges", "Edges per mesh"),
        ("edge_lengths", "Edge lengths (Å)"),
        ("face_areas", "Face areas (Å²)"),
        ("face_angles", "Face angles (°)"),
        ("dihedral_angles", "Dihedral angles (°)"),
    ]

    n_datasets = len(names)

    # =========================================================================
    # 1. CROSS-MAX COMPARISON: % of MSMS > max(Alpha_n)
    # Assumes MSMS is the LAST dataset in the list
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 1: Cross-Max Comparison")
    print("% of MSMS values exceeding max(Alpha_n) for each Alpha surface")
    print("=" * 70)

    # MSMS is assumed to be the last dataset
    msms_idx = n_datasets - 1
    msms_name = names[msms_idx]
    msms_stats = all_stats[msms_idx]

    # Alpha surfaces are all datasets except the last one
    alpha_names = names[:-1]
    alpha_stats = all_stats[:-1]

    cross_max_data = []
    for key, display_name in metrics:
        row = {"metric": display_name}
        msms_values = np.array(msms_stats[key])
        if len(msms_values) == 0:
            continue

        for alpha_name, alpha_stat in zip(alpha_names, alpha_stats):
            alpha_values = np.array(alpha_stat[key])
            if len(alpha_values) == 0:
                continue
            max_alpha = np.max(alpha_values)
            pct_exceeding = 100 * np.mean(msms_values > max_alpha)
            row[f"{msms_name} > max({alpha_name})"] = pct_exceeding

        cross_max_data.append(row)

    # Print cross-max table
    if n_datasets >= 2:
        print(f"\n{'Metric':<25}", end="")
        for alpha_name in alpha_names:
            col_name = f"MSMS>max({alpha_name[:10]})"
            print(f" {col_name:<20}", end="")
        print()
        print("-" * (25 + 21 * len(alpha_names)))

        for row in cross_max_data:
            print(f"{row['metric']:<25}", end="")
            for alpha_name in alpha_names:
                key = f"{msms_name} > max({alpha_name})"
                val = row.get(key, 0)
                print(f" {val:<20.2f}%", end="")
            print()

    # =========================================================================
    # 2. PERCENTILE COMPARISON: P95, P99, P99.9
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 2: Percentile Comparison (P95, P99, P99.9)")
    print("=" * 70)

    percentiles_to_compute = [95, 99, 99.9]
    percentile_data = {key: {} for key, _ in metrics}

    for key, display_name in metrics:
        for name, stats in zip(names, all_stats):
            values = np.array(stats[key])
            if len(values) == 0:
                continue
            percentile_data[key][name] = {
                f"P{p}": np.percentile(values, p) for p in percentiles_to_compute
            }

    # Print percentile table
    print(f"\n{'Metric':<22}", end="")
    for name in names:
        for p in percentiles_to_compute:
            print(f" {name[:6]}_P{p:<6}", end="")
    print()
    print("-" * (22 + 14 * len(names) * len(percentiles_to_compute)))

    for key, display_name in metrics:
        print(f"{display_name:<22}", end="")
        for name in names:
            if name in percentile_data[key]:
                for p in percentiles_to_compute:
                    val = percentile_data[key][name][f"P{p}"]
                    print(f" {val:<13.2f}", end="")
            else:
                for p in percentiles_to_compute:
                    print(f" {'N/A':<13}", end="")
        print()

    # =========================================================================
    # 3. TAIL RATIO: % exceeding thresholds
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 3: Tail Ratio Analysis")
    print("% of data exceeding critical thresholds")
    print("=" * 70)

    # Define thresholds based on typical problematic values
    thresholds = {
        "n_verts": [5000, 10000, 20000, 50000],
        "n_faces": [10000, 20000, 40000, 100000],
        "n_edges": [15000, 30000, 60000, 150000],
        "edge_lengths": [5, 10, 15, 20],
        "face_areas": [1, 5, 10, 20],
        "face_angles": [120, 150, 165, 175],  # Angles close to 180 are degenerate
        "dihedral_angles": [120, 150, 165, 175],
    }

    tail_ratio_data = {}
    for key, display_name in metrics:
        tail_ratio_data[key] = {}
        for name, stats in zip(names, all_stats):
            values = np.array(stats[key])
            if len(values) == 0:
                continue
            tail_ratio_data[key][name] = {}
            for thresh in thresholds.get(key, []):
                pct = 100 * np.mean(values > thresh)
                tail_ratio_data[key][name][thresh] = pct

    # Print tail ratio tables per metric
    for key, display_name in metrics:
        if key not in thresholds:
            continue
        print(f"\n{display_name}:")
        print(f"  {'Threshold':<15}", end="")
        for name in names:
            print(f" {name[:12]:<14}", end="")
        print()
        print("  " + "-" * (15 + 15 * len(names)))

        for thresh in thresholds[key]:
            print(f"  > {thresh:<12}", end="")
            for name in names:
                if (
                    name in tail_ratio_data[key]
                    and thresh in tail_ratio_data[key][name]
                ):
                    pct = tail_ratio_data[key][name][thresh]
                    print(f" {pct:<14.2f}%", end="")
                else:
                    print(f" {'N/A':<14}", end="")
            print()

    # =========================================================================
    # 4. CONDITIONAL MEAN OF EXTREMES: Mean of top N% for various thresholds
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 4: Conditional Mean of Extremes")
    print("Mean of top N% worst cases for various thresholds")
    print("=" * 70)

    # Define the percentages to analyze (top N% means we use percentile = 100 - N)
    top_percentages = [100, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    extreme_mean_data = {}

    for key, display_name in metrics:
        extreme_mean_data[key] = {}
        for name, stats in zip(names, all_stats):
            values = np.array(stats[key])
            if len(values) == 0:
                continue

            data_entry = {"overall_mean": np.mean(values)}
            for pct in top_percentages:
                percentile = 100 - pct
                threshold = np.percentile(values, percentile)
                top_values = values[values >= threshold]
                pct_key = f"top_{pct}pct"
                data_entry[f"{pct_key}_mean"] = (
                    np.mean(top_values) if len(top_values) > 0 else float("nan")
                )
                data_entry[f"{pct_key}_count"] = len(top_values)

            extreme_mean_data[key][name] = data_entry

    # Print header
    print(f"\n{'Metric':<22} {'Dataset':<15}", end="")
    print(f" {'Mean':<10}", end="")
    for pct in top_percentages:
        print(f" Top{pct}%".ljust(10), end="")
    print()
    print("-" * (22 + 15 + 10 + 10 * len(top_percentages)))

    for key, display_name in metrics:
        first_row = True
        for name in names:
            if first_row:
                print(f"{display_name:<22}", end="")
                first_row = False
            else:
                print(f"{'':<22}", end="")

            print(f" {name[:14]:<14}", end="")

            if name in extreme_mean_data[key]:
                d = extreme_mean_data[key][name]
                print(f" {d['overall_mean']:<10.2f}", end="")
                for pct in top_percentages:
                    pct_key = f"top_{pct}pct"
                    mean_val = d[f"{pct_key}_mean"]
                    if np.isnan(mean_val):
                        print(f" {'N/A':<9}", end="")
                    else:
                        print(f" {mean_val:<9.2f}", end="")
            else:
                print(f" {'N/A':<10}", end="")
                for _ in top_percentages:
                    print(f" {'N/A':<9}", end="")
            print()

    # =========================================================================
    # 5. IQR-BASED OUTLIER COUNTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 5: IQR-Based Outlier Analysis")
    print("Outliers defined as values > Q3 + 1.5*IQR")
    print("=" * 70)

    outlier_data = {}
    for key, display_name in metrics:
        outlier_data[key] = {}
        for name, stats in zip(names, all_stats):
            values = np.array(stats[key])
            if len(values) == 0:
                continue

            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            outliers = values[values > upper_fence]

            outlier_data[key][name] = {
                "count": len(outliers),
                "pct": 100 * len(outliers) / len(values),
                "upper_fence": upper_fence,
                "max_outlier": np.max(outliers) if len(outliers) > 0 else None,
            }

    print(f"\n{'Metric':<22}", end="")
    for name in names:
        print(f" {name[:8]}_n {name[:8]}_% {name[:8]}_max", end="")
    print()
    print("-" * (22 + 32 * len(names)))

    for key, display_name in metrics:
        print(f"{display_name:<22}", end="")
        for name in names:
            if name in outlier_data[key]:
                d = outlier_data[key][name]
                max_val = f"{d['max_outlier']:.1f}" if d["max_outlier"] else "N/A"
                print(f" {d['count']:<9} {d['pct']:<9.2f}% {max_val:<10}", end="")
            else:
                print(f" {'N/A':<9} {'N/A':<9} {'N/A':<10}", end="")
        print()

    # =========================================================================
    # CREATE PNG TABLES FOR ALL 5 APPROACHES
    # =========================================================================
    create_extreme_value_tables_png(
        all_stats,
        names,
        metrics,
        cross_max_data,
        alpha_names,
        msms_name,
        percentile_data,
        tail_ratio_data,
        thresholds,
        extreme_mean_data,
        outlier_data,
        output_dir,
    )

    return {
        "cross_max": cross_max_data,
        "percentiles": percentile_data,
        "tail_ratio": tail_ratio_data,
        "extreme_mean": extreme_mean_data,
        "outliers": outlier_data,
    }


def create_extreme_value_tables_png(
    all_stats,
    names,
    metrics,
    cross_max_data,
    alpha_names,
    msms_name,
    percentile_data,
    tail_ratio_data,
    thresholds,
    extreme_mean_data,
    outlier_data,
    output_dir,
):
    """Create PNG files for all 5 extreme value analysis tables."""
    n_datasets = len(names)
    n_alpha = len(alpha_names)

    # =========================================================================
    # TABLE 1: Cross-Max Comparison (MSMS > max(Alpha_n))
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(max(10, 3 + n_alpha * 2), 6))
    ax1.axis("off")
    ax1.set_title(
        f"Cross-Max: % of {msms_name} > max(Alpha_n)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    header1 = ["Metric"] + [f"> max({name})" for name in alpha_names]
    data1 = [header1]
    for row in cross_max_data:
        data_row = [row["metric"]]
        for alpha_name in alpha_names:
            key = f"{msms_name} > max({alpha_name})"
            val = row.get(key, 0)
            data_row.append(f"{val:.2f}%")
        data1.append(data_row)

    table1 = ax1.table(
        cellText=data1,
        cellLoc="center",
        loc="center",
        colWidths=[0.25] + [0.75 / n_alpha] * n_alpha,
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.0, 2.0)

    for j in range(len(data1[0])):
        table1[(0, j)].set_facecolor("#4472C4")
        table1[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(data1)):
        for j in range(len(data1[0])):
            table1[(i, j)].set_facecolor("#D9E2F3" if i % 2 == 0 else "#FFFFFF")

    path1 = os.path.join(output_dir, "extreme_1_cross_max.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path1}")

    # =========================================================================
    # TABLE 2: Percentile Comparison
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(max(14, 4 + n_datasets * 3), 6))
    ax2.axis("off")
    ax2.set_title(
        "Percentile Comparison (P95 / P99 / P99.9)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    header2 = ["Metric"] + [f"{name}\nP95 / P99 / P99.9" for name in names]
    data2 = [header2]
    for key, display_name in metrics:
        row = [display_name]
        for name in names:
            if name in percentile_data[key]:
                p = percentile_data[key][name]
                row.append(f"{p['P95']:.1f} / {p['P99']:.1f} / {p['P99.9']:.1f}")
            else:
                row.append("N/A")
        data2.append(row)

    table2 = ax2.table(
        cellText=data2,
        cellLoc="center",
        loc="center",
        colWidths=[0.2] + [0.8 / n_datasets] * n_datasets,
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.0, 1.8)

    for j in range(len(data2[0])):
        table2[(0, j)].set_facecolor("#4472C4")
        table2[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(data2)):
        for j in range(len(data2[0])):
            table2[(i, j)].set_facecolor("#D9E2F3" if i % 2 == 0 else "#FFFFFF")

    path2 = os.path.join(output_dir, "extreme_2_percentiles.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path2}")

    # =========================================================================
    # TABLE 3: Tail Ratio (one table per critical metric)
    # =========================================================================
    key_metrics_for_tail = [
        "n_verts",
        "edge_lengths",
        "face_areas",
        "face_angles",
        "dihedral_angles",
    ]

    for key in key_metrics_for_tail:
        if key not in thresholds:
            continue

        display_name = dict(metrics)[key]
        thresh_list = thresholds[key]

        fig3, ax3 = plt.subplots(figsize=(max(10, 3 + n_datasets * 2), 5))
        ax3.axis("off")
        ax3.set_title(
            f"Tail Ratio: {display_name} (% exceeding threshold)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        header3 = ["Threshold"] + names
        data3 = [header3]
        for thresh in thresh_list:
            row = [f"> {thresh}"]
            for name in names:
                if (
                    name in tail_ratio_data[key]
                    and thresh in tail_ratio_data[key][name]
                ):
                    pct = tail_ratio_data[key][name][thresh]
                    row.append(f"{pct:.2f}%")
                else:
                    row.append("N/A")
            data3.append(row)

        table3 = ax3.table(
            cellText=data3,
            cellLoc="center",
            loc="center",
            colWidths=[0.15] + [0.85 / n_datasets] * n_datasets,
        )
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1.0, 2.0)

        for j in range(len(data3[0])):
            table3[(0, j)].set_facecolor("#4472C4")
            table3[(0, j)].set_text_props(color="white", fontweight="bold")

        for i in range(1, len(data3)):
            for j in range(len(data3[0])):
                table3[(i, j)].set_facecolor("#D9E2F3" if i % 2 == 0 else "#FFFFFF")

        safe_key = key.replace("(", "").replace(")", "").replace(" ", "_")
        path3 = os.path.join(output_dir, f"extreme_3_tail_ratio_{safe_key}.png")
        plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path3}")

    # =========================================================================
    # TABLE 4: Conditional Mean of Extremes (with multiple top N% thresholds)
    # =========================================================================
    # Define the percentages to analyze
    top_percentages = [100, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    # Create a wider figure to accommodate all columns
    n_pct_cols = len(top_percentages)
    fig_width = max(20, 3 + n_pct_cols * 1.2)
    fig_height = max(10, 2 + len(metrics) * n_datasets * 0.4)
    fig4, ax4 = plt.subplots(figsize=(fig_width, fig_height))
    ax4.axis("off")
    ax4.set_title(
        "Conditional Mean of Extremes (Mean of Top N%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Header: Metric | Dataset | Mean | Top10% | Top5% | ... | Top0.0001%
    header4 = ["Metric", "Dataset", "Mean"] + [f"Top{pct}%" for pct in top_percentages]
    data4 = [header4]

    for key, display_name in metrics:
        first_row = True
        for name in names:
            if first_row:
                row = [display_name, name]
                first_row = False
            else:
                row = ["", name]

            if name in extreme_mean_data[key]:
                d = extreme_mean_data[key][name]
                row.append(f"{d['overall_mean']:.2f}")
                for pct in top_percentages:
                    pct_key = f"top_{pct}pct"
                    mean_val = d.get(f"{pct_key}_mean", float("nan"))
                    if np.isnan(mean_val):
                        row.append("N/A")
                    else:
                        row.append(f"{mean_val:.2f}")
            else:
                row.append("N/A")
                row.extend(["N/A"] * n_pct_cols)

            data4.append(row)

    # Calculate column widths
    metric_col_width = 0.12
    dataset_col_width = 0.10
    mean_col_width = 0.06
    pct_col_width = (
        1.0 - metric_col_width - dataset_col_width - mean_col_width
    ) / n_pct_cols
    col_widths = [metric_col_width, dataset_col_width, mean_col_width] + [
        pct_col_width
    ] * n_pct_cols

    table4 = ax4.table(
        cellText=data4, cellLoc="center", loc="center", colWidths=col_widths
    )
    table4.auto_set_font_size(False)
    table4.set_fontsize(6)
    table4.scale(1.0, 1.6)

    # Header row styling
    for j in range(len(data4[0])):
        table4[(0, j)].set_facecolor("#4472C4")
        table4[(0, j)].set_text_props(color="white", fontweight="bold")

    # Data row styling: alternate colors by metric group
    current_metric = 0
    for i in range(1, len(data4)):
        # Check if this is a new metric (has non-empty first column)
        if data4[i][0] != "":
            current_metric += 1
        for j in range(len(data4[0])):
            table4[(i, j)].set_facecolor(
                "#D9E2F3" if current_metric % 2 == 0 else "#FFFFFF"
            )

    path4 = os.path.join(output_dir, "extreme_4_conditional_mean.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path4}")

    # =========================================================================
    # PLOT 4b: Conditional Mean Curves (log scale on x-axis)
    # =========================================================================
    # Generate colors from a colormap
    cmap = cm.get_cmap("tab10") if n_datasets <= 10 else cm.get_cmap("tab20")
    colors = [cmap(i) for i in range(n_datasets)]

    # Create a grid of subplots, one per metric
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig_plot, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (key, display_name) in enumerate(metrics):
        ax = axes[idx]

        for i, name in enumerate(names):
            if name not in extreme_mean_data[key]:
                continue

            d = extreme_mean_data[key][name]

            # Extract conditional means for each percentage
            y_values = []
            x_values = []
            for pct in top_percentages:
                pct_key = f"top_{pct}pct"
                mean_val = d.get(f"{pct_key}_mean", float("nan"))
                if not np.isnan(mean_val):
                    x_values.append(pct)
                    y_values.append(mean_val)

            if len(x_values) > 0:
                ax.plot(
                    x_values,
                    y_values,
                    "o-",
                    label=name,
                    color=colors[i],
                    linewidth=2,
                    markersize=5,
                    alpha=0.8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Top N% (log scale)", fontsize=9)
        ax.set_ylabel("Conditional Mean", fontsize=9)
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.invert_xaxis()  # Smaller percentages (more extreme) on the right
        ax.legend(fontsize=7, loc="best")
        # Add minor gridlines for log scale
        ax.grid(True, which="minor", alpha=0.15)

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    fig_plot.suptitle(
        "Conditional Mean of Extremes: How Mean Changes at Tail Percentiles",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path4_plot = os.path.join(output_dir, "extreme_4_conditional_mean_curves.png")
    plt.savefig(path4_plot, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path4_plot}")

    # =========================================================================
    # TABLE 5: Outlier Analysis
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(max(14, 4 + n_datasets * 3), 6))
    ax5.axis("off")
    ax5.set_title(
        "IQR-Based Outlier Analysis (Count / % / Max)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    header5 = ["Metric"] + [f"{name}\nCount / % / Max" for name in names]
    data5 = [header5]
    for key, display_name in metrics:
        row = [display_name]
        for name in names:
            if name in outlier_data[key]:
                d = outlier_data[key][name]
                max_val = f"{d['max_outlier']:.1f}" if d["max_outlier"] else "N/A"
                row.append(f"{d['count']} / {d['pct']:.1f}% / {max_val}")
            else:
                row.append("N/A")
        data5.append(row)

    table5 = ax5.table(
        cellText=data5,
        cellLoc="center",
        loc="center",
        colWidths=[0.2] + [0.8 / n_datasets] * n_datasets,
    )
    table5.auto_set_font_size(False)
    table5.set_fontsize(8)
    table5.scale(1.0, 1.8)

    for j in range(len(data5[0])):
        table5[(0, j)].set_facecolor("#4472C4")
        table5[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(data5)):
        for j in range(len(data5[0])):
            table5[(i, j)].set_facecolor("#D9E2F3" if i % 2 == 0 else "#FFFFFF")

    path5 = os.path.join(output_dir, "extreme_5_outliers.png")
    plt.savefig(path5, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path5}")

    print(f"\nSaved all extreme value analysis tables to: {output_dir}")


def plot_individual_datasets(all_stats, all_results, names, output_dir):
    """Create individual plots for each dataset (one plot per dataset with 5 metrics - excluding faces and edges)."""
    # (stats_key, display_title, results_key)
    metrics = [
        ("n_verts", "Vertices per Mesh", "Vertices per mesh"),
        ("edge_lengths", "Edge Lengths (Å)", "Edge lengths (Å)"),
        ("face_areas", "Face Areas (Å²)", "Face areas (Å²)"),
        ("face_angles", "Face Angles (°)", "Face angles (°)"),
        ("dihedral_angles", "Dihedral Angles (°)", "Dihedral angles (°)"),
    ]

    output_paths = []

    for i, (name, stats, results) in enumerate(zip(names, all_stats, all_results)):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Use a consistent color for this dataset
        color = "#2ca02c"

        for j, (ax, (key, title, result_key)) in enumerate(
            zip(axes[: len(metrics)], metrics)
        ):
            values = np.array(stats[key])

            if len(values) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Filter outliers for visualization
            p99 = np.percentile(values, 99)
            filtered = values[values <= p99]

            # Use log scale for vertices
            if key == "n_verts":
                filtered = filtered[filtered > 0]  # Remove zeros for log scale
                bins = np.logspace(
                    np.log10(np.min(filtered)), np.log10(np.max(filtered)), 50
                )
            else:
                bins = 50

            # Use histogram with bins
            ax.hist(
                filtered,
                bins=bins,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                density=True,
            )

            # Set log scale for vertices
            if key == "n_verts":
                ax.set_xscale("log")

            ax.set_xlabel(title, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
            ax.grid(True, alpha=0.3)

            # Add stats annotation
            r = results[result_key]
            stats_text = (
                f"Mean: {r['mean']:.2f}\nStd: {r['std']:.2f}\nMedian: {r['median']:.2f}"
            )
            ax.text(
                0.97,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9),
            )

        # Hide unused subplot
        if len(axes) > len(metrics):
            axes[-1].axis("off")

        fig.suptitle(f"Mesh Statistics: {name}", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        # Sanitize filename
        safe_name = name.replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(output_dir, f"mesh_statistics_{safe_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Mesh Statistics Analysis for Multiple Datasets"
    )
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files to process per dataset (None = all)",
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        default=None,
        help="List of directories to analyze",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each directory (must match --dirs length)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for plots"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mesh_statistics_comparison.png",
        help="Output filename for the comparison plot",
    )

    args = parser.parse_args()

    # Use defaults if no directories specified
    if args.dirs is None:
        dirs = DEFAULT_DIRS
        names = DEFAULT_NAMES
    else:
        dirs = args.dirs
        # Auto-generate names from directory basenames if not provided
        if args.names is None:
            names = [Path(d).name for d in dirs]
        else:
            if len(args.names) != len(args.dirs):
                raise ValueError(
                    f"--names count ({len(args.names)}) must match --dirs count ({len(args.dirs)})"
                )
            names = args.names

    print("=" * 70)
    print("MESH STATISTICS ANALYSIS: Multiple Surface Datasets")
    print("=" * 70)
    print(f"Workers: {args.n_workers}")
    print(f"Max files: {args.max_files if args.max_files else 'all'}")
    print(f"Datasets: {len(dirs)}")
    for i, (d, n) in enumerate(zip(dirs, names)):
        print(f"  [{i + 1}] {n}: {d}")

    # Analyze all datasets
    all_stats = []
    all_results = []
    for directory, name in zip(dirs, names):
        if not os.path.isdir(directory):
            print(f"\nWarning: Directory not found, skipping: {directory}")
            continue

        stats = analyze_dataset(
            directory, name, max_files=args.max_files, n_workers=args.n_workers
        )
        all_stats.append(stats)
        results = print_summary_stats(stats, name)
        all_results.append(results)

    if len(all_stats) < 2:
        print("\nError: Need at least 2 valid datasets to compare")
        return

    # Create comparison plots
    os.makedirs(args.output_dir, exist_ok=True)

    # Combined plot (original)
    plot_path = plot_density_comparison_multi(
        all_stats, all_results, names, args.output_dir, args.output_name
    )

    # Extreme value analysis (5 approaches)
    print("\n" + "=" * 70)
    print("EXTREME VALUE ANALYSIS")
    print("=" * 70)
    compute_extreme_value_analysis(all_stats, names, args.output_dir)

    # Per-metric plots (one plot per metric with all datasets overlaid)
    print("\nGenerating per-metric plots (one per metric with all datasets)...")
    plot_per_metric(all_stats, all_results, names, args.output_dir)

    # Individual dataset plots
    print("\nGenerating individual dataset plots...")
    plot_individual_datasets(all_stats, all_results, names, args.output_dir)

    # Print key comparison table
    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    key_metrics = ["Vertices per mesh", "Edge lengths (Å)", "Face areas (Å²)"]

    # Print header
    header = f"{'Metric':<25}"
    for name in names:
        header += f" {name[:15]:<17}"
    print(header)
    print("-" * (25 + 18 * len(names)))

    for metric in key_metrics:
        row = f"{metric:<25}"
        for results in all_results:
            val = results[metric]["mean"]
            row += f" {val:<17.2f}"
        print(row)

    # Print recommended diffusion times relative to first dataset
    print("\n" + "=" * 70)
    print("RECOMMENDED DIFFUSION TIME ADJUSTMENT")
    print("=" * 70)
    baseline_time = 2.0  # Standard diffusion time
    baseline_edge = all_results[0]["Edge lengths (Å)"]["mean"]
    print(f"Using {names[0]} as baseline (init_time={baseline_time})")
    print(f"Baseline avg edge length: {baseline_edge:.3f} Å\n")

    for i, (name, results) in enumerate(zip(names, all_results)):
        edge_len = results["Edge lengths (Å)"]["mean"]
        recommended_time = baseline_time * (baseline_edge / edge_len)
        print(
            f"{name}: avg_edge={edge_len:.3f} Å, recommended init_time={recommended_time:.2f}"
        )


if __name__ == "__main__":
    main()
