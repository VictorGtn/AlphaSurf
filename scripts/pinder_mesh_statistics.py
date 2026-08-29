#!/usr/bin/env python3
"""Compare mesh geometry across cached Pinder surface methods."""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from tqdm import tqdm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METHODS = (
    ("msms", "MSMS Simplified"),
    ("algo2", "Alpha Complex"),
    ("nanoshaper_0.5", "NanoShaper gs = 0.5"),
    ("edtsurf_0.5", "EDTSurf gs = 0.5"),
)
TOP_PERCENTAGES = (100, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
ANGLE_METRICS = ("face_angles", "dihedral_angles")
METRICS = (
    ("n_verts", "Vertices per mesh", True),
    ("n_faces", "Faces per mesh", True),
    ("edge_lengths", "Edge length (Å)", False),
    ("face_areas", "Face area (Å²)", False),
    ("face_angles", "Triangle angle (°)", False),
    ("dihedral_angles", "Dihedral angle (°)", False),
)


def _triangle_angles(vertices, faces):
    triangles = vertices[faces]
    angles = []
    for vertex in range(3):
        edge_a = triangles[:, (vertex + 1) % 3] - triangles[:, vertex]
        edge_b = triangles[:, (vertex + 2) % 3] - triangles[:, vertex]
        denominator = np.linalg.norm(edge_a, axis=1) * np.linalg.norm(edge_b, axis=1)
        valid = denominator > 0
        cosine = (
            np.einsum("ij,ij->i", edge_a[valid], edge_b[valid]) / denominator[valid]
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.concatenate(angles)


def _mesh_metrics(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid vertex shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"invalid face shape {faces.shape}")

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    triangles = vertices[faces]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    face_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    dihedral_angles = np.degrees(mesh.face_adjacency_angles)

    return {
        "n_verts": np.array([len(vertices)], dtype=np.float64),
        "n_faces": np.array([len(faces)], dtype=np.float64),
        "edge_lengths": edge_lengths,
        "face_areas": face_areas,
        "face_angles": _triangle_angles(vertices, faces),
        "dihedral_angles": dihedral_angles,
    }


def _analyze_npz(task):
    npz_path, msms_reduction_rate = task
    try:
        from alphasurf.protein.create_surface import mesh_simplification

        with np.load(npz_path, allow_pickle=False) as data:
            required = [
                f"{key}_{field}" for key, _ in METHODS for field in ("verts", "faces")
            ]
            missing = [key for key in required if key not in data]
            if missing:
                return npz_path.stem, None, f"missing {', '.join(missing)}"

            meshes = {
                key: (data[f"{key}_verts"], data[f"{key}_faces"]) for key, _ in METHODS
            }

        alpha_no_tufting = mesh_simplification(
            *meshes["algo2"],
            out_ply=None,
            face_reduction_rate=1.0,
            min_vert_number=16,
            max_vert_number=100000,
            use_pymesh=False,
            surface_method="alpha_complex",
            tufting=False,
            allow_multiple_components=True,
        )[:2]
        msms_vertices, msms_faces = meshes["msms"]
        meshes["msms"] = mesh_simplification(
            msms_vertices,
            msms_faces,
            out_ply=None,
            face_reduction_rate=msms_reduction_rate,
            min_vert_number=16,
            max_vert_number=100000,
            use_pymesh=False,
            surface_method="msms",
            allow_multiple_components=True,
        )[:2]
        method_stats = {key: _mesh_metrics(*meshes[key]) for key, _ in METHODS}
        no_tufting_stats = _mesh_metrics(*alpha_no_tufting)
        method_stats["algo2_no_tufting"] = {
            metric_key: no_tufting_stats[metric_key] for metric_key in ANGLE_METRICS
        }
        return (
            npz_path.stem,
            method_stats,
            None,
        )
    except Exception as error:
        return npz_path.stem, None, f"{type(error).__name__}: {error}"


class MetricAccumulator:
    def __init__(self, sample_size, rng):
        self.sample_size = sample_size
        self.rng = rng
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.values = np.empty(0, dtype=np.float64)
        self.priorities = np.empty(0, dtype=np.float64)
        self.pending_values = []
        self.pending_priorities = []
        self.pending_count = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return

        self.count += len(values)
        self.total += float(values.sum(dtype=np.float64))
        self.total_squared += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

        self.pending_values.append(values)
        self.pending_priorities.append(self.rng.random(len(values)))
        self.pending_count += len(values)
        if self.pending_count >= self.sample_size:
            self._compact()

    def _compact(self):
        if self.pending_count == 0:
            return
        combined_values = np.concatenate((self.values, *self.pending_values))
        combined_priorities = np.concatenate(
            (self.priorities, *self.pending_priorities)
        )
        if len(combined_values) > self.sample_size:
            keep = np.argpartition(combined_priorities, -self.sample_size)[
                -self.sample_size :
            ]
            combined_values = combined_values[keep]
            combined_priorities = combined_priorities[keep]
        self.values = combined_values
        self.priorities = combined_priorities
        self.pending_values.clear()
        self.pending_priorities.clear()
        self.pending_count = 0

    def summary(self):
        if self.count == 0:
            raise ValueError("cannot summarize an empty metric")
        self._compact()
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        percentiles = np.percentile(self.values, [1, 5, 50, 95, 99])
        return {
            "count": self.count,
            "mean": mean,
            "std": np.sqrt(variance),
            "min": self.minimum,
            "p01": percentiles[0],
            "p05": percentiles[1],
            "median": percentiles[2],
            "p95": percentiles[3],
            "p99": percentiles[4],
            "max": self.maximum,
        }


def _write_summary(accumulators, output_path):
    fields = (
        "method",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "median",
        "p95",
        "p99",
        "max",
    )
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for method_key, method_label in METHODS:
            for metric_key, metric_label, _ in METRICS:
                writer.writerow(
                    {
                        "method": method_label,
                        "metric": metric_label,
                        **accumulators[method_key][metric_key].summary(),
                    }
                )


def _plot_distributions(
    accumulators,
    png_path,
    pdf_path,
):
    colors = ("#6A3D9A", "#E41A1C", "#238B45", "#08519C")
    panel_labels = "abcdef"
    panel_titles = (
        "Mesh vertices",
        "Mesh faces",
        "Edge lengths",
        "Triangle areas",
        "Triangle angles",
        "Dihedral angles",
    )
    style = {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    }

    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.8))

        for panel_index, (
            axis,
            (metric_key, metric_label, log_scale),
            panel_title,
        ) in enumerate(zip(axes.flat, METRICS, panel_titles)):
            if metric_key in ("face_angles", "dihedral_angles"):
                for (method_key, _), color in zip(METHODS, colors):
                    angle_method_key = (
                        "algo2_no_tufting" if method_key == "algo2" else method_key
                    )
                    means = _conditional_means(
                        accumulators[angle_method_key][metric_key]
                    )
                    axis.plot(
                        TOP_PERCENTAGES,
                        means,
                        color=color,
                        linewidth=1.25,
                        alpha=0.82,
                    )
                axis.set_xscale("log")
                axis.invert_xaxis()
                axis.set_xticks((100, 10, 1, 0.1, 0.01, 0.001))
                axis.set_xticklabels(("100", "10", "1", "0.1", "0.01", "0.001"))
                axis.set_xlabel("Top N (%)")
                axis.set_ylabel("Conditional mean (°)")
                axis.grid(
                    axis="both",
                    which="major",
                    color="#D9D9D9",
                    linewidth=0.45,
                )
                axis.grid(
                    axis="x",
                    which="minor",
                    color="#E8E8E8",
                    linewidth=0.3,
                )
            else:
                samples = [
                    accumulators[method_key][metric_key].values
                    for method_key, _ in METHODS
                ]
                pooled = np.concatenate(samples)
                if log_scale:
                    positive = pooled[pooled > 0]
                    lower, upper = positive.min(), positive.max()
                    bins = np.geomspace(lower, upper, 32)
                    axis.set_xscale("log")
                else:
                    lower, upper = np.percentile(pooled, (0.5, 99.5))
                    bins = np.linspace(lower, upper, 51)

                for values, color in zip(samples, colors):
                    counts, edges = np.histogram(values, bins=bins)
                    plotted = counts / (len(values) * np.diff(edges))
                    axis.stairs(
                        plotted,
                        edges,
                        color=color,
                        linewidth=0,
                        fill=True,
                        alpha=0.045,
                    )
                    axis.stairs(
                        plotted,
                        edges,
                        color=color,
                        linewidth=1.25,
                        alpha=0.82,
                    )

                axis.set_xlim(lower, upper)
                axis.set_xlabel(metric_label)
                axis.set_ylabel("Density")
                axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
            axis.set_title(panel_title, pad=4)
            axis.text(
                -0.17,
                1.04,
                f"({panel_labels[panel_index]})",
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
                va="bottom",
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out", length=3, width=0.7)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.6,
                alpha=0.82,
                label=label,
            )
            for (_, label), color in zip(METHODS, colors)
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=4,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.5,
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.99,
            bottom=0.10,
            top=0.87,
            wspace=0.34,
            hspace=0.42,
        )
        figure.savefig(png_path, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def _conditional_means(accumulator):
    values = accumulator.values
    means = []
    for percentage in TOP_PERCENTAGES:
        if percentage == 100:
            means.append(accumulator.total / accumulator.count)
            continue
        threshold = np.percentile(values, 100.0 - percentage)
        tail = values[values >= threshold]
        means.append(float(tail.mean()))
    return np.asarray(means)


def _write_conditional_angle_means(accumulators, output_path):
    fields = ("method", "metric", "top_percentage", "conditional_mean")
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for method_key, method_label in METHODS:
            for metric_key, metric_label in (
                ("face_angles", "Triangle angle (°)"),
                ("dihedral_angles", "Dihedral angle (°)"),
            ):
                angle_method_key = (
                    "algo2_no_tufting" if method_key == "algo2" else method_key
                )
                means = _conditional_means(accumulators[angle_method_key][metric_key])
                for percentage, mean in zip(TOP_PERCENTAGES, means):
                    writer.writerow(
                        {
                            "method": method_label,
                            "metric": metric_label,
                            "top_percentage": percentage,
                            "conditional_mean": mean,
                        }
                    )


def _select_timing_sample(pdb_dir, npz_dir, sample_lr, seed):
    pdb_paths = list(pdb_dir.glob("*.pdb"))
    left = sorted(path for path in pdb_paths if path.stem.endswith("_L"))
    right = sorted(path for path in pdb_paths if path.stem.endswith("_R"))
    if len(left) < sample_lr or len(right) < sample_lr:
        raise ValueError(
            f"requested {sample_lr} PDBs per side, found "
            f"{len(left)} L and {len(right)} R"
        )

    rng = np.random.default_rng(seed)
    selected_left = rng.choice(left, size=sample_lr, replace=False)
    selected_right = rng.choice(right, size=sample_lr, replace=False)
    selected = sorted((*selected_left, *selected_right), key=str)
    return selected, [npz_dir / f"{path.stem}.npz" for path in selected]


def _write_timing_manifest(pdb_paths, seed, output_path):
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("pdb", "side", "seed"))
        for pdb_path in pdb_paths:
            side = pdb_path.stem.rsplit("_", 1)[-1]
            writer.writerow((pdb_path.name, side, seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path(__file__).parent / "cc_sweep_output" / "surfaces",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--msms-reduction-rate", type=float, default=0.1)
    parser.add_argument(
        "--timing-sample-lr",
        type=int,
        help="Use the timing benchmark selection with this many L and R meshes",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pinder-pair" / "pdb",
    )
    parser.add_argument(
        "--distribution-sample-size",
        type=int,
        default=100000,
        help="Maximum retained values per method and metric for plots and percentiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "pinder_mesh_statistics",
    )
    args = parser.parse_args()

    if not 0 < args.msms_reduction_rate <= 1:
        parser.error("--msms-reduction-rate must be in (0, 1]")
    if args.sample_size < 1 or args.distribution_sample_size < 1:
        parser.error("sample sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_pdbs = None
    if args.timing_sample_lr is not None:
        if args.timing_sample_lr < 1:
            parser.error("--timing-sample-lr must be positive")
        try:
            timing_pdbs, candidates = _select_timing_sample(
                args.pdb_dir,
                args.npz_dir,
                args.timing_sample_lr,
                args.seed,
            )
        except ValueError as error:
            parser.error(str(error))
        _write_timing_manifest(
            timing_pdbs,
            args.seed,
            args.output_dir / "timing_sample_manifest.csv",
        )
    else:
        candidates = list(args.npz_dir.glob("*.npz"))
        if len(candidates) < args.sample_size:
            parser.error(
                f"found only {len(candidates)} NPZ files in {args.npz_dir}, "
                f"fewer than the requested {args.sample_size}"
            )
        np.random.default_rng(args.seed).shuffle(candidates)

    error_path = args.output_dir / "skipped_meshes.tsv"
    manifest_path = args.output_dir / "sampled_meshes.txt"
    rng = np.random.default_rng(args.seed)
    accumulators = {
        method_key: {
            metric_key: MetricAccumulator(args.distribution_sample_size, rng)
            for metric_key, _, _ in METRICS
        }
        for method_key, _ in METHODS
    }
    accumulators["algo2_no_tufting"] = {
        metric_key: MetricAccumulator(args.distribution_sample_size, rng)
        for metric_key in ANGLE_METRICS
    }

    accepted = []
    tasks = ((path, args.msms_reduction_rate) for path in candidates)
    pool = mp.Pool(args.workers)
    try:
        with error_path.open("w") as error_file:
            error_file.write("mesh\terror\n")
            results = pool.imap(_analyze_npz, tasks, chunksize=1)
            if timing_pdbs is None:
                progress = tqdm(total=args.sample_size, desc="Accepted Pinder meshes")
                result_iterator = results
            else:
                progress = tqdm(
                    results,
                    total=len(candidates),
                    desc="Timing sample meshes",
                )
                result_iterator = progress

            for stem, method_stats, error in result_iterator:
                if error is not None:
                    error_file.write(f"{stem}\t{error}\n")
                    continue
                accepted.append(stem)
                for method_key, _ in METHODS:
                    for metric_key, _, _ in METRICS:
                        accumulators[method_key][metric_key].update(
                            method_stats[method_key][metric_key]
                        )
                for metric_key in ANGLE_METRICS:
                    accumulators["algo2_no_tufting"][metric_key].update(
                        method_stats["algo2_no_tufting"][metric_key]
                    )
                if timing_pdbs is None:
                    progress.update()
                    if len(accepted) == args.sample_size:
                        break
            progress.close()
    finally:
        pool.terminate()
        pool.join()

    if timing_pdbs is None and len(accepted) != args.sample_size:
        raise RuntimeError(
            f"only {len(accepted)} complete meshes were found; "
            f"see {error_path} for failures"
        )
    if timing_pdbs is not None and len(accepted) != len(candidates):
        print(
            f"Using {len(accepted)} of {len(candidates)} timing meshes with all methods; "
            f"see {error_path} for missing caches"
        )

    manifest_path.write_text("\n".join(accepted) + "\n")
    summary_path = args.output_dir / "mesh_statistics_summary.csv"
    plot_path = args.output_dir / "mesh_statistics_distributions.png"
    plot_pdf_path = args.output_dir / "mesh_statistics_distributions.pdf"
    conditional_path = args.output_dir / "mesh_statistics_conditional_angle_means.csv"
    _write_summary(accumulators, summary_path)
    _plot_distributions(
        accumulators,
        plot_path,
        plot_pdf_path,
    )
    _write_conditional_angle_means(accumulators, conditional_path)
    print(f"Sample manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Distributions: {plot_path}")
    print(f"Vector distributions: {plot_pdf_path}")
    print(f"Conditional angle means (CSV): {conditional_path}")


if __name__ == "__main__":
    main()
