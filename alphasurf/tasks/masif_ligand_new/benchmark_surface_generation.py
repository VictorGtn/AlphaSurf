#!/usr/bin/env python3
"""Benchmark MaSIF-ligand surface preprocessing one protein at a time."""

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alphasurf.protein.create_operators import compute_operators, vertex_normals  # noqa: E402
from alphasurf.protein.create_surface import (  # noqa: E402
    mesh_simplification,
    pdb_to_alpha_complex,
    pdb_to_edtsurf,
    pdb_to_nanoshaper,
    pdb_to_surf_with_min,
)
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402
from alphasurf.protein.surfaces import SurfaceObject  # noqa: E402


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    grid_scale: float | None = None
    face_reduction_rate: float = 1.0
    engine: str | None = None


METHODS = (
    Method("alpha_complex", "alpha-complex"),
    Method("edtsurf", "EDTSurf", 0.3),
    Method("edtsurf", "EDTSurf", 0.4),
    Method("edtsurf", "EDTSurf", 0.5),
    Method("edtsurf", "EDTSurf", 2.0),
    Method("nanoshaper", "NanoShaper", 0.4),
    Method("nanoshaper", "NanoShaper", 0.5),
    Method("nanoshaper", "NanoShaper", 2.0),
    Method("msms", "MSMS", face_reduction_rate=0.1),
    Method("edtsurf_default", "EDTSurf", 4.0, engine="edtsurf"),
    Method("nanoshaper_default", "NanoShaper", 2.0, engine="nanoshaper"),
    Method("msms_full", "MSMS (full)", face_reduction_rate=1.0, engine="msms"),
    Method("dmasif", "dMaSIF"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serial surface-generation benchmark for MaSIF-ligand"
    )
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-proteins", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument(
        "--methods",
        default=",".join(dict.fromkeys(method.name for method in METHODS)),
        help="Comma-separated method names; selected grid-scale variants are included",
    )
    parser.add_argument(
        "--edtsurf-scales",
        help="Optional comma-separated EDTSurf scales to include",
    )
    parser.add_argument(
        "--nanoshaper-scales",
        help="Optional comma-separated NanoShaper scales to include",
    )
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atom_types_to_dmasif_onehot(atom_types):
    source = torch.as_tensor(atom_types, dtype=torch.long)
    target = torch.zeros_like(source)
    mapping = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 8: 5}
    for source_index, target_index in mapping.items():
        target[source == source_index] = target_index
    return torch.nn.functional.one_hot(target, num_classes=6).float()


def parse_protein(pdb_path):
    start = time.perf_counter()
    arrays = parse_pdb_path(str(pdb_path), use_pqr=False)
    elapsed = time.perf_counter() - start
    if arrays is None or arrays[5] is None or arrays[7] is None:
        raise RuntimeError("PDB parsing returned no atom coordinates or radii")
    return arrays, elapsed


def generate_mesh(method, pdb_path, arrays, alpha_value):
    atom_pos = arrays[5]
    atom_radius = arrays[7]
    if method.name == "alpha_complex":
        return pdb_to_alpha_complex(
            str(pdb_path),
            alpha_value=alpha_value,
            atom_pos=atom_pos,
            atom_radius=atom_radius,
        )
    engine = method.engine or method.name
    if engine == "edtsurf":
        return pdb_to_edtsurf(str(pdb_path), grid_scale=method.grid_scale)
    if engine == "nanoshaper":
        return pdb_to_nanoshaper(
            str(pdb_path),
            grid_scale=method.grid_scale,
            atom_pos=atom_pos,
            atom_radius=atom_radius,
        )
    if engine == "msms":
        return pdb_to_surf_with_min(str(pdb_path), min_number=256)
    raise ValueError(f"Unsupported mesh method: {method.name}")


def generate_dmasif(arrays, device):
    from alphasurf.network_utils.misc_arch.dmasif_utils.geometry_processing import (
        atoms_to_points_normals,
    )

    atom_pos = torch.as_tensor(arrays[5], dtype=torch.float32, device=device)
    atom_types = atom_types_to_dmasif_onehot(arrays[4]).to(device)
    batch = torch.zeros(len(atom_pos), dtype=torch.long, device=device)
    points, normals, _ = atoms_to_points_normals(
        atom_pos,
        batch,
        num_atoms=6,
        distance=1.05,
        smoothness=0.5,
        resolution=1.0,
        nits=4,
        atomtypes=atom_types,
        sup_sampling=20,
        variance=0.1,
    )
    return points, normals


def empty_row(pdb_path, method):
    return {
        "pdb": pdb_path.stem,
        "method": method.name,
        "label": method.label,
        "grid_scale": "" if method.grid_scale is None else method.grid_scale,
        "status": "error",
        "error": "",
        "n_vertices": 0,
        "n_faces": 0,
        "t_parse_s": math.nan,
        "t_surface_s": math.nan,
        "t_mesh_s": math.nan,
        "t_normals_s": math.nan,
        "t_operators_s": math.nan,
        "t_operators_full_s": math.nan,
        "t_geom_feats_s": math.nan,
        "t_pipeline_s": math.nan,
    }


def benchmark_one(pdb_path, method, args, device, sample_index):
    row = empty_row(pdb_path, method)
    total_start = time.perf_counter()
    try:
        arrays, row["t_parse_s"] = parse_protein(pdb_path)
        torch.manual_seed(args.seed + sample_index)

        if method.name == "dmasif":
            synchronize(device)
            start = time.perf_counter()
            points, _ = generate_dmasif(arrays, device)
            synchronize(device)
            row["t_surface_s"] = time.perf_counter() - start
            row["n_vertices"] = len(points)
            row["status"] = "success"
            row["t_pipeline_s"] = time.perf_counter() - total_start
            return row

        start = time.perf_counter()
        verts, faces = generate_mesh(method, pdb_path, arrays, args.alpha_value)
        row["t_surface_s"] = time.perf_counter() - start
        row["n_vertices"] = len(verts)
        row["n_faces"] = len(faces)

        if not args.skip_pipeline:
            start = time.perf_counter()
            verts, faces, _, _ = mesh_simplification(
                verts,
                faces,
                out_ply=None,
                face_reduction_rate=method.face_reduction_rate,
                min_vert_number=16,
                max_vert_number=100000,
                use_pymesh=False,
                surface_method=method.engine or method.name,
                allow_multiple_components=True,
            )
            row["t_mesh_s"] = time.perf_counter() - start
            row["n_vertices"] = len(verts)
            row["n_faces"] = len(faces)

            start = time.perf_counter()
            normals = vertex_normals(verts, faces, use_igl=False)
            row["t_normals_s"] = time.perf_counter() - start
            start = time.perf_counter()
            _, mass, L, evals, evecs, gradX, gradY = compute_operators(
                verts, faces, normals=normals
            )
            row["t_operators_s"] = time.perf_counter() - start
            row["t_operators_full_s"] = row["t_normals_s"] + row["t_operators_s"]

            surface = SurfaceObject(
                verts=verts,
                faces=faces,
                mass=mass,
                L=L,
                evals=evals,
                evecs=evecs,
                gradX=gradX,
                gradY=gradY,
                vnormals=normals,
            )
            start = time.perf_counter()
            surface.add_geom_feats()
            row["t_geom_feats_s"] = time.perf_counter() - start

        row["status"] = "success"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["t_pipeline_s"] = time.perf_counter() - total_start
    return row


def method_key(row):
    grid_scale = row["grid_scale"]
    return row["method"], None if grid_scale == "" else float(grid_scale)


def mean_ms(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return 1000 * values.mean() if len(values) else math.nan


def mean_value(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def latex_value(value):
    return "--" if not math.isfinite(value) else f"{value:.1f}"


def write_breakdown_table(summary_rows, output_dir):
    stages = (
        ("PDB parsing", "mean_parse_ms"),
        ("Surface generation", "mean_surface_ms"),
        ("Mesh processing / simplification", "mean_mesh_ms"),
        ("Vertex normals", "mean_normals_ms"),
        ("Spectral operators", "mean_operators_ms"),
        ("Normals + spectral operators", "mean_operators_full_ms"),
        ("Geometric features", "mean_geom_feats_ms"),
        ("Total preprocessing", "mean_pipeline_ms"),
    )
    body = []
    for label, field in stages:
        values = " & ".join(latex_value(row[field]) for row in summary_rows)
        body.append(f"        {label} & {values} \\\\")

    table = (
        """\\begin{table}[H]
    \\centering
    \\caption{Average MaSIF-ligand preprocessing time per protein by stage. Times are reported in milliseconds; lower values indicate faster preprocessing.}
    \\label{tab:surface_preprocessing_breakdown}

    \\resizebox{\\columnwidth}{!}{%
    \\begin{tabular}{lcccccccc}
        \\toprule
        & $\\alpha$-complex
        & \\multicolumn{3}{c}{EDTSurf}
        & \\multicolumn{2}{c}{NanoShaper}
        & MSMS
        & dMaSIF \\\\
        \\cmidrule(lr){3-5}
        \\cmidrule(lr){6-7}
        \\textbf{Grid scale}
        & -- & 0.3 & 0.4 & 0.5 & 0.4 & 0.5 & -- & -- \\\\
        \\midrule
"""
        + "\n".join(body)
        + """
        \\bottomrule
    \\end{tabular}%
    }
\\end{table}
"""
    )
    path = output_dir / "preprocessing_breakdown_table.tex"
    path.write_text(table)
    return path


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "surface_generation_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in METHODS:
        attempted = [
            row for row in rows if method_key(row) == (method.name, method.grid_scale)
        ]
        surface_successes = [
            row for row in attempted if math.isfinite(float(row["t_surface_s"]))
        ]
        completed = [row for row in attempted if row["status"] == "success"]
        times = np.asarray([row["t_surface_s"] for row in surface_successes])
        summary_rows.append(
            {
                "method": method.name,
                "label": method.label,
                "grid_scale": "" if method.grid_scale is None else method.grid_scale,
                "attempted": len(attempted),
                "surface_successes": len(surface_successes),
                "completed": len(completed),
                "pipeline_errors": sum(row["status"] != "success" for row in attempted),
                "mean_parse_ms": mean_ms(completed, "t_parse_s"),
                "mean_surface_ms": 1000 * times.mean() if len(times) else math.nan,
                "std_surface_ms": 1000 * times.std() if len(times) else math.nan,
                "mean_mesh_ms": mean_ms(completed, "t_mesh_s"),
                "mean_normals_ms": mean_ms(completed, "t_normals_s"),
                "mean_operators_ms": mean_ms(completed, "t_operators_s"),
                "mean_operators_full_ms": mean_ms(completed, "t_operators_full_s"),
                "mean_geom_feats_ms": mean_ms(completed, "t_geom_feats_s"),
                "mean_pipeline_ms": mean_ms(completed, "t_pipeline_s"),
                "mean_vertices": mean_value(completed, "n_vertices"),
                "std_vertices": float(
                    np.asarray(
                        [row["n_vertices"] for row in completed], dtype=float
                    ).std()
                )
                if completed
                else math.nan,
            }
        )

    summary_path = output_dir / "surface_generation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    values = [latex_value(row["mean_surface_ms"]) for row in summary_rows]
    latex_path = output_dir / "surface_generation_table_row.tex"
    latex_path.write_text(
        "\\textbf{Time (ms/protein)}\n& " + "\n& ".join(values) + " \\\\\n"
    )
    breakdown_path = write_breakdown_table(summary_rows, output_dir)
    return raw_path, summary_path, latex_path, breakdown_path


def main():
    args = parse_args()
    requested_methods = {name.strip() for name in args.methods.split(",")}
    known_methods = {method.name for method in METHODS}
    unknown_methods = requested_methods - known_methods
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {', '.join(sorted(unknown_methods))}")
    edtsurf_scales = (
        {float(value.strip()) for value in args.edtsurf_scales.split(",")}
        if args.edtsurf_scales
        else None
    )
    nanoshaper_scales = (
        {float(value.strip()) for value in args.nanoshaper_scales.split(",")}
        if args.nanoshaper_scales
        else None
    )
    methods = [
        method
        for method in METHODS
        if method.name in requested_methods
        and not (
            method.name == "edtsurf"
            and edtsurf_scales is not None
            and method.grid_scale not in edtsurf_scales
        )
        and not (
            method.name == "nanoshaper"
            and nanoshaper_scales is not None
            and method.grid_scale not in nanoshaper_scales
        )
    ]
    pdb_files = sorted(args.pdb_dir.glob("*.pdb"))
    if args.max_proteins is not None:
        pdb_files = pdb_files[: args.max_proteins]
    if not pdb_files:
        raise SystemExit(f"No PDB files found in {args.pdb_dir}")

    device = torch.device(args.device)
    if any(method.name == "dmasif" for method in methods) and device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("dMaSIF benchmarking requested but CUDA is unavailable")

    print(f"Proteins: {len(pdb_files)}")
    print("Workers: 0 (serial)")
    print(f"Warm-up proteins per method: {args.warmup}")
    rows = []
    for method in methods:
        suffix = "" if method.grid_scale is None else f" grid_scale={method.grid_scale}"
        print(f"\n{method.label}{suffix}", flush=True)
        for warmup_index, pdb_path in enumerate(pdb_files[: args.warmup]):
            warmup_row = benchmark_one(pdb_path, method, args, device, warmup_index)
            if warmup_row["status"] != "success":
                print(f"  warm-up {pdb_path.name}: {warmup_row['error']}", flush=True)
        for sample_index, pdb_path in enumerate(pdb_files):
            row = benchmark_one(pdb_path, method, args, device, sample_index)
            rows.append(row)
            if row["status"] != "success":
                print(f"  {pdb_path.name}: {row['error']}", flush=True)
        write_outputs(rows, args.output_dir)
        print(f"Checkpoint saved after {method.label}{suffix}", flush=True)

    raw_path, summary_path, latex_path, breakdown_path = write_outputs(
        rows, args.output_dir
    )
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"LaTeX row: {latex_path}")
    print(f"LaTeX breakdown table: {breakdown_path}")

    failures = [row for row in rows if row["status"] != "success"]
    if failures:
        raise SystemExit(f"Benchmark completed with {len(failures)} failed samples")


if __name__ == "__main__":
    main()
