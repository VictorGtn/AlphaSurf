#!/usr/bin/env python3
"""Measure the full surface-preprocessing pipeline serially on PINDER."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphasurf.tasks.masif_ligand_new.benchmark_surface_generation import (  # noqa: E402
    Method,
    benchmark_one,
)


DEFAULT_PDB_DIR = REPO_ROOT / "data" / "pinder-pair" / "pdb"


def parse_scales(value):
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serial full-pipeline preprocessing benchmark on PINDER"
    )
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(0,),
        default=0,
        help="Keep preprocessing serial; 0 is the only supported value",
    )
    parser.add_argument("--max-proteins", type=int)
    parser.add_argument(
        "--sample-lr",
        type=int,
        help="Randomly select this many left and right PDBs",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument("--edtsurf-scales", default="0.3,0.4,0.5")
    parser.add_argument("--nanoshaper-scales", default="0.4,0.5")
    parser.add_argument("--msms-face-reduction-rate", type=float, default=0.1)
    parser.add_argument(
        "--method",
        choices=("all", "alpha", "edt03", "edt04", "edt05", "nano04", "nano05", "msms"),
        default="all",
    )
    return parser.parse_args()


def build_methods(args):
    methods = [Method("alpha_complex", "Alpha-complex")]
    methods.extend(
        Method("edtsurf", "EDTSurf", scale)
        for scale in parse_scales(args.edtsurf_scales)
    )
    methods.extend(
        Method("nanoshaper", "NanoShaper", scale)
        for scale in parse_scales(args.nanoshaper_scales)
    )
    methods.append(
        Method(
            "msms_simplified",
            "MSMS Simplified",
            face_reduction_rate=args.msms_face_reduction_rate,
            engine="msms",
        )
    )
    if args.method == "all":
        return methods

    method_keys = {
        "alpha": ("alpha_complex", None),
        "edt03": ("edtsurf", 0.3),
        "edt04": ("edtsurf", 0.4),
        "edt05": ("edtsurf", 0.5),
        "nano04": ("nanoshaper", 0.4),
        "nano05": ("nanoshaper", 0.5),
        "msms": ("msms_simplified", None),
    }
    name, grid_scale = method_keys[args.method]
    selected = [
        method
        for method in methods
        if method.name == name and method.grid_scale == grid_scale
    ]
    if not selected:
        raise ValueError(f"Method {args.method} is excluded by the selected scales")
    return selected


def select_pdb_files(pdb_dir, sample_lr, seed):
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if sample_lr is None:
        return sorted(pdb_files)

    left = sorted(path for path in pdb_files if path.stem.endswith("_L"))
    right = sorted(path for path in pdb_files if path.stem.endswith("_R"))
    if len(left) < sample_lr or len(right) < sample_lr:
        raise ValueError(
            f"Requested {sample_lr} PDBs per side, found "
            f"{len(left)} left and {len(right)} right"
        )

    rng = np.random.default_rng(seed)
    selected_left = rng.choice(left, size=sample_lr, replace=False)
    selected_right = rng.choice(right, size=sample_lr, replace=False)
    return sorted((*selected_left, *selected_right), key=str)


def write_manifest(pdb_files, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pdb_sample_manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pdb", "side", "seed"))
        for pdb_path in pdb_files:
            writer.writerow((pdb_path.name, pdb_path.stem.rsplit("_", 1)[-1], seed))
    return path


def finite_values(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def mean_std_ms(rows, field):
    values = finite_values(rows, field)
    if not len(values):
        return math.nan, math.nan
    return float(values.mean() * 1000), float(values.std() * 1000)


def summarize(rows, methods):
    summary = []
    for method in methods:
        matching = [
            row
            for row in rows
            if row["method"] == method.name
            and (
                (not row["grid_scale"] and method.grid_scale is None)
                or (
                    row["grid_scale"]
                    and math.isclose(float(row["grid_scale"]), method.grid_scale)
                )
            )
        ]
        completed = [row for row in matching if row["status"] == "success"]
        surface_mean, surface_std = mean_std_ms(completed, "t_surface_s")
        mesh_mean, mesh_std = mean_std_ms(completed, "t_mesh_s")
        operators_mean, operators_std = mean_std_ms(completed, "t_operators_s")
        geom_mean, geom_std = mean_std_ms(completed, "t_geom_feats_s")
        normals_mean, normals_std = mean_std_ms(completed, "t_normals_s")
        pipeline_mean, pipeline_std = mean_std_ms(completed, "t_pipeline_s")
        summary.append(
            {
                "method": method.name,
                "label": method.label,
                "grid_scale": "" if method.grid_scale is None else method.grid_scale,
                "face_reduction_rate": method.face_reduction_rate,
                "attempted": len(matching),
                "completed": len(completed),
                "failures": len(matching) - len(completed),
                "surface_mean_ms": surface_mean,
                "surface_std_ms": surface_std,
                "mesh_mean_ms": mesh_mean,
                "mesh_std_ms": mesh_std,
                "normals_mean_ms": normals_mean,
                "normals_std_ms": normals_std,
                "operators_mean_ms": operators_mean,
                "operators_std_ms": operators_std,
                "geom_mean_ms": geom_mean,
                "geom_std_ms": geom_std,
                "pipeline_mean_ms": pipeline_mean,
                "pipeline_std_ms": pipeline_std,
            }
        )
    return summary


def format_stage(mean, std):
    if not math.isfinite(mean):
        return "--"
    return f"${mean:.1f} \\pm {std:.1f}$"


def write_outputs(rows, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "pinder_preprocessing_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "pinder_preprocessing_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    required_columns = {
        ("alpha_complex", None),
        ("edtsurf", 0.3),
        ("edtsurf", 0.4),
        ("edtsurf", 0.5),
        ("nanoshaper", 0.4),
        ("nanoshaper", 0.5),
        ("msms_simplified", None),
    }
    available_columns = {
        (
            row["method"],
            None if row["grid_scale"] == "" else float(row["grid_scale"]),
        )
        for row in summary
    }
    if required_columns != available_columns:
        return raw_path, summary_path, None

    backslash = chr(92)
    line_end = " " + backslash * 2
    columns = {
        (
            row["method"],
            None if row["grid_scale"] == "" else float(row["grid_scale"]),
        ): row
        for row in summary
    }

    def cell(method, grid_scale=None, field=None):
        row = columns[(method, grid_scale)]
        if field is None:
            return "--" if grid_scale is None else f"{grid_scale:g}"
        return format_stage(row[f"{field}_mean_ms"], row[f"{field}_std_ms"])

    alpha = ("alpha_complex", None)
    edt = [("edtsurf", scale) for scale in (0.3, 0.4, 0.5)]
    nano = [("nanoshaper", scale) for scale in (0.4, 0.5)]
    msms = ("msms_simplified", None)
    ordered = [alpha, *edt, *nano, msms]
    stage_rows = []
    for label, field in (
        ("Surface / point-cloud generation", "surface"),
        ("Mesh processing and simplification", "mesh"),
        ("Spectral-operator construction", "operators"),
        ("Geometric-feature computation", "geom"),
    ):
        stage_rows.append(
            f"        {label}\n        & "
            + "\n        & ".join(
                cell(method, scale, field) for method, scale in ordered
            )
            + line_end
        )

    total_cells = []
    for method, grid_scale in ordered:
        row = columns[(method, grid_scale)]
        means = [
            row["surface_mean_ms"],
            row["mesh_mean_ms"],
            row["operators_mean_ms"],
            row["geom_mean_ms"],
        ]
        total_cells.append(
            "--"
            if not all(math.isfinite(value) for value in means)
            else f"\\mathbf{{{sum(means):.1f}}}"
        )

    total_row = (
        "        \\textbf{Total preprocessing}\n        & "
        + "\n        & ".join(total_cells)
        + line_end
    )

    table_lines = [
        "\\begin{table}[!htbp]",
        "    \\centering",
        "    \\caption{Average preprocessing time per PINDER protein, broken down by pipeline stage. The same 2,000-protein sample as the speed and mesh-size analyses is processed serially with zero workers. Times are reported in milliseconds, and stage-level values are mean $\\pm$ standard deviation. Total preprocessing is the sum of the reported stage means. MSMS uses a face-reduction rate of 0.1.}",
        "    \\label{tab:pinder_preprocessing_breakdown}",
        "",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{lccccccc}",
        "        \\toprule",
        "        \\textbf{Pipeline stage}",
        "        & \\boldsymbol{\\alpha}\\textbf{-complex}",
        "        & \\multicolumn{3}{c}{\\textbf{EDTSurf}}",
        "        & \\multicolumn{2}{c}{\\textbf{NanoShaper}}",
        "        & \\textbf{MSMS Simplified}" + line_end,
        "        \\cmidrule(lr){3-5}",
        "        \\cmidrule(lr){6-7}",
        "",
        "        \\textbf{Grid scale} & -- & 0.3 & 0.4 & 0.5 & 0.4 & 0.5 & --"
        + line_end,
        "        \\midrule",
        "",
        *stage_rows,
        "",
        "        \\midrule",
        total_row,
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table}",
    ]
    table_path = output_dir / "pinder_preprocessing_breakdown_table.tex"
    table_path.write_text("\n".join(table_lines) + "\n")
    return raw_path, summary_path, table_path


def main():
    args = parse_args()
    if args.max_proteins is not None and args.sample_lr is not None:
        raise SystemExit("Use either --max-proteins or --sample-lr, not both")
    methods = build_methods(args)
    try:
        pdb_files = select_pdb_files(args.pdb_dir, args.sample_lr, args.seed)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.max_proteins is not None:
        pdb_files = pdb_files[: args.max_proteins]
    if not pdb_files:
        raise SystemExit(f"No PDB files found in {args.pdb_dir}")

    print(f"PDB directory: {args.pdb_dir}")
    print(f"Proteins: {len(pdb_files)}")
    print("Workers: 0 (serial)")
    print(f"Warm-up proteins per method: {args.warmup}")
    print("Methods: " + ", ".join(method.label for method in methods))
    manifest_path = write_manifest(pdb_files, args.output_dir, args.seed)
    print(f"Sample manifest: {manifest_path}")

    benchmark_args = argparse.Namespace(
        alpha_value=args.alpha_value,
        seed=args.seed,
        skip_pipeline=False,
    )
    device = torch.device("cpu")
    rows = []
    for method in methods:
        suffix = "" if method.grid_scale is None else f" gs={method.grid_scale:g}"
        print(f"\n{method.label}{suffix}", flush=True)
        failure_count = 0
        for warmup_index, pdb_path in enumerate(pdb_files[: args.warmup]):
            warmup = benchmark_one(
                pdb_path, method, benchmark_args, device, warmup_index
            )
            if warmup["status"] != "success":
                print(f"  warm-up {pdb_path.name}: {warmup['error']}", flush=True)
        for sample_index, pdb_path in enumerate(pdb_files):
            row = benchmark_one(pdb_path, method, benchmark_args, device, sample_index)
            rows.append(row)
            if row["status"] != "success":
                failure_count += 1
                if failure_count <= 20:
                    print(f"  {pdb_path.name}: {row['error']}", flush=True)
            if (sample_index + 1) % 1000 == 0:
                print(f"  {sample_index + 1}/{len(pdb_files)}", flush=True)
        if failure_count > 20:
            print(
                f"  ... {failure_count - 20} additional failures suppressed",
                flush=True,
            )

    summary = summarize(rows, methods)
    raw_path, summary_path, table_path = write_outputs(rows, summary, args.output_dir)
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")
    if table_path is not None:
        print(f"LaTeX table: {table_path}")
    print("\n" + "=" * 100)
    for row in summary:
        print(
            f"{row['label']:18s} {str(row['grid_scale']):>4s}  "
            f"completed={row['completed']}/{row['attempted']}  "
            f"surface={row['surface_mean_ms']:.1f} ms  "
            f"operators={row['operators_mean_ms']:.1f} ms  "
            f"geom={row['geom_mean_ms']:.1f} ms"
        )


if __name__ == "__main__":
    main()
