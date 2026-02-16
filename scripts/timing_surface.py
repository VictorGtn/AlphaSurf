import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_current_dir = Path(__file__).resolve().parent
_cgal_bindings_path = (
    _current_dir.parent.parent.parent.parent / "cgal_alpha_bindings" / "build"
)
sys.path.append(str(_cgal_bindings_path))
import cgal_alpha

import igl
from alphasurf.protein.create_surface import mesh_simplification
from alphasurf.protein.create_operators import compute_operators
from alphasurf.protein.surfaces import get_geom_feats
from alphasurf.utils.python_utils import silentremove


def parse_xyzr_file(xyzr_path):
    """Parse xyzr file to extract atom positions and radii."""
    # Skip first line (debug output from pdb_to_xyzr script)
    data = np.loadtxt(xyzr_path, skiprows=1)
    atom_pos = data[:, :3].astype(np.float32)
    atom_radius = data[:, 3].astype(np.float32)
    return atom_pos, atom_radius


def pdb_to_xyzr(pdb_path, xyzr_path):
    """Convert PDB to XYZR format using pdb_to_xyzr binary."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    binary_base_path = os.path.abspath(os.path.join(script_dir, "..", "..", "bin"))

    system = platform.system()
    if system == "Darwin":
        platform_dir = "msms_macos"
    elif system == "Linux":
        platform_dir = "msms_linux"
    elif system == "Windows":
        platform_dir = "msms_windows"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    binary_path = os.path.join(binary_base_path, platform_dir)
    pdb2xyzr_path = os.path.abspath(os.path.join(binary_path, "pdb_to_xyzr"))

    err_file = xyzr_path + ".err"
    try:
        with open(xyzr_path, "w") as f_out, open(err_file, "w") as f_err:
            result = subprocess.run(
                [pdb2xyzr_path, pdb_path], stdout=f_out, stderr=f_err, cwd=binary_path
            )

        if result.returncode != 0:
            with open(err_file, "r") as f:
                err_msg = f.read()
            raise RuntimeError(f"pdb_to_xyzr failed: {err_msg}")

        # Check if xyzr file has actual data (skip debug line)
        with open(xyzr_path, "r") as f:
            f.readline()  # skip debug output line
            data_line = f.readline().strip()
            if not data_line:
                raise RuntimeError("pdb_to_xyzr produced no data or invalid output")
            # Check if line contains numeric data (can start with space, minus sign, or digit)
            try:
                parts = data_line.split()
                if len(parts) < 4:  # Need at least x, y, z, r
                    raise ValueError
                float(parts[0])  # Try to parse first number
            except (ValueError, IndexError):
                raise RuntimeError(
                    f"pdb_to_xyzr produced invalid output: {data_line[:50]}"
                )
    finally:
        silentremove(err_file)


def surface_alpha_complex(atom_pos, atom_radius, alpha_value=0.001):
    """Generate alpha complex surface from atom positions and radii."""
    verts, faces = cgal_alpha.compute_alpha_complex_from_atoms(
        atom_pos,
        atom_radius,
        float(alpha_value),
        1.4,
        "singular+regular",
    )
    return verts, faces


def surface_msms(xyzr_path, min_number=256):
    """Generate MSMS surface from xyzr file, starting at density=1.0 and increasing until min_number vertices."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    binary_base_path = os.path.abspath(os.path.join(script_dir, "..", "..", "bin"))

    system = platform.system()
    if system == "Darwin":
        platform_dir = "msms_macos"
    elif system == "Linux":
        platform_dir = "msms_linux"
    elif system == "Windows":
        platform_dir = "msms_windows"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    binary_path = os.path.join(binary_base_path, platform_dir)
    msms_path = os.path.abspath(os.path.join(binary_path, "msms"))

    out_name = os.path.join(tempfile.gettempdir(), f"msms_timing_{os.getpid()}")
    vert_file = out_name + ".vert"
    face_file = out_name + ".face"
    log_name = f"{out_name}_msms.log"

    number_of_vertices = 0
    density = 1.0
    verts, faces = None, None

    try:
        while number_of_vertices < min_number:
            msms_args = [
                msms_path,
                "-if",
                xyzr_path,
                "-of",
                out_name,
                "-density",
                str(density),
            ]
            with open(log_name, "w") as f:
                result = subprocess.run(
                    msms_args, stdout=f, stderr=f, cwd=binary_path, timeout=300
                )
            if result.returncode != 0:
                raise RuntimeError(f"MSMS failed with return code {result.returncode}")

            with open(vert_file, "r", errors="ignore") as f:
                lines = f.readlines()
                n_vert = int(lines[2].split()[0])
                no_header = lines[3:]
                lines = [line.split()[:6] for line in no_header]
                lines = np.array(lines).astype(np.float32)
                verts = lines[:, :3]

            with open(face_file, "r", errors="ignore") as f:
                lines = f.readlines()
                n_faces = int(lines[2].split()[0])
                no_header = lines[3:]
                lines = [line.split() for line in no_header]
                lines = np.array(lines).astype(np.int32)
                faces = lines[:, :3]
                faces -= 1

            number_of_vertices = len(verts)
            density += 1

        return verts, faces
    finally:
        silentremove(log_name)
        silentremove(vert_file)
        silentremove(face_file)


def time_surface_generation(
    pdb_path,
    alpha_value=0.001,
    min_number=256,
    method="both",
    do_simplify=False,
    reduction_rate=1.0,
    do_operators=False,
    do_features=False,
):
    """Time surface generation methods with consistent parsing.

    Args:
        pdb_path: Path to PDB file
        alpha_value: Alpha value for alpha complex
        min_number: Minimum vertices for MSMS
        method: Which method to run - "alpha", "msms", or "both"
        do_simplify: Whether to run mesh simplification
        reduction_rate: Face reduction rate for simplification
        do_operators: Whether to compute spectral operators
        do_features: Whether to compute geometric features
    """
    xyzr_path = os.path.join(
        tempfile.gettempdir(), f"timing_surface_{os.getpid()}.xyzr"
    )

    result = {
        "pdb_name": os.path.basename(pdb_path),
        "n_atoms": 0,
        "t_parse": 0.0,
        "error": None,
    }

    # Initialize keys for all potential steps
    steps = []
    if method in ["alpha", "both"]:
        steps.append("alpha")
    if method in ["msms", "both"]:
        steps.append("msms")

    for m in steps:
        result[f"t_{m}"] = 0.0
        result[f"n_verts_{m}"] = 0
        result[f"n_faces_{m}"] = 0

        if do_simplify:
            result[f"t_simplify_{m}"] = 0.0
            result[f"n_verts_{m}_simp"] = 0

        if do_operators:
            result[f"t_operators_{m}"] = 0.0

        if do_features:
            result[f"t_features_{m}"] = 0.0

    try:
        t0 = time.time()
        pdb_to_xyzr(pdb_path, xyzr_path)
        result["t_parse"] = time.time() - t0

        atom_pos, atom_radius = parse_xyzr_file(xyzr_path)
        result["n_atoms"] = len(atom_pos)

        # Helper function to run pipeline for a method
        def run_method_pipeline(method_name, verts_in, faces_in, override_rate=None):
            # 1. Base Generation (already done, just timing recording)
            result[f"n_verts_{method_name}"] = len(verts_in)
            result[f"n_faces_{method_name}"] = len(faces_in)

            curr_verts, curr_faces = verts_in, faces_in

            # 2. Simplification
            if do_simplify:
                t_start = time.time()
                # Use override rate if provided, otherwise default
                rate = override_rate if override_rate is not None else reduction_rate

                curr_verts, curr_faces = mesh_simplification(
                    curr_verts,
                    curr_faces,
                    out_ply=None,
                    face_reduction_rate=rate,
                    min_vert_number=min_number,
                    use_pymesh=False,
                    surface_method="alpha_complex"
                    if method_name == "alpha"
                    else "msms",
                )
                result[f"t_simplify_{method_name}"] = time.time() - t_start
                result[f"n_verts_{method_name}_simp"] = len(curr_verts)

            # 3. Operators (Prerequisite for Features)
            # If features are requested, we MUST compute operators even if do_operators is False
            should_compute_ops = do_operators or do_features

            frames, mass, L, evals, evecs, gradX, gradY = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            vnormals = None

            if should_compute_ops:
                t_start = time.time()
                vnormals = igl.per_vertex_normals(curr_verts, curr_faces)
                frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
                    curr_verts, curr_faces, normals=vnormals
                )
                # Only record time if specifically asked for timing operators
                if do_operators:
                    result[f"t_operators_{method_name}"] = time.time() - t_start

            # 4. Features
            if do_features:
                t_start = time.time()
                # Ensure we have normals if operators weren't run (unlikely given logic above)
                if vnormals is None:
                    vnormals = igl.per_vertex_normals(curr_verts, curr_faces)

                _ = get_geom_feats(curr_verts, curr_faces, evecs, evals, vnormals)
                result[f"t_features_{method_name}"] = time.time() - t_start

        if method in ["alpha", "both"]:
            t0 = time.time()
            # Hardcoded alpha=0.0 as requested
            verts_alpha, faces_alpha = surface_alpha_complex(
                atom_pos, atom_radius, alpha_value=0.0
            )
            result["t_alpha"] = time.time() - t0
            # Hardcoded reduction_rate=1.0 for Alpha
            run_method_pipeline("alpha", verts_alpha, faces_alpha, override_rate=1.0)

        if method in ["msms", "both"]:
            t0 = time.time()
            verts_msms, faces_msms = surface_msms(xyzr_path, min_number)
            result["t_msms"] = time.time() - t0
            # Hardcoded reduction_rate=0.1 for MSMS
            run_method_pipeline("msms", verts_msms, faces_msms, override_rate=0.1)

    except Exception as e:
        result["error"] = str(e)
        # Uncomment for debugging
        import traceback

        traceback.print_exc()

    finally:
        silentremove(xyzr_path)

    return result


def process_pdb_folder(
    pdb_dir,
    alpha_value=0.001,
    min_number=256,
    output_csv=None,
    method="both",
    do_simplify=False,
    reduction_rate=1.0,
    do_operators=False,
    do_features=False,
):
    """Process all PDB files in a folder and generate timing statistics."""
    if not os.path.isdir(pdb_dir):
        print(f"Error: Directory not found: {pdb_dir}")
        sys.exit(1)

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])

    if not pdb_files:
        print(f"Error: No PDB files found in {pdb_dir}")
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    print(f"Method: {method.upper()}")
    if method in ["alpha", "both"]:
        print(f"Alpha value: {alpha_value}")
    if method in ["msms", "both"]:
        print(f"Min vertices (MSMS): {min_number}")
    print("Options:")
    print(f"  Simplify: {do_simplify} (Rate: {reduction_rate})")
    print(f"  Operators: {do_operators}")
    print(f"  Features: {do_features}")
    print("=" * 80)
    print()

    results = []
    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(f"[{i}/{len(pdb_files)}] Processing {pdb_file}...", end=" ", flush=True)

        result = time_surface_generation(
            pdb_path,
            alpha_value,
            min_number,
            method,
            do_simplify=do_simplify,
            reduction_rate=reduction_rate,
            do_operators=do_operators,
            do_features=do_features,
        )
        results.append(result)

        if result["error"]:
            print(f"✘ Error: {result['error']}")
        else:
            print(f"✔︎ ({result['n_atoms']} atoms)")

    df = pd.DataFrame(results)

    successful = df[df["error"].isnull()]
    failed = df[df["error"].notnull()]

    print()
    print("=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print()

    if len(successful) > 0:
        print(f"Successfully processed: {len(successful)}/{len(results)} proteins")
        print()

        print("PDB to XYZR Parsing:")
        print(
            f"  Mean: {successful['t_parse'].mean():.4f}s ± {successful['t_parse'].std():.4f}s"
        )
        print()

        def print_section_stats(method_name):
            print(f"{method_name.capitalize()} Pipeline:")

            # Base Generation
            times = successful[successful[f"t_{method_name}"] > 0][f"t_{method_name}"]
            if len(times) > 0:
                print(f"  Generation:    {times.mean():.4f}s ± {times.std():.4f}s")
                print(
                    f"    Vertices:    {successful[f'n_verts_{method_name}'].mean():.0f}"
                )

            # Simplification
            if do_simplify:
                simp_times = successful[successful[f"t_simplify_{method_name}"] > 0][
                    f"t_simplify_{method_name}"
                ]
                if len(simp_times) > 0:
                    print(
                        f"  Simplification: {simp_times.mean():.4f}s ± {simp_times.std():.4f}s"
                    )
                    print(
                        f"    Simp Verts:   {successful[f'n_verts_{method_name}_simp'].mean():.0f}"
                    )

            # Operators
            if do_operators:
                op_times = successful[successful[f"t_operators_{method_name}"] > 0][
                    f"t_operators_{method_name}"
                ]
                if len(op_times) > 0:
                    print(
                        f"  Operators:      {op_times.mean():.4f}s ± {op_times.std():.4f}s"
                    )

            # Features
            if do_features:
                feat_times = successful[successful[f"t_features_{method_name}"] > 0][
                    f"t_features_{method_name}"
                ]
                if len(feat_times) > 0:
                    print(
                        f"  Features:       {feat_times.mean():.4f}s ± {feat_times.std():.4f}s"
                    )
            print()

        if method in ["alpha", "both"]:
            print_section_stats("alpha")

        if method in ["msms", "both"]:
            print_section_stats("msms")

    if len(failed) > 0:
        print(f"Failed: {len(failed)}/{len(results)} proteins")
        print("Failed proteins:")
        for _, row in failed.iterrows():
            print(f"  {row['pdb_name']}: {row['error']}")
        print()

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Detailed results saved to: {output_csv}")
        print()

    print("=" * 80)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Time surface generation methods (Alpha Complex vs MSMS) on a folder of PDB files"
    )
    parser.add_argument(
        "pdb_dir",
        type=str,
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--alpha-value",
        type=float,
        default=0.001,
        help="Alpha value for alpha complex (default: 0.001)",
    )
    parser.add_argument(
        "--min-verts",
        type=int,
        default=256,
        help="Minimum number of vertices for MSMS (default: 256)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save detailed results as CSV",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["alpha", "msms", "both"],
        default="both",
        help="Which surface generation method to run (default: both)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run mesh simplification and cleaning",
    )
    parser.add_argument(
        "--reduction-rate",
        type=float,
        default=1.0,
        help="Target face reduction rate (default: 1.0 = cleaning only)",
    )
    parser.add_argument(
        "--compute-operators",
        action="store_true",
        help="Time the computation of spectral operators (Laplacian, etc.)",
    )
    parser.add_argument(
        "--compute-features",
        action="store_true",
        help="Time the computation of geometric features (includes operators implicitly)",
    )

    args = parser.parse_args()

    process_pdb_folder(
        args.pdb_dir,
        alpha_value=args.alpha_value,
        min_number=args.min_verts,
        output_csv=args.output_csv,
        method=args.method,
        do_simplify=args.simplify,
        reduction_rate=args.reduction_rate,
        do_operators=args.compute_operators,
        do_features=args.compute_features,
    )
