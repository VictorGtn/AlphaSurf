import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

_current_dir = Path(__file__).resolve().parent
_cgal_bindings_path = _current_dir.parent / "cgal_alpha_bindings" / "build"
sys.path.append(str(_cgal_bindings_path))
import cgal_alpha  # noqa: E402

_nanoshaper_path = _current_dir.parent.parent / "NanoShaper" / "build"
sys.path.append(str(_nanoshaper_path))
import igl  # noqa: E402
import nanoshaper  # noqa: E402
from alphasurf.protein.create_operators import compute_operators  # noqa: E402
from alphasurf.protein.create_surface import mesh_simplification  # noqa: E402
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402
from alphasurf.protein.surfaces import get_geom_feats  # noqa: E402
from alphasurf.utils.python_utils import silentremove  # noqa: E402


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
    binary_base_path = os.path.abspath(os.path.join(script_dir, "..", "bin"))

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


EDTSURF_BIN = str(Path(__file__).resolve().parent.parent.parent / "EDTSurf" / "EDTSurf")


def surface_edtsurf(pdb_path, probe_radius=1.4):
    """Generate molecular surface using EDTSurf."""
    out_base = os.path.join(tempfile.gettempdir(), f"edtsurf_{os.getpid()}")
    ply_file = out_base + ".ply"
    asa_file = out_base + ".asa"
    cav_file = out_base + "-cav.pdb"

    try:
        result = subprocess.run(
            [
                EDTSURF_BIN,
                "-i",
                pdb_path,
                "-o",
                out_base,
                "-s",
                "3",
                "-p",
                str(probe_radius),
                "-f",
                "0.5",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if not os.path.exists(ply_file):
            raise RuntimeError(f"EDTSurf produced no output: {result.stdout[-200:]}")

        t_parse = 0.0
        t_surface = 0.0
        for line in result.stdout.splitlines():
            if line.startswith("Parse time "):
                try:
                    t_parse = float(line.split()[2])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Surface time "):
                try:
                    t_surface = float(line.split()[2])
                except (IndexError, ValueError):
                    pass

        mesh = trimesh.load(ply_file, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return verts, faces, t_parse, t_surface
    finally:
        for f in [ply_file, asa_file, cav_file]:
            silentremove(f)


def surface_nanoshaper(
    atom_pos,
    atom_radius,
    atom_names=None,
    res_names=None,
    res_nums=None,
    chains=None,
    grid_scale=2.0,
    build_status=False,
    operative_mode="normal",
):
    """Generate surface using NanoShaper via Python pybind modules."""
    prm_path = os.path.join(
        tempfile.gettempdir(), f"nanoshaper_timing_{os.getpid()}.prm"
    )
    try:
        # Generate minimal NanoShaper .prm parameter file
        with open(prm_path, "w") as f:
            f.write(f"Grid_scale = {grid_scale}\n")
            f.write("Grid_perfil = 80.0\n")
            f.write("Build_status_map = true\n")
            f.write(f"Operative_Mode = {operative_mode}\n")
            f.write("Surface = ses\n")
            f.write("Triangulation = true\n")
            f.write("Probe_Radius = 1.4\n")

        t_start = time.time()
        # Parse configuration from the temp .prm file
        cf_base = nanoshaper.load_config(prm_path)
        cf = nanoshaper.parse_config(cf_base)

        scale = cf.scale
        perfil = cf.perfill

        # Load atoms natively into NanoShaper InputData
        inp = nanoshaper.InputData()
        inp.na = len(atom_pos)
        inp.x = atom_pos[:, 0].tolist()
        inp.y = atom_pos[:, 1].tolist()
        inp.z = atom_pos[:, 2].tolist()
        inp.r = atom_radius.tolist()
        inp.q = [0.0] * inp.na
        inp.d = [1] * inp.na
        if atom_names is not None:
            ai_list = []
            for i in range(inp.na):
                ai = nanoshaper.AtomInfo(
                    str(atom_names[i]),
                    int(res_nums[i]),
                    str(res_names[i]),
                    str(chains[i]),
                )
                ai_list.append(ai)
            inp.ai = ai_list
        else:
            inp.ai = []

        # Initialize DelphiShared representation for NanoShaper natively from memory
        grid = nanoshaper.DelphiShared()
        grid.init(scale, perfil, inp)

        t_parse = time.time() - t_start

        t_surf_start = time.time()
        surf = nanoshaper.createSurface(cf, grid)

        # Redirect OS-level stdout to /dev/null
        # null_fd = os.open(os.devnull, os.O_WRONLY)
        # save_fd = os.dup(1)
        # os.dup2(null_fd, 1)
        try:
            nanoshaper.normalMode(surf, grid, cf)
        finally:
            pass
            # os.dup2(save_fd, 1)
            # os.close(null_fd)
            # os.close(save_fd)

        t_surface = time.time() - t_surf_start

        # Vertices and Normals are 1D arrays [x1,y1,z1, x2...], faces are [v1,v2,v3, f2...]
        verts = np.array(surf.vertList, dtype=np.float32).reshape(-1, 3)
        faces = np.array(surf.triList, dtype=np.int32).reshape(-1, 3)

        return verts, faces, t_parse, t_surface
    finally:
        silentremove(prm_path)


def surface_msms(xyzr_path, min_number=256):
    """Generate MSMS surface from xyzr file, starting at density=1.0 and increasing until min_number vertices."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    binary_base_path = os.path.abspath(os.path.join(script_dir, "..", "bin"))

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
                no_header = lines[3:]
                lines = [line.split()[:6] for line in no_header]
                lines = np.array(lines).astype(np.float32)
                verts = lines[:, :3]

            with open(face_file, "r", errors="ignore") as f:
                lines = f.readlines()
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
    method="all",
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
        method: Which method to run - "alpha", "msms", "edtsurf", "both", or "all"
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
    if method in ["alpha", "both", "all"]:
        steps.append("alpha")
    if method in ["msms", "both", "all"]:
        steps.append("msms")
    if method in ["edtsurf", "all"]:
        steps.append("edtsurf")
    if method in ["nanoshaper", "all"]:
        steps.append("nanoshaper")

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

    needs_xyzr = method in ["alpha", "msms", "both", "all"]

    try:
        atom_pos = atom_radius = None
        if needs_xyzr:
            t0 = time.time()
            pdb_to_xyzr(pdb_path, xyzr_path)
            result["t_parse"] = time.time() - t0

            atom_pos, atom_radius = parse_xyzr_file(xyzr_path)
            result["n_atoms"] = len(atom_pos)
        else:
            # Parse PDB directly to extract atoms without XYZR file conversion (NanoShaper mode)
            (
                amino_types,  # 0
                atom_chain_id,  # 1
                atom_amino_id,  # 2
                atom_names,  # 3
                _,  # 4: atom_types
                atom_pos,  # 5
                _,  # 6: atom_charge
                atom_radius,  # 7
                _,  # 8: res_sse
                amino_ids,  # 9
                _,  # 10: atom_ids
            ) = parse_pdb_path(pdb_path, use_pqr=False)

            # Map residue info for NanoShaper
            from alphasurf.protein.graphs import res_type_dict

            res_type_idx_to_name = {v: k for k, v in res_type_dict.items()}
            atom_res_names = [
                res_type_idx_to_name[amino_types[i]] for i in atom_amino_id
            ]

            # Extract residue numbers from amino_ids (format chain:AApos e.g. B:I26)
            # Find the first digit in the part after the colon
            def extract_res_num(amino_id):
                parts = amino_id.split(":")
                if len(parts) < 2:
                    return 0
                import re

                match = re.search(r"\d+", parts[1])
                return int(match.group()) if match else 0

            res_nums_all = [extract_res_num(amino_id) for amino_id in amino_ids]
            atom_res_nums = [res_nums_all[i] for i in atom_amino_id]

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

        if method in ["alpha", "both", "all"]:
            t0 = time.time()
            verts_alpha, faces_alpha = surface_alpha_complex(
                atom_pos, atom_radius, alpha_value=0.0
            )
            result["t_alpha"] = time.time() - t0
            run_method_pipeline("alpha", verts_alpha, faces_alpha, override_rate=1.0)

        if method in ["msms", "both", "all"]:
            t0 = time.time()
            verts_msms, faces_msms = surface_msms(xyzr_path, min_number)
            result["t_msms"] = time.time() - t0
            run_method_pipeline("msms", verts_msms, faces_msms, override_rate=0.1)

        if method in ["edtsurf", "all"]:
            verts_edt, faces_edt, edt_t_parse, edt_t_surface = surface_edtsurf(pdb_path)
            result["t_parse_edtsurf"] = edt_t_parse
            result["t_edtsurf"] = edt_t_surface
            run_method_pipeline("edtsurf", verts_edt, faces_edt, override_rate=0.1)

        if method in ["nanoshaper", "all"]:
            verts_nano, faces_nano, nano_t_parse, nano_t_surface = surface_nanoshaper(
                atom_pos,
                atom_radius,
                atom_names=atom_names if not needs_xyzr else None,
                res_names=atom_res_names if not needs_xyzr else None,
                res_nums=atom_res_nums if not needs_xyzr else None,
                chains=atom_chain_id if not needs_xyzr else None,
            )
            result["t_parse_nanoshaper"] = nano_t_parse
            result["t_nanoshaper"] = nano_t_surface
            run_method_pipeline("nanoshaper", verts_nano, faces_nano, override_rate=0.1)

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
    if method in ["alpha", "both", "all"]:
        print(f"Alpha value: {alpha_value}")
    if method in ["msms", "both", "all"]:
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

        if method in ["alpha", "both", "all"]:
            print_section_stats("alpha")

        if method in ["msms", "both", "all"]:
            print_section_stats("msms")

        if method in ["edtsurf", "all"]:
            print_section_stats("edtsurf")

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
        choices=["alpha", "msms", "edtsurf", "both", "all"],
        default="all",
        help="Which surface generation method to run (default: all)",
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
