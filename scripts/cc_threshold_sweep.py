#!/usr/bin/env python3
"""
Generate surfaces with multiple methods, save as .npz, and sweep the
CC fragmentation threshold from 0% to 30% in 2% steps.

Methods: algo2, msms (-all_components), edtsurf (0.3/0.4/0.5),
         nanoshaper (0.3/0.4/0.5/0.6)

Outputs:
  <output_dir>/surfaces/<name>.npz             — saved meshes + CC sizes
  <output_dir>/cc_threshold_sweep.csv           — per-protein CC data
  <output_dir>/cc_threshold_sweep_summary.csv   — sweep table
"""

import argparse
import csv
import multiprocessing
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


class MethodTimeout(Exception):
    pass


def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    """SIGALRM-based per-call timeout. Main-thread only; safe in Pool workers."""

    def _handler(signum, frame):
        raise MethodTimeout(f"timed out after {timeout_sec}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_sec))
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _exit_desc(rc, stderr="", stdout=""):
    """Human-readable subprocess exit description with last stderr line."""
    if rc < 0:
        sig = -rc
        try:
            import signal as _sig_mod

            name = _sig_mod.Signals(sig).name
        except (ValueError, AttributeError, RuntimeError):
            name = f"signal {sig}"
        desc = f"crashed with {name} (rc={rc})"
    elif rc > 0:
        desc = f"exited with error (rc={rc})"
    else:
        desc = "exited cleanly (rc=0) but produced no usable output"
    tail = ""
    for stream in (stderr, stdout):
        if stream and stream.strip():
            last = (stream.strip().splitlines() or [""])[-1][:140]
            if last:
                tail = last
                break
    if tail:
        desc += f": {tail}"
    return desc


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.create_surface import (  # noqa: E402
    parse_verts,
    pdb_to_alpha_complex,
    pdb_to_edtsurf,
    pdb_to_nanoshaper,
    cluster_triangles_by_vertex_sharing,
)
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402

METHODS = [
    "algo2",
    "msms",
    "msms_ext",
    "edtsurf_0.3",
    "edtsurf_0.4",
    "edtsurf_0.5",
    "nanoshaper_0.3",
    "nanoshaper_0.4",
    "nanoshaper_0.5",
    "nanoshaper_0.6",
]


def run_msms_all_components(pdb_path, atom_pos, atom_radius, density=1.0):
    """Build xyzr directly from atom_pos/atom_radius (same source as nanoshaper).

    Avoids the external pdb_to_xyzr binary, which silently includes waters/ions
    that other methods filter out — causing inconsistent atom sets across methods
    and triggering MSMS's "RS component N not found" indexing bug on disconnected
    fragments.
    """
    tmpdir = tempfile.mkdtemp(prefix="msms_sweep_")
    out_base = os.path.join(tmpdir, "surf")
    xyzr_name = f"{out_base}_temp.xyzr"

    binary_path = os.path.abspath(os.path.join(project_root, "bin"))
    system = platform.system()
    platform_dir = {"Darwin": "msms_macos", "Linux": "msms_linux"}.get(
        system, "msms_windows"
    )
    msms_dir = os.path.join(binary_path, platform_dir)
    msms_path = os.path.join(msms_dir, "msms")

    try:
        with open(xyzr_name, "w") as f:
            for i in range(len(atom_pos)):
                x, y, z = atom_pos[i]
                r = atom_radius[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {float(r):.6f}\n")

        cmd = [
            msms_path,
            "-if",
            xyzr_name,
            "-of",
            out_base,
            "-density",
            str(density),
            "-all_components",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=msms_dir, timeout=300
        )

        # Read output even on non-zero exit — MSMS sometimes writes valid
        # .vert/.face before bailing.
        all_verts, all_faces = [], []
        offset = 0
        vert_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".vert"))
        for vf in vert_files:
            fp = os.path.join(tmpdir, vf.replace(".vert", ".face"))
            vp = os.path.join(tmpdir, vf)
            if not os.path.exists(fp):
                continue
            try:
                v, fc = parse_verts(vp, fp)
            except Exception:
                continue
            if len(v) == 0 or len(fc) == 0:
                continue
            all_verts.append(v)
            all_faces.append(fc + offset)
            offset += len(v)

        if all_verts:
            return np.vstack(all_verts), np.vstack(all_faces), len(vert_files)

        raise RuntimeError(
            f"MSMS {_exit_desc(result.returncode, result.stderr, result.stdout)}"
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def count_cc_open3d(verts, faces):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    _, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    return np.asarray(cluster_n_triangles)


def strip_msms_interior(verts, faces, atom_pos, atom_radius):
    """Drop MSMS -all_components pieces that don't enclose any atom.

    Each connected component from MSMS -all_components is a closed watertight
    surface. The exterior SES encloses the atoms and is kept; every cavity
    wall (including sub-resolution pockets) encloses only trapped solvent and
    is dropped.
    """
    import open3d as o3d

    if len(verts) == 0 or len(faces) == 0:
        return verts, faces

    mesh_full = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    cluster_labels, cluster_n_tri, _ = mesh_full.cluster_connected_triangles()
    cluster_labels = np.asarray(cluster_labels)
    n_clusters = len(cluster_n_tri)

    atom_pos_np = np.asarray(atom_pos, dtype=np.float32)

    keep_face_mask = np.ones(len(faces), dtype=bool)
    for ci in range(n_clusters):
        face_mask_ci = cluster_labels == ci
        comp_faces = faces[face_mask_ci]
        used_v = np.unique(comp_faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used_v] = np.arange(len(used_v))

        comp_verts_np = verts[used_v]
        bbox_min = comp_verts_np.min(axis=0)
        bbox_max = comp_verts_np.max(axis=0)
        inside_bbox = np.all(
            (atom_pos_np >= bbox_min) & (atom_pos_np <= bbox_max), axis=1
        )
        if not inside_bbox.any():
            keep_face_mask[face_mask_ci] = False
            continue

        comp_t = o3d.t.geometry.TriangleMesh()
        comp_t.vertex.positions = o3d.core.Tensor(comp_verts_np.astype(np.float32))
        comp_t.triangle.indices = o3d.core.Tensor(remap[comp_faces].astype(np.int32))

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(comp_t)
        atom_query = o3d.core.Tensor(atom_pos_np[inside_bbox])
        if bool(np.asarray(scene.compute_occupancy(atom_query).numpy()).any()):
            continue
        keep_face_mask[face_mask_ci] = False

    kept_faces = faces[keep_face_mask]
    if len(kept_faces) == 0:
        return verts[:0], faces[:0]

    used_verts = np.unique(kept_faces)
    new_idx = np.full(len(verts), -1, dtype=np.int64)
    new_idx[used_verts] = np.arange(len(used_verts))
    return verts[used_verts], new_idx[kept_faces]


def _sizes_str(sizes):
    return ";".join(str(int(s)) for s in sorted(sizes, reverse=True))


def _run_edtsurf_methods(pdb_abs, method_timeout, surface_mode, result, save_data):
    for gs in [0.3, 0.4, 0.5]:
        method = f"edtsurf_{gs}"
        try:
            verts, faces = _run_with_timeout(
                pdb_to_edtsurf,
                method_timeout,
                pdb_abs,
                grid_scale=gs,
                surface_mode=surface_mode,
            )
            if len(verts) == 0 or len(faces) == 0:
                raise ValueError("empty surface")
            sizes = count_cc_open3d(verts, faces)
            result[f"{method}_status"] = "ok"
            result[f"{method}_n_verts"] = len(verts)
            result[f"{method}_n_faces"] = len(faces)
            result[f"{method}_cc_sizes"] = _sizes_str(sizes)
            save_data[f"{method}_verts"] = verts
            save_data[f"{method}_faces"] = faces
            save_data[f"{method}_cc_sizes"] = sizes
            save_data.pop(f"{method}_error", None)
        except MethodTimeout:
            message = f"timed out after {method_timeout}s"
            result[f"{method}_status"] = f"error: {message}"
            save_data[f"{method}_error"] = message
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            result[f"{method}_status"] = f"error: {message}"
            save_data[f"{method}_error"] = message[:500]


def check_one(args):
    (
        pdb_path,
        alpha,
        output_dir,
        method_timeout,
        edtsurf_surface_mode,
        edtsurf_only,
    ) = args
    pdb_name = os.path.basename(pdb_path)
    stem = Path(pdb_name).stem

    result = {"pdb_name": pdb_name}
    for m in METHODS:
        result[f"{m}_status"] = ""
        result[f"{m}_n_verts"] = 0
        result[f"{m}_n_faces"] = 0
        result[f"{m}_cc_sizes"] = ""

    save_data = {}

    try:
        pdb_abs = os.path.abspath(pdb_path)
        parsed = parse_pdb_path(pdb_abs, use_pqr=False)
        if parsed is None or parsed[5] is None or parsed[7] is None:
            for m in METHODS:
                result[f"{m}_status"] = "error: parse_pdb_path returned None"
            return result
        atom_pos = np.asarray(parsed[5], dtype=np.float64)
        atom_radius = np.asarray(parsed[7], dtype=np.float64)
        result["n_atoms"] = len(atom_pos)
        if len(atom_pos) == 0:
            for m in METHODS:
                result[f"{m}_status"] = "error: empty atom array"
            return result
    except Exception as e:
        for m in METHODS:
            result[f"{m}_status"] = f"error: {e}"
        return result

    if edtsurf_only:
        _run_edtsurf_methods(
            pdb_abs,
            method_timeout,
            edtsurf_surface_mode,
            result,
            save_data,
        )
        np.savez_compressed(os.path.join(output_dir, f"{stem}.npz"), **save_data)
        return result

    # algo2
    try:
        v, f = _run_with_timeout(
            pdb_to_alpha_complex,
            method_timeout,
            pdb_abs,
            alpha_value=alpha,
            use_python_binding=True,
        )
        if len(v) == 0 or len(f) == 0:
            raise ValueError("empty surface")
        _, sizes = cluster_triangles_by_vertex_sharing(f)
        result["algo2_status"] = "ok"
        result["algo2_n_verts"] = len(v)
        result["algo2_n_faces"] = len(f)
        result["algo2_cc_sizes"] = _sizes_str(sizes)
        save_data["algo2_verts"] = v
        save_data["algo2_faces"] = f
        save_data["algo2_cc_sizes"] = sizes
        save_data.pop("algo2_error", None)
    except MethodTimeout:
        result["algo2_status"] = f"error: timed out after {method_timeout}s"
        save_data["algo2_error"] = f"timed out after {method_timeout}s"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        result["algo2_status"] = f"error: {msg}"
        save_data["algo2_error"] = msg[:500]

    # msms
    try:
        v, f, n_comp = _run_with_timeout(
            run_msms_all_components,
            method_timeout,
            pdb_abs,
            atom_pos,
            atom_radius,
            density=1.0,
        )
        if len(v) == 0 or len(f) == 0:
            raise ValueError("empty surface")
        sizes = count_cc_open3d(v, f)
        result["msms_status"] = "ok"
        result["msms_n_verts"] = len(v)
        result["msms_n_faces"] = len(f)
        result["msms_cc_sizes"] = _sizes_str(sizes)
        save_data["msms_verts"] = v
        save_data["msms_faces"] = f
        save_data["msms_cc_sizes"] = sizes
        save_data.pop("msms_error", None)
    except MethodTimeout:
        result["msms_status"] = f"error: timed out after {method_timeout}s"
        save_data["msms_error"] = f"timed out after {method_timeout}s"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        result["msms_status"] = f"error: {msg}"
        save_data["msms_error"] = msg[:500]

    # msms_ext — MSMS -all_components with interior cavity walls stripped
    if "msms_verts" in save_data:
        try:
            v2, f2 = _run_with_timeout(
                strip_msms_interior,
                method_timeout,
                save_data["msms_verts"],
                save_data["msms_faces"],
                atom_pos,
                atom_radius,
            )
            if len(v2) == 0 or len(f2) == 0:
                raise ValueError("empty surface after strip")
            sizes2 = count_cc_open3d(v2, f2)
            result["msms_ext_status"] = "ok"
            result["msms_ext_n_verts"] = len(v2)
            result["msms_ext_n_faces"] = len(f2)
            result["msms_ext_cc_sizes"] = _sizes_str(sizes2)
            save_data["msms_ext_verts"] = v2
            save_data["msms_ext_faces"] = f2
            save_data["msms_ext_cc_sizes"] = sizes2
            save_data.pop("msms_ext_error", None)
        except MethodTimeout:
            result["msms_ext_status"] = f"error: timed out after {method_timeout}s"
            save_data["msms_ext_error"] = f"timed out after {method_timeout}s"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            result["msms_ext_status"] = f"error: {msg}"
            save_data["msms_ext_error"] = msg[:500]
    elif "msms_error" in save_data:
        save_data["msms_ext_error"] = save_data["msms_error"]
        result["msms_ext_status"] = result["msms_status"]

    _run_edtsurf_methods(
        pdb_abs,
        method_timeout,
        edtsurf_surface_mode,
        result,
        save_data,
    )

    # nanoshaper
    for gs in [0.3, 0.4, 0.5, 0.6]:
        m = f"nanoshaper_{gs}"
        try:
            v, f = _run_with_timeout(
                pdb_to_nanoshaper,
                method_timeout,
                pdb_abs,
                grid_scale=gs,
                atom_pos=atom_pos,
                atom_radius=atom_radius,
            )
            if len(v) == 0 or len(f) == 0:
                raise ValueError("empty surface")
            sizes = count_cc_open3d(v, f)
            result[f"{m}_status"] = "ok"
            result[f"{m}_n_verts"] = len(v)
            result[f"{m}_n_faces"] = len(f)
            result[f"{m}_cc_sizes"] = _sizes_str(sizes)
            save_data[f"{m}_verts"] = v
            save_data[f"{m}_faces"] = f
            save_data[f"{m}_cc_sizes"] = sizes
            save_data.pop(f"{m}_error", None)
        except MethodTimeout:
            result[f"{m}_status"] = f"error: timed out after {method_timeout}s"
            save_data[f"{m}_error"] = f"timed out after {method_timeout}s"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            result[f"{m}_status"] = f"error: {msg}"
            save_data[f"{m}_error"] = msg[:500]

    # Always save so the loader can read back error reasons on partial failures
    np.savez_compressed(os.path.join(output_dir, f"{stem}.npz"), **save_data)

    return result


def evaluate_at_threshold(sizes_str, threshold_pct, n_faces):
    if not sizes_str or n_faces == 0:
        return True, 0, 0.0
    sizes = np.array([int(s) for s in sizes_str.split(";")])
    largest = int(np.max(sizes))
    cutoff = int(threshold_pct / 100.0 * largest)
    n_large = int(np.sum(sizes >= cutoff))
    dropped = int(np.sum(sizes[sizes < cutoff]))
    return n_large <= 1, n_large, dropped / n_faces


def _load_npz(args_tuple):
    p, surf_dir, alpha = args_tuple
    stem = Path(p).stem
    npz_path = os.path.join(surf_dir, f"{stem}.npz")
    if not os.path.exists(npz_path):
        return None
    r = {
        "pdb_name": os.path.basename(p),
        "_npz_path": npz_path,
        "_pdb_path": p,
        "_alpha": alpha,
    }
    try:
        parsed = parse_pdb_path(os.path.abspath(p), use_pqr=False)
        if parsed is None or parsed[5] is None:
            r["n_atoms"] = 0
        else:
            r["n_atoms"] = len(parsed[5])
    except Exception:
        r["n_atoms"] = 0
    try:
        d = np.load(npz_path, allow_pickle=True)
    except (EOFError, ValueError, OSError, zipfile.BadZipFile):
        return ("corrupted", npz_path, p, alpha)
    for m in METHODS:
        faces_key = f"{m}_faces"
        sizes_key = f"{m}_cc_sizes"
        err_key = f"{m}_error"
        if faces_key in d:
            r[f"{m}_status"] = "ok"
            r[f"{m}_n_verts"] = len(d[f"{m}_verts"]) if f"{m}_verts" in d else 0
            r[f"{m}_n_faces"] = len(d[faces_key])
            r[f"{m}_cc_sizes"] = _sizes_str(d[sizes_key]) if sizes_key in d else ""
        elif err_key in d:
            err_msg = str(d[err_key])
            r[f"{m}_status"] = f"error: {err_msg}"
            r[f"{m}_n_verts"] = 0
            r[f"{m}_n_faces"] = 0
            r[f"{m}_cc_sizes"] = ""
        else:
            r[f"{m}_status"] = "error: missing from npz (pre-error-capture)"
            r[f"{m}_n_verts"] = 0
            r[f"{m}_n_faces"] = 0
            r[f"{m}_cc_sizes"] = ""
    return r


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.scale as mscale  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from dataclasses import dataclass  # noqa: E402


# Custom scale that stretches the high end of [0, 1] — useful for pass-rate data
# clustered near 100%. transform: -log((sup_lim + offset - a) / divider)
class CustomScale(mscale.ScaleBase):
    name = "custom"

    def __init__(self, axis, offset=0.02, sup_lim=1, divider=1, power=1):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = offset
        self.divider = divider
        self.sup_lim = sup_lim
        self.power = power
        self.thresh = None

    def get_transform(self):
        return self.CustomTransform(
            thresh=self.thresh,
            offset=self.offset,
            sup_lim=self.sup_lim,
            divider=self.divider,
            power=self.power,
        )

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim, divider, power):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.offset = offset
            self.sup_lim = sup_lim
            self.divider = divider
            self.power = power

        def transform_non_affine(self, a):
            log_distance = (
                np.log((self.sup_lim + self.offset) / (self.sup_lim + self.offset - a))
                / self.divider
            )
            return np.sign(log_distance) * np.abs(log_distance) ** self.power

        def inverted(self):
            return CustomScale.InvertedCustomTransform(
                thresh=self.thresh,
                offset=self.offset,
                sup_lim=self.sup_lim,
                divider=self.divider,
                power=self.power,
            )

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh, sup_lim, divider, power):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.thresh = thresh
            self.sup_lim = sup_lim
            self.divider = divider
            self.power = power

        def transform_non_affine(self, a):
            log_distance = np.sign(a) * np.abs(a) ** (1 / self.power)
            upper = self.sup_lim + self.offset
            return upper * (1 - np.exp(-log_distance * self.divider))

        def inverted(self):
            return CustomScale.CustomTransform(
                offset=self.offset,
                thresh=self.thresh,
                sup_lim=self.sup_lim,
                divider=self.divider,
                power=self.power,
            )


mscale.register_scale(CustomScale)


# Mirror of CustomScale that stretches the LOW end of [0, sup_lim] instead of the high end.
class CustomScaleLow(mscale.ScaleBase):
    name = "custom_low"

    def __init__(self, axis, offset=0.02, sup_lim=1, divider=1):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = offset
        self.divider = divider
        self.sup_lim = sup_lim

    def get_transform(self):
        return self.CustomTransform(
            offset=self.offset,
            sup_lim=self.sup_lim,
            divider=self.divider,
        )

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, sup_lim, divider):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.sup_lim = sup_lim
            self.divider = divider

        def transform_non_affine(self, a):
            return np.log((a + self.offset) / self.divider)

        def inverted(self):
            return CustomScaleLow.InvertedCustomTransform(
                offset=self.offset,
                sup_lim=self.sup_lim,
                divider=self.divider,
            )

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, sup_lim, divider):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.sup_lim = sup_lim
            self.divider = divider

        def transform_non_affine(self, a):
            return self.divider * np.exp(a) - self.offset

        def inverted(self):
            return CustomScaleLow.CustomTransform(
                offset=self.offset,
                sup_lim=self.sup_lim,
                divider=self.divider,
            )


mscale.register_scale(CustomScaleLow)


# Plot methods omit the raw `msms` column — only the cavity-stripped msms_ext is plotted.
PLOT_METHODS = [
    "algo2",
    "msms_ext",
    "edtsurf_0.3",
    "edtsurf_0.4",
    "edtsurf_0.5",
    "nanoshaper_0.3",
    "nanoshaper_0.4",
    "nanoshaper_0.5",
    "nanoshaper_0.6",
]

# Colorblind-friendly family palette with a light-to-dark gradient only within
# parameterized families.  The four method families remain easy to distinguish.
_edtsurf_shades = ["#9ECAE1", "#4292C6", "#08519C"]
_nanoshaper_shades = ["#A1D99B", "#41AB5D", "#238B45", "#005A32"]
COLORS_BY_METHOD = {
    "algo2": "#E41A1C",
    "msms_ext": "#6A3D9A",
    "edtsurf_0.3": _edtsurf_shades[0],
    "edtsurf_0.4": _edtsurf_shades[1],
    "edtsurf_0.5": _edtsurf_shades[2],
    "nanoshaper_0.3": _nanoshaper_shades[0],
    "nanoshaper_0.4": _nanoshaper_shades[1],
    "nanoshaper_0.5": _nanoshaper_shades[2],
    "nanoshaper_0.6": _nanoshaper_shades[3],
}

DISPLAY_NAME = {
    "algo2": "Alpha Complex",
    "msms_ext": "MSMS",
    "edtsurf_0.3": "EDTsurf  gs=0.3",
    "edtsurf_0.4": "EDTsurf  gs=0.4",
    "edtsurf_0.5": "EDTsurf  gs=0.5",
    "nanoshaper_0.3": "NanoShaper  gs=0.3",
    "nanoshaper_0.4": "NanoShaper  gs=0.4",
    "nanoshaper_0.5": "NanoShaper  gs=0.5",
    "nanoshaper_0.6": "NanoShaper  gs=0.6",
}

# Stable family order shared by all legends.
METHOD_FAMILY_ORDER = [
    (None, ["algo2", "msms_ext"]),
    ("edtsurf", ["edtsurf_0.3", "edtsurf_0.4", "edtsurf_0.5"]),
    (
        "nanoshaper",
        ["nanoshaper_0.3", "nanoshaper_0.4", "nanoshaper_0.5", "nanoshaper_0.6"],
    ),
]
ORDERED_METHODS = [m for _, ms in METHOD_FAMILY_ORDER for m in ms]


@dataclass
class _PlotPaths:
    out_dir: str
    sweep_csv: str
    per_protein_csv: str
    atom_csv: str
    classify_csv: str = ""
    include_categories: tuple = ("full_atom",)


def _load_category_map(paths):
    """Return {pdb_name: category} from classify_csv, or {} if not provided."""
    if not paths.classify_csv or not os.path.exists(paths.classify_csv):
        return {}
    out = {}
    with open(paths.classify_csv) as f:
        for row in csv.DictReader(f):
            out[row["pdb_name"]] = row.get("category", "full_atom")
    return out


def _load_atom_keep(paths):
    if not paths.atom_csv or not os.path.exists(paths.atom_csv):
        return None
    with open(paths.atom_csv) as f:
        return {
            row["pdb_name"]
            for row in csv.DictReader(f)
            if not row.get("error") and int(row["n_components"]) == 1
        }


def _iter_per_protein_rows(paths):
    """Yield per-protein rows, filtered to paths.include_categories when
    classify_csv is set. Adds a `category` field to each row."""
    cat_map = _load_category_map(paths)
    atom_keep = _load_atom_keep(paths)
    filtering = bool(cat_map) or atom_keep is not None
    n_in = n_out = 0
    if filtering:
        skipped = Counter()
    with open(paths.per_protein_csv) as f:
        for row in csv.DictReader(f):
            if atom_keep is not None and row["pdb_name"] not in atom_keep:
                n_out += 1
                skipped["disconnected_atom_graph"] += 1
                continue
            cat = (
                cat_map.get(row["pdb_name"], "full_atom") if filtering else "full_atom"
            )
            row["category"] = cat
            if filtering and cat not in paths.include_categories:
                n_out += 1
                skipped[cat] += 1
                continue
            n_in += 1
            yield row
    if filtering:
        skipped_str = ", ".join(f"{k}={v}" for k, v in skipped.most_common())
        print(f"  input filters: kept {n_in:,}, dropped {n_out:,} [{skipped_str}]")


def _color_for(method):
    return COLORS_BY_METHOD.get(method, "#888888")


def _label_for(method):
    return DISPLAY_NAME.get(method, method)


def plot_sweep(paths, y_scale_offset=0.005, y_scale_power=1.35, x_max=5):
    # When a classify_csv is set, re-aggregate from filtered per-protein rows
    # instead of using the (unfiltered) pre-aggregated summary CSV.
    use_filtered = bool(paths.classify_csv) and os.path.exists(paths.classify_csv)
    if use_filtered:
        thresholds = [0, 1, 2, 3, 4, 5, 6] + list(range(8, 32, 2))
        agg = {m: {t: {"pass": 0, "fail": 0} for t in thresholds} for m in METHODS}
        for row in _iter_per_protein_rows(paths):
            for m in METHODS:
                s = row.get(f"{m}_status", "")
                if not s.startswith("ok"):
                    continue
                sizes_str = row.get(f"{m}_cc_sizes", "")
                n_faces = int(row.get(f"{m}_n_faces", 0) or 0)
                if not sizes_str or n_faces == 0:
                    for t in thresholds:
                        agg[m][t]["pass"] += 1
                    continue
                try:
                    sizes = np.array([int(size) for size in sizes_str.split(";")])
                except ValueError:
                    continue
                largest = int(np.max(sizes))
                second_largest = (
                    int(np.partition(sizes, -2)[-2]) if len(sizes) > 1 else None
                )
                for t in thresholds:
                    cutoff = int(t / 100.0 * largest)
                    p = second_largest is None or second_largest < cutoff
                    if p:
                        agg[m][t]["pass"] += 1
                    else:
                        agg[m][t]["fail"] += 1
        rows = []
        for t in thresholds:
            r = {"threshold_pct": t}
            for m in METHODS:
                r[f"{m}_pass"] = agg[m][t]["pass"]
                r[f"{m}_fail"] = agg[m][t]["fail"]
            rows.append(r)
    else:
        rows = []
        with open(paths.sweep_csv) as f:
            for r in csv.DictReader(f):
                rows.append(r)

    fig, ax = plt.subplots(figsize=(11, 6))
    thresholds = [int(r["threshold_pct"]) for r in rows]
    last_total = 0

    for m in PLOT_METHODS:
        passes = [int(r[f"{m}_pass"]) for r in rows]
        fails = [int(r[f"{m}_fail"]) for r in rows]
        total = [p + f for p, f in zip(passes, fails)]
        last_total = max(total) if total else 0
        pass_rate = [p / t if t else 0 for p, t in zip(passes, total)]
        ax.plot(
            thresholds,
            pass_rate,
            marker="o",
            color=_color_for(m),
            label=_label_for(m),
            lw=1.8,
            ms=5,
        )

    legend_handles = [
        plt.Line2D(
            [],
            [],
            color=_color_for(m),
            marker="o",
            linestyle="-",
            lw=2.0,
            markersize=6,
            label=_label_for(m),
        )
        for m in ORDERED_METHODS
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        ncol=1,
        title="Surface method",
        title_fontsize=9.5,
        frameon=True,
        edgecolor="#CCCCCC",
    )

    ax.set_xlabel("CC size threshold (% of largest)")
    ax.set_ylabel("% proteins passing (≤1 CC above threshold)")
    ax.set_title(f"CC threshold sweep — {last_total:,} proteins")
    ax.set_yscale("custom", offset=y_scale_offset, power=y_scale_power)
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:g}%"))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x_max)
    ax.grid(alpha=0.3)
    ax.axhline(0.95, ls="--", color="gray", alpha=0.5)
    ax.axhline(0.99, ls=":", color="gray", alpha=0.4)

    plt.tight_layout()
    out = f"{paths.out_dir}/sweep_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def _bucket_error(msg):
    """Classify an error message string into a bucket."""
    if "timed out" in msg:
        return "timed out"
    if "missing from npz" in msg:
        return "missing from npz (pre-capture)"
    if "empty atom array" in msg:
        return "empty atom array (no protein atoms)"
    if "empty surface" in msg:
        return "empty surface"
    if "returned None" in msg or "parse_pdb_path None" in msg:
        return "parse_pdb_path None"
    if (
        "exit=-11" in msg
        or "rc=-11" in msg
        or "rc=139" in msg
        or "Segmentation fault" in msg
    ):
        return "subprocess SIGSEGV (−11)"
    if "exit=-6" in msg or "rc=-6" in msg or "rc=134" in msg:
        return "subprocess SIGABRT (−6, heap corruption)"
    if "exit=1" in msg or "rc=1" in msg:
        return "subprocess error exit (1)"
    if "RS component" in msg:
        return "MSMS RS component missing"
    if "CalledProcessError" in msg:
        return "subprocess failed"
    if "TimeoutExpired" in msg:
        return "subprocess timeout"
    if "RuntimeError" in msg:
        return "other RuntimeError"
    if "ValueError" in msg:
        return "other ValueError"
    return "other"


def plot_status_breakdown(paths):
    """Errors only, stratified by error type. Drops ok / ok-with-skipped."""
    status_counts = {m: Counter() for m in PLOT_METHODS}
    examples = {m: {} for m in PLOT_METHODS}
    for row in _iter_per_protein_rows(paths):
        for m in PLOT_METHODS:
            s = row.get(f"{m}_status", "")
            if not s.startswith("error"):
                continue
            msg = s.split("error:", 1)[1].strip() if "error:" in s else s
            b = _bucket_error(msg)
            status_counts[m][b] += 1
            if b not in examples[m]:
                ex = msg
                if "stdout=" in ex:
                    ex = ex.split("stdout=")[0]
                examples[m][b] = ex[:90]

    total_per_bucket = Counter()
    for m in PLOT_METHODS:
        for b, c in status_counts[m].items():
            total_per_bucket[b] += c
    all_buckets = [b for b, _ in total_per_bucket.most_common()]

    if not all_buckets:
        print("no errors to plot")
        return

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(all_buckets), 3)))

    fig, (ax_bar, ax_legend) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    x = np.arange(len(PLOT_METHODS))
    bottom = np.zeros(len(PLOT_METHODS))
    for bi, bucket in enumerate(all_buckets):
        heights = np.array([status_counts[m].get(bucket, 0) for m in PLOT_METHODS])
        ax_bar.bar(
            x,
            heights,
            bottom=bottom,
            color=palette[bi],
            label=bucket,
            edgecolor="white",
            lw=0.5,
        )
        bottom += heights

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [_label_for(m) for m in PLOT_METHODS], rotation=30, ha="right"
    )
    ax_bar.set_ylabel("# error proteins")
    ax_bar.set_title("Per-method error breakdown")
    ax_bar.grid(axis="y", alpha=0.3)

    for i, m in enumerate(PLOT_METHODS):
        total = sum(status_counts[m].values())
        if total > 0:
            ax_bar.text(
                i,
                total + max(bottom) * 0.01,
                f"{total:,}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax_legend.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[bi]) for bi in range(len(all_buckets))
    ]
    labels = []
    for bucket in all_buckets:
        ex = next(
            (
                examples[m].get(bucket, "")
                for m in PLOT_METHODS
                if examples[m].get(bucket)
            ),
            "",
        )
        total = total_per_bucket[bucket]
        labels.append(
            f"{bucket} (n={total})\n   e.g. {ex}" if ex else f"{bucket} (n={total})"
        )
    ax_legend.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=9,
        title=f"Error types ({sum(total_per_bucket.values()):,} total)",
        title_fontsize=10,
        frameon=False,
    )

    plt.suptitle("cc_threshold_sweep — error stratification per method", fontsize=13)
    plt.tight_layout()
    out = f"{paths.out_dir}/status_breakdown.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def plot_per_method_errors(paths):
    """Two separate figures: charts grid + table. Avoids label cramping."""
    method_errors = {m: Counter() for m in PLOT_METHODS}
    for row in _iter_per_protein_rows(paths):
        for m in PLOT_METHODS:
            s = row.get(f"{m}_status", "")
            if not s.startswith("error"):
                continue
            msg = s.split("error:", 1)[1].strip() if "error:" in s else s
            if "stdout=" in msg:
                msg = msg.split("stdout=")[0].strip()
            msg = msg.replace("RuntimeError: ", "")
            msg = msg.replace("EDTSurf produced no output ", "EDTSurf ")
            msg = msg.replace("MSMS failed ", "MSMS ")
            method_errors[m][msg[:100]] += 1

    n_cols = 3
    n_rows = (len(PLOT_METHODS) + n_cols - 1) // n_cols
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.2 * n_rows + 1))
    axes = axes.flatten()
    for i, m in enumerate(PLOT_METHODS):
        ax = axes[i]
        c = method_errors[m]
        total = sum(c.values())
        if total == 0:
            ax.text(
                0.5,
                0.5,
                "no errors",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="gray",
            )
            ax.set_title(f"{_label_for(m)}  (0 errors)", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue
        items = c.most_common(6)
        labels = [k if len(k) <= 65 else k[:62] + "..." for k, _ in items]
        counts = [v for _, v in items]
        y = np.arange(len(labels))
        bars = ax.barh(
            y, counts, color=_color_for(m), edgecolor="white", lw=0.5, alpha=0.85
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("# proteins", fontsize=9)
        ax.set_title(
            f"{_label_for(m)}   ({total:,} errors)", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        for bar, n in zip(bars, counts):
            ax.text(
                bar.get_width() + max(counts) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{n:,}",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, max(counts) * 1.25)
        ax.tick_params(axis="y", pad=4)
    for j in range(len(PLOT_METHODS), len(axes)):
        axes[j].axis("off")
    fig1.suptitle("Per-method error detail", fontsize=15, y=0.995, fontweight="bold")
    fig1.subplots_adjust(
        top=0.95, bottom=0.05, left=0.08, right=0.97, wspace=0.85, hspace=0.55
    )
    out1 = f"{paths.out_dir}/per_method_errors_charts.png"
    fig1.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig1)
    print(f"wrote {out1}")

    rows = []
    for m in PLOT_METHODS:
        c = method_errors[m]
        total = sum(c.values())
        if total == 0:
            rows.append([_label_for(m), "(no errors)", 0, ""])
            continue
        for msg, n in c.most_common(6):
            display = msg if len(msg) <= 110 else msg[:107] + "..."
            rows.append([_label_for(m), display, n, f"{100 * n / total:.1f}%"])

    n_rows_table = len(rows) + 1
    fig2, ax = plt.subplots(figsize=(16, 0.42 * n_rows_table + 1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "Error message", "Count", "% of method"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    widths = [0.12, 0.66, 0.09, 0.13]
    for (r, c), cell in table.get_celld().items():
        cell.set_width(widths[c])
        if r == 0:
            cell.set_facecolor("#2c2c2c")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")
        cell.set_edgecolor("#bbbbbb")
    ax.set_title("Per-method error table", fontsize=14, pad=20, fontweight="bold")
    out2 = f"{paths.out_dir}/per_method_errors_table.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")


def plot_failure_overlap(paths):
    """Pairwise overlap of PDBs that fail per method + distribution of # failures per PDB."""
    failed_per_pdb = {}
    for row in _iter_per_protein_rows(paths):
        failed_per_pdb[row["pdb_name"]] = {
            m for m in PLOT_METHODS if row.get(f"{m}_status", "").startswith("error")
        }

    dist = Counter(len(s) for s in failed_per_pdb.values())

    sets_per_method = {
        m: {p for p, s in failed_per_pdb.items() if m in s} for m in PLOT_METHODS
    }
    n = len(PLOT_METHODS)
    inter = np.zeros((n, n), dtype=int)
    for i, mi in enumerate(PLOT_METHODS):
        for j, mj in enumerate(PLOT_METHODS):
            inter[i, j] = len(sets_per_method[mi] & sets_per_method[mj])

    fig, (ax_dist, ax_hm) = plt.subplots(
        1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    nfails = sorted(dist.keys())
    counts = [dist[k] for k in nfails]
    bars = ax_dist.bar(
        nfails,
        counts,
        color=plt.cm.viridis(np.linspace(0.2, 0.85, len(nfails))),
        edgecolor="white",
        lw=0.5,
    )
    ax_dist.set_xlabel("# methods that failed on this PDB")
    ax_dist.set_ylabel("# PDBs")
    ax_dist.set_title("Failure distribution\n(how many methods fail per PDB?)")
    ax_dist.set_yscale("log")
    ax_dist.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        ax_dist.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{cnt:,}",
            ha="center",
            fontsize=9,
        )
    total = sum(counts)
    succ = dist.get(0, 0)
    ax_dist.text(
        0.95,
        0.95,
        f"total PDBs: {total:,}\nsuccess on all {len(PLOT_METHODS)}: {succ:,} ({100 * succ / total:.1f}%)",
        transform=ax_dist.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9),
    )

    im = ax_hm.imshow(inter, cmap="Blues", aspect="auto")
    ax_hm.set_xticks(range(n), labels=[_label_for(m) for m in PLOT_METHODS])
    ax_hm.set_yticks(range(n), labels=[_label_for(m) for m in PLOT_METHODS])
    plt.setp(ax_hm.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(ax_hm.get_yticklabels(), fontsize=9)
    ax_hm.set_title("Pairwise overlap\n(# PDBs failing in BOTH methods)")
    for i in range(n):
        for j in range(n):
            v = inter[i, j]
            color = "white" if v > inter.max() * 0.5 else "black"
            ax_hm.text(
                j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=8
            )
    plt.colorbar(im, ax=ax_hm, label="# PDBs")

    plt.suptitle("Per-method failure overlap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = f"{paths.out_dir}/failure_overlap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def _evaluate_at_threshold(sizes_str, n_faces, threshold_frac):
    """Pass = at most 1 CC with size >= threshold_frac * largest."""
    if not sizes_str or n_faces == 0:
        return True
    try:
        sizes = np.array([int(s) for s in sizes_str.split(";")])
    except ValueError:
        return True
    if len(sizes) == 0:
        return True
    largest = int(np.max(sizes))
    cutoff = int(threshold_frac * largest)
    n_large = int(np.sum(sizes >= cutoff))
    return n_large <= 1


def _size_vs_failure_lines(
    paths, rows_with_atoms, methods_to_plot, out_name, title, y_label
):
    """Methods as raw colored lines (binned, no smoothing) on custom_low y-scale
    (stretches near 0), with KDE protein-size distribution as subtle background fill."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    n_atoms_arr = np.array([r[0] for r in rows_with_atoms])
    x_min, x_max = n_atoms_arr.min(), n_atoms_arr.max()

    bin_edges = np.logspace(np.log10(max(x_min, 1)), np.log10(x_max), 21)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_idx = np.digitize(n_atoms_arr, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)
    bin_total = np.bincount(bin_idx, minlength=len(bin_centers))

    valid_bin_mask = bin_total >= 20
    if valid_bin_mask.any():
        plot_x_min = bin_centers[valid_bin_mask].min()
        plot_x_max = bin_centers[valid_bin_mask].max()
    else:
        plot_x_min, plot_x_max = x_min, x_max

    fig, ax = plt.subplots(figsize=(14, 8))

    try:
        from scipy.stats import gaussian_kde

        log_atoms = np.log10(n_atoms_arr)
        kde = gaussian_kde(log_atoms)
        kde.set_bandwidth(kde.factor * 1.35)
        log_x = np.linspace(log_atoms.min(), log_atoms.max(), 800)
        kde_vals = kde(log_x)
        x_kde = 10**log_x
        kde_scaled = kde_vals / kde_vals.max()
        ax.fill_between(
            x_kde,
            0,
            kde_scaled,
            color="#aaaaaa",
            alpha=0.15,
            edgecolor="none",
            zorder=0,
        )
    except ImportError:
        counts, _, patches = ax.hist(
            n_atoms_arr,
            bins=bin_edges,
            weights=np.ones_like(n_atoms_arr),
            color="#888888",
            alpha=0.18,
            edgecolor="none",
        )
        counts_arr = np.array(counts, dtype=float)
        if counts_arr.max() > 0:
            scaled = counts_arr / counts_arr.max()
            for c, p in zip(scaled, patches):
                p.set_height(c)

    ax.set_xscale("log")
    ax.set_yscale("custom_low")
    ax.set_ylim(0, 1)
    ax.set_xlim(plot_x_min, plot_x_max)

    failure_ticks = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    ax.yaxis.set_major_locator(FixedLocator(failure_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:g}%"))

    ax.set_xlabel("n_atoms (log scale)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(False)

    ax.grid(axis="both", which="major", alpha=0.3, linestyle="-")
    ax.grid(axis="x", which="minor", alpha=0.1, linestyle=":")

    n_fail_per_method = {}
    for m in methods_to_plot:
        failed_arr = np.array([r[1][m] for r in rows_with_atoms])
        bin_failed = np.bincount(
            bin_idx, weights=failed_arr.astype(float), minlength=len(bin_centers)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(bin_total > 0, bin_failed / bin_total, np.nan)

        mask = valid_bin_mask & np.isfinite(rate)
        x = bin_centers[mask]
        y = rate[mask]

        ax.plot(x, y, color=_color_for(m), lw=2.0, zorder=5)

        n_fail_per_method[m] = int(failed_arr.sum())

    ax.set_title(title, fontsize=13, fontweight="bold")

    legend_handles = []
    for family, members in METHOD_FAMILY_ORDER:
        for m in members:
            if m not in methods_to_plot:
                continue
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=_color_for(m),
                    linestyle="-",
                    lw=2.0,
                    label=f"{_label_for(m)}   ({n_fail_per_method.get(m, 0):,} fail)",
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(-0.12, 1.0),
        fontsize=9,
        ncol=1,
        title="Surface method",
        title_fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    out = f"{paths.out_dir}/{out_name}"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_error_vs_size(paths):
    """3x3 bubble grid: subprocess-error failure rate vs n_atoms."""
    rows_with_atoms = []
    for row in _iter_per_protein_rows(paths):
        try:
            n_atoms = int(row["n_atoms"])
        except (ValueError, KeyError):
            continue
        if n_atoms == 0:
            continue
        failed = {
            m: row.get(f"{m}_status", "").startswith("error") for m in PLOT_METHODS
        }
        rows_with_atoms.append((n_atoms, failed))

    if not rows_with_atoms:
        print("no data for size-vs-error plot")
        return

    _size_vs_failure_lines(
        paths,
        rows_with_atoms,
        methods_to_plot=PLOT_METHODS,
        out_name="error_vs_size.png",
        title=f"Subprocess failure rate vs protein size   (n={len(rows_with_atoms):,} PDBs)",
        y_label="% failed",
    )


def plot_error_vs_size_1pct(paths, threshold_frac=0.01):
    """CC-fragmentation failure (methods whose surface has >1 CC ≥ threshold*largest)."""
    rows_with_atoms = []
    for row in _iter_per_protein_rows(paths):
        try:
            n_atoms = int(row["n_atoms"])
        except (ValueError, KeyError):
            continue
        if n_atoms == 0:
            continue
        failed = {}
        for m in PLOT_METHODS:
            s = row.get(f"{m}_status", "")
            if not s.startswith("ok"):
                failed[m] = True
                continue
            sizes_str = row.get(f"{m}_cc_sizes", "")
            n_faces = int(row.get(f"{m}_n_faces", 0) or 0)
            failed[m] = not _evaluate_at_threshold(sizes_str, n_faces, threshold_frac)
        rows_with_atoms.append((n_atoms, failed))

    if not rows_with_atoms:
        print(f"no data for {threshold_frac * 100:g}%-threshold size-vs-error plot")
        return

    methods_to_plot = ["algo2", "msms_ext", "edtsurf_0.5", "nanoshaper_0.5"]
    threshold_pct_str = f"{threshold_frac * 100:g}%"
    _size_vs_failure_lines(
        paths,
        rows_with_atoms,
        methods_to_plot=methods_to_plot,
        out_name=f"error_vs_size_{threshold_pct_str}.png",
        title=f"CC-fragmentation failure ({threshold_pct_str} threshold) vs protein size   "
        f"(n={len(rows_with_atoms):,} PDBs)",
        y_label=f"% with >1 CC ≥ {threshold_pct_str} of largest",
    )


def plot_usable_vs_subprocess_success(
    paths,
    threshold_frac=0.01,
    include_legend=False,
    output_name="validity_vs_subprocess_success",
    conditional=False,
):
    """Plot surface usability against subprocess success."""
    counts = {m: {"total": 0, "success": 0, "usable": 0} for m in PLOT_METHODS}
    for row in _iter_per_protein_rows(paths):
        for m in PLOT_METHODS:
            counts[m]["total"] += 1
            status = row.get(f"{m}_status", "")
            if status.startswith("ok"):
                counts[m]["success"] += 1
            if status.startswith("ok"):
                sizes_str = row.get(f"{m}_cc_sizes", "")
                n_faces = int(row.get(f"{m}_n_faces", 0) or 0)
                if _evaluate_at_threshold(sizes_str, n_faces, threshold_frac):
                    counts[m]["usable"] += 1

    if not counts or not any(c["total"] for c in counts.values()):
        print("no data for usable-vs-subprocess-success plot")
        return

    points = []
    for m in ORDERED_METHODS:
        c = counts[m]
        x = c["success"] / c["total"] if c["total"] else 0
        denominator = c["success"] if conditional else c["total"]
        y = c["usable"] / denominator if denominator else 0
        points.append((m, x, y))

    overlap_counts = Counter((round(x, 12), round(y, 12)) for _, x, y in points)
    overlap_seen = Counter()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for m, x, y in points:
        key = (round(x, 12), round(y, 12))
        rank = overlap_seen[key]
        overlap_seen[key] += 1
        marker_radius = np.sqrt(140) + 5 * (overlap_counts[key] - rank - 1)
        marker_size = marker_radius**2
        ax.scatter(
            x,
            y,
            s=marker_size,
            color=_color_for(m),
            edgecolor="white",
            linewidth=2.0 if overlap_counts[key] > 1 else 0.9,
            label=_label_for(m),
            zorder=4 + rank,
            clip_on=False,
        )

    ax.set_xlim(0.92, 1)
    ax.set_ylim(0.60, 1)
    ax.set_box_aspect(1)
    ax.set_xscale("custom", offset=0.005, sup_lim=1, power=1.35)
    ax.set_yscale("custom", offset=0.005, power=1.35)
    ax.set_xlabel("1 − subprocess error rate" if conditional else "Surface generated")
    if conditional:
        ax.set_ylabel(
            f"Usable surface rate among successful runs ({threshold_frac * 100:g}% CC threshold)"
        )
        ax.set_title("Usable surfaces vs subprocess success")
    else:
        ax.set_ylabel(f"Valid surface rate ({threshold_frac * 100:g}% CC threshold)")
        ax.set_title("Surface validity vs surface generation")
    from matplotlib.ticker import FixedLocator, FuncFormatter

    percent_formatter = FuncFormatter(lambda x, _: f"{x * 100:g}%")
    ax.xaxis.set_major_locator(FixedLocator([0.92, 0.95, 0.97, 0.99, 1.0]))
    ax.yaxis.set_major_locator(FixedLocator([0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]))
    ax.xaxis.set_major_formatter(percent_formatter)
    ax.yaxis.set_major_formatter(percent_formatter)

    ax.set_axisbelow(True)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if include_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=7.5,
            ncol=3,
            columnspacing=0.9,
            handletextpad=0.4,
            frameon=False,
        )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.30)
    out = f"{paths.out_dir}/{output_name}"
    fig.savefig(f"{out}.png", dpi=300)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    print(f"wrote {out}.png and {out}.pdf")


def run_plots(
    output_dir,
    csv_suffix="",
    atom_csv=None,
    threshold_frac=0.01,
    classify_csv=None,
    include_categories=("full_atom",),
    sweep_y_scale_offset=0.005,
    sweep_y_scale_power=1.35,
    sweep_x_max=5,
):
    """Re-render all plots from existing CSVs in output_dir.

    When classify_csv is provided, plot functions will only include PDBs whose
    `category` column matches one of include_categories (default: full_atom
    only — drops ca_only and backbone_only). The classify_csv must have
    columns pdb_name,category.
    """
    paths = _PlotPaths(
        out_dir=output_dir,
        sweep_csv=os.path.join(
            output_dir, f"cc_threshold_sweep_summary{csv_suffix}.csv"
        ),
        per_protein_csv=os.path.join(output_dir, f"cc_threshold_sweep{csv_suffix}.csv"),
        atom_csv=atom_csv or "",
        classify_csv=classify_csv or "",
        include_categories=tuple(include_categories or ("full_atom",)),
    )
    print("plot inputs:")
    print(f"  per-protein csv: {paths.per_protein_csv}")
    print(f"  sweep csv:       {paths.sweep_csv}")
    print(f"  atom csv:        {paths.atom_csv or '(none)'}")
    print(f"  classify csv:    {paths.classify_csv or '(none)'}")
    if paths.classify_csv:
        print(f"  include cats:    {','.join(paths.include_categories)}")
    print(f"  output dir:      {paths.out_dir}")

    plot_sweep(
        paths,
        y_scale_offset=sweep_y_scale_offset,
        y_scale_power=sweep_y_scale_power,
        x_max=sweep_x_max,
    )
    plot_status_breakdown(paths)
    plot_per_method_errors(paths)
    plot_failure_overlap(paths)
    plot_error_vs_size(paths)
    plot_error_vs_size_1pct(paths, threshold_frac=threshold_frac)
    plot_usable_vs_subprocess_success(paths, threshold_frac=threshold_frac)


def main():
    parser = argparse.ArgumentParser(
        description="Generate surfaces (9 methods), save, CC threshold sweep, and plot."
    )
    parser.add_argument("--pdb-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="Parallel workers (0 = all CPUs)"
    )
    parser.add_argument(
        "--atom-csv",
        default=None,
        help="If set, exclude PDBs that have an error or n_components != 1 in this CSV",
    )
    parser.add_argument(
        "--method-timeout",
        type=int,
        default=120,
        help="Per-method timeout in seconds (default 120). "
        "A timed-out method is marked as error and the next method is tried.",
    )
    parser.add_argument(
        "--edtsurf-surface-mode",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="EDTSurf -h mode: 1 outer+inner, 2 outer, 3 inner (default: 1).",
    )
    parser.add_argument(
        "--edtsurf-only",
        action="store_true",
        help="Generate only the three EDTSurf grid scales.",
    )
    parser.add_argument(
        "--csv-suffix",
        default="",
        help="Append to CSV filenames (e.g. '_nofilter'). Lets multiple "
        "sweep variants share the same --output-dir and .npz cache.",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Only process PDBs whose .npz already exists in surf_dir. "
        "Skips surface generation entirely; loads from cache.",
    )
    parser.add_argument(
        "--skip-plot", action="store_true", help="Skip plotting after the sweep."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Don't run surface generation/sweep; only re-render plots "
        "from existing CSVs in --output-dir.",
    )
    parser.add_argument(
        "--threshold-frac",
        type=float,
        default=0.01,
        help="CC threshold (fraction of largest) for the size-vs-error plot.",
    )
    parser.add_argument(
        "--sweep-y-scale-offset",
        type=float,
        default=0.005,
        help="Offset for the sweep plot's custom y-scale (default: 0.005). "
        "Smaller values push 95%% lower and magnify differences near 100%%; "
        "must be greater than zero.",
    )
    parser.add_argument(
        "--sweep-y-scale-power",
        type=float,
        default=1.35,
        help="Curvature exponent for the sweep plot's custom y-scale "
        "(default: 1.35). Larger values push 95%% lower; "
        "1.0 gives the original logarithmic shape.",
    )
    parser.add_argument(
        "--sweep-x-max",
        type=float,
        default=5,
        help="Maximum CC threshold shown in the sweep summary (default: 5%%).",
    )
    parser.add_argument(
        "--classify-csv",
        default=None,
        help="CSV with columns pdb_name,category. When set, plot "
        "functions filter PDBs by --include-categories.",
    )
    parser.add_argument(
        "--include-categories",
        default="full_atom",
        help="Comma-separated categories to include in plots when "
        "--classify-csv is set (default: full_atom). Use "
        "'full_atom,ca_only,backbone_only' to include all.",
    )
    args = parser.parse_args()

    include_cats = tuple(
        c.strip() for c in args.include_categories.split(",") if c.strip()
    )

    if args.sweep_y_scale_offset <= 0:
        parser.error("--sweep-y-scale-offset must be greater than zero")
    if args.sweep_y_scale_power <= 0:
        parser.error("--sweep-y-scale-power must be greater than zero")
    if args.sweep_x_max <= 0:
        parser.error("--sweep-x-max must be greater than zero")

    if args.plot_only:
        run_plots(
            args.output_dir,
            args.csv_suffix,
            args.atom_csv,
            args.threshold_frac,
            classify_csv=args.classify_csv,
            include_categories=include_cats,
            sweep_y_scale_offset=args.sweep_y_scale_offset,
            sweep_y_scale_power=args.sweep_y_scale_power,
            sweep_x_max=args.sweep_x_max,
        )
        return

    if not args.pdb_dir:
        parser.error("--pdb-dir is required unless --plot-only is given")

    atom_keep = None
    if args.atom_csv:
        atom_keep = set()
        with open(args.atom_csv) as f:
            for row in csv.DictReader(f):
                if not row["error"] and int(row["n_components"]) == 1:
                    atom_keep.add(row["pdb_name"])
        print(f"Atom CSV filter: keeping {len(atom_keep):,} single-component PDBs")

    surf_dir = os.path.join(args.output_dir, "surfaces")
    os.makedirs(surf_dir, exist_ok=True)

    pdb_files = sorted(str(x) for x in Path(args.pdb_dir).glob("*.pdb"))
    print(f"PDB files found: {len(pdb_files):,}")
    if atom_keep is not None:
        before = len(pdb_files)
        pdb_files = [p for p in pdb_files if os.path.basename(p) in atom_keep]
        print(
            f"After atom-csv filter: {len(pdb_files):,} (dropped {before - len(pdb_files):,})"
        )
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]
        print(f"Limited to {args.max_files} files")
    if not pdb_files:
        return

    task_args = []
    if args.cached_only:
        cached = []
        for p in pdb_files:
            stem = Path(p).stem
            if os.path.exists(os.path.join(surf_dir, f"{stem}.npz")):
                cached.append(p)
        pdb_files = cached
        print(f"Cached-only: {len(pdb_files):,} PDBs have .npz")
    else:
        for p in pdb_files:
            stem = Path(p).stem
            if not os.path.exists(os.path.join(surf_dir, f"{stem}.npz")):
                task_args.append(
                    (
                        p,
                        args.alpha,
                        surf_dir,
                        args.method_timeout,
                        args.edtsurf_surface_mode,
                        args.edtsurf_only,
                    )
                )
        print(f"Already processed: {len(pdb_files) - len(task_args)}")
    print(f"Remaining: {len(task_args)}")

    new_results = []
    if task_args:
        n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        print(f"Workers: {n_workers}")
        print()

        t0 = time.time()
        if n_workers == 1:
            new_results = []
            for i, a in enumerate(task_args):
                new_results.append(check_one(a))
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(task_args) - i - 1) / rate if rate > 0 else 0
                print(
                    f"\r  [{i + 1}/{len(task_args)}] {rate:.1f}/s  ETA {eta / 60:.0f}m",
                    end="",
                    flush=True,
                )
            print()
        else:
            n_total = len(task_args)
            # Outer safety net: per-method timeout * 9 methods + parse + buffer
            method_count = 3 if args.edtsurf_only else len(METHODS)
            default_outer = args.method_timeout * method_count + 60
            task_timeout = float(os.environ.get("CC_SWEEP_TIMEOUT", default_outer))
            n_timeouts = 0

            max_tasks_per_child = 100 if args.edtsurf_only else 1
            with multiprocessing.Pool(
                n_workers, maxtasksperchild=max_tasks_per_child
            ) as pool:
                async_results = [pool.apply_async(check_one, (a,)) for a in task_args]
                new_results = []
                for i, ar in enumerate(async_results, 1):
                    try:
                        res = ar.get(timeout=task_timeout)
                        new_results.append(res)
                    except multiprocessing.TimeoutError:
                        n_timeouts += 1
                        print(
                            f"\n  OUTER TIMEOUT ({task_timeout:.0f}s) on task {i}/{n_total}: {task_args[i - 1][0]}",
                            flush=True,
                        )
                    if i % 100 == 0 or i == n_total:
                        elapsed = time.time() - t0
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (n_total - i) / rate if rate > 0 else 0
                        print(
                            f"\r  [{i}/{n_total}] {rate:.1f}/s  ETA {eta / 60:.0f}m",
                            end="",
                            flush=True,
                        )
                pool.terminate()
            print()
            print(f"Timeouts: {n_timeouts}/{n_total}")
        print(f"Generation done in {time.time() - t0:.1f}s")

    # Load previously-processed results
    prev_results = []
    done_stems = {Path(t[0]).stem for t in task_args}
    load_tasks = [
        (p, surf_dir, args.alpha) for p in pdb_files if Path(p).stem not in done_stems
    ]
    n_workers_load = (
        min(n_workers if "n_workers" in dir() else 30, len(load_tasks))
        if load_tasks
        else 1
    )
    if load_tasks:
        print(f"Loading {len(load_tasks)} npz files with {n_workers_load} workers...")
        with multiprocessing.Pool(n_workers_load) as pool:
            for res in pool.imap_unordered(_load_npz, load_tasks, chunksize=64):
                if isinstance(res, tuple) and res[0] == "corrupted":
                    _, npz_path_bad, p_bad, alpha_bad = res
                    print(f"  Skipping corrupted npz: {npz_path_bad}")
                    os.remove(npz_path_bad)
                    task_args.append(
                        (
                            p_bad,
                            alpha_bad,
                            surf_dir,
                            args.method_timeout,
                            args.edtsurf_surface_mode,
                            args.edtsurf_only,
                        )
                    )
                elif res is not None:
                    res.pop("_npz_path", None)
                    res.pop("_pdb_path", None)
                    res.pop("_alpha", None)
                    prev_results.append(res)
        print(f"Loaded {len(prev_results)} npz files")

    results = prev_results + new_results

    # Per-method error counts
    print(f"\nTotal results: {len(results)}")
    for m in METHODS:
        errors = sum(1 for r in results if r.get(f"{m}_status", "").startswith("error"))
        ok = sum(1 for r in results if r.get(f"{m}_status", "").startswith("ok"))
        print(f"  {m}: ok={ok} errors={errors}")

    ok_results = [
        r
        for r in results
        if any(r.get(f"{m}_status", "").startswith("ok") for m in METHODS)
    ]
    if atom_keep is not None:
        before = len(ok_results)
        ok_results = [r for r in ok_results if r["pdb_name"] in atom_keep]
        print(
            f"  after atom-csv filter: {len(ok_results):,} (dropped {before - len(ok_results):,})"
        )
    if not ok_results:
        return

    # Threshold sweep
    thresholds = [0, 1, 2, 3, 4, 5, 6] + list(range(8, 32, 2))

    print(f"\n{'=' * 200}")
    print(f"CC threshold sweep (0%-30%) — {len(ok_results)} proteins")
    print(f"{'=' * 200}")
    print(f"{'Thresh':>6}", end="")
    for m in METHODS:
        print(f"  | {m:>12}_pass  fail   drop", end="")
    print()
    print("-" * (6 + len(METHODS) * 28))

    sweep_rows = []
    for thresh in thresholds:
        row = {"threshold_pct": thresh}
        print(f"{thresh:>5}%", end="")
        for m in METHODS:
            passes = fails = 0
            drops = []
            for r in ok_results:
                if not r.get(f"{m}_status", "").startswith("ok"):
                    continue
                p, _, d = evaluate_at_threshold(
                    r.get(f"{m}_cc_sizes", ""), thresh, r.get(f"{m}_n_faces", 0)
                )
                passes += p
                fails += not p
                drops.append(d)
            mean_drop = float(np.mean(drops)) if drops else 0.0
            max_drop = float(np.max(drops)) if drops else 0.0
            row[f"{m}_pass"] = passes
            row[f"{m}_fail"] = fails
            row[f"{m}_mean_drop"] = mean_drop
            row[f"{m}_max_drop"] = max_drop
            print(f"  | {passes:>8} {fails:>5} {mean_drop:>7.4f}", end="")
        print()
        sweep_rows.append(row)

    # Per-protein CSV
    csv_path = os.path.join(args.output_dir, f"cc_threshold_sweep{args.csv_suffix}.csv")
    fields = ["pdb_name", "n_atoms"]
    for m in METHODS:
        fields.extend([f"{m}_status", f"{m}_n_verts", f"{m}_n_faces", f"{m}_cc_sizes"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-protein CSV: {csv_path}")

    # Sweep summary CSV
    sweep_path = os.path.join(
        args.output_dir, f"cc_threshold_sweep_summary{args.csv_suffix}.csv"
    )
    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"Sweep summary: {sweep_path}")

    if not args.skip_plot:
        run_plots(
            args.output_dir,
            args.csv_suffix,
            args.atom_csv,
            args.threshold_frac,
            classify_csv=args.classify_csv,
            include_categories=include_cats,
            sweep_y_scale_offset=args.sweep_y_scale_offset,
            sweep_y_scale_power=args.sweep_y_scale_power,
            sweep_x_max=args.sweep_x_max,
        )


if __name__ == "__main__":
    main()
