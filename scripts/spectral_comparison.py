#!/usr/bin/env python3
"""
Heat-diffusion agreement between alpha-complex and MSMS surfaces of one protein.

For each pair of surfaces the script carries three heat quantities onto the
shared atom correspondence and compares them: the atom heat coupling H_t(i, j),
and the heat kernel signature HKS_t(i). The coupling is also compared pair by
pair within geodesic distance windows, in angstrom and in diffusion lengths sqrt(4t), measured
on the full MSMS mesh between the vertex of each atom's patch closest to the atom
centre, so every pair of a protein is filtered on the same atom pairs.

Operators follow the network exactly (alphasurf.protein.create_operators): cotan
Laplacian and lumped vertex-area mass on the mesh in angstrom, with
k = min(k_eig, (n_verts - 1) // 3) eigenpairs per mesh. Diffusion times are
therefore in angstrom^2, the unit of the learned DiffusionNet times.

The correspondence is exact rather than geometric: MSMS reports the sphere each
vertex sits on, and alpha-complex vertices are atom centres. Both are mapped
back to the atom array, and every metric is evaluated on the atoms the two
meshes share.

Diffusion always runs on the full meshes. With --support patch, heat is
released over and read from each atom's whole patch. With sampled, vertices of
the full MSMS mesh are drawn by farthest-point sampling, each is mapped to the
nearest vertex of every mesh, and the vertex kernel K_t(x, y) is compared there.
Samples whose alpha vertex is duplicated by tufting are dropped, so every kept
sample sits on the same vertex of the tufted and untufted alpha meshes, and
their kernels differ only through the paths heat takes between samples.

Outputs:
  <output_dir>/per_protein/<name>__<a>_vs_<b>.npz  - per-atom HKS per protein
  <output_dir>/spectral_comparison.csv             - one row per protein/pair
"""

import argparse
import csv
import os
import signal
import sys
import traceback
from multiprocessing import Pool

# Must precede numpy: a threaded BLAS deadlocks when the worker pool forks.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(
    0, os.environ.get("CGAL_BINDINGS_DIR", os.path.join(project_root, "cgal_alpha_bindings", "build"))
)
sys.path.append(os.path.join(project_root, "cgal_alpha_bindings", "build"))

from alphasurf.protein.create_surface import (  # noqa: E402
    mesh_simplification,
    pdb_to_alpha_complex,
    pdb_to_edtsurf,
    pdb_to_nanoshaper,
    pdb_to_surf,
)
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402

# A grid kind may carry its grid scale, e.g. edtsurf@0.3; bare names use --grid-scale.
GRID_KINDS = ("nanoshaper", "edtsurf")
MESH_KINDS = ("alpha", "alpha_tuft", "msms_full", "msms_dec") + GRID_KINDS
DEFAULT_PAIRS = ("alpha:msms_dec", "alpha_tuft:msms_dec", "msms_full:msms_dec")
DEFAULT_TIMES = (1, 2, 3, 5, 10, 20)
GEODESIC_WINDOWS = (("0-10", 0, 10), ("0-15", 0, 15), ("5-15", 5, 15), ("10-50", 10, 50), ("10-20", 10, 20),
                    ("20-30", 20, 30), ("30-40", 30, 40), ("40-50", 40, 50))
# Windows in units of each time's diffusion length sqrt(4t), so every time is compared at the same reach of its heat.
DIFFUSION_WINDOWS = (("0-2L", 0, 2), ("1-3L", 1, 3), ("0-1L", 0, 1), ("1-2L", 1, 2), ("2-3L", 2, 3), ("3-4L", 3, 4))
MIN_WINDOW_PAIRS = 30
# Window of the per-source Spearman, and the targets a source needs in it.
SOURCE_WINDOW = "10-50"
MIN_SOURCE_TARGETS = 10
# Unbiased against exact edge-flip geodesics on MSMS meshes; the default 1.0 overestimates by ~2%.
GEODESIC_T_COEF = 0.5


class StepTimeout(Exception):
    pass


def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    """SIGALRM-based per-call timeout. Main-thread only; safe in Pool workers."""

    def _handler(signum, frame):
        raise StepTimeout(f"timed out after {timeout_sec}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_sec))
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def largest_component(verts, faces):
    """Keep the largest vertex-connected component; each extra one adds a zero eigenvalue."""
    n_verts = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adjacency = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts)
    )
    n_components, labels = scipy.sparse.csgraph.connected_components(
        adjacency, directed=False
    )
    if n_components == 1:
        return verts, faces, n_components
    keep = labels == np.bincount(labels).argmax()
    reindex = np.full(n_verts, -1, dtype=np.int64)
    reindex[keep] = np.arange(int(keep.sum()))
    faces = faces[keep[faces].all(axis=1)]
    return verts[keep], reindex[faces].astype(np.int32), n_components


def total_area(verts, faces):
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def build_laplacian(verts, faces):
    import potpourri3d as pp3d

    verts32 = np.asarray(verts, dtype=np.float32)
    faces32 = np.asarray(faces, dtype=np.int32)
    L = pp3d.cotan_laplacian(verts32, faces32, denom_eps=1e-10)
    mass = np.asarray(pp3d.vertex_areas(verts32, faces32), dtype=np.float64)
    if np.isnan(L.data).any() or np.isnan(mass).any():
        raise RuntimeError("NaN in Laplacian or mass")
    return scipy.sparse.csc_matrix(L), mass


def eigendecomposition(L, mass, k):
    """Generalised eigenproblem L phi = lambda M phi, normalised so Phi^T M Phi = I."""
    eps = 1e-8
    mass = mass + eps * mass.mean()
    M_mat = scipy.sparse.diags(mass)
    L_shift = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
    evals, evecs = sla.eigsh(L_shift, k=k, M=M_mat, sigma=eps)
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)
    return evals[order], evecs[:, order], mass


def assign_atom_index(verts, ref_points, ref_atom_idx):
    """Carry an atom index through mesh cleaning by nearest reference vertex.

    Returns the mapping and the largest distance used, which is zero whenever the
    cleaning preserved vertex positions.
    """
    dist, idx = cKDTree(ref_points).query(verts, k=1)
    return ref_atom_idx[idx].astype(np.int64), float(dist.max())


def atom_representatives(verts, atom_idx, atom_pos):
    """Per atom, the vertex of its patch closest to the atom centre."""
    dist = np.linalg.norm(verts - atom_pos[atom_idx], axis=1)
    order = np.lexsort((dist, atom_idx))
    first = np.r_[True, atom_idx[order][1:] != atom_idx[order][:-1]]
    return dict(zip(atom_idx[order][first].tolist(), order[first].tolist()))


def farthest_point_sampling(verts, n, seed):
    """Indices of n vertices spread evenly over the mesh, by greedy farthest-point sampling."""
    n = min(n, len(verts))
    picked = np.empty(n, dtype=np.int64)
    picked[0] = np.random.default_rng(seed).integers(len(verts))
    dist = np.linalg.norm(verts - verts[picked[0]], axis=1)
    for i in range(1, n):
        picked[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(verts - verts[picked[i]], axis=1))
    return picked


def sample_points(meshes, n, seed):
    """Sample full-MSMS vertices and map each to the nearest vertex of every mesh.

    Samples whose alpha vertex is duplicated by tufting are dropped, and so is any
    sample landing on a vertex that an earlier sample already holds.
    """
    full = np.asarray(meshes["msms_full"]["verts"], dtype=float)
    pos = full[farthest_point_sampling(full, n, seed)]
    mapped = {kind: cKDTree(np.asarray(m["verts"], dtype=float)).query(pos)[1] for kind, m in meshes.items()}
    keep = np.ones(len(pos), dtype=bool)
    n_tufted = 0
    if "alpha" in meshes and "alpha_tuft" in meshes:
        copies = np.bincount(meshes["alpha_tuft"]["atom_idx"])
        tufted = copies[meshes["alpha"]["atom_idx"][mapped["alpha"]]] > 1
        keep &= ~tufted
        n_tufted = int(tufted.sum())
    for vert in mapped.values():
        idx = np.flatnonzero(keep)
        _, first = np.unique(vert[idx], return_index=True)
        repeated = np.ones(len(idx), dtype=bool)
        repeated[first] = False
        keep[idx[repeated]] = False
    for kind, mesh in meshes.items():
        mesh["sample_vertex"] = mapped[kind][keep]
    return {"n_samples_drawn": len(pos), "n_samples_tufted": n_tufted, "n_samples_kept": int(keep.sum())}


def geodesic_matrix(mesh, vert):
    """Heat-method geodesic distances between mesh vertices; -1 entries stay at infinity."""
    import potpourri3d as pp3d

    solver = pp3d.MeshHeatMethodDistanceSolver(
        np.asarray(mesh["verts"], dtype=np.float64), np.asarray(mesh["faces"]), t_coef=GEODESIC_T_COEF
    )
    vert = np.asarray(vert)
    ok = np.flatnonzero(vert >= 0)
    G = np.full((len(vert), len(vert)), np.inf)
    for i in ok:
        G[i, ok] = solver.compute_distance(int(vert[i]))[vert[ok]]
    block = G[np.ix_(ok, ok)]
    G[np.ix_(ok, ok)] = 0.5 * (block + block.T)
    return G


def needed_kinds(pairs):
    return {kind for pair in pairs for kind in pair.split(":")}


def known_kind(kind):
    tool, sep, scale = kind.partition("@")
    if sep:
        return tool in GRID_KINDS and scale.replace(".", "", 1).isdigit()
    return kind in MESH_KINDS


def build_mesh_set(pdb_path, args):
    wanted = needed_kinds(args.pairs)
    if args.support == "sampled":
        wanted.add("msms_full")
    parsed = parse_pdb_path(pdb_path, use_pqr=False)
    atom_pos = np.asarray(parsed[5], dtype=np.float32)
    atom_radius = np.asarray(parsed[7], dtype=np.float32)
    if len(atom_pos) == 0:
        raise RuntimeError("no atoms parsed")

    simplify_kwargs = dict(
        out_ply=None,
        min_vert_number=args.min_vert_number,
        max_vert_number=args.max_vert_number,
        use_pymesh=False,
        allow_multiple_components=args.allow_multiple_components,
    )

    meshes = {}

    def store(kind, verts, faces, ref_points, ref_atom_idx):
        verts, faces, n_components = largest_component(verts, faces)
        atom_idx, remap = assign_atom_index(verts, ref_points, ref_atom_idx)
        meshes[kind] = dict(
            verts=verts,
            faces=faces,
            atom_idx=atom_idx,
            remap=remap,
            n_components=n_components,
            raw_area=total_area(verts, faces),
        )

    if {"alpha", "alpha_tuft"} & wanted:
        alpha_v, alpha_f = pdb_to_alpha_complex(
            pdb_path,
            alpha_value=args.alpha_value,
            atom_pos=atom_pos,
            atom_radius=atom_radius,
        )
        for kind, tufting in (("alpha", False), ("alpha_tuft", True)):
            if kind not in wanted:
                continue
            verts, faces, _, _ = mesh_simplification(
                verts=alpha_v,
                faces=alpha_f,
                face_reduction_rate=1.0,
                surface_method="alpha_complex",
                tufting=tufting,
                **simplify_kwargs,
            )
            store(kind, verts, faces, atom_pos, np.arange(len(atom_pos)))

    if {"msms_full", "msms_dec"} & wanted:
        raw_v, raw_f, raw_idx = pdb_to_surf(
            pdb_path,
            density=args.msms_density,
            atom_pos=atom_pos,
            atom_radius=atom_radius + args.msms_radius_offset,
            keep_atom_idx=True,
            probe_radius=args.msms_probe,
        )
        for kind, rate in (("msms_full", 1.0), ("msms_dec", args.msms_reduction)):
            if kind not in wanted:
                continue
            verts, faces, _, _ = mesh_simplification(
                verts=raw_v,
                faces=raw_f,
                face_reduction_rate=rate,
                surface_method="msms",
                tufting=False,
                **simplify_kwargs,
            )
            store(kind, verts, faces, raw_v, raw_idx)

    radius = atom_radius + args.msms_radius_offset
    for kind in sorted(wanted):
        tool, _, scale = kind.partition("@")
        if tool not in GRID_KINDS:
            continue
        scale = float(scale) if scale else args.grid_scale
        if tool == "edtsurf":
            raw_v, raw_f = pdb_to_edtsurf(pdb_path, grid_scale=scale, surface_mode=2)
        else:
            raw_v, raw_f = pdb_to_nanoshaper(pdb_path, probe_radius=args.msms_probe or 1.4,
                                             grid_scale=scale, atom_pos=atom_pos, atom_radius=radius)
        verts, faces, _, _ = mesh_simplification(
            verts=raw_v,
            faces=raw_f,
            face_reduction_rate=1.0,
            surface_method=tool,
            tufting=False,
            **simplify_kwargs,
        )
        verts = np.asarray(verts, dtype=float)
        dist, idx = cKDTree(atom_pos).query(verts, k=min(16, len(atom_pos)))
        atom_idx = idx[np.arange(len(verts)), (dist - radius[idx]).argmin(axis=1)]
        store(kind, verts, np.asarray(faces), verts, atom_idx)

    if "msms_full" in meshes:
        full = meshes["msms_full"]
        full["rep"] = atom_representatives(np.asarray(full["verts"]), full["atom_idx"], atom_pos)
    if args.support == "sampled":
        meshes["msms_full"]["sample_stats"] = sample_points(meshes, args.n_samples, args.seed)
    return meshes, len(atom_pos)


def spectra_for(mesh, k_max):
    n_verts = len(mesh["verts"])
    k = min(k_max, (n_verts - 1) // 3)
    if k < 1:
        raise RuntimeError(f"mesh too small: {n_verts} verts")
    L, mass = build_laplacian(mesh["verts"], mesh["faces"])
    evals, evecs, mass = eigendecomposition(L, mass, k)
    return dict(evals=evals, evecs=evecs, mass=mass)


def rescale_spectrum(spec, area_ratio):
    """Spectrum of the mesh scaled so that its area is divided by area_ratio."""
    return dict(
        evals=spec["evals"] * area_ratio,
        evecs=spec["evecs"] * np.sqrt(area_ratio),
        mass=spec["mass"] / area_ratio,
    )


def pair_operators(mesh, spec, atoms, atom_slot):
    """B = Phi^T M A (k x n) and D = A^T M 1 (n), with A the atom indicator."""
    atom_idx = mesh["atom_idx"]
    slots = np.array([atom_slot.get(int(a), -1) for a in atom_idx])
    keep = slots >= 0
    rows = np.flatnonzero(keep)
    cols = slots[keep]
    A = scipy.sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(atom_idx), len(atoms)),
    )
    MA = A.multiply(spec["mass"][:, None]).tocsr()
    B = (MA.T @ spec["evecs"]).T
    D = np.asarray(MA.sum(axis=0)).ravel()
    return B, D


def heat_kernel_atoms(evals, B, D, t):
    """H[i, j]: mass arriving on the patch of atom i from a unit released uniformly over atom j."""
    return (B.T @ (np.exp(-evals * t)[:, None] * B)) / D[None, :]


def point_kernel(evals, phi_atoms, t):
    """K_t(x_i, x_j): temperature at vertex x_i from a unit Dirac at vertex x_j."""
    return (phi_atoms * np.exp(-evals * t)[None, :]) @ phi_atoms.T


def hks_atoms(evals, phi_atoms, times):
    return np.exp(-np.outer(times, evals)) @ (phi_atoms**2).T


def rel_frobenius(x, y):
    denom = np.linalg.norm(y)
    return float(np.linalg.norm(x - y) / denom) if denom > 0 else float("nan")


def row_correlations(x, y):
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    den = np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (xc * yc).sum(axis=1) / den, np.nan)


def per_source_spearman(H_a, H_b, mask):
    """Spearman between H_a[:, j] and H_b[:, j] on the targets in mask[:, j], for every source j with enough."""
    return np.array([spearmanr(H_a[i, j], H_b[i, j]).statistic
                     for j in range(H_a.shape[1])
                     for i in [np.flatnonzero(mask[:, j])] if len(i) >= MIN_SOURCE_TARGETS])


def analyse_pair(name_a, name_b, meshes, specs, atoms, times, geo, support="patch"):
    spec_a, spec_b = specs[name_a], specs[name_b]

    if support == "patch":
        atom_slot = {int(a): j for j, a in enumerate(atoms)}
        B_a, D_a = pair_operators(meshes[name_a], spec_a, atoms, atom_slot)
        B_b, D_b = pair_operators(meshes[name_b], spec_b, atoms, atom_slot)
        alive = (D_a > 0) & (D_b > 0)
    else:
        vert_a = meshes[name_a]["sample_vertex"][atoms]
        vert_b = meshes[name_b]["sample_vertex"][atoms]
        alive = np.ones(len(atoms), dtype=bool)
    atoms = atoms[alive]
    if geo is not None:
        geo = geo[np.ix_(alive, alive)]
    if len(atoms) < 16:
        raise RuntimeError(f"only {len(atoms)} shared atoms")

    evals_a, evals_b = spec_a["evals"], spec_b["evals"]
    if support == "patch":
        B_a, D_a = B_a[:, alive], D_a[alive]
        B_b, D_b = B_b[:, alive], D_b[alive]
        phi_a = (B_a / D_a[None, :]).T
        phi_b = (B_b / D_b[None, :]).T
    else:
        phi_a = spec_a["evecs"][vert_a[alive]]
        phi_b = spec_b["evecs"][vert_b[alive]]

    row = {
        "k_a": len(evals_a),
        "k_b": len(evals_b),
        "n_atoms_pairwise": len(atoms),
    }
    # H is not symmetric, so both directions of each pair are kept.
    off = ~np.eye(len(atoms), dtype=bool)
    windows = {}
    source_window = None
    if geo is not None:
        g = geo[off]
        for label, lo, hi in GEODESIC_WINDOWS:
            windows[label] = (g >= lo) & (g < hi)
            row[f"npairs_g{label}"] = int(windows[label].sum())
            if label == SOURCE_WINDOW:
                source_window = (geo >= lo) & (geo < hi)

    for t in times:
        tag = f"t{t:g}"
        if support == "patch":
            H_a = heat_kernel_atoms(evals_a, B_a, D_a, t)
            H_b = heat_kernel_atoms(evals_b, B_b, D_b, t)
        else:
            H_a = point_kernel(evals_a, phi_a, t)
            H_b = point_kernel(evals_b, phi_b, t)
        row[f"heat_relfrob_{tag}"] = rel_frobenius(H_a, H_b)
        row[f"heat_rowcorr_median_{tag}"] = float(np.nanmedian(row_correlations(H_a, H_b)))

        pairs_a, pairs_b = H_a[off], H_b[off]
        row[f"heat_spearman_{tag}"] = float(spearmanr(pairs_a, pairs_b).statistic)
        row[f"heat_pearson_{tag}"] = float(np.corrcoef(pairs_a, pairs_b)[0, 1])
        row[f"heat_pair_relfrob_{tag}"] = rel_frobenius(pairs_a, pairs_b)
        masks = dict(windows)
        if geo is not None:
            length = np.sqrt(4 * t)
            for label, lo, hi in DIFFUSION_WINDOWS:
                masks[label] = (g >= lo * length) & (g < hi * length)
                row[f"npairs_{tag}_g{label}"] = int(masks[label].sum())
        for label, mask in masks.items():
            enough = mask.sum() >= MIN_WINDOW_PAIRS
            row[f"heat_spearman_{tag}_g{label}"] = (
                float(spearmanr(pairs_a[mask], pairs_b[mask]).statistic) if enough else float("nan")
            )
            row[f"heat_pearson_{tag}_g{label}"] = (
                float(np.corrcoef(pairs_a[mask], pairs_b[mask])[0, 1]) if enough else float("nan")
            )
            row[f"heat_pair_relfrob_{tag}_g{label}"] = (
                rel_frobenius(pairs_a[mask], pairs_b[mask]) if enough else float("nan")
            )
        if source_window is not None:
            rhos = per_source_spearman(H_a, H_b, source_window)
            empty = len(rhos) == 0
            row[f"heat_spearman_src_median_{tag}_g{SOURCE_WINDOW}"] = float("nan") if empty else float(np.nanmedian(rhos))
            row[f"heat_spearman_src_mean_{tag}_g{SOURCE_WINDOW}"] = float("nan") if empty else float(np.nanmean(rhos))
            row[f"n_sources_{tag}_g{SOURCE_WINDOW}"] = len(rhos)

    hks_a = hks_atoms(evals_a, phi_a, np.asarray(times, dtype=float))
    hks_b = hks_atoms(evals_b, phi_b, np.asarray(times, dtype=float))
    for i, t in enumerate(times):
        tag = f"t{t:g}"
        row[f"hks_relerr_{tag}"] = rel_frobenius(hks_a[i], hks_b[i])
        row[f"hks_corr_{tag}"] = float(row_correlations(hks_a[i][None, :], hks_b[i][None, :])[0])
        row[f"hks_spearman_{tag}"] = float(spearmanr(hks_a[i], hks_b[i]).statistic)

    curves = {"atoms": atoms, "hks_a": hks_a, "hks_b": hks_b}
    return row, curves


def units_on(mesh, support):
    """Atoms (patch support) or kept samples (sampled support) a mesh carries."""
    if support == "patch":
        return set(mesh["atom_idx"].tolist())
    return set(range(len(mesh["sample_vertex"])))


def process_one(task):
    pdb_path, args = task
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    rows, saved = [], []
    try:
        meshes, n_atoms = _run_with_timeout(build_mesh_set, args.step_timeout, pdb_path, args)
    except Exception as exc:
        return [{"pdb_id": name, "pair": "", "status": f"mesh_failed: {type(exc).__name__}: {exc}"}], []

    try:
        specs = {
            kind: _run_with_timeout(spectra_for, args.step_timeout, meshes[kind], args.k_eig)
            for kind in meshes
        }
    except Exception as exc:
        return [{"pdb_id": name, "pair": "", "status": f"spectrum_failed: {type(exc).__name__}: {exc}"}], []
    for kind, ratio in args.area_scale.items():
        if kind in specs:
            specs[kind] = rescale_spectrum(specs[kind], ratio)

    # One ranking of the protein's atoms (or samples), so the pairs subsample
    # overlapping sets and share a single geodesic matrix.
    support = args.support
    n_units = n_atoms if support == "patch" else len(meshes["msms_full"]["sample_vertex"])
    rank = np.random.default_rng(args.seed).permutation(n_units)
    picks = {}
    for pair in args.pairs:
        name_a, name_b = pair.split(":")
        shared = np.array(
            sorted(units_on(meshes[name_a], support) & units_on(meshes[name_b], support)), dtype=np.int64
        )
        picks[pair] = (shared, np.sort(shared[np.argsort(rank[shared])[: args.max_atoms]]))

    geo = {}
    if "msms_full" in meshes:
        full = meshes["msms_full"]
        union = np.unique(np.concatenate([picked for _, picked in picks.values()]))
        vert = (
            np.array([full["rep"].get(int(a), -1) for a in union])
            if support == "patch"
            else full["sample_vertex"][union]
        )
        try:
            G = _run_with_timeout(geodesic_matrix, args.step_timeout, full, vert)
        except Exception as exc:
            return [{"pdb_id": name, "pair": "", "status": f"geodesic_failed: {type(exc).__name__}: {exc}"}], []
        slot = {int(a): i for i, a in enumerate(union)}
        for pair, (_, picked) in picks.items():
            ix = np.array([slot[int(a)] for a in picked], dtype=np.int64)
            geo[pair] = G[np.ix_(ix, ix)]

    for pair in args.pairs:
        name_a, name_b = pair.split(":")
        base = {
            "pdb_id": name,
            "pair": pair,
            "n_atoms": n_atoms,
            "n_verts_a": len(meshes[name_a]["verts"]),
            "n_verts_b": len(meshes[name_b]["verts"]),
            "raw_area_a": meshes[name_a]["raw_area"],
            "raw_area_b": meshes[name_b]["raw_area"],
            "remap_dist_a": meshes[name_a]["remap"],
            "remap_dist_b": meshes[name_b]["remap"],
            "n_components_a": meshes[name_a]["n_components"],
            "n_components_b": meshes[name_b]["n_components"],
        }
        if support == "sampled":
            base.update(meshes["msms_full"]["sample_stats"])
        shared, picked = picks[pair]
        base["n_shared_atoms"] = len(shared)
        base["coverage_a"] = len(shared) / max(len(units_on(meshes[name_a], support)), 1)
        base["coverage_b"] = len(shared) / max(len(units_on(meshes[name_b], support)), 1)
        try:
            metrics, curves = _run_with_timeout(
                analyse_pair,
                args.step_timeout,
                name_a,
                name_b,
                meshes,
                specs,
                picked,
                args.times,
                geo.get(pair),
                support,
            )
        except Exception as exc:
            base["status"] = f"pair_failed: {type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        base.update(metrics)
        base["status"] = "ok"
        rows.append(base)
        saved.append((f"{name}__{name_a}_vs_{name_b}", curves))
    return rows, saved


def collect_pdbs(pdb_dir, max_files, shuffle=False, seed=0):
    paths = []
    for root, _, files in os.walk(pdb_dir):
        for fname in sorted(files):
            if fname.endswith((".pdb", ".cif", ".mmcif")):
                paths.append(os.path.join(root, fname))
    paths.sort()
    if not max_files or max_files >= len(paths):
        return paths
    if shuffle:
        pick = np.random.default_rng(seed).choice(len(paths), max_files, replace=False)
        return [paths[i] for i in sorted(pick)]
    return paths[:max_files]


def write_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument(
        "--output-dir", default=os.path.join(script_dir, "outputs", "spectral_heat")
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true",
                        help="sample --max-files at random rather than taking the first")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument("--msms-density", type=float, default=1.0)
    parser.add_argument("--msms-radius-offset", type=float, default=0.0,
                        help="added to every atom radius before MSMS; 1.4 with a tiny probe approximates the SAS")
    parser.add_argument("--msms-probe", type=float, default=None,
                        help="MSMS probe radius (MSMS default 1.5)")
    parser.add_argument("--grid-scale", type=float, default=0.5,
                        help="grid points per angstrom of the nanoshaper and edtsurf meshes named without @scale")
    parser.add_argument("--area-scale", default="",
                        help="comma-separated kind=ratio; each listed mesh diffuses as if scaled by "
                             "1/sqrt(ratio), with ratio a fixed area over the reference area for that kind")
    parser.add_argument("--msms-reduction", type=float, default=0.1,
                        help="face_reduction_rate for the decimated MSMS mesh, as used in training")
    parser.add_argument("--k-eig", type=int, default=128)
    parser.add_argument("--min-vert-number", type=int, default=16)
    parser.add_argument("--max-vert-number", type=int, default=1000000)
    parser.add_argument("--allow-multiple-components", action="store_true",
                        help="keep meshes with several large components instead of failing the protein; "
                             "the largest component is analysed, as for every mesh")
    parser.add_argument("--max-atoms", type=int, default=1500,
                        help="subsample size for the n x n heat quantities")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--times", default=",".join(f"{t:g}" for t in DEFAULT_TIMES),
                        help="comma-separated diffusion times in angstrom^2")
    parser.add_argument("--step-timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--support", choices=["patch", "sampled"], default="patch",
                        help="release and read heat over each atom's patch, or at sampled full-MSMS vertices")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="full-MSMS vertices drawn per protein with --support sampled")
    args = parser.parse_args()

    args.pairs = [p for p in args.pairs.split(",") if p]
    for pair in args.pairs:
        left, right = pair.split(":")
        if not (known_kind(left) and known_kind(right)):
            parser.error(f"unknown mesh kind in pair {pair}; choose from {MESH_KINDS}, grid kinds optionally as kind@scale")
    args.times = [float(t) for t in args.times.split(",")]
    args.area_scale = {kind: float(ratio) for kind, ratio in (item.split("=") for item in args.area_scale.split(",") if item)}
    unknown = [kind for kind in args.area_scale if not known_kind(kind)]
    if unknown:
        parser.error(f"unknown mesh kind in --area-scale: {sorted(unknown)}; choose from {MESH_KINDS}")

    os.makedirs(args.output_dir, exist_ok=True)
    npz_dir = os.path.join(args.output_dir, "per_protein")
    os.makedirs(npz_dir, exist_ok=True)

    pdbs = collect_pdbs(args.pdb_dir, args.max_files, args.shuffle, args.seed)
    if not pdbs:
        parser.error(f"no structures found under {args.pdb_dir}")
    print(f"{len(pdbs)} structures, pairs={args.pairs}, times={args.times}")

    tasks = [(p, args) for p in pdbs]
    all_rows = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.imap_unordered(process_one, tasks)
            for i, (rows, saved) in enumerate(results, 1):
                all_rows.extend(rows)
                for key, curves in saved:
                    np.savez_compressed(os.path.join(npz_dir, f"{key}.npz"), **curves)
                if i % 20 == 0 or i == len(tasks):
                    print(f"{i}/{len(tasks)} done", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            rows, saved = process_one(task)
            all_rows.extend(rows)
            for key, curves in saved:
                np.savez_compressed(os.path.join(npz_dir, f"{key}.npz"), **curves)
            print(f"{i}/{len(tasks)} done", flush=True)

    csv_path = os.path.join(args.output_dir, "spectral_comparison.csv")
    write_csv(all_rows, csv_path)
    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    print(f"{ok}/{len(all_rows)} rows ok -> {csv_path}")
    for row in all_rows:
        if row.get("status", "ok") != "ok":
            print(f"  {row['pdb_id']} {row.get('pair', '')}: {row['status']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
