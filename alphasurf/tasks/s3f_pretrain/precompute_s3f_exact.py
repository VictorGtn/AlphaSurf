"""
Precompute dMaSIF point cloud + features for S3F-exact encoder.

For each PDB in the input directory, produces <output_dir>/<name>.pt containing:
  - sequence: str (1-letter AA codes, len n_res)
  - ca_pos: (n_res, 3) Cα coords
  - bb_pos: (n_res, 3, 3) N/CA/C coords per residue (for fusion pooling)
  - surf_pos: (M, 3) dMaSIF point cloud
  - surf_normals: (M, 3)
  - surf_feat: (M, 42) concatenated [hks(32), curv(10)]
  - res2surf: (n_res, 3, 21) full-protein residue-to-surface map

dMaSIF point cloud uses N/CA/C backbone atoms only (S3F-exact). Curvature
is computed at scales [1, 2, 3, 5, 10] (mean + Gauss = 10 dims). HKS uses
robust_laplacian point-cloud Laplacian + eigsh, 32 time bins.

The geometry comes from `s3f_official/surface.py`, the verbatim upstream S3F
module, so this script and `script/process_surface.py` produce the same
features. Caches written before that switch used our own dMaSIF and curvature
copies (`reg=0.01` instead of `1e-10`) and must be regenerated.

Edges are recomputed at load time.  The full-protein res2surf map is stored
because S3F uses it to select and reindex the surface when cropping residues.

Usage:
  python -m alphasurf.tasks.s3f_pretrain.precompute_s3f_exact \\
      --pdb_dir /path/to/cath/dompdb \\
      --output_dir /path/to/precomputed \\
      [--limit N] [--overwrite] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import os
from multiprocessing import get_context

import numpy as np
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm

logger = logging.getLogger(__name__)

_keops_cache = os.environ.get("KEOPS_CACHE_FOLDER")
if _keops_cache:
    os.makedirs(_keops_cache, exist_ok=True)
    try:
        import pykeops

        pykeops.set_build_folder(_keops_cache)
    except Exception as e:
        logger.warning("pykeops.set_build_folder failed: %s", e)

AA_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

RADIUS_CUTOFF = 10.0
RBF_D_MAX = 20.0
RBF_DIM = 16
SURF_KNN = 16
DMASIF_DISTANCE = 1.05
DMASIF_SMOOTHNESS = 0.5
DMASIF_RESOLUTION = 1.0
DMASIF_NITS = 5
DMASIF_SUPSAMPLING = 20
DMASIF_VARIANCE = 0.5
CURV_SCALES = [1.0, 2.0, 3.0, 5.0, 10.0]
HKS_DIM = 32
HKS_T_MIN = 0.1
HKS_T_MAX = 1000.0
HKS_SCALE = 1000.0
HKS_MIN_EIGS = 50
HKS_EIGS_RATIO = 0.06
HKS_LARGE_EIGS_RATIO = 0.01
HKS_LARGE_SURFACE = 20_000
FUSION_K_PER_ATOM = 21


def parse_backbone(pdb_path):
    """Extract N/CA/C backbone coords + 1-letter sequence.

    Returns (bb_pos[n_res, 3, 3], sequence: str) or (None, None) on failure.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("p", pdb_path)
    except Exception as e:
        logger.warning("PDB parse failed for %s: %s", pdb_path, e)
        return None, None

    bb_pos = []
    sequence = []
    for residue in structure.get_residues():
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname().upper()
        if resname not in AA_3_TO_1:
            continue
        try:
            n = residue["N"].get_coord()
            ca = residue["CA"].get_coord()
            c = residue["C"].get_coord()
        except KeyError:
            continue
        bb_pos.append([n, ca, c])
        sequence.append(AA_3_TO_1[resname])

    if len(bb_pos) == 0:
        return None, None
    return np.asarray(bb_pos, dtype=np.float32), "".join(sequence)


def _backbone_atoms(bb_pos):
    """(n_res, 3, 3) numpy array or tensor -> flat (n_res*3, 3) float tensor.

    The precompute script passes numpy; the training batch hook passes CUDA
    tensors, which must not be routed through numpy.
    """
    if torch.is_tensor(bb_pos):
        return bb_pos.detach().to(torch.float32).reshape(-1, 3)
    return torch.as_tensor(np.asarray(bb_pos), dtype=torch.float32).reshape(-1, 3)


def generate_dmasif_cloud(bb_pos_list, device):
    """Run dMaSIF atoms_to_points_normals on N/CA/C atoms for several proteins.

    S3F atom-type convention: [3, 0, 0] per residue (N, C, C) one-hot over
    {C, H, O, N, S, SE} (index 3 = N, index 0 = C).

    `bb_pos_list` is a list of (n_res, 3, 3) arrays or tensors. One batched
    call is issued for the whole list; `batch_points` assigns each generated
    point to its protein.

    Returns (points, normals, batch_points) on `device` — the caller moves to
    CPU where needed, since HKS is scipy-only.
    """
    from alphasurf.tasks.s3f_pretrain.s3f_official.surface import (
        atoms_to_points_normals,
    )

    atoms, atom_batch, atom_type_idx = [], [], []
    for i, bb_pos in enumerate(bb_pos_list):
        bb = _backbone_atoms(bb_pos)
        n_res = bb.shape[0] // 3
        atoms.append(bb)
        atom_batch.append(torch.full((bb.shape[0],), i, dtype=torch.long))
        atom_type_idx.append(torch.tensor([3, 0, 0], dtype=torch.long).repeat(n_res))

    atoms_flat = torch.cat(atoms).to(device)
    atom_batch = torch.cat(atom_batch).to(device)
    atomtypes = torch.nn.functional.one_hot(
        torch.cat(atom_type_idx).to(device), num_classes=6
    ).float()

    points, normals, batch_points = atoms_to_points_normals(
        atoms_flat,
        atom_batch,
        distance=DMASIF_DISTANCE,
        smoothness=DMASIF_SMOOTHNESS,
        resolution=DMASIF_RESOLUTION,
        nits=DMASIF_NITS,
        atomtypes=atomtypes,
        sup_sampling=DMASIF_SUPSAMPLING,
        variance=DMASIF_VARIANCE,
    )
    return points.detach(), normals.detach(), batch_points.detach()


def build_s3f_surfaces(bb_pos_list, device, min_points=16):
    """dMaSIF cloud + 42-d features + res2surf for a list of proteins.

    The point cloud and multi-scale curvature are computed for the whole list
    in one batched GPU call; HKS and res2surf are per protein.

    Returns a list with one dict per input protein, holding CPU tensors
    `surf_pos`, `surf_normals`, `surf_feat` (M, 42) and `res2surf`
    (n_res, 3, k), or None where the cloud came out too small to use.
    """
    from alphasurf.tasks.s3f_pretrain.s3f_official.surface import knn_atoms
    from alphasurf.utils.timing_stats import Timer

    with Timer("s3f_dmasif_cloud"):
        points, normals, batch_points = generate_dmasif_cloud(bb_pos_list, device)
    with Timer("s3f_curvatures"):
        curv = compute_curvatures(points, normals, batch=batch_points)

    out = []
    for i, bb_pos in enumerate(bb_pos_list):
        sel = batch_points == i
        pts = points[sel]
        if pts.shape[0] < min_points:
            out.append(None)
            continue
        nrm = normals[sel]
        crv = curv[sel]

        with Timer("s3f_hks"):
            hks = torch.from_numpy(compute_hks(pts)).float()
        # load_surface() in released S3F concatenates HKS before curvatures.
        surf_feat = torch.cat([hks, crv.cpu()], dim=-1)

        bb = _backbone_atoms(bb_pos).to(pts.device)
        res2surf = (
            knn_atoms(bb, pts, k=FUSION_K_PER_ATOM - 1)[0]
            .view(bb.shape[0] // 3, 3, -1)
            .cpu()
        )

        out.append(
            {
                "surf_pos": pts.cpu().float(),
                "surf_normals": nrm.cpu().float(),
                "surf_feat": surf_feat.float(),
                "res2surf": res2surf.long(),
            }
        )
    return out


def compute_curvatures(points, normals, batch=None):
    """KeOps curvatures — runs on GPU if input tensors are on GPU."""
    from alphasurf.tasks.s3f_pretrain.s3f_official.surface import (
        compute_curvatures as official_compute_curvatures,
    )

    if batch is None:
        batch = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    return official_compute_curvatures(
        points.float(), normals.float(), batch, CURV_SCALES
    ).detach()


def compute_hks(points, faces=None):
    """32-dim HKS via S3F's compute_HKS.

    The eigenbasis comes from S3F's compute_eigens (point-cloud Laplacian), or,
    when `faces` is given, from the cotan eigenbasis of `compute_operators`
    (`laplacian_eigenbasis`), with S3F's eigenpair count in both cases.
    Accepts CPU or GPU tensors; converts internally. The upstream functions
    assert on degenerate spectra, which would abort a training run, so a
    failure yields zeros for this protein instead.
    """
    from alphasurf.protein.create_operators import laplacian_eigenbasis
    from alphasurf.tasks.s3f_pretrain.s3f_official.surface import (
        compute_eigens,
        compute_HKS,
    )

    pts_np = points.detach().cpu().numpy()
    n = len(pts_np)
    eigs_ratio = HKS_LARGE_EIGS_RATIO if n > HKS_LARGE_SURFACE else HKS_EIGS_RATIO

    try:
        if faces is None:
            evals, evecs, _ = compute_eigens(
                n, pts_np, min_n_eigs=HKS_MIN_EIGS, eigs_ratio=eigs_ratio
            )
        else:
            n_eigs = max(HKS_MIN_EIGS, int(eigs_ratio * n) + 1)
            _, _, evals, evecs = laplacian_eigenbasis(
                pts_np, np.asarray(faces, dtype=np.int64), k_eig=n_eigs
            )
        hks = compute_HKS(
            evecs,
            evals,
            num_t=HKS_DIM,
            t_min=HKS_T_MIN,
            t_max=HKS_T_MAX,
            scale=HKS_SCALE,
        )
    except (AssertionError, ValueError, RuntimeError) as e:
        logger.warning("HKS failed (n=%d): %s", n, e)
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    return hks.astype(np.float32)


def process_one(pdb_path, output_path, device, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return "skip"

    bb_pos, sequence = parse_backbone(pdb_path)
    if bb_pos is None or len(bb_pos) < 16:
        return "skip"

    ca_pos = torch.from_numpy(bb_pos[:, 1]).float()

    try:
        surface = build_s3f_surfaces([bb_pos], device)[0]
    except Exception as e:
        logger.warning("dMaSIF failed for %s: %s", pdb_path, e)
        return "fail"
    if surface is None:
        return "fail"

    data = {
        "sequence": sequence,
        "ca_pos": ca_pos,
        "bb_pos": torch.from_numpy(bb_pos).float(),
        "surf_feature_order": "hks_curv",
        **surface,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    return "ok"


def _worker(args):
    pdb_path, output_path, device, overwrite = args
    return process_one(pdb_path, output_path, device, overwrite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel workers (each gets one GPU if device=cuda). "
        "Use 1 for single-GPU; increase for multi-GPU node.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pdbs = sorted(
        f
        for f in os.listdir(args.pdb_dir)
        if not f.startswith(".") and os.path.isfile(os.path.join(args.pdb_dir, f))
    )
    if args.limit:
        pdbs = pdbs[: args.limit]

    logger.info("Processing %d PDBs -> %s", len(pdbs), args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    tasks = []
    for i, pdb in enumerate(pdbs):
        pdb_path = os.path.join(args.pdb_dir, pdb)
        out_name = pdb
        if out_name.endswith(".pdb"):
            out_name = out_name[: -len(".pdb")]
        output_path = os.path.join(args.output_dir, out_name + ".pt")
        if args.num_workers > 1 and args.device == "cuda":
            worker_device = f"cuda:{i % torch.cuda.device_count()}"
        else:
            worker_device = args.device
        tasks.append((pdb_path, output_path, worker_device, args.overwrite))

    if args.num_workers == 1:
        counts = {"ok": 0, "skip": 0, "fail": 0}
        for t in tqdm(tasks):
            r = _worker(t)
            counts[r] = counts.get(r, 0) + 1
    else:
        # Workers are spawned, not forked: CUDA is unusable in a forked child.
        with get_context("spawn").Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)))
        counts = {"ok": 0, "skip": 0, "fail": 0}
        for r in results:
            counts[r] = counts.get(r, 0) + 1

    logger.info(
        "Done. ok=%d skip=%d fail=%d", counts["ok"], counts["skip"], counts["fail"]
    )


if __name__ == "__main__":
    main()
