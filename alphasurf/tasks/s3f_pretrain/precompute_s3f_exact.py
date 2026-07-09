"""
Precompute dMaSIF point cloud + features for S3F-exact encoder.

For each PDB in the input directory, produces <output_dir>/<name>.pt containing:
  - sequence: str (1-letter AA codes, len n_res)
  - ca_pos: (n_res, 3) Cα coords
  - bb_pos: (n_res, 3, 3) N/CA/C coords per residue (for fusion pooling)
  - surf_pos: (M, 3) dMaSIF point cloud
  - surf_normals: (M, 3)
  - surf_feat: (M, 42) concatenated [curv(10), hks(32)]

dMaSIF point cloud uses N/CA/C backbone atoms only (S3F-exact). Curvature
is computed at scales [1, 2, 3, 5, 10] (mean + Gauss = 10 dims). HKS uses
robust_laplacian point-cloud Laplacian + eigsh, 32 time bins.

Edges (Cα radius=10, surf kNN=16) and cross-mappings (res2surf 60-NN,
surf2res 3-NN) are recomputed at load time after cropping — they are
cheap and depend on the crop window.

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
from multiprocessing import Pool

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


def generate_dmasif_cloud(bb_pos, device):
    """Run dMaSIF atoms_to_points_normals on N/CA/C atoms (GPU).

    S3F atom-type convention: [3, 0, 0] per residue (N, C, C) one-hot over
    {C, H, O, N, S, SE} (index 3 = N, index 0 = C).

    Returns GPU tensors — caller is responsible for moving to CPU when needed
    (HKS uses scipy/numpy which is CPU-only).
    """
    from alphasurf.network_utils.misc_arch.dmasif_utils.geometry_processing import (
        atoms_to_points_normals,
    )

    n_res = bb_pos.shape[0]
    atoms_flat = torch.from_numpy(bb_pos.reshape(-1, 3)).float().to(device)
    batch = torch.zeros(n_res * 3, dtype=torch.long, device=device)

    atom_type_idx = torch.tensor([3, 0, 0], dtype=torch.long, device=device).repeat(
        n_res
    )
    atomtypes = torch.nn.functional.one_hot(atom_type_idx, num_classes=6).float()

    points, normals, _ = atoms_to_points_normals(
        atoms_flat,
        batch,
        num_atoms=6,
        distance=DMASIF_DISTANCE,
        smoothness=DMASIF_SMOOTHNESS,
        resolution=DMASIF_RESOLUTION,
        nits=DMASIF_NITS,
        atomtypes=atomtypes,
        sup_sampling=DMASIF_SUPSAMPLING,
        variance=DMASIF_VARIANCE,
    )
    return points.detach(), normals.detach()


def compute_curvatures(points, normals, batch=None):
    """KeOps curvatures — runs on GPU if input tensors are on GPU."""
    from alphasurf.network_utils.misc_arch.dmasif_utils.geometry_processing import (
        curvatures,
    )

    return curvatures(
        points.float(),
        triangles=None,
        normals=normals.float(),
        scales=CURV_SCALES,
        batch=batch,
    ).detach()


def compute_hks(points):
    """32-dim HKS via robust_laplacian point-cloud Laplacian + eigsh.

    Mirrors S3F script/process_surface.py settings. CPU-only (scipy).
    Accepts CPU or GPU tensor; converts internally.
    """
    import robust_laplacian
    from scipy.sparse.linalg import eigsh

    pts_np = points.detach().cpu().numpy().astype(np.float64)
    n = len(pts_np)
    if n < 10:
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    try:
        L, M = robust_laplacian.point_cloud_laplacian(pts_np)
    except Exception as e:
        logger.warning("robust_laplacian failed (n=%d): %s", n, e)
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    n_eigs = max(min(HKS_MIN_EIGS, n - 2), int(n * HKS_EIGS_RATIO))
    n_eigs = min(n_eigs, n - 2)
    if n_eigs < 4:
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    try:
        evals, evecs = eigsh(L, k=n_eigs, M=M, sigma=1e-8, which="LM")
    except Exception as e:
        logger.warning("eigsh failed (n=%d, k=%d): %s", n, n_eigs, e)
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)
    if evals[1] <= 0:
        return np.zeros((n, HKS_DIM), dtype=np.float32)

    t_list = np.geomspace(HKS_T_MIN, HKS_T_MAX, HKS_DIM, dtype=np.float64)
    phase = np.exp(-np.outer(t_list, evals[1:]))
    wphi = phase[:, None, :] * evecs[None, :, 1:]
    hks = np.einsum("tnk,nk->nt", wphi, evecs[:, 1:]) * HKS_SCALE
    heat_trace = np.sum(phase, axis=1)
    hks /= heat_trace
    return hks.astype(np.float32)


def process_one(pdb_path, output_path, device, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return "skip"

    bb_pos, sequence = parse_backbone(pdb_path)
    if bb_pos is None or len(bb_pos) < 16:
        return "skip"

    ca_pos = torch.from_numpy(bb_pos[:, 1]).float()

    try:
        surf_pos, surf_normals = generate_dmasif_cloud(bb_pos, device)
    except Exception as e:
        logger.warning("dMaSIF failed for %s: %s", pdb_path, e)
        return "fail"
    if surf_pos.shape[0] < 16:
        return "fail"

    curv = compute_curvatures(surf_pos, surf_normals)
    if curv.shape[0] != surf_pos.shape[0]:
        curv = curv[: surf_pos.shape[0]]

    hks = compute_hks(surf_pos)
    if hks.shape[0] != surf_pos.shape[0]:
        hks = hks[: surf_pos.shape[0]]

    surf_pos_cpu = surf_pos.cpu()
    surf_normals_cpu = surf_normals.cpu()
    curv_cpu = curv.cpu()
    surf_feat = torch.cat([curv_cpu, torch.from_numpy(hks).float()], dim=-1)

    data = {
        "sequence": sequence,
        "ca_pos": ca_pos,
        "bb_pos": torch.from_numpy(bb_pos).float(),
        "surf_pos": surf_pos_cpu.float(),
        "surf_normals": surf_normals_cpu.float(),
        "surf_feat": surf_feat.float(),
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
        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)))
        counts = {"ok": 0, "skip": 0, "fail": 0}
        for r in results:
            counts[r] = counts.get(r, 0) + 1

    logger.info(
        "Done. ok=%d skip=%d fail=%d", counts["ok"], counts["skip"], counts["fail"]
    )


if __name__ == "__main__":
    main()
