#!/usr/bin/env python
"""Test curvature_ext against igl and analytical ground truth on various shapes."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.test_principal_curvature import make_icosphere

# ---- Mesh generators ----


def make_torus(R=1.0, r=0.4, n_major=40, n_minor=20):
    theta = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    V = np.column_stack([x, y, z])
    faces = []
    for i in range(n_minor):
        for j in range(n_major):
            v0 = i * n_major + j
            v1 = i * n_major + (j + 1) % n_major
            v2 = ((i + 1) % n_minor) * n_major + (j + 1) % n_major
            v3 = ((i + 1) % n_minor) * n_major + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(V, dtype=np.float64), np.array(faces, dtype=np.int64), phi


def make_cylinder(R=1.0, height=4.0, n_theta=40, n_z=20):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-height / 2, height / 2, n_z)
    theta, z = np.meshgrid(theta, z_vals)
    theta = theta.ravel()
    z = z.ravel()
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    V = np.column_stack([x, y, z])
    faces = []
    for i in range(n_z - 1):
        for j in range(n_theta):
            v0 = i * n_theta + j
            v1 = i * n_theta + (j + 1) % n_theta
            v2 = (i + 1) * n_theta + (j + 1) % n_theta
            v3 = (i + 1) * n_theta + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(V, dtype=np.float64), np.array(faces, dtype=np.int64)


def make_saddle(a=1.0, b=1.0, n=30, extent=1.0):
    u = np.linspace(-extent, extent, n)
    v = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(u, v)
    U = U.ravel()
    V = V.ravel()
    Z = a * U**2 - b * V**2
    verts = np.column_stack([U, V, Z])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j + 1
            v3 = (i + 1) * n + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def make_ellipsoid(a=1.5, b=1.0, c=0.6, n_lat=30, n_lon=40):
    theta = np.linspace(0, np.pi, n_lat)
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    V = np.column_stack([x, y, z])
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + (j + 1) % n_lon
            v3 = (i + 1) * n_lon + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(V, dtype=np.float64), np.array(faces, dtype=np.int64)


def make_bumpy_sphere(R=1.0, n_bumps=5, amp=0.05, n_lat=30, n_lon=40):
    theta = np.linspace(0, np.pi, n_lat)
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    r = R + amp * np.sin(n_bumps * theta) * np.cos(n_bumps * phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    V = np.column_stack([x, y, z])
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + (j + 1) % n_lon
            v3 = (i + 1) * n_lon + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(V, dtype=np.float64), np.array(faces, dtype=np.int64)


def make_gaussian_bump(n=30, extent=2.0, sigma=0.5):
    u = np.linspace(-extent, extent, n)
    v = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(u, v)
    U = U.ravel()
    V = V.ravel()
    Z = np.exp(-(U**2 + V**2) / sigma**2)
    verts = np.column_stack([U, V, Z])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j + 1
            v3 = (i + 1) * n + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# ---- Helpers ----


def safe_normals(V, F):
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(norms > 1e-12, fn / norms, 0.0)
    vn = np.zeros_like(V)
    np.add.at(vn, F[:, 0], fn)
    np.add.at(vn, F[:, 1], fn)
    np.add.at(vn, F[:, 2], fn)
    norms_v = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(norms_v > 1e-12, vn / norms_v, 0.0)
    return vn


def compare_vs_igl(V, F, label):
    import curvature_ext
    import igl

    PD1, PD2, PV1, PV2 = curvature_ext.principal_curvature(V, F)
    PD1_i, PD2_i, PV1_i, PV2_i = igl.principal_curvature(V, F)

    mask = (np.abs(PV1_i) > 1e-10) | (np.abs(PV2_i) > 1e-10)
    n_valid = np.sum(mask)
    if n_valid == 0:
        print(f"  [{label}] no valid igl vertices")
        return 0.0, 0.0, 0.0

    corr_k1 = np.corrcoef(np.abs(PV1[mask]), np.abs(PV1_i[mask]))[0, 1]
    corr_k2 = np.corrcoef(np.abs(PV2[mask]), np.abs(PV2_i[mask]))[0, 1]
    mae_k1 = np.mean(np.abs(np.abs(PV1[mask]) - np.abs(PV1_i[mask])))
    mae_k2 = np.mean(np.abs(np.abs(PV2[mask]) - np.abs(PV2_i[mask])))

    dmask = (np.linalg.norm(PD1, axis=1) > 1e-10) & (
        np.linalg.norm(PD1_i, axis=1) > 1e-10
    )
    cos_pd1, cos_pd2 = float("nan"), float("nan")
    if np.sum(dmask) > 0:
        cos_pd1 = np.mean(np.abs(np.sum(PD1[dmask] * PD1_i[dmask], axis=1)))
        cos_pd2 = np.mean(np.abs(np.sum(PD2[dmask] * PD2_i[dmask], axis=1)))

    print(f"[{label}]  V={len(V)}  F={len(F)}  valid={n_valid}")
    print(f"  |k1| corr={corr_k1:.6f}  MAE={mae_k1:.6f}")
    print(f"  |k2| corr={corr_k2:.6f}  MAE={mae_k2:.6f}")
    print(f"  PD1 |cos|={cos_pd1:.6f}   PD2 |cos|={cos_pd2:.6f}")
    return corr_k1, corr_k2, cos_pd1


# ---- Tests ----


def test_sphere_ground_truth():
    import curvature_ext

    V, F = make_icosphere(subdivisions=4)
    _, _, PV1, PV2 = curvature_ext.principal_curvature(V, F)

    mask = np.abs(PV1) > 1e-10
    mean_k1 = np.mean(PV1[mask])
    mean_k2 = np.mean(PV2[mask])

    print(f"[sphere_gt]  V={len(V)}  truth: k1=k2=1.0")
    print(f"  mean k1={mean_k1:.6f}  err={abs(mean_k1 - 1.0):.6f}")
    print(f"  mean k2={mean_k2:.6f}  err={abs(mean_k2 - 1.0):.6f}")
    assert abs(mean_k1 - 1.0) < 0.05
    assert abs(mean_k2 - 1.0) < 0.05
    print("  PASSED\n")


def test_torus_vs_igl():
    V, F, _ = make_torus(R=1.0, r=0.4, n_major=40, n_minor=20)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "torus")
    assert corr_k1 > 0.90, f"torus k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.95, f"torus k2 corr too low: {corr_k2}"
    assert cos_pd1 > 0.99, f"torus PD1 alignment too low: {cos_pd1}"
    print("  PASSED\n")


def test_cylinder_vs_igl():
    V, F = make_cylinder(R=1.0, height=4.0, n_theta=40, n_z=20)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "cylinder")
    assert corr_k1 > 0.90, f"cylinder k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.90, f"cylinder k2 corr too low: {corr_k2}"
    print("  PASSED\n")


def test_ellipsoid_vs_igl():
    V, F = make_ellipsoid(a=1.5, b=1.0, c=0.6, n_lat=25, n_lon=35)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "ellipsoid")
    assert corr_k1 > 0.95, f"ellipsoid k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.95, f"ellipsoid k2 corr too low: {corr_k2}"
    assert cos_pd1 > 0.95, f"ellipsoid PD1 alignment too low: {cos_pd1}"
    print("  PASSED\n")


def test_saddle_vs_igl():
    V, F = make_saddle(a=1.0, b=1.0, n=25, extent=1.0)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "saddle")
    assert corr_k1 > 0.95, f"saddle k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.95, f"saddle k2 corr too low: {corr_k2}"
    print("  PASSED\n")


def test_bumpy_sphere_vs_igl():
    V, F = make_bumpy_sphere(R=1.0, n_bumps=5, amp=0.05, n_lat=25, n_lon=35)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "bumpy_sphere")
    assert corr_k1 > 0.95, f"bumpy sphere k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.95, f"bumpy sphere k2 corr too low: {corr_k2}"
    assert cos_pd1 > 0.95, f"bumpy sphere PD1 alignment too low: {cos_pd1}"
    print("  PASSED\n")


def test_gaussian_bump_vs_igl():
    V, F = make_gaussian_bump(n=25, extent=2.0, sigma=0.5)
    corr_k1, corr_k2, cos_pd1 = compare_vs_igl(V, F, "gaussian_bump")
    assert corr_k1 > 0.95, f"gaussian bump k1 corr too low: {corr_k1}"
    assert corr_k2 > 0.95, f"gaussian bump k2 corr too low: {corr_k2}"
    print("  PASSED\n")


def test_normals_match():
    """Passing normals explicitly should give identical results to computing them internally."""
    import curvature_ext

    V, F, _ = make_torus(R=1.0, r=0.4, n_major=30, n_minor=15)
    vn = safe_normals(V, F)

    _, _, PV1_no, PV2_no = curvature_ext.principal_curvature(V, F)
    _, _, PV1_wn, PV2_wn = curvature_ext.principal_curvature(V, F, normals=vn)

    diff_k1 = np.max(np.abs(PV1_no - PV1_wn))
    diff_k2 = np.max(np.abs(PV2_no - PV2_wn))

    print("[normals_match]  torus  computed-vs-passed normals")
    print(f"  k1 max diff: {diff_k1:.2e}")
    print(f"  k2 max diff: {diff_k2:.2e}")
    assert diff_k1 < 1e-8
    assert diff_k2 < 1e-8
    print("  PASSED\n")


def test_degenerate_stress():
    """Inject 10% degenerate faces -- must not crash."""
    import curvature_ext

    V, F = make_icosphere(subdivisions=2)

    rng = np.random.RandomState(42)
    n_degen = max(1, len(F) // 10)
    degen_idx = rng.choice(len(F), n_degen, replace=False)
    F_bad = F.copy()
    for idx in degen_idx:
        kind = rng.randint(3)
        if kind == 0:
            F_bad[idx] = [F_bad[idx, 0]] * 3
        elif kind == 1:
            F_bad[idx] = [F_bad[idx, 0], F_bad[idx, 0], F_bad[idx, 2]]
        else:
            F_bad[idx] = [F_bad[idx, 1], F_bad[idx, 2], F_bad[idx, 2]]

    vn = safe_normals(V, F_bad)
    n_zero = np.sum(np.linalg.norm(vn, axis=1) < 0.5)

    print(
        f"[degenerate_stress]  V={len(V)}  F={len(F_bad)}  degen={n_degen}  zero_normals={n_zero}"
    )

    PD1, PD2, PV1, PV2 = curvature_ext.principal_curvature(V, F_bad, normals=vn)

    n_computed = np.sum(np.abs(PV1) > 0)
    print(f"  computed={n_computed}  skipped={len(V) - n_computed}")
    print("  PASSED (no crash)\n")


def test_zero_normals():
    """Explicitly zero some normals -- those vertices must be skipped."""
    import curvature_ext

    V, F = make_icosphere(subdivisions=2)
    vn = safe_normals(V, F)

    rng = np.random.RandomState(7)
    zero_idx = rng.choice(len(V), 5, replace=False)
    vn[zero_idx] = 0.0

    print(f"[zero_normals]  zeroed vertices: {sorted(zero_idx)}")

    PD1, PD2, PV1, PV2 = curvature_ext.principal_curvature(V, F, normals=vn)

    for vi in zero_idx:
        assert PV1[vi] == 0.0, f"vertex {vi} should have zero curvature, got {PV1[vi]}"
    n_ok = np.sum(np.abs(PV1) > 0)
    assert n_ok > len(V) * 0.5, f"too few computed curvatures: {n_ok}/{len(V)}"
    print(f"  all zeroed vertices skipped, {n_ok}/{len(V)} computed OK")
    print("  PASSED\n")


def test_protein_surfaces():
    """Test curvature on real protein surfaces from MaSIF dataset."""
    import glob

    import curvature_ext
    import igl
    import torch

    surface_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "masif_ligand",
        "msms_01",
        "surfaces_full_msms_fr0.1",
    )
    surface_dir = os.path.abspath(surface_dir)

    pt_files = sorted(glob.glob(os.path.join(surface_dir, "*.pt")))
    if not pt_files:
        print("[protein_surfaces]  SKIP: no .pt files found in " + surface_dir)
        return

    n_test = min(10, len(pt_files))
    print(
        f"[protein_surfaces]  testing on {n_test} real surfaces from {os.path.basename(surface_dir)}"
    )

    corr_k1_all = []
    corr_k2_all = []
    for pt_file in pt_files[:n_test]:
        data = torch.load(pt_file, weights_only=False)
        V = data.verts.numpy().astype(np.float64)
        F = data.faces.numpy().astype(np.int64)
        vn = data.vnormals.numpy().astype(np.float64)

        PD1, PD2, PV1, PV2 = curvature_ext.principal_curvature(V, F, normals=vn)
        _, _, PV1_i, PV2_i = igl.principal_curvature(V, F)

        mask = (np.abs(PV1_i) > 1e-10) | (np.abs(PV2_i) > 1e-10)
        if mask.sum() < 10:
            print(f"  {os.path.basename(pt_file)}: too few valid vertices, skipping")
            continue

        corr_k1 = np.corrcoef(np.abs(PV1[mask]), np.abs(PV1_i[mask]))[0, 1]
        corr_k2 = np.corrcoef(np.abs(PV2[mask]), np.abs(PV2_i[mask]))[0, 1]
        corr_k1_all.append(corr_k1)
        corr_k2_all.append(corr_k2)
        print(
            f"  {os.path.basename(pt_file):20s}  V={len(V):5d}  F={len(F):5d}  k1_corr={corr_k1:.4f}  k2_corr={corr_k2:.4f}"
        )

    mean_k1 = np.mean(corr_k1_all)
    mean_k2 = np.mean(corr_k2_all)
    print(
        f"  mean k1_corr={mean_k1:.4f}  mean k2_corr={mean_k2:.4f}  ({len(corr_k1_all)} surfaces)"
    )
    assert mean_k1 > 0.90, f"mean k1 corr too low: {mean_k1}"
    assert mean_k2 > 0.90, f"mean k2 corr too low: {mean_k2}"
    print("  PASSED\n")


def test_speed():
    import curvature_ext
    import igl

    V, F = make_icosphere(subdivisions=4)
    N = 5

    t0 = time.time()
    for _ in range(N):
        curvature_ext.principal_curvature(V, F)
    t_ours = (time.time() - t0) / N

    t0 = time.time()
    for _ in range(N):
        igl.principal_curvature(V, F)
    t_igl = (time.time() - t0) / N

    print(f"[speed]  sphere subdiv=4  V={len(V)}")
    print(f"  curvature_ext: {t_ours:.4f}s")
    print(f"  igl:           {t_igl:.4f}s")
    print(f"  ratio:         {t_ours / t_igl:.2f}x\n")


def main():
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("  curvature_ext - comprehensive test suite")
    print("=" * 60 + "\n")

    test_sphere_ground_truth()
    test_torus_vs_igl()
    test_cylinder_vs_igl()
    test_ellipsoid_vs_igl()
    test_saddle_vs_igl()
    test_bumpy_sphere_vs_igl()
    test_gaussian_bump_vs_igl()
    test_normals_match()
    test_degenerate_stress()
    test_zero_normals()
    test_protein_surfaces()
    test_speed()

    print("All tests passed.")


if __name__ == "__main__":
    main()
