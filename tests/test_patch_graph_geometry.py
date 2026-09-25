from pathlib import Path

import numpy as np
import pytest

from alphasurf.protein.patch_operators import extract_patch_graph


BINDING_DIR = Path(__file__).resolve().parents[1] / "cgal_alpha_bindings" / "build_py310"
if not list(BINDING_DIR.glob("cgal_patch_graph*.so")):
    pytest.skip("cgal_patch_graph binding is not built", allow_module_level=True)


def extract(positions: np.ndarray):
    return extract_patch_graph(
        positions,
        np.ones(len(positions), dtype=np.float32),
        alpha=0.0,
        probe_radius=0.0,
        binding_dir=BINDING_DIR,
    )


def incident_conormal_sum(graph):
    result = np.zeros((graph.num_patches, 3))
    np.add.at(result, graph.edge_index[0], graph.integrated_conormal[0])
    np.add.at(result, graph.edge_index[1], graph.integrated_conormal[1])
    return result


def test_two_sphere_patch_moments():
    graph = extract(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32))

    np.testing.assert_allclose(graph.patch_area, 3.0 * np.pi)
    for atom, normal, centroid, center in (
        (0, [-1, 0, 0], [-0.25, 0, 0], [-1, 0, 0]),
        (1, [1, 0, 0], [1.25, 0, 0], [2, 0, 0]),
    ):
        patch = np.flatnonzero(graph.patch_atom_index == atom)[0]
        np.testing.assert_allclose(graph.patch_normal[patch], normal, atol=1e-12)
        np.testing.assert_allclose(
            graph.patch_area_centroid[patch], centroid, atol=1e-12
        )
        np.testing.assert_allclose(graph.patch_center[patch], center, atol=1e-12)


def test_patch_geometry_is_rigid_transform_covariant():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]],
        dtype=np.float32,
    )
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 1.0], [1.0, 0.0, 2.0]])
    )
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = np.array([7.0, -3.0, 2.0])

    graph = extract(positions)
    transformed = extract((positions @ rotation.T + translation).astype(np.float32))
    for atom in range(3):
        patch = np.flatnonzero(graph.patch_atom_index == atom)[0]
        transformed_patch = np.flatnonzero(
            transformed.patch_atom_index == atom
        )[0]
        np.testing.assert_allclose(
            transformed.patch_normal[transformed_patch],
            rotation @ graph.patch_normal[patch],
            atol=3e-6,
        )
        np.testing.assert_allclose(
            transformed.patch_center[transformed_patch],
            rotation @ graph.patch_center[patch] + translation,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            transformed.patch_area_centroid[transformed_patch],
            rotation @ graph.patch_area_centroid[patch] + translation,
            atol=3e-6,
        )


def test_integrated_conormals_satisfy_stokes_boundary_identity():
    graph = extract(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]],
            dtype=np.float32,
        )
    )
    normal_moment = (
        graph.patch_area[:, None]
        / graph.patch_radius[:, None]
        * (graph.patch_area_centroid - graph.patch_sphere_center)
    )
    expected = -2.0 / graph.patch_radius[:, None] * normal_moment
    np.testing.assert_allclose(
        incident_conormal_sum(graph), expected, atol=1e-11
    )


def test_full_sphere_has_no_preferred_normal():
    graph = extract(np.zeros((1, 3), dtype=np.float32))

    np.testing.assert_allclose(graph.patch_area, [4.0 * np.pi])
    np.testing.assert_array_equal(graph.patch_normal_valid, [False])
    np.testing.assert_allclose(graph.patch_normal, 0.0)
    np.testing.assert_allclose(graph.patch_center, graph.patch_sphere_center)
