import importlib.util
from pathlib import Path

import numpy as np
import pytest

from alphasurf.protein.patch_operators import (
    build_patch_operators,
    extract_patch_graph,
)


BINDING_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "cgal_alpha_bindings" / "build_py310"
)
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cgal_patch_graph") is None
    and not any(BINDING_DIRECTORY.glob("cgal_patch_graph*.so")),
    reason="cgal_patch_graph has not been built",
)


def test_single_sphere_area_and_empty_adjacency():
    graph = extract_patch_graph(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        alpha=0.0,
        probe_radius=0.0,
        binding_dir=BINDING_DIRECTORY,
    )

    assert graph.num_patches == 1
    assert graph.num_edges == 0
    assert graph.integrated_conormal.shape == (2, 0, 3)
    np.testing.assert_allclose(graph.patch_area, [4.0 * np.pi], rtol=1e-12)
    np.testing.assert_array_equal(graph.patch_atom_index, [0])


def test_two_overlapping_spheres_have_exact_shared_circle():
    graph = extract_patch_graph(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        alpha=0.0,
        probe_radius=0.0,
        binding_dir=BINDING_DIRECTORY,
    )

    expected_patch_area = 3.0 * np.pi
    expected_arc_length = 2.0 * np.pi * np.sqrt(3.0 / 4.0)
    assert graph.num_patches == 2
    assert graph.num_edges == 1
    np.testing.assert_allclose(graph.patch_area, expected_patch_area, rtol=1e-12)
    np.testing.assert_allclose(graph.shared_arc_length, expected_arc_length, rtol=1e-12)
    np.testing.assert_array_equal(graph.arc_count, [1])

    expected_conormal = 1.5 * np.pi
    for atom, direction in ((0, 1.0), (1, -1.0)):
        patch = np.flatnonzero(graph.patch_atom_index == atom)[0]
        endpoint = np.flatnonzero(graph.edge_index[:, 0] == patch)[0]
        np.testing.assert_allclose(
            graph.integrated_conormal[endpoint, 0],
            [direction * expected_conormal, 0.0, 0.0],
            atol=1e-12,
        )

    operators = build_patch_operators(graph, k_eig=2)
    np.testing.assert_allclose(operators.stiffness @ np.ones(2), 0.0, atol=1e-12)


def test_one_atom_can_own_multiple_disconnected_patches():
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ring = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles), np.zeros(6)))
    positions = np.vstack(([0.0, 0.0, 0.0], ring)).astype(np.float32)
    radii = np.array([3.0] + [2.0] * 6, dtype=np.float32)

    graph = extract_patch_graph(
        positions,
        radii,
        alpha=0.0,
        probe_radius=0.0,
        binding_dir=BINDING_DIRECTORY,
    )

    assert np.count_nonzero(graph.patch_atom_index == 0) == 2
    operators = build_patch_operators(graph, k_eig=graph.num_patches)
    assert np.min(operators.patch_distance) > 0.0
    assert np.min(operators.eigenvalues) >= 0.0
