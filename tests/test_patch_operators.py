import numpy as np
import pytest

from alphasurf.protein.patch_operators import (
    SurfacePatchGraph,
    build_patch_operators,
    center_distance,
    induced_patch_subgraph,
    load_patch_graph,
    save_patch_graph,
)


def chain_graph() -> SurfacePatchGraph:
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    return SurfacePatchGraph(
        patch_area=np.array([1.0, 2.0, 3.0]),
        patch_center=centers,
        patch_sphere_center=centers,
        patch_area_centroid=centers,
        patch_normal=np.tile([0.0, 0.0, 1.0], (3, 1)),
        patch_normal_valid=np.ones(3, dtype=bool),
        patch_radius=np.ones(3),
        patch_atom_index=np.arange(3),
        edge_index=np.array([[0, 1], [1, 2]]),
        shared_arc_length=np.array([2.0, 4.0]),
        integrated_conormal=np.array(
            [
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
            ]
        ),
        arc_count=np.ones(2, dtype=np.int64),
    )


def test_mass_stiffness_and_conductance():
    graph = chain_graph()
    operators = build_patch_operators(graph, k_eig=3)

    np.testing.assert_allclose(operators.patch_distance, [1.0, 2.0])
    np.testing.assert_allclose(operators.edge_conductance, [2.0, 2.0])
    np.testing.assert_allclose(
        operators.stiffness.toarray(),
        [[2.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 2.0]],
    )
    np.testing.assert_allclose(operators.mass.diagonal(), graph.patch_area)
    np.testing.assert_allclose(operators.stiffness @ np.ones(3), 0.0, atol=1e-12)
    np.testing.assert_allclose(operators.apply_laplacian(np.ones(3)), 0.0)
    np.testing.assert_allclose(
        operators.symmetric_stiffness.toarray(),
        operators.symmetric_stiffness.toarray().T,
    )
    assert np.min(operators.eigenvalues) >= 0

    fields = operators.diffusionnet_fields()
    assert fields["mass"].shape == (3, 3)
    assert fields["L"].shape == (3, 3)
    assert fields["evals"].shape == (3,)
    assert fields["evecs"].shape == (3, 3)
    assert fields["gradX"].nnz > 0
    assert fields["gradY"].nnz == 0
    np.testing.assert_allclose(fields["gradX"] @ np.ones(3), 0.0, atol=1e-7)
    np.testing.assert_allclose(fields["gradY"] @ np.ones(3), 0.0, atol=1e-7)


def test_patch_gradient_approximates_planar_affine_function():
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    graph = SurfacePatchGraph(
        patch_area=np.ones(3),
        patch_center=centers,
        patch_sphere_center=centers,
        patch_area_centroid=centers,
        patch_normal=np.tile([0.0, 0.0, 1.0], (3, 1)),
        patch_normal_valid=np.ones(3, dtype=bool),
        patch_radius=np.ones(3),
        patch_atom_index=np.arange(3),
        edge_index=np.array([[0, 0, 1], [1, 2, 2]]),
        shared_arc_length=np.ones(3),
        integrated_conormal=np.array(
            [
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.2, 0.0]],
                [[-2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.8, 0.0, 0.0]],
            ]
        ),
    )
    operators = build_patch_operators(graph, k_eig=3)
    values = 2.0 * centers[:, 0] - 3.0 * centers[:, 1] + 4.0

    np.testing.assert_allclose(operators.grad_x @ values, 2.0, rtol=0.0, atol=0.04)
    np.testing.assert_allclose(operators.grad_y @ values, -3.0, rtol=0.0, atol=0.04)


def test_diffusion_preserves_constant_and_mass_integral():
    operators = build_patch_operators(chain_graph(), k_eig=3)
    constant = np.full((3, 2), 2.5)
    np.testing.assert_allclose(
        operators.diffuse(constant, np.array([0.2, 3.0])),
        constant,
        atol=1e-10,
    )

    values = np.array([[1.0], [-2.0], [4.0]])
    diffused = operators.diffuse(values, 1.7)
    before = operators.mass_vector @ values[:, 0]
    after = operators.mass_vector @ diffused[:, 0]
    np.testing.assert_allclose(after, before, atol=1e-10)


def test_multiple_patches_can_share_parent_atom():
    graph = chain_graph()
    graph = SurfacePatchGraph(
        patch_area=graph.patch_area,
        patch_center=graph.patch_center,
        patch_sphere_center=graph.patch_sphere_center,
        patch_area_centroid=graph.patch_area_centroid,
        patch_normal=graph.patch_normal,
        patch_normal_valid=graph.patch_normal_valid,
        patch_radius=graph.patch_radius,
        patch_atom_index=np.array([0, 0, 1]),
        edge_index=graph.edge_index,
        shared_arc_length=graph.shared_arc_length,
        integrated_conormal=graph.integrated_conormal,
    )
    operators = build_patch_operators(graph, k_eig=3)
    assert operators.stiffness.shape == (3, 3)


def test_center_distance_rejects_adjacent_same_center_patches():
    graph = SurfacePatchGraph(
        patch_area=np.ones(2),
        patch_center=np.zeros((2, 3)),
        patch_sphere_center=np.zeros((2, 3)),
        patch_area_centroid=np.zeros((2, 3)),
        patch_normal=np.tile([1.0, 0.0, 0.0], (2, 1)),
        patch_normal_valid=np.ones(2, dtype=bool),
        patch_radius=np.ones(2),
        patch_atom_index=np.zeros(2, dtype=np.int64),
        edge_index=np.array([[0], [1]]),
        shared_arc_length=np.ones(1),
        integrated_conormal=np.zeros((2, 1, 3)),
    )
    np.testing.assert_allclose(center_distance(graph), [0.0])
    with pytest.raises(ValueError, match="same-center"):
        build_patch_operators(graph)


def test_rejects_noncanonical_undirected_edges():
    with pytest.raises(ValueError, match="canonical"):
        SurfacePatchGraph(
            patch_area=np.ones(2),
            patch_center=np.zeros((2, 3)),
            patch_sphere_center=np.zeros((2, 3)),
            patch_area_centroid=np.zeros((2, 3)),
            patch_normal=np.tile([1.0, 0.0, 0.0], (2, 1)),
            patch_normal_valid=np.ones(2, dtype=bool),
            patch_radius=np.ones(2),
            patch_atom_index=np.arange(2),
            edge_index=np.array([[1], [0]]),
            shared_arc_length=np.ones(1),
            integrated_conormal=np.zeros((2, 1, 3)),
        )


def test_requires_all_disconnected_constant_modes():
    graph = SurfacePatchGraph(
        patch_area=np.ones(3),
        patch_center=np.eye(3),
        patch_sphere_center=np.zeros((3, 3)),
        patch_area_centroid=np.zeros((3, 3)),
        patch_normal=np.eye(3),
        patch_normal_valid=np.ones(3, dtype=bool),
        patch_radius=np.ones(3),
        patch_atom_index=np.arange(3),
        edge_index=np.empty((2, 0), dtype=np.int64),
        shared_arc_length=np.empty(0),
        integrated_conormal=np.empty((2, 0, 3)),
    )
    with pytest.raises(ValueError, match="connected components"):
        build_patch_operators(graph, k_eig=2)


def test_induced_patch_subgraph_remaps_nodes_and_edges():
    graph = chain_graph()
    subgraph = induced_patch_subgraph(graph, np.array([False, True, True]))
    np.testing.assert_allclose(subgraph.patch_area, [2.0, 3.0])
    np.testing.assert_array_equal(subgraph.edge_index, [[0], [1]])
    np.testing.assert_allclose(subgraph.shared_arc_length, [4.0])
    np.testing.assert_allclose(
        subgraph.integrated_conormal,
        graph.integrated_conormal[:, 1:2],
    )
    np.testing.assert_array_equal(subgraph.patch_atom_index, [1, 2])


def test_patch_graph_cache_round_trip_and_geometry_validation(tmp_path):
    graph = chain_graph()
    atom_positions = graph.patch_center.astype(np.float32)
    atom_radii = graph.patch_radius.astype(np.float32)
    cache_path = tmp_path / "protein.npz"

    save_patch_graph(
        cache_path,
        graph,
        alpha=0.0,
        probe_radius=1.4,
        atom_positions=atom_positions,
        atom_radii=atom_radii,
    )
    loaded = load_patch_graph(
        cache_path,
        alpha=0.0,
        probe_radius=1.4,
        atom_positions=atom_positions,
        atom_radii=atom_radii,
    )
    np.testing.assert_allclose(loaded.patch_area, graph.patch_area)
    np.testing.assert_allclose(loaded.patch_center, graph.patch_center)
    np.testing.assert_array_equal(loaded.edge_index, graph.edge_index)
    np.testing.assert_allclose(
        loaded.shared_arc_length, graph.shared_arc_length
    )
    np.testing.assert_allclose(
        loaded.integrated_conormal, graph.integrated_conormal
    )

    changed_positions = atom_positions.copy()
    changed_positions[0, 0] += 0.1
    with pytest.raises(ValueError, match="does not match the atom geometry"):
        load_patch_graph(
            cache_path,
            alpha=0.0,
            probe_radius=1.4,
            atom_positions=changed_positions,
            atom_radii=atom_radii,
        )
