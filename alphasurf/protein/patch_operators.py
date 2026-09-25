"""Finite-volume-inspired diffusion operators on spherical-patch graphs."""

from __future__ import annotations

import importlib
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


Array = np.ndarray
DistanceFunction = Callable[["SurfacePatchGraph"], Array]


@dataclass(frozen=True)
class SurfacePatchGraph:
    """Geometry exported by the ``cgal_patch_graph`` extension.

    ``edge_index`` contains each undirected adjacent pair once. Different
    geometric arcs between the same pair have already been summed in
    ``shared_arc_length``.
    """

    patch_area: Array
    patch_center: Array
    patch_sphere_center: Array
    patch_area_centroid: Array
    patch_normal: Array
    patch_normal_valid: Array
    patch_radius: Array
    patch_atom_index: Array
    edge_index: Array
    shared_arc_length: Array
    integrated_conormal: Array
    arc_count: Array | None = None
    node_features: Array | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "patch_area", np.asarray(self.patch_area, dtype=np.float64)
        )
        object.__setattr__(
            self, "patch_center", np.asarray(self.patch_center, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "patch_sphere_center",
            np.asarray(self.patch_sphere_center, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "patch_area_centroid",
            np.asarray(self.patch_area_centroid, dtype=np.float64),
        )
        object.__setattr__(
            self, "patch_normal", np.asarray(self.patch_normal, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "patch_normal_valid",
            np.asarray(self.patch_normal_valid, dtype=bool),
        )
        object.__setattr__(
            self, "patch_radius", np.asarray(self.patch_radius, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "patch_atom_index",
            np.asarray(self.patch_atom_index, dtype=np.int64),
        )
        object.__setattr__(
            self, "edge_index", np.asarray(self.edge_index, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "shared_arc_length",
            np.asarray(self.shared_arc_length, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "integrated_conormal",
            np.asarray(self.integrated_conormal, dtype=np.float64),
        )
        if self.arc_count is not None:
            object.__setattr__(
                self, "arc_count", np.asarray(self.arc_count, dtype=np.int64)
            )
        if self.node_features is not None:
            object.__setattr__(self, "node_features", np.asarray(self.node_features))
        self.validate()

    @property
    def num_patches(self) -> int:
        return int(self.patch_area.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @classmethod
    def from_mapping(cls, values: Mapping[str, Array], **kwargs) -> "SurfacePatchGraph":
        return cls(
            patch_area=values["patch_area"],
            patch_center=values["patch_center"],
            patch_sphere_center=values["patch_sphere_center"],
            patch_area_centroid=values["patch_area_centroid"],
            patch_normal=values["patch_normal"],
            patch_normal_valid=values["patch_normal_valid"],
            patch_radius=values["patch_radius"],
            patch_atom_index=values["patch_atom_index"],
            edge_index=values["edge_index"],
            shared_arc_length=values["shared_arc_length"],
            integrated_conormal=values["integrated_conormal"],
            arc_count=values.get("arc_count"),
            **kwargs,
        )

    def validate(self) -> None:
        n = self.num_patches
        if self.patch_area.ndim != 1:
            raise ValueError("patch_area must have shape [N]")
        if self.patch_center.shape != (n, 3):
            raise ValueError("patch_center must have shape [N, 3]")
        if self.patch_sphere_center.shape != (n, 3):
            raise ValueError("patch_sphere_center must have shape [N, 3]")
        if self.patch_area_centroid.shape != (n, 3):
            raise ValueError("patch_area_centroid must have shape [N, 3]")
        if self.patch_normal.shape != (n, 3):
            raise ValueError("patch_normal must have shape [N, 3]")
        if self.patch_normal_valid.shape != (n,):
            raise ValueError("patch_normal_valid must have shape [N]")
        if self.patch_radius.shape != (n,):
            raise ValueError("patch_radius must have shape [N]")
        if self.patch_atom_index.shape != (n,):
            raise ValueError("patch_atom_index must have shape [N]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.shared_arc_length.shape != (self.num_edges,):
            raise ValueError("shared_arc_length must have shape [E]")
        if self.integrated_conormal.shape != (2, self.num_edges, 3):
            raise ValueError("integrated_conormal must have shape [2, E, 3]")
        if self.arc_count is not None and self.arc_count.shape != (self.num_edges,):
            raise ValueError("arc_count must have shape [E]")
        if self.node_features is not None:
            if self.node_features.ndim != 2 or self.node_features.shape[0] != n:
                raise ValueError("node_features must have shape [N, C]")

        if not np.all(np.isfinite(self.patch_area)) or np.any(self.patch_area <= 0):
            raise ValueError("all patch areas must be finite and strictly positive")
        if not np.all(np.isfinite(self.patch_center)):
            raise ValueError("patch centers must be finite")
        if not np.all(np.isfinite(self.patch_sphere_center)):
            raise ValueError("patch sphere centers must be finite")
        if not np.all(np.isfinite(self.patch_area_centroid)):
            raise ValueError("patch area centroids must be finite")
        if not np.all(np.isfinite(self.patch_normal)):
            raise ValueError("patch normals must be finite")
        normal_norm = np.linalg.norm(self.patch_normal, axis=1)
        if not np.allclose(normal_norm[self.patch_normal_valid], 1.0, atol=1e-8):
            raise ValueError("valid patch normals must have unit length")
        if np.any(normal_norm[~self.patch_normal_valid] != 0.0):
            raise ValueError("invalid patch normals must be zero")
        if not np.all(np.isfinite(self.patch_radius)) or np.any(self.patch_radius <= 0):
            raise ValueError("all patch radii must be finite and strictly positive")
        if np.any(self.patch_atom_index < 0):
            raise ValueError("patch_atom_index must be non-negative")
        if self.num_edges:
            if np.any(self.edge_index < 0) or np.any(self.edge_index >= n):
                raise ValueError("edge_index contains an invalid patch index")
            if np.any(self.edge_index[0] >= self.edge_index[1]):
                raise ValueError(
                    "edge_index must contain canonical undirected pairs i < j"
                )
            if not np.all(np.isfinite(self.shared_arc_length)) or np.any(
                self.shared_arc_length <= 0
            ):
                raise ValueError(
                    "shared arc lengths must be finite and strictly positive"
                )
            if not np.all(np.isfinite(self.integrated_conormal)):
                raise ValueError("integrated conormals must be finite")


def atom_geometry_digest(positions: Array, radii: Array) -> str:
    """Return a stable digest for the exact atom geometry used by the binding."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    radii = np.ascontiguousarray(radii, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(np.asarray(positions.shape, dtype=np.int64).tobytes())
    digest.update(positions.tobytes())
    digest.update(np.asarray(radii.shape, dtype=np.int64).tobytes())
    digest.update(radii.tobytes())
    return digest.hexdigest()


def save_patch_graph(
    path: str | os.PathLike[str],
    graph: SurfacePatchGraph,
    *,
    alpha: float,
    probe_radius: float,
    atom_positions: Array,
    atom_radii: Array,
) -> None:
    """Atomically save a full-protein patch graph and its geometry metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        format_version=np.asarray(3, dtype=np.int64),
        alpha=np.asarray(alpha, dtype=np.float64),
        probe_radius=np.asarray(probe_radius, dtype=np.float64),
        atom_count=np.asarray(len(atom_positions), dtype=np.int64),
        atom_geometry_digest=np.asarray(
            atom_geometry_digest(atom_positions, atom_radii)
        ),
        patch_area=graph.patch_area,
        patch_center=graph.patch_center,
        patch_sphere_center=graph.patch_sphere_center,
        patch_area_centroid=graph.patch_area_centroid,
        patch_normal=graph.patch_normal,
        patch_normal_valid=graph.patch_normal_valid,
        patch_radius=graph.patch_radius,
        patch_atom_index=graph.patch_atom_index,
        edge_index=graph.edge_index,
        shared_arc_length=graph.shared_arc_length,
        integrated_conormal=graph.integrated_conormal,
        arc_count=(
            graph.arc_count
            if graph.arc_count is not None
            else np.ones(graph.num_edges, dtype=np.int64)
        ),
    )
    os.replace(temporary_path, path)


def load_patch_graph(
    path: str | os.PathLike[str],
    *,
    alpha: float,
    probe_radius: float,
    atom_positions: Array,
    atom_radii: Array,
) -> SurfacePatchGraph:
    """Load a cached graph, rejecting incompatible or stale atom geometry."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as values:
        version = int(values["format_version"].item())
        cached_alpha = float(values["alpha"].item())
        cached_probe_radius = float(values["probe_radius"].item())
        cached_atom_count = int(values["atom_count"].item())
        cached_digest = str(values["atom_geometry_digest"].item())
        if version != 3:
            raise ValueError(f"unsupported patch-graph cache version {version}")
        if not np.isclose(cached_alpha, alpha, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"cached alpha {cached_alpha} does not match requested alpha {alpha}"
            )
        if not np.isclose(
            cached_probe_radius, probe_radius, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "cached probe radius "
                f"{cached_probe_radius} does not match requested {probe_radius}"
            )
        if cached_atom_count != len(atom_positions):
            raise ValueError(
                f"cached atom count {cached_atom_count} does not match "
                f"requested {len(atom_positions)}"
            )
        requested_digest = atom_geometry_digest(atom_positions, atom_radii)
        if cached_digest != requested_digest:
            raise ValueError("cached patch graph does not match the atom geometry")
        return SurfacePatchGraph.from_mapping(
            {
                "patch_area": values["patch_area"],
                "patch_center": values["patch_center"],
                "patch_sphere_center": values["patch_sphere_center"],
                "patch_area_centroid": values["patch_area_centroid"],
                "patch_normal": values["patch_normal"],
                "patch_normal_valid": values["patch_normal_valid"],
                "patch_radius": values["patch_radius"],
                "patch_atom_index": values["patch_atom_index"],
                "edge_index": values["edge_index"],
                "shared_arc_length": values["shared_arc_length"],
                "integrated_conormal": values["integrated_conormal"],
                "arc_count": values["arc_count"],
            }
        )


@dataclass(frozen=True)
class SurfacePatchOperators:
    """Sparse mass/stiffness matrices and a mass-orthonormal spectral basis."""

    mass_vector: Array
    mass: scipy.sparse.csc_matrix
    stiffness: scipy.sparse.csc_matrix
    symmetric_stiffness: scipy.sparse.csc_matrix
    eigenvalues: Array
    eigenvectors: Array
    patch_distance: Array
    edge_conductance: Array
    grad_x: scipy.sparse.csc_matrix
    grad_y: scipy.sparse.csc_matrix

    def apply_laplacian(self, values: Array) -> Array:
        """Apply ``M^-1 L`` without constructing an inverse mass matrix."""
        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.mass_vector.shape[0]:
            raise ValueError("values must have shape [N] or [N, C]")
        denominator = self.mass_vector
        if values.ndim == 2:
            denominator = denominator[:, None]
        return (self.stiffness @ values) / denominator

    def diffusionnet_fields(self) -> dict[str, Array | scipy.sparse.csc_matrix]:
        """Return fields accepted by the existing ``SurfaceObject``."""
        return {
            "mass": self.mass.astype(np.float32),
            "L": self.stiffness.astype(np.float32),
            "evals": self.eigenvalues.astype(np.float32),
            "evecs": self.eigenvectors.astype(np.float32),
            "gradX": self.grad_x.astype(np.float32),
            "gradY": self.grad_y.astype(np.float32),
        }

    def diffuse(self, values: Array, time: float | Array) -> Array:
        """Apply truncated spectral heat diffusion to ``[N, C]`` values."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
            squeeze = True
        else:
            squeeze = False
        if values.ndim != 2 or values.shape[0] != self.mass_vector.shape[0]:
            raise ValueError("values must have shape [N] or [N, C]")

        times = np.asarray(time, dtype=np.float64)
        if np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("diffusion time must be finite and non-negative")
        if times.ndim > 1 or (times.ndim == 1 and times.shape[0] != values.shape[1]):
            raise ValueError("time must be scalar or have shape [C]")

        coefficients = self.eigenvectors.T @ (self.mass_vector[:, None] * values)
        decay = np.exp(-self.eigenvalues[:, None] * times)
        result = self.eigenvectors @ (decay * coefficients)
        return result[:, 0] if squeeze else result


def center_distance(graph: SurfacePatchGraph) -> Array:
    """Euclidean distance between patch representative points for each edge."""
    source, target = graph.edge_index
    return np.linalg.norm(
        graph.patch_center[source] - graph.patch_center[target], axis=1
    )


def build_patch_gradients(
    graph: SurfacePatchGraph,
) -> tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
    """Fit tangent gradients from neighboring patch representative points."""
    n = graph.num_patches
    eps_reg = 1e-2
    reference = np.tile([1.0, 0.0, 0.0], (n, 1))
    reference[np.abs(graph.patch_normal[:, 0]) > 0.9] = [0.0, 1.0, 0.0]
    tangent_x = reference - (
        np.sum(reference * graph.patch_normal, axis=1)[:, None]
        * graph.patch_normal
    )
    tangent_x_norm = np.linalg.norm(tangent_x, axis=1)
    valid = graph.patch_normal_valid & (tangent_x_norm > 0.0)
    tangent_x[valid] /= tangent_x_norm[valid, None]
    tangent_x[~valid] = 0.0
    tangent_y = np.cross(graph.patch_normal, tangent_x)

    edge_source, edge_target = graph.edge_index
    source = np.concatenate((edge_source, edge_target))
    target = np.concatenate((edge_target, edge_source))
    order = np.argsort(source, kind="stable")
    source = source[order]
    target = target[order]
    offsets = np.searchsorted(source, np.arange(n + 1))

    rows: list[int] = []
    columns: list[int] = []
    values_x: list[float] = []
    values_y: list[float] = []
    points = graph.patch_center

    for node in np.flatnonzero(valid):
        start, stop = offsets[node], offsets[node + 1]
        neighbors = target[start:stop]
        if neighbors.size == 0:
            continue

        displacement = points[neighbors] - points[node]
        projected = np.column_stack(
            (displacement @ tangent_x[node], displacement @ tangent_y[node])
        )
        projected_t = projected.T
        coefficients = np.linalg.inv(
            projected_t @ projected + eps_reg * np.identity(2)
        ) @ projected_t

        rows.extend([node] * (neighbors.size + 1))
        columns.extend(neighbors.tolist())
        columns.append(node)
        values_x.extend(coefficients[0].tolist())
        values_x.append(float(-np.sum(coefficients[0])))
        values_y.extend(coefficients[1].tolist())
        values_y.append(float(-np.sum(coefficients[1])))

    grad_x = scipy.sparse.coo_matrix(
        (values_x, (rows, columns)), shape=(n, n)
    ).tocsc()
    grad_y = scipy.sparse.coo_matrix(
        (values_y, (rows, columns)), shape=(n, n)
    ).tocsc()
    grad_x.eliminate_zeros()
    grad_y.eliminate_zeros()
    return grad_x, grad_y


def induced_patch_subgraph(
    graph: SurfacePatchGraph, node_mask: Array
) -> SurfacePatchGraph:
    """Return the node-induced subgraph, with compact canonical edge indices."""
    node_mask = np.asarray(node_mask, dtype=bool)
    if node_mask.shape != (graph.num_patches,):
        raise ValueError("node_mask must have shape [N]")
    selected = np.flatnonzero(node_mask)
    if selected.size == 0:
        raise ValueError("node_mask selects no patches")

    old_to_new = np.full(graph.num_patches, -1, dtype=np.int64)
    old_to_new[selected] = np.arange(selected.size)
    source, target = graph.edge_index
    edge_mask = node_mask[source] & node_mask[target]
    edge_index = old_to_new[graph.edge_index[:, edge_mask]]

    node_features = None
    if graph.node_features is not None:
        node_features = graph.node_features[selected]
    arc_count = None
    if graph.arc_count is not None:
        arc_count = graph.arc_count[edge_mask]

    return SurfacePatchGraph(
        patch_area=graph.patch_area[selected],
        patch_center=graph.patch_center[selected],
        patch_sphere_center=graph.patch_sphere_center[selected],
        patch_area_centroid=graph.patch_area_centroid[selected],
        patch_normal=graph.patch_normal[selected],
        patch_normal_valid=graph.patch_normal_valid[selected],
        patch_radius=graph.patch_radius[selected],
        patch_atom_index=graph.patch_atom_index[selected],
        edge_index=edge_index,
        shared_arc_length=graph.shared_arc_length[edge_mask],
        integrated_conormal=graph.integrated_conormal[:, edge_mask],
        arc_count=arc_count,
        node_features=node_features,
    )


def extract_patch_graph(
    positions: Array,
    radii: Array,
    alpha: float,
    probe_radius: float = 1.4,
    *,
    binding_dir: str | os.PathLike[str] | None = None,
) -> SurfacePatchGraph:
    """Run the separately-built ``cgal_patch_graph`` extension."""
    candidate_directories: list[Path] = []
    if binding_dir is not None:
        candidate_directories.append(Path(binding_dir))
    elif os.environ.get("CGAL_BINDINGS_DIR"):
        candidate_directories.append(Path(os.environ["CGAL_BINDINGS_DIR"]))
    else:
        repository_root = Path(__file__).resolve().parents[2]
        candidate_directories.extend(
            (
                repository_root / "cgal_alpha_bindings" / "build",
                repository_root / "cgal_alpha_bindings" / "build_py310",
            )
        )

    for candidate in candidate_directories:
        if not candidate.is_dir():
            continue
        path = str(candidate.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        binding = importlib.import_module("cgal_patch_graph")
    except ImportError as error:
        raise ImportError(
            "cgal_patch_graph is not available; build the cgal_patch_graph "
            "CMake target or pass binding_dir"
        ) from error

    geometry = binding.compute_patch_graph_from_atoms(
        np.asarray(positions, dtype=np.float32),
        np.asarray(radii, dtype=np.float32),
        float(alpha),
        float(probe_radius),
    )
    return SurfacePatchGraph.from_mapping(geometry)


def build_patch_operators(
    graph: SurfacePatchGraph,
    *,
    distance_function: DistanceFunction = center_distance,
    k_eig: int = 128,
    eigensolver_tolerance: float = 1e-8,
    numerical_tolerance: float = 1e-8,
) -> SurfacePatchOperators:
    """Construct ``M``, ``L``, ``M^-1/2 L M^-1/2`` and low eigenpairs."""
    n = graph.num_patches
    if n == 0:
        raise ValueError("cannot construct operators for an empty patch graph")
    if k_eig < 1:
        raise ValueError("k_eig must be positive")

    distance = np.asarray(distance_function(graph), dtype=np.float64)
    if distance.shape != (graph.num_edges,):
        raise ValueError("distance_function must return shape [E]")
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError(
            "all patch distances must be finite and strictly positive; "
            "center distance is invalid for an adjacent same-center patch pair"
        )

    conductance = graph.shared_arc_length / distance
    if np.any(~np.isfinite(conductance)) or np.any(conductance <= 0):
        raise ValueError("all edge conductances must be finite and positive")
    grad_x, grad_y = build_patch_gradients(graph)

    source, target = graph.edge_index
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, source, conductance)
    np.add.at(degree, target, conductance)

    rows = np.concatenate((source, target, np.arange(n)))
    columns = np.concatenate((target, source, np.arange(n)))
    entries = np.concatenate((-conductance, -conductance, degree))
    stiffness = scipy.sparse.coo_matrix(
        (entries, (rows, columns)), shape=(n, n)
    ).tocsc()
    stiffness.sum_duplicates()

    mass_vector = graph.patch_area.copy()
    mass = scipy.sparse.diags(mass_vector, format="csc")
    inverse_sqrt_mass = 1.0 / np.sqrt(mass_vector)
    scaling = scipy.sparse.diags(inverse_sqrt_mass, format="csc")
    symmetric_stiffness = (scaling @ stiffness @ scaling).tocsc()

    requested_eigenpairs = min(int(k_eig), n)
    component_count = scipy.sparse.csgraph.connected_components(
        stiffness, directed=False, return_labels=False
    )
    if requested_eigenpairs < component_count:
        raise ValueError(
            f"k_eig={requested_eigenpairs} is smaller than the graph's "
            f"{component_count} connected components; all constant null modes "
            "are required for conservative diffusion"
        )
    if requested_eigenpairs == n or n <= 3:
        eigenvalues, symmetric_eigenvectors = np.linalg.eigh(
            symmetric_stiffness.toarray()
        )
        eigenvalues = eigenvalues[:requested_eigenpairs]
        symmetric_eigenvectors = symmetric_eigenvectors[:, :requested_eigenpairs]
    else:
        eigenvalues, symmetric_eigenvectors = scipy.sparse.linalg.eigsh(
            symmetric_stiffness,
            k=requested_eigenpairs,
            sigma=-max(numerical_tolerance, 1e-12),
            which="LM",
            tol=eigensolver_tolerance,
        )
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        symmetric_eigenvectors = symmetric_eigenvectors[:, order]

    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -numerical_tolerance:
        raise ValueError(
            f"L_sym is not positive semidefinite: min eigenvalue={minimum_eigenvalue}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    generalized_eigenvectors = inverse_sqrt_mass[:, None] * symmetric_eigenvectors

    operators = SurfacePatchOperators(
        mass_vector=mass_vector,
        mass=mass,
        stiffness=stiffness,
        symmetric_stiffness=symmetric_stiffness,
        eigenvalues=eigenvalues,
        eigenvectors=generalized_eigenvectors,
        patch_distance=distance,
        edge_conductance=conductance,
        grad_x=grad_x,
        grad_y=grad_y,
    )
    validate_patch_operators(operators, tolerance=numerical_tolerance)
    return operators


def validate_patch_operators(
    operators: SurfacePatchOperators, *, tolerance: float = 1e-8
) -> None:
    """Validate symmetry, nullspace, PSD, and mass orthonormality."""
    stiffness = operators.stiffness
    symmetric_stiffness = operators.symmetric_stiffness
    n = stiffness.shape[0]
    ones = np.ones(n, dtype=np.float64)

    stiffness_asymmetry = stiffness - stiffness.T
    if stiffness_asymmetry.nnz and np.max(np.abs(stiffness_asymmetry.data)) > tolerance:
        raise ValueError("stiffness matrix is not symmetric")
    symmetric_asymmetry = symmetric_stiffness - symmetric_stiffness.T
    if symmetric_asymmetry.nnz and np.max(np.abs(symmetric_asymmetry.data)) > tolerance:
        raise ValueError("symmetric stiffness matrix is not symmetric")
    if not np.allclose(stiffness @ ones, 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("stiffness matrix does not annihilate constants")
    if np.min(operators.eigenvalues) < -tolerance:
        raise ValueError("negative eigenvalue outside numerical tolerance")
    for name, gradient in (
        ("grad_x", operators.grad_x),
        ("grad_y", operators.grad_y),
    ):
        if gradient.shape != (n, n):
            raise ValueError(f"{name} must have shape [N, N]")
        if np.any(~np.isfinite(gradient.data)):
            raise ValueError(f"{name} contains non-finite values")
        if not np.allclose(gradient @ ones, 0.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"{name} does not annihilate constants")

    gram = operators.eigenvectors.T @ (
        operators.mass_vector[:, None] * operators.eigenvectors
    )
    if not np.allclose(
        gram, np.eye(gram.shape[0]), atol=10 * tolerance, rtol=10 * tolerance
    ):
        raise ValueError("eigenvectors are not mass-orthonormal")
