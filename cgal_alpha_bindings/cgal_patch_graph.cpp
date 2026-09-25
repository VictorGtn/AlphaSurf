#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_spherical_kernel_3.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Interval_nt.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/number_utils.h>

#include <SBL/GT/Spherical_kernel_extension_3.hpp>
#include <SBL/GT/Union_of_balls_boundary_3_builder.hpp>
#include <SBL/GT/Union_of_balls_boundary_3_data_structure.hpp>
#include <SBL/GT/Union_of_balls_boundary_area_face_3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Regular_vertex_base = CGAL::Regular_triangulation_vertex_base_3<Kernel>;
using Alpha_vertex_base =
    CGAL::Fixed_alpha_shape_vertex_base_3<Kernel, Regular_vertex_base>;
using Regular_cell_base = CGAL::Regular_triangulation_cell_base_3<Kernel>;
using Alpha_cell_base =
    CGAL::Fixed_alpha_shape_cell_base_3<Kernel, Regular_cell_base>;
using Triangulation_data_structure =
    CGAL::Triangulation_data_structure_3<Alpha_vertex_base, Alpha_cell_base>;
using Triangulation =
    CGAL::Regular_triangulation_3<Kernel, Triangulation_data_structure>;
using Alpha_complex = CGAL::Fixed_alpha_shape_3<Triangulation>;
using Weighted_point = Triangulation::Weighted_point;
using Point_3 = Triangulation::Bare_point;

using Boundary =
    SBL::GT::T_Union_of_balls_boundary_3_data_structure<Alpha_complex>;
using Boundary_builder =
    SBL::GT::T_Union_of_balls_boundary_3_builder<Boundary, Kernel::FT>;

// SBL's inexact spherical reconstruction can classify a boundary vertex as
// valid while subsequently finding no common point for its three supporting
// spheres. Use its exact mode for patch areas and circular arcs so full
// molecular assemblies do not depend on that inconsistent reconstruction.
using Spherical_kernel = CGAL::Exact_spherical_kernel_3;
using Interval_kernel = CGAL::Simple_cartesian<CGAL::Interval_nt<false>>;
using Patch_area =
    SBL::GT::T_Union_of_balls_boundary_area_face_3<
        Boundary, Spherical_kernel, Interval_kernel>;
using Spherical_extension =
    SBL::GT::T_Spherical_kernel_extension_3<Spherical_kernel>;
using Arc_squared_length =
    Spherical_extension::T_Construct_circular_arc_squared_length_3<double>;
struct Vector_3d {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vector_3d operator+(const Vector_3d &lhs, const Vector_3d &rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vector_3d operator-(const Vector_3d &lhs, const Vector_3d &rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vector_3d operator*(double scalar, const Vector_3d &vector) {
  return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

double norm(const Vector_3d &vector) {
  return std::sqrt(vector.x * vector.x + vector.y * vector.y +
                   vector.z * vector.z);
}

double dot(const Vector_3d &lhs, const Vector_3d &rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vector_3d cross(const Vector_3d &lhs, const Vector_3d &rhs) {
  return {lhs.y * rhs.z - lhs.z * rhs.y,
          lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}

Vector_3d normalized(const Vector_3d &vector) {
  const double length = norm(vector);
  if (!(length > 0.0)) {
    throw std::runtime_error("cannot normalize a zero-length vector");
  }
  return (1.0 / length) * vector;
}

template <class Point>
Vector_3d to_vector(const Point &point) {
  return {CGAL::to_double(point.x()), CGAL::to_double(point.y()),
          CGAL::to_double(point.z())};
}

struct Handle_hash {
  template <class Handle> std::size_t operator()(const Handle &handle) const {
    return std::hash<const void *>{}(
        static_cast<const void *>(handle.operator->()));
  }
};

struct Input_sphere {
  double x;
  double y;
  double z;
  double squared_radius;
};

struct Edge_geometry {
  double shared_arc_length = 0.0;
  std::int64_t arc_count = 0;
  Vector_3d source_integrated_conormal;
  Vector_3d target_integrated_conormal;
};

int find_parent_atom(const Alpha_complex::Vertex_handle &vertex,
                     const std::vector<Input_sphere> &input_spheres) {
  const auto point = vertex->point().point();
  const double weight = CGAL::to_double(vertex->point().weight());

  int best_index = -1;
  double best_error = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < input_spheres.size(); ++i) {
    const auto &sphere = input_spheres[i];
    const double dx = CGAL::to_double(point.x()) - sphere.x;
    const double dy = CGAL::to_double(point.y()) - sphere.y;
    const double dz = CGAL::to_double(point.z()) - sphere.z;
    const double dw = weight - sphere.squared_radius;
    const double error = dx * dx + dy * dy + dz * dz + dw * dw;
    if (error < best_error) {
      best_error = error;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

double circular_arc_length(const Boundary &boundary,
                           const Boundary::Halfedge_handle &halfedge,
                           const Arc_squared_length &arc_squared_length) {
  if (halfedge->is_degenerated()) {
    return 0.0;
  }

  if (halfedge->is_full_circle() ||
      halfedge->is_degenerated_full_circle()) {
    const auto circle =
        boundary.template get_supporting_circle<Spherical_kernel>(halfedge);
    const double radius =
        std::sqrt(std::max(0.0, CGAL::to_double(circle.squared_radius())));
    return 2.0 * std::acos(-1.0) * radius;
  }

  const auto arc =
      boundary.template get_circular_arc<Spherical_kernel>(halfedge);
  return std::sqrt(std::max(0.0, arc_squared_length(arc)));
}

Vector_3d circular_arc_vector_area(
    const Boundary &boundary, const Boundary::Halfedge_handle &halfedge,
    const Vector_3d &sphere_center) {
  const auto arc =
      boundary.template get_circular_arc<Spherical_kernel>(halfedge);
  const auto circle = arc.supporting_circle();
  const auto opposite_sphere =
      boundary.template get_supporting_sphere<Spherical_kernel>(
          halfedge->opposite()->face());
  const Vector_3d axis =
      normalized(sphere_center - to_vector(opposite_sphere.center()));
  const double radius =
      std::sqrt(std::max(0.0, CGAL::to_double(circle.squared_radius())));

  if (arc.is_full()) {
    return (2.0 * std::acos(-1.0) * radius * radius) * axis;
  }

  const Vector_3d circle_center = to_vector(circle.center()) - sphere_center;
  const Vector_3d source =
      normalized(to_vector(arc.target()) - sphere_center - circle_center);
  const Vector_3d target =
      normalized(to_vector(arc.source()) - sphere_center - circle_center);
  double angle = std::atan2(dot(axis, cross(source, target)),
                            std::clamp(dot(source, target), -1.0, 1.0));
  if (angle <= 0.0) {
    angle += 2.0 * std::acos(-1.0);
  }
  const Vector_3d u = source;
  const Vector_3d v = cross(axis, u);
  return radius * ((std::cos(angle) - 1.0) * cross(circle_center, u) +
                   std::sin(angle) * cross(circle_center, v)) +
         (radius * radius * angle) * axis;
}

py::dict compute_patch_graph_from_atoms(
    const py::array_t<float, py::array::c_style | py::array::forcecast>
        &positions,
    const py::array_t<float, py::array::c_style | py::array::forcecast>
        &radii,
    double alpha, double probe_radius) {
  const auto position = positions.unchecked<2>();
  const auto radius = radii.unchecked<1>();
  if (position.ndim() != 2 || position.shape(1) != 3) {
    throw std::invalid_argument("positions must have shape (N, 3)");
  }
  if (radius.ndim() != 1 || radius.shape(0) != position.shape(0)) {
    throw std::invalid_argument("radii must have shape (N,)");
  }
  if (!std::isfinite(alpha) || !std::isfinite(probe_radius)) {
    throw std::invalid_argument("alpha and probe_radius must be finite");
  }

  const std::size_t atom_count = static_cast<std::size_t>(position.shape(0));
  std::vector<Weighted_point> weighted_points;
  std::vector<Input_sphere> input_spheres;
  weighted_points.reserve(atom_count);
  input_spheres.reserve(atom_count);

  for (std::size_t i = 0; i < atom_count; ++i) {
    const double x = static_cast<double>(position(i, 0));
    const double y = static_cast<double>(position(i, 1));
    const double z = static_cast<double>(position(i, 2));
    const double inflated_radius =
        static_cast<double>(radius(i)) + probe_radius;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(inflated_radius) || inflated_radius <= 0.0) {
      throw std::invalid_argument(
          "positions must be finite and radii + probe_radius must be positive");
    }
    const double squared_radius = inflated_radius * inflated_radius;
    weighted_points.emplace_back(Point_3(x, y, z), squared_radius);
    input_spheres.push_back({x, y, z, squared_radius});
  }

  py::dict output;
  if (atom_count == 0) {
    output["patch_area"] = py::array_t<double>({0});
    output["patch_center"] =
        py::array_t<double>(std::vector<py::ssize_t>{0, 3});
    output["patch_sphere_center"] =
        py::array_t<double>(std::vector<py::ssize_t>{0, 3});
    output["patch_area_centroid"] =
        py::array_t<double>(std::vector<py::ssize_t>{0, 3});
    output["patch_normal"] =
        py::array_t<double>(std::vector<py::ssize_t>{0, 3});
    output["patch_normal_valid"] = py::array_t<bool>({0});
    output["patch_radius"] = py::array_t<double>({0});
    output["patch_atom_index"] = py::array_t<std::int64_t>({0});
    output["edge_index"] =
        py::array_t<std::int64_t>(std::vector<py::ssize_t>{2, 0});
    output["shared_arc_length"] = py::array_t<double>({0});
    output["integrated_conormal"] = py::array_t<double>(
        std::vector<py::ssize_t>{2, 0, 3});
    output["arc_count"] = py::array_t<std::int64_t>({0});
    return output;
  }

  Triangulation triangulation(weighted_points.begin(), weighted_points.end());
  Alpha_complex alpha_complex(triangulation, alpha);
  Boundary boundary(alpha_complex);
  Boundary_builder builder;
  builder(boundary, 2);

  Patch_area patch_area_functor(boundary);
  Arc_squared_length arc_squared_length;

  std::vector<Boundary::Face_handle> patches;
  std::unordered_map<Boundary::Face_handle, int, Handle_hash> patch_index;
  for (auto face_it = boundary.faces_begin(); face_it != boundary.faces_end();
       ++face_it) {
    Boundary::Face_handle face = face_it;
    if (!face->is_exterior() || face->is_degenerated()) {
      continue;
    }
    const double area = CGAL::to_double(patch_area_functor(face));
    if (!(area > 0.0) || !std::isfinite(area)) {
      continue;
    }
    patch_index.emplace(face, static_cast<int>(patches.size()));
    patches.push_back(face);
  }

  const std::size_t patch_count = patches.size();
  py::array_t<double> patch_areas(patch_count);
  py::array_t<double> patch_centers(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(patch_count), 3});
  py::array_t<double> patch_sphere_centers(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(patch_count), 3});
  py::array_t<double> patch_area_centroids(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(patch_count), 3});
  py::array_t<double> patch_normals(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(patch_count), 3});
  py::array_t<bool> patch_normal_validity(patch_count);
  py::array_t<double> patch_radii(patch_count);
  py::array_t<std::int64_t> patch_atom_indices(patch_count);
  auto area_out = patch_areas.mutable_unchecked<1>();
  auto center_out = patch_centers.mutable_unchecked<2>();
  auto sphere_center_out = patch_sphere_centers.mutable_unchecked<2>();
  auto area_centroid_out = patch_area_centroids.mutable_unchecked<2>();
  auto normal_out = patch_normals.mutable_unchecked<2>();
  auto normal_valid_out = patch_normal_validity.mutable_unchecked<1>();
  auto radius_out = patch_radii.mutable_unchecked<1>();
  auto atom_out = patch_atom_indices.mutable_unchecked<1>();

  std::vector<Vector_3d> sphere_centers(patch_count);
  std::vector<double> sphere_radii(patch_count);
  std::vector<Vector_3d> normal_moments(patch_count);
  for (std::size_t i = 0; i < patch_count; ++i) {
    const auto face = patches[i];
    const auto vertex = face->get_dual_simplex();
    const auto center = vertex->point().point();
    const double squared_radius =
        CGAL::to_double(vertex->point().weight()) + alpha;

    const Vector_3d sphere_center = to_vector(center);
    sphere_centers[i] = sphere_center;
    area_out(i) = CGAL::to_double(patch_area_functor(face));
    sphere_center_out(i, 0) = sphere_center.x;
    sphere_center_out(i, 1) = sphere_center.y;
    sphere_center_out(i, 2) = sphere_center.z;
    sphere_radii[i] = std::sqrt(std::max(0.0, squared_radius));
    radius_out(i) = sphere_radii[i];
    atom_out(i) = find_parent_atom(vertex, input_spheres);
  }

  std::map<std::pair<int, int>, Edge_geometry> edges;
  for (auto halfedge_it = boundary.halfedges_begin();
       halfedge_it != boundary.halfedges_end(); ++halfedge_it) {
    Boundary::Halfedge_handle halfedge = halfedge_it;
    if (halfedge->is_degenerated()) {
      continue;
    }

    const auto source_it = patch_index.find(halfedge->face());
    Vector_3d source_vector_area;
    if (source_it != patch_index.end()) {
      source_vector_area = circular_arc_vector_area(
          boundary, halfedge, sphere_centers[source_it->second]);
      normal_moments[source_it->second] =
          normal_moments[source_it->second] +
          0.5 * source_vector_area;
    }
    const auto target_it = patch_index.find(halfedge->opposite()->face());
    if (source_it == patch_index.end() || target_it == patch_index.end()) {
      continue;
    }

    const int source = source_it->second;
    const int target = target_it->second;
    if (source >= target) {
      continue;
    }

    const double length =
        circular_arc_length(boundary, halfedge, arc_squared_length);
    if (!(length > 0.0) || !std::isfinite(length)) {
      continue;
    }
    auto &edge = edges[{source, target}];
    edge.shared_arc_length += length;
    edge.arc_count += 1;
    edge.source_integrated_conormal =
        edge.source_integrated_conormal +
        (-1.0 / sphere_radii[source]) * source_vector_area;
    const Vector_3d target_vector_area = circular_arc_vector_area(
        boundary, halfedge->opposite(), sphere_centers[target]);
    edge.target_integrated_conormal =
        edge.target_integrated_conormal +
        (-1.0 / sphere_radii[target]) * target_vector_area;
  }

  for (std::size_t i = 0; i < patch_count; ++i) {
    const double moment_norm = norm(normal_moments[i]);
    const bool normal_valid = moment_norm > 1e-12 * area_out(i);
    const Vector_3d normal =
        normal_valid ? (1.0 / moment_norm) * normal_moments[i] : Vector_3d{};
    const Vector_3d area_centroid =
        sphere_centers[i] + (radius_out(i) / area_out(i)) * normal_moments[i];
    const Vector_3d patch_center =
        normal_valid ? sphere_centers[i] + radius_out(i) * normal
                     : sphere_centers[i];

    center_out(i, 0) = patch_center.x;
    center_out(i, 1) = patch_center.y;
    center_out(i, 2) = patch_center.z;
    area_centroid_out(i, 0) = area_centroid.x;
    area_centroid_out(i, 1) = area_centroid.y;
    area_centroid_out(i, 2) = area_centroid.z;
    normal_out(i, 0) = normal.x;
    normal_out(i, 1) = normal.y;
    normal_out(i, 2) = normal.z;
    normal_valid_out(i) = normal_valid;
  }

  const std::size_t edge_count = edges.size();
  py::array_t<std::int64_t> edge_indices(
      std::vector<py::ssize_t>{2, static_cast<py::ssize_t>(edge_count)});
  py::array_t<double> shared_arc_lengths(edge_count);
  py::array_t<double> integrated_conormals(std::vector<py::ssize_t>{
      2, static_cast<py::ssize_t>(edge_count), 3});
  py::array_t<std::int64_t> arc_counts(edge_count);
  auto edge_out = edge_indices.mutable_unchecked<2>();
  auto length_out = shared_arc_lengths.mutable_unchecked<1>();
  auto conormal_out = integrated_conormals.mutable_unchecked<3>();
  auto count_out = arc_counts.mutable_unchecked<1>();

  std::size_t edge_id = 0;
  for (const auto &[endpoints, geometry] : edges) {
    edge_out(0, edge_id) = endpoints.first;
    edge_out(1, edge_id) = endpoints.second;
    length_out(edge_id) = geometry.shared_arc_length;
    conormal_out(0, edge_id, 0) = geometry.source_integrated_conormal.x;
    conormal_out(0, edge_id, 1) = geometry.source_integrated_conormal.y;
    conormal_out(0, edge_id, 2) = geometry.source_integrated_conormal.z;
    conormal_out(1, edge_id, 0) = geometry.target_integrated_conormal.x;
    conormal_out(1, edge_id, 1) = geometry.target_integrated_conormal.y;
    conormal_out(1, edge_id, 2) = geometry.target_integrated_conormal.z;
    count_out(edge_id) = geometry.arc_count;
    ++edge_id;
  }

  output["patch_area"] = std::move(patch_areas);
  output["patch_center"] = std::move(patch_centers);
  output["patch_sphere_center"] = std::move(patch_sphere_centers);
  output["patch_area_centroid"] = std::move(patch_area_centroids);
  output["patch_normal"] = std::move(patch_normals);
  output["patch_normal_valid"] = std::move(patch_normal_validity);
  output["patch_radius"] = std::move(patch_radii);
  output["patch_atom_index"] = std::move(patch_atom_indices);
  output["edge_index"] = std::move(edge_indices);
  output["shared_arc_length"] = std::move(shared_arc_lengths);
  output["integrated_conormal"] = std::move(integrated_conormals);
  output["arc_count"] = std::move(arc_counts);
  return output;
}

} // namespace

PYBIND11_MODULE(cgal_patch_graph, module) {
  module.doc() =
      "Spherical-patch graph extraction from an SBL union-of-balls boundary.";
  module.def(
      "compute_patch_graph_from_atoms", &compute_patch_graph_from_atoms,
      py::arg("positions"), py::arg("radii"), py::arg("alpha"),
      py::arg("probe_radius") = 1.4,
      R"doc(
Construct the exposed spherical-patch graph of a union of inflated balls.

The returned graph uses one node per connected exterior SBL boundary face.
`edge_index` contains each undirected adjacent patch pair once. Multiple
boundary arcs between a pair are summed in `shared_arc_length`, and their
number is returned in `arc_count`.

Returns a dict containing:
  patch_area:        float64 [N]
  patch_center:      float64 [N, 3] representative points on the spheres
  patch_sphere_center: float64 [N, 3] supporting-sphere centers
  patch_area_centroid: float64 [N, 3] Euclidean surface-area centroids
  patch_normal:      float64 [N, 3] normalized area-normal moments
  patch_normal_valid: bool [N]
  patch_radius:      float64 [N]
  patch_atom_index:  int64 [N]
  edge_index:        int64 [2, E], one entry per undirected pair
  shared_arc_length: float64 [E]
  integrated_conormal: float64 [2, E, 3], outward for each endpoint
  arc_count:         int64 [E]
)doc");
}
