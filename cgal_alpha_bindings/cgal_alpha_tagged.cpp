// CGAL Alpha Complex bindings — tagged variant.
//
// Emits every non-EXTERIOR facet of the alpha complex with a per-face tag:
//   1 if the facet is SINGULAR or REGULAR (boundary of the alpha complex)
//   0 if the facet is INTERIOR            (between two interior tetrahedra)
//
// Also emits every INTERIOR cell as a tetrahedron (4 vertex indices).
//
// No `filter` argument, no singular-edge repair step.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Regular_triangulation_3.h>

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_vertex_base_3<K> Vbb;
typedef CGAL::Fixed_alpha_shape_vertex_base_3<K, Vbb> Vb;
typedef CGAL::Regular_triangulation_cell_base_3<K> Rcb;
typedef CGAL::Fixed_alpha_shape_cell_base_3<K, Rcb> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Regular_triangulation_3<K, Tds> Triangulation;
typedef CGAL::Fixed_alpha_shape_3<Triangulation> Fixed_alpha_shape;

typedef Triangulation::Weighted_point Weighted_point;
typedef Triangulation::Bare_point Point_3;
typedef Fixed_alpha_shape::Vertex_handle Vertex_handle;
typedef Fixed_alpha_shape::Cell_handle Cell_handle;
typedef Fixed_alpha_shape::Facet Facet;

struct PtrHash {
  template <typename T> size_t operator()(T const &p) const {
    return std::hash<void *>{}(p.operator->());
  }
};

std::tuple<py::array_t<float>, py::array_t<int32_t>, py::array_t<uint8_t>,
           py::array_t<int32_t>>
compute_alpha_complex_tagged(py::array_t<float> positions,
                             py::array_t<float> radii, float alpha,
                             float probe_radius = 1.4f) {
  auto pos_buf = positions.unchecked<2>();
  auto rad_buf = radii.unchecked<1>();

  if (pos_buf.shape(1) != 3)
    throw std::invalid_argument("positions must have shape (N, 3)");
  size_t n_atoms = pos_buf.shape(0);
  if (rad_buf.shape(0) != n_atoms)
    throw std::invalid_argument("radii must have shape (N,)");

  try {
    std::vector<Weighted_point> wpoints;
    wpoints.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; i++) {
      double r =
          static_cast<double>(rad_buf(i)) + static_cast<double>(probe_radius);
      wpoints.emplace_back(
          Point_3(pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2)), r * r);
    }

    Triangulation T(wpoints.begin(), wpoints.end());
    Fixed_alpha_shape A(T, alpha);

    std::vector<Vertex_handle> verts;
    std::unordered_map<Vertex_handle, int, PtrHash> v_to_idx;
    v_to_idx.reserve(n_atoms);

    auto get_or_create_idx = [&](Vertex_handle v) -> int {
      auto it = v_to_idx.find(v);
      if (it != v_to_idx.end())
        return it->second;
      int idx = static_cast<int>(verts.size());
      verts.push_back(v);
      v_to_idx[v] = idx;
      return idx;
    };

    std::vector<std::array<int, 3>> faces;
    std::vector<uint8_t> face_tags;

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      auto cls = A.classify(*fit);
      uint8_t tag;
      if (cls == Fixed_alpha_shape::SINGULAR || cls == Fixed_alpha_shape::REGULAR)
        tag = 1;
      else if (cls == Fixed_alpha_shape::INTERIOR)
        tag = 0;
      else
        continue;

      Cell_handle cell = fit->first;
      int opp = fit->second;
      std::array<int, 3> face;
      int idx = 0;
      for (int i = 0; i < 4; i++)
        if (i != opp)
          face[idx++] = get_or_create_idx(cell->vertex(i));
      faces.push_back(face);
      face_tags.push_back(tag);
    }

    std::vector<std::array<int, 4>> tets;
    for (auto cit = A.finite_cells_begin(); cit != A.finite_cells_end();
         ++cit) {
      if (A.classify(cit) != Fixed_alpha_shape::INTERIOR)
        continue;
      std::array<int, 4> tet;
      for (int i = 0; i < 4; i++)
        tet[i] = get_or_create_idx(cit->vertex(i));
      tets.push_back(tet);
    }

    size_t nv = verts.size();
    size_t nf = faces.size();
    size_t nt = tets.size();

    py::array_t<float> vout({nv, 3UL});
    auto vb = vout.mutable_unchecked<2>();
    for (size_t i = 0; i < nv; i++) {
      auto p = verts[i]->point().point();
      vb(i, 0) = static_cast<float>(p.x());
      vb(i, 1) = static_cast<float>(p.y());
      vb(i, 2) = static_cast<float>(p.z());
    }

    py::array_t<int32_t> fout({nf, 3UL});
    auto fb = fout.mutable_unchecked<2>();
    py::array_t<uint8_t> tout({nf});
    auto tb = tout.mutable_unchecked<1>();
    for (size_t i = 0; i < nf; i++) {
      fb(i, 0) = faces[i][0];
      fb(i, 1) = faces[i][1];
      fb(i, 2) = faces[i][2];
      tb(i) = face_tags[i];
    }

    py::array_t<int32_t> tetout({nt, 4UL});
    auto tetb = tetout.mutable_unchecked<2>();
    for (size_t i = 0; i < nt; i++) {
      tetb(i, 0) = tets[i][0];
      tetb(i, 1) = tets[i][1];
      tetb(i, 2) = tets[i][2];
      tetb(i, 3) = tets[i][3];
    }

    return std::make_tuple(vout, fout, tout, tetout);
  } catch (const std::bad_alloc &e) {
    throw std::runtime_error(std::string("CGAL out of memory (n_atoms=") +
                             std::to_string(n_atoms) + "): " + e.what());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("CGAL error (n_atoms=") +
                             std::to_string(n_atoms) + "): " + e.what());
  } catch (...) {
    throw std::runtime_error(std::string("Unknown CGAL error (n_atoms=") +
                             std::to_string(n_atoms) + ")");
  }
}

PYBIND11_MODULE(cgal_alpha_tagged, m) {
  m.doc() = "Alpha complex binding: returns all non-EXTERIOR facets tagged "
            "1 (SINGULAR/REGULAR) or 0 (INTERIOR), plus INTERIOR tetrahedra";
  m.attr("__version__") = "1.1.0";

  m.def("compute_alpha_complex_tagged", &compute_alpha_complex_tagged,
        py::arg("positions"), py::arg("radii"), py::arg("alpha"),
        py::arg("probe_radius") = 1.4f,
        R"doc(
Compute alpha complex facets with per-face classification tags and
INTERIOR tetrahedra.

Every non-EXTERIOR facet is emitted (SINGULAR, REGULAR, INTERIOR). The
tag encodes the facet classification:

    1  if SINGULAR or REGULAR  (boundary of the alpha complex)
    0  if INTERIOR             (sits between two interior tetrahedra)

Every INTERIOR cell is emitted as a tetrahedron (4 vertex indices).

Args:
    positions:    (N, 3) float32 atom centers
    radii:        (N,)   float32 atom VDW radii
    alpha:        alpha value for the alpha complex (e.g. 0.0)
    probe_radius: probe radius added to each atom radius (default 1.4)

Returns:
    (vertices, faces, face_tags, tetrahedra) where
        vertices:    (V, 3) float32
        faces:       (F, 3) int32  — triangle indices into `vertices`
        face_tags:   (F,)   uint8  — 1 = boundary, 0 = interior
        tetrahedra:  (T, 4) int32  — vertex indices of INTERIOR cells
)doc");
}
