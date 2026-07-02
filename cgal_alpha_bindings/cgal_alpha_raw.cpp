#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Regular_triangulation_3.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
typedef Fixed_alpha_shape::Edge Edge;
typedef Fixed_alpha_shape::Facet Facet;

struct PtrHash {
  template <typename T> size_t operator()(T const &p) const {
    return std::hash<void *>{}(p.operator->());
  }
};

static std::unordered_set<Fixed_alpha_shape::Classification_type>
parse_filter(const std::string &filter_str) {
  std::unordered_set<Fixed_alpha_shape::Classification_type> allowed;
  if (filter_str == "all") {
    allowed = {Fixed_alpha_shape::SINGULAR, Fixed_alpha_shape::REGULAR,
               Fixed_alpha_shape::INTERIOR};
  } else if (filter_str == "solid") {
    allowed = {Fixed_alpha_shape::REGULAR, Fixed_alpha_shape::INTERIOR};
  } else {
    if (filter_str.find("singular") != std::string::npos)
      allowed.insert(Fixed_alpha_shape::SINGULAR);
    if (filter_str.find("regular") != std::string::npos)
      allowed.insert(Fixed_alpha_shape::REGULAR);
    if (filter_str.find("interior") != std::string::npos)
      allowed.insert(Fixed_alpha_shape::INTERIOR);
  }
  return allowed;
}

py::tuple compute_alpha_raw_from_atoms(
    py::array_t<float> positions, py::array_t<float> radii, float alpha,
    float probe_radius = 1.4f, const std::string &filter = "singular+regular",
    bool return_face_types = false, bool return_singular_edges = false) {
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
      double r = rad_buf(i) + probe_radius;
      wpoints.emplace_back(Point_3(pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2)),
                           r * r);
    }

    Triangulation T(wpoints.begin(), wpoints.end());
    Fixed_alpha_shape A(T, alpha);
    auto allowed = parse_filter(filter);

    // Exterior-from-infinity classification (same as original binding)
    std::vector<Cell_handle> exterior_cells;
    exterior_cells.reserve(1024);
    std::unordered_map<Cell_handle, int, PtrHash> cell_to_id;

    int inf_id = -1;

    for (auto cit = A.all_cells_begin(); cit != A.all_cells_end(); ++cit) {
      if (A.classify(cit) == Fixed_alpha_shape::EXTERIOR) {
        int id = static_cast<int>(exterior_cells.size());
        cell_to_id[cit] = id;
        exterior_cells.push_back(cit);
        if (inf_id == -1 && A.is_infinite(cit)) {
          inf_id = id;
        }
      }
    }

    const int n_ext = static_cast<int>(exterior_cells.size());
    std::vector<int> uf(n_ext);
    std::iota(uf.begin(), uf.end(), 0);

    auto uf_find = [&](int x) -> int {
      while (uf[x] != x) {
        uf[x] = uf[uf[x]];
        x = uf[x];
      }
      return x;
    };
    auto uf_union = [&](int a, int b) {
      a = uf_find(a);
      b = uf_find(b);
      if (a != b)
        uf[a] = b;
    };

    if (inf_id != -1) {
      for (int id = 0; id < n_ext; ++id) {
        if (A.is_infinite(exterior_cells[id])) {
          uf_union(inf_id, id);
        }
      }
    }

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      if (A.classify(*fit) == Fixed_alpha_shape::EXTERIOR) {
        Cell_handle c1 = fit->first;
        Cell_handle c2 = c1->neighbor(fit->second);
        auto it1 = cell_to_id.find(c1);
        auto it2 = cell_to_id.find(c2);
        if (it1 != cell_to_id.end() && it2 != cell_to_id.end()) {
          uf_union(it1->second, it2->second);
        }
      }
    }

    std::vector<bool> ext_from_infinite(n_ext, false);
    if (inf_id != -1) {
      int inf_root = uf_find(inf_id);
      for (int id = 0; id < n_ext; ++id) {
        if (uf_find(id) == inf_root) {
          ext_from_infinite[id] = true;
        }
      }
    }

    auto is_ext = [&](Cell_handle c) -> bool {
      auto it = cell_to_id.find(c);
      if (it == cell_to_id.end())
        return false;
      return ext_from_infinite[it->second];
    };

    auto is_valid_facet = [&](const Facet &f) {
      Cell_handle c1 = f.first;
      Cell_handle c2 = c1->neighbor(f.second);
      bool e1 = is_ext(c1);
      bool e2 = is_ext(c2);
      auto cls = A.classify(f);
      if (e1 != e2)
        return true;
      if (e1 && e2 && cls == Fixed_alpha_shape::SINGULAR &&
          allowed.count(Fixed_alpha_shape::SINGULAR))
        return true;
      return false;
    };

    // Extract ALL valid facets as a raw triangle mesh — no patch splitting
    std::vector<Vertex_handle> verts;
    std::unordered_map<Vertex_handle, int, PtrHash> v_to_idx;

    auto get_or_create_idx = [&](Vertex_handle v) -> int {
      auto it = v_to_idx.find(v);
      if (it != v_to_idx.end())
        return it->second;
      int idx = static_cast<int>(verts.size());
      verts.push_back(v);
      v_to_idx[v] = idx;
      return idx;
    };

    struct Tri {
      int v[3];
      bool operator==(const Tri &o) const {
        return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2];
      }
    };
    struct TriHash {
      size_t operator()(const Tri &t) const {
        return static_cast<size_t>(t.v[0]) ^
               (static_cast<size_t>(t.v[1]) * 2654435761ULL) ^
               (static_cast<size_t>(t.v[2]) * 40503ULL);
      }
    };

    std::vector<std::array<int, 3>> faces;
    std::vector<uint8_t> face_types;
    std::unordered_set<Tri, TriHash> seen_tri;
    seen_tri.reserve(4096);

    auto add_triangle = [&](int v1, int v2, int v3) {
      Tri key;
      key.v[0] = v1;
      key.v[1] = v2;
      key.v[2] = v3;
      std::sort(key.v, key.v + 3);
      if (seen_tri.insert(key).second) {
        faces.push_back({v1, v2, v3});
        face_types.push_back(0);
      }
    };

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      if (!is_valid_facet(*fit))
        continue;

      Cell_handle c1 = fit->first;
      int opp = fit->second;
      Cell_handle c2 = c1->neighbor(opp);

      int vidx = 0;
      Vertex_handle v[3];
      for (int i = 0; i < 4; i++) {
        if (i != opp)
          v[vidx++] = c1->vertex(i);
      }

      Cell_handle ext_cell = is_ext(c1) ? c1 : c2;
      int opp_ext = (ext_cell == c1) ? opp : c2->index(c1);
      int idx1 = (opp_ext + 1) & 3;
      int idx2 = (opp_ext + 2) & 3;
      int idx3 = (opp_ext + 3) & 3;
      if ((opp_ext % 2) == 1)
        std::swap(idx2, idx3);

      int out_v1 = get_or_create_idx(ext_cell->vertex(idx1));
      int out_v2 = get_or_create_idx(ext_cell->vertex(idx2));
      int out_v3 = get_or_create_idx(ext_cell->vertex(idx3));
      add_triangle(out_v1, out_v2, out_v3);
    }

    // Extract singular edges (1D wire-frame elements)
    std::vector<std::array<int, 2>> singular_edges;
    if (return_singular_edges) {
      for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end();
           ++eit) {
        auto cls = A.classify(*eit);
        if (cls == Fixed_alpha_shape::SINGULAR &&
            allowed.count(Fixed_alpha_shape::SINGULAR)) {
          Cell_handle c = eit->first;
          int i = eit->second, j = eit->third;
          Vertex_handle va = c->vertex(i);
          Vertex_handle vb = c->vertex(j);
          int ia = get_or_create_idx(va);
          int ib = get_or_create_idx(vb);
          singular_edges.push_back({ia, ib});
        }
      }
    }

    // Compact vertices — only keep those referenced by faces or edges
    std::vector<int> old_to_new(verts.size(), -1);
    std::vector<Vertex_handle> final_V;

    for (const auto &tri : faces) {
      for (int k = 0; k < 3; k++) {
        int old_v = tri[k];
        if (old_to_new[old_v] == -1) {
          old_to_new[old_v] = static_cast<int>(final_V.size());
          final_V.push_back(verts[old_v]);
        }
      }
    }
    for (const auto &edge : singular_edges) {
      for (int k = 0; k < 2; k++) {
        int old_v = edge[k];
        if (old_to_new[old_v] == -1) {
          old_to_new[old_v] = static_cast<int>(final_V.size());
          final_V.push_back(verts[old_v]);
        }
      }
    }

    size_t nv = final_V.size();
    size_t nf = faces.size();

    py::array_t<float> vout({nv, 3UL});
    auto vb = vout.mutable_unchecked<2>();
    for (size_t i = 0; i < nv; i++) {
      auto pt = final_V[i]->point().point();
      vb(i, 0) = static_cast<float>(pt.x());
      vb(i, 1) = static_cast<float>(pt.y());
      vb(i, 2) = static_cast<float>(pt.z());
    }

    py::array_t<int32_t> fout({nf, 3UL});
    auto fb = fout.mutable_unchecked<2>();
    for (size_t i = 0; i < nf; i++) {
      fb(i, 0) = old_to_new[faces[i][0]];
      fb(i, 1) = old_to_new[faces[i][1]];
      fb(i, 2) = old_to_new[faces[i][2]];
    }

    if (return_face_types && return_singular_edges) {
      py::array_t<uint8_t> tout({nf});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < nf; i++) {
        tb(i) = face_types[i];
      }
      size_t ne = singular_edges.size();
      py::array_t<int32_t> eout({ne, 2UL});
      auto eb = eout.mutable_unchecked<2>();
      for (size_t i = 0; i < ne; i++) {
        eb(i, 0) = old_to_new[singular_edges[i][0]];
        eb(i, 1) = old_to_new[singular_edges[i][1]];
      }
      return py::make_tuple(vout, fout, tout, eout);
    }

    if (return_face_types) {
      py::array_t<uint8_t> tout({nf});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < nf; i++) {
        tb(i) = face_types[i];
      }
      return py::make_tuple(vout, fout, tout);
    }

    if (return_singular_edges) {
      size_t ne = singular_edges.size();
      py::array_t<int32_t> eout({ne, 2UL});
      auto eb = eout.mutable_unchecked<2>();
      for (size_t i = 0; i < ne; i++) {
        eb(i, 0) = old_to_new[singular_edges[i][0]];
        eb(i, 1) = old_to_new[singular_edges[i][1]];
      }
      return py::make_tuple(vout, fout, eout);
    }

    return py::make_tuple(vout, fout);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("CGAL Error: ") + e.what());
  }
}

PYBIND11_MODULE(cgal_alpha_raw, m) {
  m.def("compute_alpha_raw_from_atoms", &compute_alpha_raw_from_atoms,
        py::arg("positions"), py::arg("radii"), py::arg("alpha"),
        py::arg("probe_radius") = 1.4f, py::arg("filter") = "singular+regular",
        py::arg("return_face_types") = false,
        py::arg("return_singular_edges") = false,
        R"doc(
Extract raw alpha complex boundary facets — no SBL patch splitting.

Same alpha shape construction and exterior-from-infinity classification
as the full binding, but outputs all valid facets directly as a triangle
mesh without any vertex duplication. Nonmanifold vertices are preserved
as-is.

This is intended to be followed by a Python-side nonmanifold vertex
splitting step for comparison with the SBL-based pipeline.

Args:
    positions: (N, 3) float32 atom centers
    radii: (N,) float32 atom VDW radii
    alpha: Alpha value (e.g. 0.0)
    probe_radius: added to radii (default 1.4)
    filter: classification filter ("singular+regular", "all", "solid")
    return_face_types: if True, returns (V, F, face_types)
    return_singular_edges: if True, appends (E, 2) singular edge array

Returns:
    (vertices, faces)
    (vertices, faces, face_types)                     if return_face_types
    (vertices, faces, singular_edges)                 if return_singular_edges
    (vertices, faces, face_types, singular_edges)     if both
)doc");
}
