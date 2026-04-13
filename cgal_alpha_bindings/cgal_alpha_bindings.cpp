#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Regular_triangulation_3.h>

#include <SBL/GT/Union_of_balls_boundary_3_builder.hpp>
#include <SBL/GT/Union_of_balls_boundary_3_data_structure.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
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
typedef Fixed_alpha_shape::Facet Facet;

typedef SBL::GT::T_Union_of_balls_boundary_3_data_structure<Fixed_alpha_shape>
    UBB_DS;
typedef SBL::GT::T_Union_of_balls_boundary_3_builder<UBB_DS, K::FT> UBB_Builder;

struct PtrHash {
  template <typename T> size_t operator()(T const &p) const {
    return std::hash<void *>{}(p.operator->());
  }
};

struct PtrPairHash {
  size_t operator()(const std::pair<Vertex_handle, Vertex_handle> &p) const {
    size_t h1 = std::hash<void *>{}(p.first.operator->());
    size_t h2 = std::hash<void *>{}(p.second.operator->());
    return h1 ^ (h2 * 2654435761ULL);
  }
};

struct PtrTripleHash {
  size_t operator()(
      const std::tuple<Vertex_handle, Vertex_handle, Vertex_handle> &t) const {
    size_t h1 = std::hash<void *>{}(std::get<0>(t).operator->());
    size_t h2 =
        std::hash<void *>{}(std::get<1>(t).operator->()) * 2654435761ULL;
    size_t h3 = std::hash<void *>{}(std::get<2>(t).operator->()) * 40503ULL;
    return h1 ^ h2 ^ h3;
  }
};

typedef std::pair<Vertex_handle, Vertex_handle> EdgeKey;
typedef std::tuple<Vertex_handle, Vertex_handle, Vertex_handle> FacetKey;
typedef std::unordered_set<EdgeKey, PtrPairHash> EdgeSet;
typedef std::unordered_set<FacetKey, PtrTripleHash> FacetSet;

static EdgeKey make_edge_key(Vertex_handle u, Vertex_handle v) {
  return (u < v) ? EdgeKey{u, v} : EdgeKey{v, u};
}

static FacetKey make_facet_key(Vertex_handle u, Vertex_handle v,
                               Vertex_handle w) {
  std::array<Vertex_handle, 3> arr = {u, v, w};
  std::sort(arr.begin(), arr.end());
  return {arr[0], arr[1], arr[2]};
}

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

py::tuple compute_alpha_complex_from_atoms(
    py::array_t<float> positions, py::array_t<float> radii, float alpha,
    float probe_radius = 1.4f, const std::string &filter = "singular+regular",
    bool return_face_types = false) {
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

    // OPT #1: Vector-based union-find
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
      if (a != b) uf[a] = b;
    };

    if (inf_id != -1) {
      for (int id = 0; id < n_ext; ++id) {
        if (A.is_infinite(exterior_cells[id])) {
          uf_union(inf_id, id);
        }
      }
    }

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
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

    // OPT #5: is_ext as vector<bool>
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
      if (it == cell_to_id.end()) return false;
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

    UBB_DS ubb(A);
    UBB_Builder builder;
    builder(ubb, 2);

    using UBB_Face = decltype(ubb.faces_begin());
    using UBB_Halfedge = decltype(ubb.halfedges_begin());
    using UBB_Vertex = decltype(ubb.vertices_begin());

    // Étape 1 : Détection des boules multi-patches (extérieur uniquement)
    std::unordered_map<Vertex_handle, std::vector<UBB_Face>, PtrHash>
        ball_to_patches;
    for (auto fit = ubb.faces_begin(); fit != ubb.faces_end(); ++fit) {
      UBB_Face p = fit;
      if (!p->is_exterior())
        continue;
      ball_to_patches[p->get_dual_simplex()].push_back(p);
    }

    std::unordered_set<Vertex_handle, PtrHash> M;
    for (const auto &kv : ball_to_patches) {
      if (kv.second.size() > 1) {
        M.insert(kv.first);
      }
    }

    // Étape 2 : Duplication des vertex
    std::vector<Vertex_handle> tilde_V;
    std::unordered_map<Vertex_handle, int, PtrHash> ac_to_idx;
    std::unordered_map<UBB_Face, int, PtrHash> p_to_idx;

    for (auto vit = A.finite_vertices_begin(); vit != A.finite_vertices_end();
         ++vit) {
      if (M.count(vit) == 0) {
        ac_to_idx[vit] = tilde_V.size();
        tilde_V.push_back(vit);
      }
    }

    for (const auto &kv : ball_to_patches) {
      Vertex_handle c = kv.first;
      if (M.count(c)) {
        for (auto p : kv.second) {
          p_to_idx[p] = tilde_V.size();
          tilde_V.push_back(c);
        }
      }
    }

    auto get_patch_index = [&](UBB_Face patch) -> int {
      if (!patch->is_exterior())
        return -1;
      Vertex_handle v = patch->get_dual_simplex();
      if (M.count(v)) {
        auto it = p_to_idx.find(patch);
        return (it == p_to_idx.end()) ? -1 : it->second;
      }
      auto it = ac_to_idx.find(v);
      return (it == ac_to_idx.end()) ? -1 : it->second;
    };

    struct Tri {
      int v[3];
      bool operator==(const Tri &o) const {
        return (v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2]);
      }
    };
    struct TriHash {
      size_t operator()(const Tri &t) const {
        return t.v[0] ^ (t.v[1] * 2654435761ULL) ^ (t.v[2] * 40503ULL);
      }
    };

    std::vector<std::array<int, 3>> tilde_F;
    std::vector<uint8_t> tilde_face_types;

    // OPT #2: single unordered_map replaces set + map
    std::unordered_map<Tri, size_t, TriHash> tri_to_index;
    int n_tilde_V = static_cast<int>(tilde_V.size());

    auto add_triangle = [&](int v1, int v2, int v3, uint8_t ftype) {
      if (v1 < 0 || v1 >= n_tilde_V ||
          v2 < 0 || v2 >= n_tilde_V ||
          v3 < 0 || v3 >= n_tilde_V)
        return;

      Tri t;
      t.v[0] = v1;
      t.v[1] = v2;
      t.v[2] = v3;
      std::sort(t.v, t.v + 3);

      auto [it, inserted] = tri_to_index.try_emplace(t, tilde_F.size());
      if (inserted) {
        tilde_F.push_back({v1, v2, v3});
        tilde_face_types.push_back(ftype);
      } else {
        tilde_face_types[it->second] =
            std::max<uint8_t>(tilde_face_types[it->second], ftype);
      }
    };

    // Extraction of original non-M faces
    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      if (!is_valid_facet(*fit))
        continue;

      Cell_handle c1 = fit->first;
      int opp = fit->second;
      int vidx = 0;
      Vertex_handle v[3];
      for (int i = 0; i < 4; i++) {
        if (i != opp)
          v[vidx++] = c1->vertex(i);
      }

      if (M.count(v[0]) == 0 && M.count(v[1]) == 0 && M.count(v[2]) == 0) {
        if (ac_to_idx.count(v[0]) && ac_to_idx.count(v[1]) &&
            ac_to_idx.count(v[2])) {
          Cell_handle c2 = c1->neighbor(fit->second);
          Cell_handle ext_cell = is_ext(c1) ? c1 : c2;
          int opp_ext = ext_cell == c1 ? fit->second : c2->index(c1);
          int idx1 = (opp_ext + 1) & 3;
          int idx2 = (opp_ext + 2) & 3;
          int idx3 = (opp_ext + 3) & 3;
          if ((opp_ext % 2) == 1)
            std::swap(idx2, idx3);

          int out_v1 = ac_to_idx[ext_cell->vertex(idx1)];
          int out_v2 = ac_to_idx[ext_cell->vertex(idx2)];
          int out_v3 = ac_to_idx[ext_cell->vertex(idx3)];
          add_triangle(out_v1, out_v2, out_v3, static_cast<uint8_t>(0));
        }
      }
    }

    // Étape 3 : Reconstruction des edges et faces (extérieur uniquement)
    for (Vertex_handle c : M) {
      for (auto p : ball_to_patches[c]) {
        if (!p->is_exterior())
          continue;

        std::vector<UBB_Halfedge> boundaries;
        if (p->halfedge() != UBB_Halfedge()) {
          boundaries.push_back(p->halfedge());
        }
        for (auto hit = ubb.holes_begin(p); hit != ubb.holes_end(p); ++hit) {
          if (*hit != UBB_Halfedge()) {
            boundaries.push_back(*hit);
          }
        }

        for (auto start_a : boundaries) {
          auto a = start_a;
          do {
            auto v_sbl = a->vertex();
            if (v_sbl == UBB_Vertex()) {
              a = a->next();
              continue;
            }

            Facet f_ac = v_sbl->get_dual_simplex();
            if (!is_valid_facet(f_ac)) {
              a = a->next();
              continue;
            }

            auto opp_a = a->opposite();
            if (opp_a == UBB_Halfedge()) {
              a = a->next();
              continue;
            }
            auto p_prime = opp_a->face();
            if (p_prime == UBB_Face() || !p_prime->is_exterior()) {
              a = a->next();
              continue;
            }

            auto a_next = a->next();
            if (a_next == UBB_Halfedge()) {
              a = a->next();
              continue;
            }

            auto opp_next = a_next->opposite();
            if (opp_next == UBB_Halfedge()) {
              a = a->next();
              continue;
            }

            auto p_sec = opp_next->face();
            if (p_sec == UBB_Face() || !p_sec->is_exterior()) {
              a = a->next();
              continue;
            }

            int v_p = get_patch_index(p);
            int v_pprime = get_patch_index(p_prime);
            int v_psec = get_patch_index(p_sec);

            if (v_p >= 0 && v_pprime >= 0 && v_psec >= 0) {
              add_triangle(v_p, v_pprime, v_psec, static_cast<uint8_t>(2));
            }

            a = a->next();
          } while (a != start_a && a != UBB_Halfedge());
        }
      }
    }

    // Clean up orphan vertices by only outputting those referenced by faces
    std::vector<int> old_to_new(tilde_V.size(), -1);
    std::vector<Vertex_handle> final_V;

    for (const auto &tri : tilde_F) {
      for (int k = 0; k < 3; k++) {
        int old_v = tri[k];
        if (old_to_new[old_v] == -1) {
          old_to_new[old_v] = final_V.size();
          final_V.push_back(tilde_V[old_v]);
        }
      }
    }

    // Output to numpy arrays
    size_t nv = final_V.size();
    size_t nf = tilde_F.size();

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
      fb(i, 0) = old_to_new[tilde_F[i][0]];
      fb(i, 1) = old_to_new[tilde_F[i][1]];
      fb(i, 2) = old_to_new[tilde_F[i][2]];
    }

    if (return_face_types) {
      py::array_t<uint8_t> tout({nf});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < nf; i++) {
        tb(i) = tilde_face_types[i];
      }
      return py::make_tuple(vout, fout, tout);
    }

    return py::make_tuple(vout, fout);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("CGAL Error: ") + e.what());
  }
}

PYBIND11_MODULE(cgal_alpha, m) {
  m.def("compute_alpha_complex_from_atoms", &compute_alpha_complex_from_atoms,
        py::arg("positions"), py::arg("radii"), py::arg("alpha"),
        py::arg("probe_radius") = 1.4f, py::arg("filter") = "singular+regular",
        py::arg("return_face_types") = false,
        R"doc(
Compute a surface triangle mesh from weighted atomic spheres using a CGAL fixed alpha shape.
The output applies singular-edge repair.

Args:
    positions: numpy array of shape (N, 3), float32 - atom centers
    radii: numpy array of shape (N,), float32 - atom VDW radii
    alpha: Alpha value for the alpha complex (e.g., 0.0)
    probe_radius: Probe radius to add to atomic radii (default 1.4 for water)
    filter: Filter for facet classification ("all", "solid", "singular+regular")

Returns:
    Tuple of (vertices, faces)
)doc");
}