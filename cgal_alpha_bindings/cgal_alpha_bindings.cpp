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
    bool return_face_types = false, bool return_patch_info = false,
    bool return_singular_edges = false) {
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
    int total_patches = 0;
    for (const auto &kv : ball_to_patches) {
      total_patches += static_cast<int>(kv.second.size());
      if (kv.second.size() > 1) {
        M.insert(kv.first);
      }
    }
    int n_multi_patch = static_cast<int>(M.size());

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
      if (v1 < 0 || v1 >= n_tilde_V || v2 < 0 || v2 >= n_tilde_V || v3 < 0 ||
          v3 >= n_tilde_V)
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

    // Count singular edges, classified by endpoint isolation.
    // Type A: one endpoint has no incident alpha-shape edges or facets
    //         (isolated vertex — its star in the alpha shape is empty).
    // Type B: both endpoints have other connections in the alpha shape.
    int singular_edge_count = 0;
    int singular_edge_type_a = 0;
    int singular_edge_type_b = 0;
    {
      // Count non-EXTERIOR incident edges per vertex.  For a singular edge
      // (u,v), if an endpoint has count == 1 then its only alpha-shape
      // connection is this singular edge — its star is "empty" beyond it.
      std::unordered_map<Vertex_handle, int, PtrHash> alpha_edge_count;
      for (auto vit = A.finite_vertices_begin(); vit != A.finite_vertices_end();
           ++vit) {
        std::vector<Fixed_alpha_shape::Edge> inc_edges;
        A.finite_incident_edges(vit, std::back_inserter(inc_edges));
        int cnt = 0;
        for (const auto &e : inc_edges)
          if (A.classify(e) != Fixed_alpha_shape::EXTERIOR)
            cnt++;
        alpha_edge_count[vit] = cnt;
      }

      for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end();
           ++eit) {
        if (A.classify(*eit) != Fixed_alpha_shape::SINGULAR)
          continue;
        singular_edge_count++;
        Vertex_handle eu = eit->first->vertex(eit->second);
        Vertex_handle ev = eit->first->vertex(eit->third);
        if (alpha_edge_count[eu] <= 1 || alpha_edge_count[ev] <= 1)
          singular_edge_type_a++;
        else
          singular_edge_type_b++;
      }
    }

    // Connected component analysis: check if singular edges bridge components
    // of the output surface.  Build UF on alpha-shape vertex handles from
    // face adjacency, then count singular edges whose endpoints sit in
    // different components.
    int n_components = 0;
    int singular_bridge_count = 0;
    {
      std::unordered_map<Vertex_handle, int, PtrHash> vh_to_ufid;
      for (const auto &tri : tilde_F) {
        for (int k = 0; k < 3; k++) {
          Vertex_handle vh = tilde_V[tri[k]];
          if (!vh_to_ufid.count(vh))
            vh_to_ufid[vh] = static_cast<int>(vh_to_ufid.size());
        }
      }

      std::vector<int> cc_uf(vh_to_ufid.size());
      std::iota(cc_uf.begin(), cc_uf.end(), 0);
      auto cc_find = [&](int x) -> int {
        while (cc_uf[x] != x) {
          cc_uf[x] = cc_uf[cc_uf[x]];
          x = cc_uf[x];
        }
        return x;
      };
      auto cc_union = [&](int a, int b) {
        a = cc_find(a);
        b = cc_find(b);
        if (a != b)
          cc_uf[a] = b;
      };

      for (const auto &tri : tilde_F)
        for (int k = 0; k < 3; k++)
          cc_union(vh_to_ufid[tilde_V[tri[k]]],
                   vh_to_ufid[tilde_V[tri[(k + 1) % 3]]]);

      std::unordered_set<int> roots;
      for (const auto &kv : vh_to_ufid)
        roots.insert(cc_find(kv.second));
      n_components = static_cast<int>(roots.size());

      for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end();
           ++eit) {
        if (A.classify(*eit) != Fixed_alpha_shape::SINGULAR)
          continue;
        Vertex_handle eu = eit->first->vertex(eit->second);
        Vertex_handle ev = eit->first->vertex(eit->third);
        auto it_u = vh_to_ufid.find(eu);
        auto it_v = vh_to_ufid.find(ev);
        if (it_u == vh_to_ufid.end() || it_v == vh_to_ufid.end())
          continue;
        if (cc_find(it_u->second) != cc_find(it_v->second))
          singular_bridge_count++;
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

    if (return_singular_edges) {
      if (return_face_types && return_patch_info) {
        py::array_t<uint8_t> tout({nf});
        auto tb = tout.mutable_unchecked<1>();
        for (size_t i = 0; i < nf; i++)
          tb(i) = tilde_face_types[i];
        return py::make_tuple(vout, fout, tout, n_multi_patch, total_patches,
                              singular_edge_count, singular_edge_type_a,
                              singular_edge_type_b, singular_bridge_count,
                              n_components);
      }
      if (return_face_types) {
        py::array_t<uint8_t> tout({nf});
        auto tb = tout.mutable_unchecked<1>();
        for (size_t i = 0; i < nf; i++)
          tb(i) = tilde_face_types[i];
        return py::make_tuple(vout, fout, tout, singular_edge_count,
                              singular_edge_type_a, singular_edge_type_b,
                              singular_bridge_count, n_components);
      }
      if (return_patch_info) {
        return py::make_tuple(vout, fout, n_multi_patch, total_patches,
                              singular_edge_count, singular_edge_type_a,
                              singular_edge_type_b, singular_bridge_count,
                              n_components);
      }
      return py::make_tuple(vout, fout, singular_edge_count,
                            singular_edge_type_a, singular_edge_type_b,
                            singular_bridge_count, n_components);
    }
    if (return_face_types && return_patch_info) {
      py::array_t<uint8_t> tout({nf});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < nf; i++)
        tb(i) = tilde_face_types[i];
      return py::make_tuple(vout, fout, tout, n_multi_patch, total_patches);
    }
    if (return_face_types) {
      py::array_t<uint8_t> tout({nf});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < nf; i++)
        tb(i) = tilde_face_types[i];
      return py::make_tuple(vout, fout, tout);
    }
    if (return_patch_info) {
      return py::make_tuple(vout, fout, n_multi_patch, total_patches);
    }

    return py::make_tuple(vout, fout);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("CGAL Error: ") + e.what());
  }
}

struct UFInt_SBL {
  std::vector<int> p;
  std::vector<uint8_t> r;
  int make_set() {
    int i = static_cast<int>(p.size());
    p.push_back(i);
    r.push_back(0);
    return i;
  }
  int find(int x) {
    while (p[x] != x) {
      p[x] = p[p[x]];
      x = p[x];
    }
    return x;
  }
  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (r[a] < r[b])
      std::swap(a, b);
    p[b] = a;
    if (r[a] == r[b])
      r[a]++;
  }
};

py::dict debug_multipatch_sbl(py::array_t<float> positions,
                              py::array_t<float> radii, float alpha,
                              float probe_radius = 1.4f) {
  auto pos_buf = positions.unchecked<2>();
  auto rad_buf = radii.unchecked<1>();
  if (pos_buf.shape(1) != 3)
    throw std::invalid_argument("positions must have shape (N, 3)");
  size_t n_atoms = pos_buf.shape(0);
  if (rad_buf.shape(0) != n_atoms)
    throw std::invalid_argument("radii must have shape (N,)");

  std::vector<Weighted_point> wpoints;
  wpoints.reserve(n_atoms);
  for (size_t i = 0; i < n_atoms; i++) {
    double r =
        static_cast<double>(rad_buf(i)) + static_cast<double>(probe_radius);
    wpoints.emplace_back(Point_3(pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2)),
                         r * r);
  }

  Triangulation T(wpoints.begin(), wpoints.end());
  Fixed_alpha_shape A(T, alpha);

  std::vector<Cell_handle> exterior_cells;
  exterior_cells.reserve(1024);
  std::unordered_map<Cell_handle, int, PtrHash> cell_to_id;

  int inf_id = -1;
  for (auto cit = A.all_cells_begin(); cit != A.all_cells_end(); ++cit) {
    if (A.classify(cit) == Fixed_alpha_shape::EXTERIOR) {
      int id = static_cast<int>(exterior_cells.size());
      cell_to_id[cit] = id;
      exterior_cells.push_back(cit);
      if (inf_id == -1 && A.is_infinite(cit))
        inf_id = id;
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
    for (int id = 0; id < n_ext; ++id)
      if (A.is_infinite(exterior_cells[id]))
        uf_union(inf_id, id);
  }

  for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
       ++fit) {
    if (A.classify(*fit) == Fixed_alpha_shape::EXTERIOR) {
      Cell_handle c1 = fit->first;
      Cell_handle c2 = c1->neighbor(fit->second);
      auto it1 = cell_to_id.find(c1);
      auto it2 = cell_to_id.find(c2);
      if (it1 != cell_to_id.end() && it2 != cell_to_id.end())
        uf_union(it1->second, it2->second);
    }
  }

  std::vector<bool> ext_from_infinite(n_ext, false);
  if (inf_id != -1) {
    int inf_root = uf_find(inf_id);
    for (int id = 0; id < n_ext; ++id)
      if (uf_find(id) == inf_root)
        ext_from_infinite[id] = true;
  }

  auto is_ext = [&](Cell_handle c) -> bool {
    auto it = cell_to_id.find(c);
    if (it == cell_to_id.end())
      return false;
    return ext_from_infinite[it->second];
  };

  UBB_DS ubb(A);
  UBB_Builder builder;
  builder(ubb, 2);

  using UBB_Face = decltype(ubb.faces_begin());

  std::unordered_map<Vertex_handle, std::vector<UBB_Face>, PtrHash>
      ball_to_patches;
  for (auto fit = ubb.faces_begin(); fit != ubb.faces_end(); ++fit) {
    UBB_Face p = fit;
    if (!p->is_exterior())
      continue;
    ball_to_patches[p->get_dual_simplex()].push_back(p);
  }

  py::list result;
  for (const auto &kv : ball_to_patches) {
    if (kv.second.size() <= 1)
      continue;

    Vertex_handle v = kv.first;
    py::dict d;
    auto pt = v->point().point();
    py::list pos_list;
    pos_list.append(static_cast<float>(pt.x()));
    pos_list.append(static_cast<float>(pt.y()));
    pos_list.append(static_cast<float>(pt.z()));
    d["pos"] = pos_list;
    d["n_exterior_patches"] = static_cast<int>(kv.second.size());

    py::list patch_info;
    for (auto p : kv.second) {
      py::dict pd;
      // Count halfedges on this patch boundary as a fingerprint
      int n_halfedges = 0;
      if (p->halfedge() != decltype(ubb.halfedges_begin())()) {
        auto start = p->halfedge();
        auto h = start;
        do {
          n_halfedges++;
          h = h->next();
        } while (h != start && h != decltype(ubb.halfedges_begin())());
      }
      pd["n_halfedges"] = n_halfedges;
      patch_info.append(pd);
    }
    d["patches"] = patch_info;

    // Also run algo2-style local UF for comparison
    std::vector<Cell_handle> inc_cells;
    A.incident_cells(v, std::back_inserter(inc_cells));

    std::vector<Cell_handle> ext_inc_cells;
    for (Cell_handle c : inc_cells)
      if (A.classify(c) == Fixed_alpha_shape::EXTERIOR && is_ext(c))
        ext_inc_cells.push_back(c);

    d["n_outside_cells"] = static_cast<int>(ext_inc_cells.size());

    // Local UF on outside cells (algo2 logic)
    UFInt_SBL ufl;
    std::unordered_map<Cell_handle, int, PtrHash> id_of_cell;
    for (Cell_handle c : ext_inc_cells)
      id_of_cell[c] = ufl.make_set();

    struct EdgeInfo {
      int a, b;
      std::string cls;
    };
    std::vector<EdgeInfo> edges;

    auto cls_to_str =
        [](Fixed_alpha_shape::Classification_type cls) -> std::string {
      switch (cls) {
      case Fixed_alpha_shape::EXTERIOR:
        return "EXTERIOR";
      case Fixed_alpha_shape::SINGULAR:
        return "SINGULAR";
      case Fixed_alpha_shape::REGULAR:
        return "REGULAR";
      case Fixed_alpha_shape::INTERIOR:
        return "INTERIOR";
      default:
        return "UNKNOWN";
      }
    };

    for (Cell_handle c : ext_inc_cells) {
      for (int i = 0; i < 4; ++i) {
        if (c->vertex(i) == v)
          continue;
        Cell_handle n = c->neighbor(i);
        auto itn = id_of_cell.find(n);
        if (itn == id_of_cell.end())
          continue;
        int id_c = id_of_cell[c];
        int id_n = itn->second;
        if (id_c > id_n)
          continue;
        Facet f(c, i);
        auto cls = A.classify(f);
        if (cls == Fixed_alpha_shape::EXTERIOR)
          ufl.unite(id_c, id_n);
        edges.push_back({id_c, id_n, cls_to_str(cls)});
      }
    }

    std::unordered_map<int, std::vector<int>> root_to_ids;
    for (auto &kv2 : id_of_cell)
      root_to_ids[ufl.find(kv2.second)].push_back(kv2.second);

    d["n_local_uf_components"] = static_cast<int>(root_to_ids.size());

    py::list comps;
    for (auto it = root_to_ids.begin(); it != root_to_ids.end(); ++it) {
      py::dict comp;
      comp["root"] = it->first;
      comp["size"] = static_cast<int>(it->second.size());
      comps.append(comp);
    }
    d["local_uf_components"] = comps;

    py::list edge_list;
    for (auto &e : edges) {
      py::dict ed;
      ed["a"] = e.a;
      ed["b"] = e.b;
      ed["facet_cls"] = e.cls;
      ed["same_component"] = (ufl.find(e.a) == ufl.find(e.b));
      edge_list.append(ed);
    }
    d["edges"] = edge_list;

    result.append(d);
  }

  py::dict out;
  out["n_multipatch"] = static_cast<int>(result.size());
  out["multipatch_vertices"] = result;
  return out;
}

PYBIND11_MODULE(cgal_alpha, m) {
  m.def("compute_alpha_complex_from_atoms", &compute_alpha_complex_from_atoms,
        py::arg("positions"), py::arg("radii"), py::arg("alpha"),
        py::arg("probe_radius") = 1.4f, py::arg("filter") = "singular+regular",
        py::arg("return_face_types") = false,
        py::arg("return_patch_info") = false,
        py::arg("return_singular_edges") = false,
        R"doc(
Compute a surface triangle mesh from weighted atomic spheres using a CGAL fixed alpha shape.

Args:
    positions: numpy array of shape (N, 3), float32 - atom centers
    radii: numpy array of shape (N,), float32 - atom VDW radii
    alpha: Alpha value for the alpha complex (e.g., 0.0)
    probe_radius: Probe radius to add to atomic radii (default 1.4 for water)
    filter: Filter for facet classification ("all", "solid", "singular+regular")
    return_face_types: if True, append face type array to output
    return_patch_info: if True, append (n_multi_patch, total_patches) to output
    return_singular_edges: if True, append singular edge stats and component info to output

Returns:
    Tuple of (vertices, faces [, face_types] [, n_multi_patch, total_patches]
              [, singular_edge_count, type_a, type_b, bridge_count, n_components])
)doc");

  m.def("debug_multipatch_sbl", &debug_multipatch_sbl, py::arg("positions"),
        py::arg("radii"), py::arg("alpha"), py::arg("probe_radius") = 1.4f,
        R"doc(
Debug: per-vertex UBB patch count and local UF analysis for multi-patch vertices.

Returns dict with:
    n_multipatch: number of multi-patch vertices (UBB exterior patches > 1)
    multipatch_vertices: list of dicts with:
        pos: [x, y, z]
        n_exterior_patches: number of UBB exterior patches (SBL algo1)
        n_outside_cells: number of incident outside cells
        n_local_uf_components: number of UF components via algo2 logic
        patches: list of {n_halfedges} per UBB patch
        local_uf_components: list of {root, size}
        edges: list of {a, b, facet_cls, same_component}
)doc");
}
