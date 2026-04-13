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
#include <cstdint>
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
typedef Fixed_alpha_shape::Classification_type Cls;

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

typedef std::pair<Vertex_handle, Vertex_handle> EdgeKeyVH;
typedef std::tuple<Vertex_handle, Vertex_handle, Vertex_handle> TriKeyVH;

static EdgeKeyVH make_edge_key(Vertex_handle u, Vertex_handle v) {
  return (u < v) ? EdgeKeyVH{u, v} : EdgeKeyVH{v, u};
}

static TriKeyVH make_tri_key(Vertex_handle a, Vertex_handle b,
                             Vertex_handle c) {
  std::array<Vertex_handle, 3> arr = {a, b, c};
  std::sort(arr.begin(), arr.end());
  return {arr[0], arr[1], arr[2]};
}

struct UFInt {
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

py::tuple compute_alpha_complex_algo2_from_atoms(
    py::array_t<float> positions, py::array_t<float> radii, float alpha,
    float probe_radius = 1.4f, bool return_face_types = false) {
  auto pos = positions.unchecked<2>();
  auto rad = radii.unchecked<1>();
  if (pos.shape(1) != 3)
    throw std::invalid_argument("positions must have shape (N, 3)");
  size_t n_atoms = pos.shape(0);
  if (rad.shape(0) != n_atoms)
    throw std::invalid_argument("radii must have shape (N,)");

  try {
    std::vector<Weighted_point> wpoints;
    wpoints.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; ++i) {
      const double r =
          static_cast<double>(rad(i)) + static_cast<double>(probe_radius);
      wpoints.emplace_back(Point_3(pos(i, 0), pos(i, 1), pos(i, 2)), r * r);
    }

    Triangulation T(wpoints.begin(), wpoints.end());
    Fixed_alpha_shape A(T, alpha);

    // ------------------------------------------------------------------------
    // Global exterior-from-infinity mask on EXTERIOR cells
    // ------------------------------------------------------------------------
    std::unordered_map<Cell_handle, Cell_handle, PtrHash> uf_ext;
    std::vector<Cell_handle> exterior_cells;
    exterior_cells.reserve(1024);

    auto uf_ext_find = [&](Cell_handle x) {
      while (uf_ext[x] != x) {
        uf_ext[x] = uf_ext[uf_ext[x]];
        x = uf_ext[x];
      }
      return x;
    };
    auto uf_ext_union = [&](Cell_handle a, Cell_handle b) {
      a = uf_ext_find(a);
      b = uf_ext_find(b);
      if (a != b)
        uf_ext[a] = b;
    };

    Cell_handle inf_seed = nullptr;
    bool has_inf = false;
    for (auto cit = A.all_cells_begin(); cit != A.all_cells_end(); ++cit) {
      if (A.classify(cit) == Fixed_alpha_shape::EXTERIOR) {
        uf_ext[cit] = cit;
        exterior_cells.push_back(cit);
        if (!has_inf && A.is_infinite(cit)) {
          inf_seed = cit;
          has_inf = true;
        }
      }
    }

    if (has_inf) {
      for (Cell_handle c : exterior_cells) {
        if (A.is_infinite(c))
          uf_ext_union(inf_seed, c);
      }
    }

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      if (A.classify(*fit) == Fixed_alpha_shape::EXTERIOR) {
        Cell_handle c1 = fit->first;
        Cell_handle c2 = c1->neighbor(fit->second);
        if (uf_ext.find(c1) != uf_ext.end() &&
            uf_ext.find(c2) != uf_ext.end()) {
          uf_ext_union(c1, c2);
        }
      }
    }

    std::unordered_set<Cell_handle, PtrHash> outside_cells;
    if (has_inf) {
      Cell_handle inf_root = uf_ext_find(inf_seed);
      for (Cell_handle c : exterior_cells) {
        if (uf_ext_find(c) == inf_root)
          outside_cells.insert(c);
      }
    }

    auto is_outside = [&](Cell_handle c) -> bool {
      return outside_cells.count(c) > 0;
    };

    auto is_boundary_facet = [&](const Facet &f) -> bool {
      Cell_handle c1 = f.first;
      Cell_handle c2 = c1->neighbor(f.second);
      const bool out1 = is_outside(c1);
      const bool out2 = is_outside(c2);

      // Match current surface logic:
      // 1) outside vs non-outside interface
      // 2) outside-outside singular facets (kept to avoid missing duplicated
      // patches)
      if (out1 != out2)
        return true;

      if (out1 && out2 && A.classify(f) == Fixed_alpha_shape::SINGULAR)
        return true;

      return false;
    };

    // ------------------------------------------------------------------------
    // Step 0: collect boundary facets and base boundary vertices (algorithm 2
    // base)
    // ------------------------------------------------------------------------
    std::vector<Facet> boundary_facets;
    boundary_facets.reserve(4096);

    std::unordered_set<Vertex_handle, PtrHash> boundary_vertices;
    boundary_vertices.reserve(2048);

    std::unordered_set<TriKeyVH, PtrTripleHash> boundary_tri_keys;
    boundary_tri_keys.reserve(4096);

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end();
         ++fit) {
      if (!is_boundary_facet(*fit))
        continue;

      boundary_facets.push_back(*fit);

      Cell_handle c = fit->first;
      int opp = fit->second;
      std::array<Vertex_handle, 3> tv;
      int k = 0;
      for (int i = 0; i < 4; ++i)
        if (i != opp)
          tv[k++] = c->vertex(i);

      boundary_vertices.insert(tv[0]);
      boundary_vertices.insert(tv[1]);
      boundary_vertices.insert(tv[2]);
      boundary_tri_keys.insert(make_tri_key(tv[0], tv[1], tv[2]));
    }

    // ------------------------------------------------------------------------
    // Step 1-2: per-vertex local UF on incident outside cells, separated by
    // non-EXTERIOR shared facets
    // ------------------------------------------------------------------------
    std::unordered_set<Vertex_handle, PtrHash> multi_patch_vertices; // M
    multi_patch_vertices.reserve(512);

    std::unordered_map<Vertex_handle, std::vector<int>, PtrHash>
        comp_ids_per_vertex;
    comp_ids_per_vertex.reserve(512);

    std::unordered_map<Vertex_handle, UFInt, PtrHash> local_uf_map;
    local_uf_map.reserve(512);

    std::unordered_map<Vertex_handle,
                       std::unordered_map<Cell_handle, int, PtrHash>, PtrHash>
        outside_cell_to_local_id;
    outside_cell_to_local_id.reserve(512);

    // map local component root -> cloned vertex global index in tilde_V index
    // space
    struct IntPairHash {
      size_t operator()(const std::pair<Vertex_handle, int> &p) const {
        return std::hash<void *>{}(p.first.operator->()) ^
               (std::hash<int>{}(p.second) * 2654435761ULL);
      }
    };
    std::unordered_map<std::pair<Vertex_handle, int>, int, IntPairHash>
        clone_index_for_comp;

    // We still need global vertex indexing containers first.
    std::vector<Vertex_handle> tilde_V;
    tilde_V.reserve(boundary_vertices.size() + 1024);

    std::unordered_map<Vertex_handle, int, PtrHash> base_v_to_idx;
    base_v_to_idx.reserve(boundary_vertices.size());

    // For each boundary vertex, build local connectivity among outside incident
    // cells.
    for (Vertex_handle v : boundary_vertices) {
      std::vector<Cell_handle> inc_cells;
      A.incident_cells(v, std::back_inserter(inc_cells));

      std::vector<Cell_handle> ext_inc_cells;
      ext_inc_cells.reserve(inc_cells.size());
      for (Cell_handle c : inc_cells) {
        if (A.classify(c) == Fixed_alpha_shape::EXTERIOR && is_outside(c))
          ext_inc_cells.push_back(c);
      }

      if (ext_inc_cells.empty()) {
        // keep as non-duplicated fallback
        base_v_to_idx[v] = static_cast<int>(tilde_V.size());
        tilde_V.push_back(v);
        continue;
      }

      UFInt ufl;
      std::unordered_map<Cell_handle, int, PtrHash> id_of_cell;
      id_of_cell.reserve(ext_inc_cells.size());

      for (Cell_handle c : ext_inc_cells) {
        int id = ufl.make_set();
        id_of_cell[c] = id;
      }

      // union if neighbor across a facet containing v with facet type EXTERIOR
      for (Cell_handle c : ext_inc_cells) {
        for (int i = 0; i < 4; ++i) {
          // facet opposite i contains v iff vertex(i) != v
          if (c->vertex(i) == v)
            continue;

          Cell_handle n = c->neighbor(i);
          auto itn = id_of_cell.find(n);
          if (itn == id_of_cell.end())
            continue;

          Facet f(c, i);
          if (A.classify(f) == Fixed_alpha_shape::EXTERIOR) {
            ufl.unite(id_of_cell[c], itn->second);
          }
        }
      }

      // collect roots
      std::unordered_set<int> roots;
      for (auto &kv : id_of_cell)
        roots.insert(ufl.find(kv.second));

      if (roots.size() <= 1) {
        // not multi-patch
        base_v_to_idx[v] = static_cast<int>(tilde_V.size());
        tilde_V.push_back(v);
      } else {
        multi_patch_vertices.insert(v);
        local_uf_map[v] = std::move(ufl);

        std::vector<int> roots_vec(roots.begin(), roots.end());
        comp_ids_per_vertex[v] = roots_vec;

        // Persist per-vertex outside-cell local ids so Step 3 uses the exact
        // same UF id space as Step 2.
        outside_cell_to_local_id[v] = id_of_cell;

        // create one clone per root (same coordinate now)
        for (int r : roots_vec) {
          int idx = static_cast<int>(tilde_V.size());
          tilde_V.push_back(v);
          clone_index_for_comp[{v, r}] = idx;
        }
      }
    }

    // ------------------------------------------------------------------------
    // Step 3: rebuild faces
    // - keep original boundary faces with no vertex in M
    // - for faces touching M, map each v in M to clone picked by outside cell
    // adjacent to face
    // ------------------------------------------------------------------------
    std::vector<std::array<int, 3>> tilde_F;
    std::vector<uint8_t>
        face_types; // 0 original kept, 2 rebuilt via Algorithm 2 mapping
    tilde_F.reserve(boundary_facets.size() * 2);
    face_types.reserve(boundary_facets.size() * 2);

    struct TriExact {
      int a, b, c;
      bool operator==(const TriExact &o) const {
        return a == o.a && b == o.b && c == o.c;
      }
    };
    struct TriExactHash {
      size_t operator()(const TriExact &t) const {
        size_t h1 = static_cast<size_t>(t.a);
        size_t h2 = static_cast<size_t>(t.b) * 2654435761ULL;
        size_t h3 = static_cast<size_t>(t.c) * 40503ULL;
        return h1 ^ h2 ^ h3;
      }
    };
    std::unordered_set<TriExact, TriExactHash> seen_tri_exact;
    seen_tri_exact.reserve(boundary_facets.size() * 2);

    auto add_tri = [&](int a, int b, int c, uint8_t t) {
      TriExact key{a, b, c};
      if (seen_tri_exact.insert(key).second) {
        tilde_F.push_back({a, b, c});
        face_types.push_back(t);
      }
    };

    auto oriented_face_indices_from_outside =
        [&](const Facet &f) -> std::array<Vertex_handle, 3> {
      Cell_handle c1 = f.first;
      Cell_handle c2 = c1->neighbor(f.second);
      Cell_handle ext_cell = is_outside(c1) ? c1 : c2;

      int opp_ext = (ext_cell == c1) ? f.second : c2->index(c1);
      int idx1 = (opp_ext + 1) & 3;
      int idx2 = (opp_ext + 2) & 3;
      int idx3 = (opp_ext + 3) & 3;
      if ((opp_ext % 2) == 1)
        std::swap(idx2, idx3);

      return {ext_cell->vertex(idx1), ext_cell->vertex(idx2),
              ext_cell->vertex(idx3)};
    };

    for (const Facet &f : boundary_facets) {
      auto vv = oriented_face_indices_from_outside(f);

      bool touches_multi = multi_patch_vertices.count(vv[0]) ||
                           multi_patch_vertices.count(vv[1]) ||
                           multi_patch_vertices.count(vv[2]);

      // collect all relevant outside adjacent cells for this boundary facet
      // - always include the outside cell directly adjacent to the facet
      // - if facet is outside-outside singular, include both outside incident
      // cells
      Cell_handle c1 = f.first;
      Cell_handle c2 = c1->neighbor(f.second);

      std::vector<Cell_handle> candidate_outside_cells;
      candidate_outside_cells.reserve(2);

      if (is_outside(c1))
        candidate_outside_cells.push_back(c1);
      if (is_outside(c2) && c2 != c1)
        candidate_outside_cells.push_back(c2);

      if (candidate_outside_cells.empty())
        continue;

      // de-duplicate candidate cells
      {
        std::unordered_set<Cell_handle, PtrHash> seen_cells;
        std::vector<Cell_handle> uniq;
        uniq.reserve(candidate_outside_cells.size());
        for (Cell_handle c_ext : candidate_outside_cells) {
          if (seen_cells.insert(c_ext).second)
            uniq.push_back(c_ext);
        }
        candidate_outside_cells.swap(uniq);
      }

      for (Cell_handle c_ext : candidate_outside_cells) {
        int out_idx[3];
        bool ok = true;

        for (int i = 0; i < 3; ++i) {
          Vertex_handle v = vv[i];
          if (!multi_patch_vertices.count(v)) {
            auto it = base_v_to_idx.find(v);
            if (it == base_v_to_idx.end()) {
              // fallback create
              int idx = static_cast<int>(tilde_V.size());
              tilde_V.push_back(v);
              base_v_to_idx[v] = idx;
              out_idx[i] = idx;
            } else {
              out_idx[i] = it->second;
            }
            continue;
          }

          // find local component of c_ext in UF_v
          auto ituf = local_uf_map.find(v);
          if (ituf == local_uf_map.end()) {
            ok = false;
            break;
          }

          auto it_map = outside_cell_to_local_id.find(v);
          if (it_map == outside_cell_to_local_id.end()) {
            ok = false;
            break;
          }

          auto itid = it_map->second.find(c_ext);
          if (itid == it_map->second.end()) {
            ok = false;
            break;
          }

          int root = ituf->second.find(itid->second);
          auto itclone = clone_index_for_comp.find({v, root});
          if (itclone == clone_index_for_comp.end()) {
            ok = false;
            break;
          }
          out_idx[i] = itclone->second;
        }

        if (!ok)
          continue;

        add_tri(out_idx[0], out_idx[1], out_idx[2], touches_multi ? 2 : 0);
      }
    }

    // ------------------------------------------------------------------------
    // compact vertices actually used
    // ------------------------------------------------------------------------
    std::vector<int> old_to_new(tilde_V.size(), -1);
    std::vector<Vertex_handle> final_V;
    final_V.reserve(tilde_V.size());

    for (const auto &tr : tilde_F) {
      for (int k = 0; k < 3; ++k) {
        int ov = tr[k];
        if (old_to_new[ov] == -1) {
          old_to_new[ov] = static_cast<int>(final_V.size());
          final_V.push_back(tilde_V[ov]);
        }
      }
    }

    py::array_t<float> vout({final_V.size(), static_cast<size_t>(3)});
    auto vb = vout.mutable_unchecked<2>();
    for (size_t i = 0; i < final_V.size(); ++i) {
      auto pt = final_V[i]->point().point();
      vb(i, 0) = static_cast<float>(pt.x());
      vb(i, 1) = static_cast<float>(pt.y());
      vb(i, 2) = static_cast<float>(pt.z());
    }

    py::array_t<int32_t> fout({tilde_F.size(), static_cast<size_t>(3)});
    auto fb = fout.mutable_unchecked<2>();
    for (size_t i = 0; i < tilde_F.size(); ++i) {
      fb(i, 0) = old_to_new[tilde_F[i][0]];
      fb(i, 1) = old_to_new[tilde_F[i][1]];
      fb(i, 2) = old_to_new[tilde_F[i][2]];
    }

    if (return_face_types) {
      py::array_t<uint8_t> tout({tilde_F.size()});
      auto tb = tout.mutable_unchecked<1>();
      for (size_t i = 0; i < tilde_F.size(); ++i)
        tb(i) = face_types[i];
      return py::make_tuple(vout, fout, tout);
    }

    return py::make_tuple(vout, fout);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("CGAL Error (algo2): ") + e.what());
  }
}

PYBIND11_MODULE(cgal_alpha_algo2, m) {
  m.def("compute_alpha_complex_algo2_from_atoms",
        &compute_alpha_complex_algo2_from_atoms, py::arg("positions"),
        py::arg("radii"), py::arg("alpha"), py::arg("probe_radius") = 1.4f,
        py::arg("return_face_types") = false,
        R"doc(
CGAL-only Algorithm 2 tufting (no SBL data structure dependency in the algorithm itself).

Args:
    positions: (N,3) float32 atom centers
    radii: (N,) float32 atom radii
    alpha: alpha value
    probe_radius: probe radius added to radii
    return_face_types: if True returns (V,F,face_types)
                      face_types: 0 original boundary-like, 2 rebuilt for multi-patch assignment

Returns:
    if return_face_types=False: (vertices, faces)
    else: (vertices, faces, face_types)
)doc");
}
