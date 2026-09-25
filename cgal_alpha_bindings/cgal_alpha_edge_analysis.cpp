// Diagnostic module: builds the same Fixed_alpha_shape_3 as algo2, then
// returns (a) the surface-mesh connected components (clustered by triangle
// vertex sharing, like Open3D's cluster_connected_triangles), and (b) every
// finite alpha-complex edge (REGULAR or SINGULAR) with its endpoints as atom
// indices. Lets us check whether an alpha-complex edge bridges two surface
// components that the surface mesh reports as disconnected.

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

struct UFInt {
  std::vector<int> p;
  std::vector<uint8_t> r;
  int make_set() { int i = p.size(); p.push_back(i); r.push_back(0); return i; }
  int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  void unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return;
    if (r[a] < r[b]) std::swap(a, b);
    p[b] = a;
    if (r[a] == r[b]) r[a]++;
  }
};

static std::pair<Vertex_handle, Vertex_handle> make_edge_key(Vertex_handle u, Vertex_handle v) {
  return (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
}

// Reproduces algo2's "outside" computation: union-find over EXTERIOR cells
// connected through EXTERIOR facets; outside = component of the infinite cell.
static std::unordered_set<Cell_handle, PtrHash> compute_outside_mask(
    Fixed_alpha_shape &A, Cell_handle &inf_seed_out) {
  std::unordered_map<Cell_handle, Cell_handle, PtrHash> uf_ext;
  std::vector<Cell_handle> exterior_cells;
  exterior_cells.reserve(1024);

  auto uf_find = [&](Cell_handle x) {
    while (uf_ext[x] != x) { uf_ext[x] = uf_ext[uf_ext[x]]; x = uf_ext[x]; }
    return x;
  };
  auto uf_union = [&](Cell_handle a, Cell_handle b) {
    a = uf_find(a); b = uf_find(b);
    if (a != b) uf_ext[a] = b;
  };

  inf_seed_out = nullptr;
  bool has_inf = false;
  for (auto cit = A.all_cells_begin(); cit != A.all_cells_end(); ++cit) {
    if (A.classify(cit) == Fixed_alpha_shape::EXTERIOR) {
      uf_ext[cit] = cit;
      exterior_cells.push_back(cit);
      if (!has_inf && A.is_infinite(cit)) {
        inf_seed_out = cit;
        has_inf = true;
      }
    }
  }
  if (has_inf) {
    for (Cell_handle c : exterior_cells)
      if (A.is_infinite(c)) uf_union(inf_seed_out, c);
  }
  for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
    if (A.classify(*fit) == Fixed_alpha_shape::EXTERIOR) {
      Cell_handle c1 = fit->first;
      Cell_handle c2 = c1->neighbor(fit->second);
      if (uf_ext.find(c1) != uf_ext.end() && uf_ext.find(c2) != uf_ext.end())
        uf_union(c1, c2);
    }
  }

  std::unordered_set<Cell_handle, PtrHash> outside;
  if (!has_inf) return outside;
  Cell_handle inf_root = uf_find(inf_seed_out);
  for (Cell_handle c : exterior_cells)
    if (uf_find(c) == inf_root) outside.insert(c);
  return outside;
}

py::dict analyze_alpha_edges(
    py::array_t<float> positions, py::array_t<float> radii,
    float alpha, float probe_radius = 1.4f) {
  auto pos = positions.unchecked<2>();
  auto rad = radii.unchecked<1>();
  if (pos.shape(1) != 3)
    throw std::invalid_argument("positions must have shape (N, 3)");
  size_t n_atoms = pos.shape(0);
  if (rad.shape(0) != n_atoms)
    throw std::invalid_argument("radii must have shape (N,)");

  std::vector<Weighted_point> wpoints;
  wpoints.reserve(n_atoms);
  for (size_t i = 0; i < n_atoms; ++i) {
    const double r = static_cast<double>(rad(i)) + static_cast<double>(probe_radius);
    wpoints.emplace_back(Point_3(pos(i, 0), pos(i, 1), pos(i, 2)), r * r);
  }
  Triangulation T(wpoints.begin(), wpoints.end());
  Fixed_alpha_shape A(T, alpha);

  // Map Vertex_handle -> original atom index. Hidden vertices (not in the
  // triangulation) have no entry and won't appear in edges.
  // NOTE: Fixed_alpha_shape_3 SWAPS T's contents into itself on construction,
  // so we must iterate A's vertices, not T's.
  std::unordered_map<Vertex_handle, int, PtrHash> vh_to_atom;
  // Build coord->atom_index lookup (rounded to 1e-4 to avoid FP noise)
  struct Coord {
    int x, y, z;
    bool operator==(const Coord &o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct CoordHash {
    size_t operator()(const Coord &c) const {
      return std::hash<long long>{}((long long)c.x * 73856093LL ^
                                    (long long)c.y * 19349663LL ^
                                    (long long)c.z * 83492791LL);
    }
  };
  std::unordered_map<Coord, int, CoordHash> coord_to_atom;
  for (size_t i = 0; i < n_atoms; ++i) {
    Coord c{(int)std::llround(pos(i, 0) * 10000.0),
            (int)std::llround(pos(i, 1) * 10000.0),
            (int)std::llround(pos(i, 2) * 10000.0)};
    coord_to_atom[c] = (int)i;
  }
  for (auto vit = A.finite_vertices_begin(); vit != A.finite_vertices_end(); ++vit) {
    const Point_3 &p = vit->point().point();
    Coord c{(int)std::llround(p.x() * 10000.0),
            (int)std::llround(p.y() * 10000.0),
            (int)std::llround(p.z() * 10000.0)};
    auto it = coord_to_atom.find(c);
    if (it != coord_to_atom.end())
      vh_to_atom[vit] = it->second;
  }

  // Step 1: compute outside cells (same as algo2)
  Cell_handle inf_seed = nullptr;
  std::unordered_set<Cell_handle, PtrHash> outside_cells =
      compute_outside_mask(A, inf_seed);
  auto is_outside = [&](Cell_handle c) { return outside_cells.count(c) > 0; };

  // Step 2: collect boundary facets (same rule as algo2)
  auto is_boundary_facet = [&](const Facet &f) -> bool {
    Cell_handle c1 = f.first;
    Cell_handle c2 = c1->neighbor(f.second);
    bool out1 = is_outside(c1);
    bool out2 = is_outside(c2);
    if (out1 != out2) return true;
    if (out1 && out2 && A.classify(f) == Fixed_alpha_shape::SINGULAR) return true;
    return false;
  };

  // Step 3: cluster boundary triangles by SHARED VERTEX (Open3D's
  // cluster_connected_triangles semantics). Build per-vertex triangle-list
  // and BFS over triangles sharing a vertex.
  struct TriVH { Vertex_handle a, b, c; };
  std::vector<TriVH> boundary_tris;
  boundary_tris.reserve(4096);
  for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
    if (!is_boundary_facet(*fit)) continue;
    Cell_handle c = fit->first;
    int opp = fit->second;
    Vertex_handle tv[3];
    int k = 0;
    for (int i = 0; i < 4; ++i) if (i != opp) tv[k++] = c->vertex(i);
    boundary_tris.push_back({tv[0], tv[1], tv[2]});
  }

  // Triangle-triangle adjacency through ANY shared vertex (matches Open3D).
  std::unordered_map<Vertex_handle, std::vector<int>, PtrHash> tris_per_vh;
  for (int i = 0; i < (int)boundary_tris.size(); ++i) {
    auto &t = boundary_tris[i];
    tris_per_vh[t.a].push_back(i);
    tris_per_vh[t.b].push_back(i);
    tris_per_vh[t.c].push_back(i);
  }
  UFInt uf_tri;
  for (size_t i = 0; i < boundary_tris.size(); ++i) uf_tri.make_set();
  for (auto &kv : tris_per_vh) {
    auto &v = kv.second;
    for (size_t j = 1; j < v.size(); ++j) uf_tri.unite(v[0], v[j]);
  }
  // Build component id per triangle (compact)
  std::unordered_map<int, int> root_to_compact;
  std::vector<int> tri_compact(boundary_tris.size(), -1);
  for (size_t i = 0; i < boundary_tris.size(); ++i) {
    int r = uf_tri.find((int)i);
    auto it = root_to_compact.find(r);
    int cid;
    if (it == root_to_compact.end()) {
      cid = (int)root_to_compact.size();
      root_to_compact[r] = cid;
    } else {
      cid = it->second;
    }
    tri_compact[i] = cid;
  }
  int n_comp = (int)root_to_compact.size();

  // Map atom index -> surface component (if any). Note: if a vertex is
  // multi-patch this could be ambiguous, but in practice the simple map is
  // what we need for the edge analysis.
  std::unordered_map<int, int> atom_to_compact;
  for (size_t i = 0; i < boundary_tris.size(); ++i) {
    auto &t = boundary_tris[i];
    int cid = tri_compact[i];
    for (Vertex_handle vh : {t.a, t.b, t.c}) {
      auto ita = vh_to_atom.find(vh);
      if (ita == vh_to_atom.end()) continue;
      atom_to_compact[ita->second] = cid;
    }
  }

  // Step 4: enumerate EVERY finite edge of the alpha complex (every
  // non-EXTERIOR 1-simplex). Covers INTERIOR, REGULAR, and SINGULAR edges —
  // including singular-only edges whose every incident cell is EXTERIOR.
  auto add_edge_once = [&](Vertex_handle u, Vertex_handle v,
                           std::unordered_set<std::pair<Vertex_handle, Vertex_handle>, PtrPairHash> &seen,
                           std::vector<std::pair<Vertex_handle, Vertex_handle>> &out) {
    auto key = make_edge_key(u, v);
    if (seen.insert(key).second) out.push_back(key);
  };
  std::unordered_set<std::pair<Vertex_handle, Vertex_handle>, PtrPairHash> edge_seen;
  std::vector<std::pair<Vertex_handle, Vertex_handle>> alpha_edges;
  for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end(); ++eit) {
    Cls cls = A.classify(*eit);
    if (cls == Fixed_alpha_shape::EXTERIOR) continue;
    Cell_handle c = eit->first;
    int i1 = eit->second, i2 = eit->third;
    add_edge_once(c->vertex(i1), c->vertex(i2), edge_seen, alpha_edges);
  }
  // Also iterate non-outside cells so we capture everything even if
  // classify(edge) under-reports (defensive — should be a superset).
  int n_nonsingular_nonext_cells = 0;
  for (auto cit = A.finite_cells_begin(); cit != A.finite_cells_end(); ++cit) {
    if (is_outside(cit)) continue;
    n_nonsingular_nonext_cells++;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        add_edge_once(cit->vertex(i), cit->vertex(j), edge_seen, alpha_edges);
  }

  std::vector<std::array<int, 4>> edges;  // [atom_u, atom_v, cls_int_unused, spans_comps]
  for (auto &e : alpha_edges) {
    auto iu = vh_to_atom.find(e.first);
    auto iv = vh_to_atom.find(e.second);
    if (iu == vh_to_atom.end() || iv == vh_to_atom.end()) continue;
    int au = iu->second, av = iv->second;
    auto cu_it = atom_to_compact.find(au);
    auto cv_it = atom_to_compact.find(av);
    int cu = (cu_it == atom_to_compact.end()) ? -1 : cu_it->second;
    int cv = (cv_it == atom_to_compact.end()) ? -1 : cv_it->second;
    int spans = (cu >= 0 && cv >= 0 && cu != cv) ? 1 : 0;
    edges.push_back({au, av, 0, spans});
  }

  // Aggregate
  int n_span_total = 0;
  for (auto &e : edges) if (e[3]) n_span_total++;

  // Build surface mesh output (verts + faces) directly from boundary_tris.
  // No vertex cloning — multi-patch vertices are shared, so two components
  // touching the same atom would merge in this representation. That's fine
  // for visualization; the component counts above used Open3D-style clustering
  // which also merges on vertex sharing.
  std::unordered_map<Vertex_handle, int, PtrHash> vh_to_idx;
  vh_to_idx.reserve(boundary_tris.size() * 2);
  std::vector<std::array<float, 3>> surf_verts;
  surf_verts.reserve(boundary_tris.size() * 2);
  std::vector<std::array<int, 3>> surf_faces;
  surf_faces.reserve(boundary_tris.size());
  auto vh_idx = [&](Vertex_handle v) -> int {
    auto it = vh_to_idx.find(v);
    if (it != vh_to_idx.end()) return it->second;
    int idx = (int)surf_verts.size();
    const Point_3 &p = v->point().point();
    surf_verts.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
    vh_to_idx[v] = idx;
    return idx;
  };
  for (auto &t : boundary_tris) {
    int a = vh_idx(t.a), b = vh_idx(t.b), c = vh_idx(t.c);
    surf_faces.push_back({a, c, b});  // swap b,c for outward normal (matches algo2)
  }

  // Pack verts/faces into numpy arrays
  auto verts_arr = py::array_t<float>({(ssize_t)surf_verts.size(), (ssize_t)3});
  auto va = verts_arr.mutable_unchecked<2>();
  for (size_t i = 0; i < surf_verts.size(); ++i) {
    va(i,0) = surf_verts[i][0];
    va(i,1) = surf_verts[i][1];
    va(i,2) = surf_verts[i][2];
  }
  auto faces_arr = py::array_t<int>({(ssize_t)surf_faces.size(), (ssize_t)3});
  auto fa = faces_arr.mutable_unchecked<2>();
  for (size_t i = 0; i < surf_faces.size(); ++i) {
    fa(i,0) = surf_faces[i][0];
    fa(i,1) = surf_faces[i][1];
    fa(i,2) = surf_faces[i][2];
  }

  py::dict out;
  out["n_surface_components"] = n_comp;
  out["n_boundary_tris"] = (int)boundary_tris.size();
  out["n_nonoutside_cells"] = n_nonsingular_nonext_cells;
  out["n_alpha_edges_raw"] = (int)alpha_edges.size();
  out["n_vh_to_atom_entries"] = (int)vh_to_atom.size();
  out["n_alpha_edges_total"] = (int)edges.size();
  out["n_alpha_edges_interior"] = 0;
  out["n_alpha_edges_regular"] = 0;
  out["n_alpha_edges_singular"] = 0;
  out["n_edges_spanning_components"] = n_span_total;
  out["n_span_interior"] = 0;
  out["n_span_regular"] = 0;
  out["n_span_singular"] = 0;
  std::vector<std::array<int, 3>> span_edges;
  for (auto &e : edges)
    if (e[3]) span_edges.push_back({e[0], e[1], e[2]});
  out["spanning_edges"] = span_edges;
  out["surface_verts"] = verts_arr;
  out["surface_faces"] = faces_arr;
  return out;
}

PYBIND11_MODULE(cgal_alpha_edge_analysis, m) {
  m.def("analyze_alpha_edges", &analyze_alpha_edges,
        py::arg("positions"), py::arg("radii"),
        py::arg("alpha"), py::arg("probe_radius") = 1.4f);
}
