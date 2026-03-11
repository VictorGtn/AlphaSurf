#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Fixed_alpha_shape_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>

#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <numeric>
#include <list>

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

struct PtrPairHash {
    size_t operator()(const std::pair<Vertex_handle, Vertex_handle>& p) const {
        size_t h1 = std::hash<void*>{}(p.first.operator->());
        size_t h2 = std::hash<void*>{}(p.second.operator->());
        return h1 ^ (h2 * 2654435761ULL);
    }
};
struct PtrTripleHash {
    size_t operator()(const std::tuple<Vertex_handle, Vertex_handle, Vertex_handle>& t) const {
        size_t h1 = std::hash<void*>{}(std::get<0>(t).operator->());
        size_t h2 = std::hash<void*>{}(std::get<1>(t).operator->()) * 2654435761ULL;
        size_t h3 = std::hash<void*>{}(std::get<2>(t).operator->()) * 40503ULL;
        return h1 ^ h2 ^ h3;
    }
};
struct PtrHash {
    template<typename T>
    size_t operator()(T const& p) const {
        return std::hash<void*>{}(p.operator->());
    }
};

typedef std::pair<Vertex_handle, Vertex_handle> EdgeKey;
typedef std::tuple<Vertex_handle, Vertex_handle, Vertex_handle> FacetKey;
typedef std::unordered_set<EdgeKey, PtrPairHash> EdgeSet;
typedef std::unordered_set<FacetKey, PtrTripleHash> FacetSet;

static EdgeKey make_edge_key(Vertex_handle u, Vertex_handle v) {
    return (u < v) ? EdgeKey{u, v} : EdgeKey{v, u};
}

static FacetKey make_facet_key(Vertex_handle u, Vertex_handle v, Vertex_handle w) {
    std::array<Vertex_handle, 3> arr = {u, v, w};
    std::sort(arr.begin(), arr.end());
    return {arr[0], arr[1], arr[2]};
}

static std::unordered_set<Fixed_alpha_shape::Classification_type>
parse_filter(const std::string& filter_str) {
    std::unordered_set<Fixed_alpha_shape::Classification_type> allowed;
    if (filter_str == "all") {
        allowed = {Fixed_alpha_shape::SINGULAR, Fixed_alpha_shape::REGULAR, Fixed_alpha_shape::INTERIOR};
    } else if (filter_str == "solid") {
        allowed = {Fixed_alpha_shape::REGULAR, Fixed_alpha_shape::INTERIOR};
    } else {
        if (filter_str.find("singular") != std::string::npos) allowed.insert(Fixed_alpha_shape::SINGULAR);
        if (filter_str.find("regular") != std::string::npos) allowed.insert(Fixed_alpha_shape::REGULAR);
        if (filter_str.find("interior") != std::string::npos) allowed.insert(Fixed_alpha_shape::INTERIOR);
    }
    return allowed;
}

std::tuple<py::array_t<float>, py::array_t<int32_t>>
compute_alpha_complex_from_atoms(
    py::array_t<float> positions,
    py::array_t<float> radii,
    float alpha,
    float probe_radius = 1.4f,
    const std::string& filter = "singular+regular"
) {
    auto pos_buf = positions.unchecked<2>();
    auto rad_buf = radii.unchecked<1>();
    if (pos_buf.shape(1) != 3) throw std::invalid_argument("positions must have shape (N, 3)");
    size_t n_atoms = pos_buf.shape(0);

    try {
    std::vector<Weighted_point> wpoints; wpoints.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; i++) wpoints.emplace_back(Point_3(pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2)), (rad_buf(i) + probe_radius) * (rad_buf(i) + probe_radius));
    Triangulation T(wpoints.begin(), wpoints.end());
    Fixed_alpha_shape A(T, alpha);
    auto allowed = parse_filter(filter);
    K::Compute_squared_radius_smallest_orthogonal_sphere_3 sq_radius_ortho;

    std::unordered_set<Vertex_handle, PtrHash> used_vertices;
    std::vector<std::array<Vertex_handle, 3>> face_verts;
    EdgeSet face_edges;
    FacetSet face_set;

    auto add_facet = [&](const Facet& f) {
        auto cell = f.first; int opp = f.second;
        std::array<Vertex_handle, 3> vh; int idx = 0;
        for (int i = 0; i < 4; i++) if (i != opp) vh[idx++] = cell->vertex(i);
        for (int i = 0; i < 3; i++) used_vertices.insert(vh[i]);
        face_verts.push_back(vh);
        for (int i = 0; i < 3; i++) face_edges.insert(make_edge_key(vh[i], vh[(i+1)%3]));
        face_set.insert(make_facet_key(vh[0], vh[1], vh[2]));
    };

    // Largest connected component selection
    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit)
        if (allowed.count(A.classify(*fit))) add_facet(*fit);

    size_t n = face_verts.size();
    if (n > 1) {
        std::unordered_map<Vertex_handle, std::vector<size_t>, PtrHash> vtx_to_faces;
        for (size_t i = 0; i < n; i++) for (auto& vh : face_verts[i]) vtx_to_faces[vh].push_back(i);
        std::vector<size_t> parent(n); std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](size_t x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        for (auto& [vh, flist] : vtx_to_faces) for (size_t i = 1; i < flist.size(); i++) { size_t r1 = find(flist[0]), r2 = find(flist[i]); if (r1 != r2) parent[r1] = r2; }
        std::unordered_map<size_t, size_t> csize; for (size_t i = 0; i < n; i++) csize[find(i)]++;
        size_t best = std::max_element(csize.begin(), csize.end(), [](const auto& a, const auto& b) { return a.second < b.second; })->first;
        std::vector<std::array<Vertex_handle, 3>> kept;
        for (size_t i = 0; i < n; i++) if (find(i) == best) kept.push_back(face_verts[i]);
        face_verts = std::move(kept);
        face_edges.clear(); face_set.clear(); used_vertices.clear();
        for (auto& vh : face_verts) { face_set.insert(make_facet_key(vh[0], vh[1], vh[2])); for (int i = 0; i < 3; i++) used_vertices.insert(vh[i]); for (int i = 0; i < 3; i++) face_edges.insert(make_edge_key(vh[i], vh[(i+1)%3])); }
    }

    // Build exterior component Union-Find (excludes pockets from infinite exterior)
    std::unordered_map<Cell_handle, Cell_handle, PtrHash> uf;
    auto uf_find = [&](Cell_handle x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
    auto uf_union = [&](Cell_handle a, Cell_handle b) { a = uf_find(a); b = uf_find(b); if (a != b) uf[a] = b; };
    Cell_handle inf_repr; bool has_inf = false;
    for (auto cit = A.all_cells_begin(); cit != A.all_cells_end(); ++cit) if (A.classify(cit) == Fixed_alpha_shape::EXTERIOR) { uf[cit] = cit; if (A.is_infinite(cit)) { if (!has_inf) { inf_repr = cit; has_inf = true; } else uf_union(inf_repr, cit); } }
    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) if (A.classify(*fit) == Fixed_alpha_shape::EXTERIOR) uf_union(fit->first, fit->first->neighbor(fit->second));
    auto is_ext = [&](Cell_handle c) { auto it = uf.find(c); return (it != uf.end() && uf_find(c) == uf_find(inf_repr)); };
    std::unordered_map<Cell_handle, bool, PtrHash> solid;
    for (auto cit = A.finite_cells_begin(); cit != A.finite_cells_end(); ++cit) solid[cit] = (A.classify(cit) == Fixed_alpha_shape::INTERIOR) || !is_ext(cit);

    // Rebuild face list with pocket exclusion
    face_verts.clear(); face_set.clear();
    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
        Cell_handle c1 = fit->first, c2 = c1->neighbor(fit->second);
        bool s1 = !A.is_infinite(c1) && solid[c1], s2 = !A.is_infinite(c2) && solid[c2];
        auto cls = A.classify(*fit);
        if (s1 != s2) { std::array<Vertex_handle, 3> fvh; int idx = 0; for (int i = 0; i < 4; i++) if (i != fit->second) fvh[idx++] = c1->vertex(i); face_verts.push_back(fvh); face_set.insert(make_facet_key(fvh[0], fvh[1], fvh[2])); }
        else if (!s1 && !s2 && cls == Fixed_alpha_shape::SINGULAR && allowed.count(Fixed_alpha_shape::SINGULAR) && (is_ext(c1) || is_ext(c2))) { std::array<Vertex_handle, 3> fvh; int idx = 0; for (int i = 0; i < 4; i++) if (i != fit->second) fvh[idx++] = c1->vertex(i); face_verts.push_back(fvh); face_set.insert(make_facet_key(fvh[0], fvh[1], fvh[2])); }
    }
    used_vertices.clear(); face_edges.clear();
    for (const auto& vh : face_verts) { for (int i = 0; i < 3; i++) used_vertices.insert(vh[i]); for (int i = 0; i < 3; i++) face_edges.insert(make_edge_key(vh[i], vh[(i+1)%3])); }

    // Singular edge repair
    std::vector<Fixed_alpha_shape::Edge> sedges; A.get_alpha_shape_edges(std::back_inserter(sedges), Fixed_alpha_shape::SINGULAR);
    for (const auto& e : sedges) {
        Vertex_handle u = e.first->vertex(e.second), v = e.first->vertex(e.third);
        if (face_edges.count(make_edge_key(u, v))) continue;
        auto pre_c = A.incident_cells(e), pre_d = pre_c; bool surf = false;
        do { if (A.is_infinite(pre_c) || is_ext(pre_c)) { surf = true; break; } } while (++pre_c != pre_d);
        if (!surf) continue;
        auto circ = A.incident_cells(e), done = circ; double best_mu = 1e30; Facet best_f; bool found = false; FacetSet checked;
        do {
            Cell_handle c = circ; for (int i = 0; i < 4; i++) {
                if (c->vertex(i) == u || c->vertex(i) == v) continue;
                std::array<Vertex_handle, 3> fvh; int fi = 0; for (int j = 0; j < 4; j++) if (j != i) fvh[fi++] = c->vertex(j);
                if (!checked.insert(make_facet_key(fvh[0], fvh[1], fvh[2])).second) continue;
                Cell_handle c2 = c->neighbor(i); if (A.is_infinite(c) || A.is_infinite(c2) || (!is_ext(c) && !is_ext(c2))) continue;
                double mu = CGAL::to_double(sq_radius_ortho(fvh[0]->point(), fvh[1]->point(), fvh[2]->point()));
                if (mu < best_mu) { best_mu = mu; best_f = Facet(c, i); found = true; }
            }
        } while (++circ != done);
        if (found) { add_facet(best_f); }
    }

    // Pinched vertex repair
    auto get_v_info = [&](Vertex_handle v, const std::vector<size_t>& findices, std::vector<std::vector<size_t>>& comp_flist) {
        std::unordered_map<Vertex_handle, std::unordered_set<Vertex_handle, PtrHash>, PtrHash> ladj; std::unordered_set<Vertex_handle, PtrHash> lverts;
        for (size_t fi : findices) {
            auto& fvh = face_verts[fi]; Vertex_handle u = nullptr, w = nullptr;
            for (int k = 0; k < 3; k++) if (fvh[k] != v) { if (u == nullptr) u = fvh[k]; else w = fvh[k]; }
            if (u != nullptr && w != nullptr) { ladj[u].insert(w); ladj[w].insert(u); lverts.insert(u); lverts.insert(w); }
        }
        int ncomps = 0; std::unordered_map<Vertex_handle, int, PtrHash> vcomp;
        for (auto u : lverts) { if (vcomp.count(u)) continue; std::vector<Vertex_handle> q = {u}; vcomp[u] = ncomps; while (!q.empty()) { auto c = q.back(); q.pop_back(); for (auto nb : ladj[c]) if (!vcomp.count(nb)) { vcomp[nb] = ncomps; q.push_back(nb); } } ncomps++; }
        if (ncomps <= 1) return 1;
        comp_flist.assign(ncomps, {}); for (size_t fi : findices) { auto& fvh = face_verts[fi]; for (int k = 0; k < 3; k++) if (fvh[k] != v && vcomp.count(fvh[k])) { comp_flist[vcomp[fvh[k]]].push_back(fi); break; } }
        std::vector<Cell_handle> extv; std::list<Cell_handle> inc; A.incident_cells(v, std::back_inserter(inc)); for (auto c : inc) if (is_ext(c)) extv.push_back(c);
        std::unordered_set<Cell_handle, PtrHash> eset(extv.begin(), extv.end()); int nxt = 0; std::unordered_map<Cell_handle, int, PtrHash> ecomp;
        for (auto s : extv) { if (ecomp.count(s)) continue; std::vector<Cell_handle> q = {s}; ecomp[s] = nxt; while (!q.empty()) { auto c = q.back(); q.pop_back(); int vi = c->index(v); for (int i = 0; i < 4; i++) if (i != vi) { auto nb = c->neighbor(i); if (eset.count(nb) && !ecomp.count(nb)) { ecomp[nb] = nxt; q.push_back(nb); } } } nxt++; }
        return (nxt == 1) ? 2 : 0;
    };

    std::unordered_map<Vertex_handle, std::vector<size_t>, PtrHash> v2f;
    for (size_t i = 0; i < face_verts.size(); i++) for (auto& vh : face_verts[i]) v2f[vh].push_back(i);
    for (auto& [v, findices] : v2f) {
        std::vector<std::vector<size_t>> comp_flist; int info = get_v_info(v, findices, comp_flist);
        if (info != 2) continue;
        int ncomps = comp_flist.size(); FacetSet checked; struct Cand { int c1, c2; double mu; Facet f; }; std::vector<Cand> cands;
        std::list<Cell_handle> inc; A.incident_cells(v, std::back_inserter(inc));
        for (auto c : inc) {
            int vi = c->index(v); for (int o = 0; o < 4; o++) {
                if (o == vi || A.is_infinite(Facet(c, o)) || (!is_ext(c) && !is_ext(c->neighbor(o)))) continue;
                std::array<Vertex_handle, 3> fvh; int idx = 0; for (int i = 0; i < 4; i++) if (i != o) fvh[idx++] = c->vertex(i);
                if (!checked.insert(make_facet_key(fvh[0], fvh[1], fvh[2])).second) continue;
                Vertex_handle u = nullptr, w = nullptr; for (int k = 0; k < 3; k++) if (fvh[k] != v) { if (u == nullptr) u = fvh[k]; else w = fvh[k]; }
                if (u == nullptr || w == nullptr) continue;
                int c_u = -1, c_w = -1; for (int i = 0; i < ncomps; i++) for (size_t fi : comp_flist[i]) { auto& fv = face_verts[fi]; if (fv[0] == u || fv[1] == u || fv[2] == u) c_u = i; if (fv[0] == w || fv[1] == w || fv[2] == w) c_w = i; }
                if (c_u != -1 && c_w != -1 && c_u != c_w) { double mu = 0; try { mu = CGAL::to_double(sq_radius_ortho(fvh[0]->point(), fvh[1]->point(), fvh[2]->point())); } catch (...) {} cands.push_back({c_u, c_w, mu, Facet(c, o)}); }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.mu < b.mu; });
        std::vector<int> bp(ncomps); std::iota(bp.begin(), bp.end(), 0);
        auto bf = [&](int x) { while (bp[x] != x) { bp[x] = bp[bp[x]]; x = bp[x]; } return x; };
        for (const auto& b : cands) { int p1 = bf(b.c1), p2 = bf(b.c2); if (p1 != p2) { bp[p1] = p2; add_facet(b.f); } }
    }

    std::unordered_map<Vertex_handle, int, PtrHash> vidx; std::vector<Vertex_handle> vlist;
    for (auto v : used_vertices) { vidx[v] = vlist.size(); vlist.push_back(v); }
    size_t nv = vlist.size(), nf = face_verts.size();
    py::array_t<float> vout({nv, 3UL}); py::array_t<int32_t> fout({nf, 3UL});
    auto vb = vout.mutable_unchecked<2>(); auto fb = fout.mutable_unchecked<2>();
    for (size_t i = 0; i < nv; i++) { auto p = vlist[i]->point().point(); vb(i,0)=(float)p.x(); vb(i,1)=(float)p.y(); vb(i,2)=(float)p.z(); }
    for (size_t i = 0; i < nf; i++) { auto& vh = face_verts[i]; fb(i,0)=vidx[vh[0]]; fb(i,1)=vidx[vh[1]]; fb(i,2)=vidx[vh[2]]; }
    return std::make_tuple(vout, fout);
    } catch (const std::exception& e) { throw std::runtime_error(std::string("CGAL Error: ") + e.what()); }
}

PYBIND11_MODULE(cgal_alpha, m) {
    m.def("compute_alpha_complex_from_atoms", &compute_alpha_complex_from_atoms,
        py::arg("positions"), py::arg("radii"), py::arg("alpha"),
        py::arg("probe_radius")=1.4f, py::arg("filter")="singular+regular",
        R"doc(
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