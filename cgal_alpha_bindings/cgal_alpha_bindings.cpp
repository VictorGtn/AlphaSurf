// CGAL Alpha Complex Python Bindings
// pybind11 wrapper using CGAL's Fixed_alpha_shape_3 for optimal performance

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
#include <set>
#include <map>
#include <string>
#include <stdexcept>

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

std::set<Fixed_alpha_shape::Classification_type> parse_filter(const std::string& filter_str) {
    std::set<Fixed_alpha_shape::Classification_type> allowed;
    if (filter_str == "all") {
        allowed = {Fixed_alpha_shape::SINGULAR, Fixed_alpha_shape::REGULAR, Fixed_alpha_shape::INTERIOR};
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

typedef Fixed_alpha_shape::Vertex_handle Vertex_handle;
typedef std::pair<Vertex_handle, Vertex_handle> EdgeKey;
typedef Fixed_alpha_shape::Facet Facet;

static EdgeKey make_edge_key(Vertex_handle u, Vertex_handle v) {
    return (u < v) ? EdgeKey{u, v} : EdgeKey{v, u};
}

std::tuple<py::array_t<float>, py::array_t<int32_t>>
extract_mesh_from_alpha_shape(Fixed_alpha_shape& A,
                               const std::set<Fixed_alpha_shape::Classification_type>& allowed) {
    K::Compute_squared_radius_smallest_orthogonal_sphere_3 sq_radius_ortho;

    std::set<Vertex_handle> used_vertices;
    std::vector<std::array<Vertex_handle, 3>> face_vertex_handles;
    std::set<EdgeKey> face_edges;

    auto add_facet = [&](const Facet& f) {
        auto cell = f.first;
        int opp = f.second;
        std::array<Vertex_handle, 3> vh;
        int idx = 0;
        for (int i = 0; i < 4; i++)
            if (i != opp) vh[idx++] = cell->vertex(i);
        for (int i = 0; i < 3; i++) used_vertices.insert(vh[i]);
        face_vertex_handles.push_back(vh);
        for (int i = 0; i < 3; i++)
            face_edges.insert(make_edge_key(vh[i], vh[(i+1)%3]));
    };

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
        if (allowed.count(A.classify(*fit)))
            add_facet(*fit);
    }

    // Repair singular edges by adding the cheapest incident Delaunay facet.
    for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end(); ++eit) {
        if (A.classify(*eit) != Fixed_alpha_shape::SINGULAR)
            continue;

        auto u = eit->first->vertex(eit->second);
        auto v = eit->first->vertex(eit->third);
        if (face_edges.count(make_edge_key(u, v)))
            continue;

        double best_mu = std::numeric_limits<double>::max();
        std::optional<Facet> best_facet;

        auto fcirc = A.incident_facets(*eit);
        auto fdone = fcirc;
        do {
            Facet f = *fcirc;
            if (!A.is_infinite(f)) {
                auto cell = f.first;
                int opp = f.second;
                std::array<Weighted_point, 3> wpts;
                int idx = 0;
                for (int i = 0; i < 4; i++)
                    if (i != opp) wpts[idx++] = cell->vertex(i)->point();
                double mu = CGAL::to_double(sq_radius_ortho(wpts[0], wpts[1], wpts[2]));
                if (mu < best_mu) {
                    best_mu = mu;
                    best_facet = f;
                }
            }
            ++fcirc;
        } while (fcirc != fdone);

        if (best_facet)
            add_facet(*best_facet);
    }

    std::map<Vertex_handle, int> vertex_index;
    std::vector<Vertex_handle> vertices;
    for (auto v : used_vertices) {
        vertex_index[v] = vertices.size();
        vertices.push_back(v);
    }

    size_t num_vertices = vertices.size();
    size_t num_faces = face_vertex_handles.size();

    py::array_t<float> vertices_array({num_vertices, size_t(3)});
    py::array_t<int32_t> faces_array({num_faces, size_t(3)});

    auto v_buf = vertices_array.mutable_unchecked<2>();
    auto f_buf = faces_array.mutable_unchecked<2>();

    for (size_t i = 0; i < num_vertices; i++) {
        auto p = vertices[i]->point().point();
        v_buf(i, 0) = static_cast<float>(p.x());
        v_buf(i, 1) = static_cast<float>(p.y());
        v_buf(i, 2) = static_cast<float>(p.z());
    }

    for (size_t i = 0; i < num_faces; i++) {
        auto& vh = face_vertex_handles[i];
        f_buf(i, 0) = vertex_index[vh[0]];
        f_buf(i, 1) = vertex_index[vh[1]];
        f_buf(i, 2) = vertex_index[vh[2]];
    }

    return std::make_tuple(vertices_array, faces_array);
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

    if (pos_buf.shape(1) != 3) {
        throw std::invalid_argument("positions must have shape (N, 3)");
    }
    if (pos_buf.shape(0) != rad_buf.shape(0)) {
        throw std::invalid_argument("positions and radii must have same number of atoms");
    }

    size_t n_atoms = pos_buf.shape(0);

    try {
    std::vector<Weighted_point> weighted_points;
    weighted_points.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; i++) {
        Point_3 p(pos_buf(i, 0), pos_buf(i, 1), pos_buf(i, 2));
        double r = rad_buf(i) + probe_radius;
        double weight = r * r;
        weighted_points.emplace_back(p, weight);
    }

    Triangulation T(weighted_points.begin(), weighted_points.end());
    Fixed_alpha_shape A(T, alpha);

    auto allowed = parse_filter(filter);
        auto result = extract_mesh_from_alpha_shape(A, allowed);
        
        A.clear();
        T.clear();
        weighted_points.clear();
        
        return result;
        
    } catch (const std::bad_alloc& e) {
        std::string msg = "CGAL out of memory during alpha complex computation (n_atoms=" 
                        + std::to_string(n_atoms) + "): " + std::string(e.what());
        throw std::runtime_error(msg);
    } catch (const std::exception& e) {
        std::string msg = "CGAL error during alpha complex (n_atoms=" 
                        + std::to_string(n_atoms) + "): " + std::string(e.what());
        throw std::runtime_error(msg);
    } catch (...) {
        std::string msg = "Unknown error in CGAL alpha complex (n_atoms=" 
                        + std::to_string(n_atoms) + ")";
        throw std::runtime_error(msg);
    }
}

PYBIND11_MODULE(cgal_alpha, m) {
    m.doc() = "Fast Alpha Complex Python bindings using CGAL Fixed_alpha_shape_3";
    m.attr("__version__") = "1.1.0-error-handling";

    m.def("compute_alpha_complex_from_atoms", &compute_alpha_complex_from_atoms,
          py::arg("positions"),
          py::arg("radii"),
          py::arg("alpha"),
          py::arg("probe_radius") = 1.4f,
          py::arg("filter") = "singular+regular",
          R"doc(
Compute alpha complex mesh from atom positions and radii.

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