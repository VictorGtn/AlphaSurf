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

std::tuple<py::array_t<float>, py::array_t<int32_t>, int, int>
extract_mesh_from_alpha_shape(Fixed_alpha_shape& A,
                               const std::set<Fixed_alpha_shape::Classification_type>& allowed) {
    typedef Fixed_alpha_shape::Vertex_handle Vertex_handle;

    std::set<Vertex_handle> used_vertices;
    std::vector<std::array<Vertex_handle, 3>> face_vertex_handles;
    std::set<std::pair<Vertex_handle, Vertex_handle>> face_edges;
    
    int singular_faces_count = 0;

    for (auto fit = A.finite_facets_begin(); fit != A.finite_facets_end(); ++fit) {
        auto classif = A.classify(*fit);
        
        if (classif == Fixed_alpha_shape::SINGULAR) {
            singular_faces_count++;
        }
        
        if (allowed.count(classif) == 0) continue;

        auto cell = fit->first;
        int opposite = fit->second;

        std::array<int, 3> indices;
        int idx = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != opposite) indices[idx++] = i;
        }

        std::array<Vertex_handle, 3> vh;
        for (int i = 0; i < 3; i++) {
            vh[i] = cell->vertex(indices[i]);
            used_vertices.insert(vh[i]);
        }
        face_vertex_handles.push_back(vh);

        // Record edges (canonical order)
        for (int i = 0; i < 3; i++) {
            auto u = vh[i];
            auto v = vh[(i + 1) % 3];
            if (u < v) face_edges.insert({u, v});
            else face_edges.insert({v, u});
        }
    }

    // Count singular/regular edges that are NOT in the mesh faces
    int dropped_edges_count = 0;
    for (auto eit = A.finite_edges_begin(); eit != A.finite_edges_end(); ++eit) {
        auto type = A.classify(*eit);
        // User requested: "not interior edges only singular regular"
        if (type == Fixed_alpha_shape::SINGULAR || type == Fixed_alpha_shape::REGULAR) {
            auto cell = eit->first;
            int i = eit->second;
            int j = eit->third;
            auto u = cell->vertex(i);
            auto v = cell->vertex(j);
            
            std::pair<Vertex_handle, Vertex_handle> edge_pair;
            if (u < v) edge_pair = {u, v};
            else edge_pair = {v, u};

            if (face_edges.find(edge_pair) == face_edges.end()) {
                dropped_edges_count++;
            }
        }
    }

    std::map<Vertex_handle, int> vertex_index;
    std::vector<Vertex_handle> vertices;
    for (auto v : used_vertices) {
        vertex_index[v] = vertices.size();
        vertices.push_back(v);
    }

    std::vector<std::array<int, 3>> faces;
    for (const auto& vh : face_vertex_handles) {
        std::array<int, 3> face;
        for (int i = 0; i < 3; i++) {
            face[i] = vertex_index[vh[i]];
        }
        faces.push_back(face);
    }

    size_t num_vertices = vertices.size();
    size_t num_faces = faces.size();

    py::array_t<float> vertices_array({num_vertices, size_t(3)});
    py::array_t<int32_t> faces_array({num_faces, size_t(3)});

    auto v_buf = vertices_array.mutable_unchecked<2>();
    auto f_buf = faces_array.mutable_unchecked<2>();

    for (size_t i = 0; i < num_vertices; i++) {
        auto wp = vertices[i]->point();
        auto p = wp.point();
        v_buf(i, 0) = static_cast<float>(p.x());
        v_buf(i, 1) = static_cast<float>(p.y());
        v_buf(i, 2) = static_cast<float>(p.z());
    }

    for (size_t i = 0; i < num_faces; i++) {
        f_buf(i, 0) = faces[i][0];
        f_buf(i, 1) = faces[i][1];
        f_buf(i, 2) = faces[i][2];
    }

    return std::make_tuple(vertices_array, faces_array, dropped_edges_count, singular_faces_count);
}

std::tuple<py::array_t<float>, py::array_t<int32_t>, int, int>
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
        
        // Explicit cleanup to avoid destructor issues
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
    Tuple of (vertices, faces, dropped_singular_regular_edge_count, singular_faces_count)
)doc");
}