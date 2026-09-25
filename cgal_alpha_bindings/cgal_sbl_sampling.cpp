#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Fixed_alpha_shape_vertex_base_3.h>
#include <CGAL/Fixed_alpha_shape_cell_base_3.h>
#include <CGAL/Fixed_alpha_shape_3.h>

#include <SBL/GT/Union_of_balls_boundary_3_data_structure.hpp>
#include <SBL/GT/Union_of_balls_boundary_3_builder.hpp>
#include <SBL/GT/Union_of_balls_epsilon_mesh_3.hpp>
#include <SBL/GT/Betti_numbers_2.hpp>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>

#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/facets_in_complex_3_to_triangle_mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
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
typedef CGAL::Regular_triangulation_3<K, Tds> Triangulation_3;
typedef CGAL::Fixed_alpha_shape_3<Triangulation_3> Alpha_complex;

typedef SBL::GT::T_Union_of_balls_boundary_3_data_structure<Alpha_complex> Boundary;
typedef SBL::GT::T_Union_of_balls_boundary_3_builder<Boundary> Boundary_builder;
typedef SBL::GT::T_Union_of_balls_epsilon_mesh_3<Boundary> Mesher;

// Epsilon-sampling of the boundary of the union of balls (SBL Union_of_balls_mesh_3).
py::array_t<double> sample_union_of_balls(py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                                          py::array_t<double, py::array::c_style | py::array::forcecast> radii,
                                          double epsilon) {
    auto c = centers.unchecked<2>();
    auto r = radii.unchecked<1>();
    std::vector<K::Sphere_3> spheres;
    spheres.reserve(r.shape(0));
    for (py::ssize_t i = 0; i < r.shape(0); ++i)
        spheres.emplace_back(K::Point_3(c(i, 0), c(i, 1), c(i, 2)), r(i) * r(i));

    Boundary boundary(spheres.begin(), spheres.end());
    Boundary_builder builder;
    builder(boundary);

    std::vector<K::Point_3> points;
    Mesher mesher;
    mesher(boundary, epsilon, std::back_inserter(points));

    py::array_t<double> out({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
    auto o = out.mutable_unchecked<2>();
    for (size_t i = 0; i < points.size(); ++i) {
        o(i, 0) = points[i].x();
        o(i, 1) = points[i].y();
        o(i, 2) = points[i].z();
    }
    return out;
}

// Duplicates the vertices of non-manifold edges so that every edge has at most two faces.
py::tuple split_nonmanifold_edges(py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                                  py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces) {
    auto vb = vertices.unchecked<2>();
    auto fb = faces.unchecked<2>();
    std::vector<K::Point_3> points;
    points.reserve(vb.shape(0));
    for (py::ssize_t i = 0; i < vb.shape(0); ++i)
        points.emplace_back(vb(i, 0), vb(i, 1), vb(i, 2));
    std::vector<std::vector<std::size_t>> polygons(fb.shape(0));
    for (py::ssize_t i = 0; i < fb.shape(0); ++i)
        polygons[i] = {static_cast<std::size_t>(fb(i, 0)), static_cast<std::size_t>(fb(i, 1)),
                       static_cast<std::size_t>(fb(i, 2))};

    CGAL::Polygon_mesh_processing::duplicate_non_manifold_edges_in_polygon_soup(points, polygons);

    py::array_t<double> vout({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
    auto vo = vout.mutable_unchecked<2>();
    for (size_t i = 0; i < points.size(); ++i) {
        vo(i, 0) = points[i].x();
        vo(i, 1) = points[i].y();
        vo(i, 2) = points[i].z();
    }
    py::array_t<std::int64_t> fout({static_cast<py::ssize_t>(polygons.size()), static_cast<py::ssize_t>(3)});
    auto fo = fout.mutable_unchecked<2>();
    for (size_t i = 0; i < polygons.size(); ++i)
        for (int j = 0; j < 3; ++j)
            fo(i, j) = static_cast<std::int64_t>(polygons[i][j]);
    return py::make_tuple(vout, fout);
}

// Balls bucketed in a uniform grid of cell size max radius, so that a query only visits 27 cells.
struct Ball_grid {
    std::vector<double> centers, sq_radii;
    double cell;
    double origin[3];
    int dims[3];
    std::vector<int> start, items;

    Ball_grid(const std::vector<double>& c, const std::vector<double>& r) : centers(c) {
        const std::size_t n = r.size();
        cell = *std::max_element(r.begin(), r.end());
        double hi[3];
        for (int a = 0; a < 3; ++a) {
            origin[a] = hi[a] = c[a];
            for (std::size_t i = 0; i < n; ++i) {
                origin[a] = std::min(origin[a], c[3 * i + a]);
                hi[a] = std::max(hi[a], c[3 * i + a]);
            }
            origin[a] -= cell;
            dims[a] = static_cast<int>((hi[a] - origin[a]) / cell) + 2;
        }
        sq_radii.resize(n);
        std::vector<int> key(n);
        start.assign(static_cast<std::size_t>(dims[0]) * dims[1] * dims[2] + 1, 0);
        for (std::size_t i = 0; i < n; ++i) {
            sq_radii[i] = r[i] * r[i];
            key[i] = index(c[3 * i], c[3 * i + 1], c[3 * i + 2]);
            ++start[key[i] + 1];
        }
        for (std::size_t k = 1; k < start.size(); ++k)
            start[k] += start[k - 1];
        items.resize(n);
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (std::size_t i = 0; i < n; ++i)
            items[fill[key[i]]++] = static_cast<int>(i);
    }

    int index(double x, double y, double z) const {
        const int i = static_cast<int>((x - origin[0]) / cell);
        const int j = static_cast<int>((y - origin[1]) / cell);
        const int k = static_cast<int>((z - origin[2]) / cell);
        return (i * dims[1] + j) * dims[2] + k;
    }

    // Smallest power distance |p - c|^2 - r^2 over the balls near p; positive when p is outside the union.
    double power(double x, double y, double z) const {
        const double p[3] = {x, y, z};
        int lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            const int c = static_cast<int>(std::floor((p[a] - origin[a]) / cell));
            lo[a] = std::max(c - 1, 0);
            hi[a] = std::min(c + 1, dims[a] - 1);
            if (lo[a] > hi[a])
                return 1.0;
        }
        double best = 1.0;
        for (int i = lo[0]; i <= hi[0]; ++i)
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int k = lo[2]; k <= hi[2]; ++k) {
                    const int key = (i * dims[1] + j) * dims[2] + k;
                    for (int s = start[key]; s < start[key + 1]; ++s) {
                        const int b = items[s];
                        const double dx = x - centers[3 * b], dy = y - centers[3 * b + 1], dz = z - centers[3 * b + 2];
                        best = std::min(best, dx * dx + dy * dy + dz * dz - sq_radii[b]);
                    }
                }
        return best;
    }
};

struct Union_of_balls_function {
    std::shared_ptr<const Ball_grid> grid;
    K::FT operator()(const K::Point_3& p) const { return grid->power(p.x(), p.y(), p.z()); }
};

typedef CGAL::Labeled_mesh_domain_3<K> Implicit_domain;
typedef CGAL::Mesh_domain_with_polyline_features_3<Implicit_domain> Mesh_domain;
typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Mesh_tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Mesh_tr, Mesh_domain::Corner_index, Mesh_domain::Curve_index> C3t3;
typedef CGAL::Mesh_criteria_3<Mesh_tr> Mesh_criteria;
typedef Mesher::Spherical_kernel SK;
typedef std::vector<K::Point_3> Polyline;

K::Point_3 to_point(const SK::Circular_arc_point_3& p) {
    return K::Point_3(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
}

// Points of the circle from angle 0 (at `from`) over `sweep` radians, one every `step` radians at most.
Polyline arc_points(const K::Point_3& center, const K::Vector_3& u, const K::Vector_3& w, double rho,
                    double sweep, double step) {
    const int n = std::max(2, static_cast<int>(std::ceil(std::abs(sweep) / step)));
    Polyline points;
    for (int i = 0; i <= n; ++i) {
        const double t = sweep * i / n;
        points.push_back(center + rho * (std::cos(t) * u + std::sin(t) * w));
    }
    return points;
}

// Crease arcs of the boundary of the union as polylines sharing their end points exactly.
std::vector<Polyline> crease_polylines(Boundary& boundary, const Ball_grid& grid, double step_length) {
    std::unordered_map<const void*, K::Point_3> corners;
    auto corner = [&](Boundary::Vertex_handle v) {
        auto it = corners.find(&*v);
        if (it == corners.end())
            it = corners.emplace(&*v, to_point(boundary.get_point<SK>(v))).first;
        return it->second;
    };
    auto buried = [&](const Polyline& points) {
        int n = 0;
        for (std::size_t i = 1; i + 1 < points.size(); ++i)
            n += grid.power(points[i].x(), points[i].y(), points[i].z()) < -1e-6;
        return n;
    };

    std::vector<Polyline> polylines;
    std::unordered_set<const void*> visited;
    for (Boundary::Halfedge_iterator h = boundary.halfedges_begin(); h != boundary.halfedges_end(); ++h) {
        if (h->is_degenerated() || visited.count(&*h->opposite()))
            continue;
        visited.insert(&*h);

        const SK::Circle_3 circle = boundary.get_supporting_circle<SK>(h);
        const K::Point_3 center(circle.center().x(), circle.center().y(), circle.center().z());
        const double rho = std::sqrt(circle.squared_radius());
        const SK::Vector_3 nv = circle.supporting_plane().orthogonal_vector();
        K::Vector_3 normal(nv.x(), nv.y(), nv.z());
        normal = normal / std::sqrt(normal.squared_length());
        const double step = std::min(M_PI / 60, step_length / rho);

        const bool closed = h->is_full_circle() || h->is_degenerated_full_circle();
        K::Point_3 source, target;
        if (h->is_full_circle()) {
            K::Vector_3 any = std::abs(normal.x()) < 0.9 ? K::Vector_3(1, 0, 0) : K::Vector_3(0, 1, 0);
            source = center + rho * CGAL::cross_product(normal, any) / std::sqrt(CGAL::cross_product(normal, any).squared_length());
        } else {
            source = corner(closed ? h->vertex() : h->opposite()->vertex());
            target = corner(h->vertex());
        }
        K::Vector_3 u = source - center;
        u = u / std::sqrt(u.squared_length());
        const K::Vector_3 w = CGAL::cross_product(normal, u);

        Polyline points;
        if (closed || source == target) {
            points = arc_points(center, u, w, rho, 2 * M_PI, step);
        } else {
            const K::Vector_3 d = target - center;
            double angle = std::atan2(d * w, d * u);
            if (angle <= 0)
                angle += 2 * M_PI;
            Polyline ccw = arc_points(center, u, w, rho, angle, step);
            Polyline cw = arc_points(center, u, w, rho, angle - 2 * M_PI, step);
            points = buried(ccw) <= buried(cw) ? ccw : cw;
            points.back() = target;
        }
        points.front() = source;
        if (closed || source == target)
            points.back() = source;
        polylines.push_back(points);
    }
    return polylines;
}

// Surface mesh of the boundary of the union of balls by CGAL Mesh_3, with the crease arcs protected as features.
py::tuple mesh_union_of_balls(py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                              py::array_t<double, py::array::c_style | py::array::forcecast> radii,
                              double facet_size, double facet_distance, double facet_angle, double edge_size,
                              double min_size, int mode) {
    auto c = centers.unchecked<2>();
    auto r = radii.unchecked<1>();
    const py::ssize_t n = r.shape(0);
    std::vector<double> flat(3 * n), rad(n);
    std::vector<K::Sphere_3> spheres;
    spheres.reserve(n);
    for (py::ssize_t i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a)
            flat[3 * i + a] = c(i, a);
        rad[i] = r(i);
        spheres.emplace_back(K::Point_3(c(i, 0), c(i, 1), c(i, 2)), r(i) * r(i));
    }
    auto grid = std::make_shared<const Ball_grid>(flat, rad);

    Boundary boundary(spheres.begin(), spheres.end());
    Boundary_builder builder;
    builder(boundary);
    const std::vector<Polyline> polylines = crease_polylines(boundary, *grid, 0.1);

    K::Point_3 mean(0, 0, 0);
    for (py::ssize_t i = 0; i < n; ++i)
        mean = mean + (spheres[i].center() - CGAL::ORIGIN) / static_cast<double>(n);
    py::ssize_t middle = 0;
    for (py::ssize_t i = 1; i < n; ++i)
        if (CGAL::squared_distance(spheres[i].center(), mean) < CGAL::squared_distance(spheres[middle].center(), mean))
            middle = i;
    double reach = 0;
    for (py::ssize_t i = 0; i < n; ++i)
        reach = std::max(reach, std::sqrt(CGAL::squared_distance(spheres[i].center(), spheres[middle].center())) + rad[i]);
    reach += 1.0;

    namespace params = CGAL::parameters;
    Mesh_domain domain = Mesh_domain::create_implicit_mesh_domain(
        Union_of_balls_function{grid}, K::Sphere_3(spheres[middle].center(), reach * reach),
        params::relative_error_bound(1e-7));
    if (mode & 1)
        domain.add_features(polylines.begin(), polylines.end());
    Mesh_criteria criteria(params::edge_size(edge_size).facet_angle(facet_angle).facet_size(facet_size)
                               .facet_distance(facet_distance).edge_min_size(min_size).facet_min_size(min_size));
    C3t3 c3t3 = (mode & 2) ? CGAL::make_mesh_3<C3t3>(domain, criteria, params::manifold().no_perturb().no_exude())
                           : CGAL::make_mesh_3<C3t3>(domain, criteria, params::no_perturb().no_exude());

    std::vector<K::Point_3> points;
    std::vector<std::array<std::size_t, 3>> faces;
    std::vector<C3t3::Surface_patch_index> patches;
    CGAL::SMDS_3::internal::facets_in_complex_3_to_triangle_soup(c3t3, C3t3::Subdomain_index(1), points, faces, patches, true, false);

    py::array_t<double> vout({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
    auto vo = vout.mutable_unchecked<2>();
    for (size_t i = 0; i < points.size(); ++i)
        for (int a = 0; a < 3; ++a)
            vo(i, a) = points[i][a];
    py::array_t<std::int64_t> fout({static_cast<py::ssize_t>(faces.size()), static_cast<py::ssize_t>(3)});
    auto fo = fout.mutable_unchecked<2>();
    for (size_t i = 0; i < faces.size(); ++i)
        for (int a = 0; a < 3; ++a)
            fo(i, a) = static_cast<std::int64_t>(faces[i][a]);
    return py::make_tuple(vout, fout, static_cast<py::ssize_t>(polylines.size()));
}

// Betti numbers (components, tunnels, cavities) of the union of balls, from its alpha-complex at alpha = 0.
py::tuple union_betti_numbers(py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                              py::array_t<double, py::array::c_style | py::array::forcecast> radii) {
    auto c = centers.unchecked<2>();
    auto r = radii.unchecked<1>();
    std::vector<Triangulation_3::Weighted_point> balls;
    for (py::ssize_t i = 0; i < r.shape(0); ++i)
        balls.emplace_back(K::Point_3(c(i, 0), c(i, 1), c(i, 2)), r(i) * r(i));
    const auto betti = SBL::GT::T_Betti_numbers_2<Alpha_complex>()(balls.begin(), balls.end());
    return py::make_tuple(betti.get<0>(), betti.get<1>(), betti.get<2>());
}

PYBIND11_MODULE(cgal_sbl_sampling, m) {
    m.def("sample_union_of_balls", &sample_union_of_balls, py::arg("centers"), py::arg("radii"), py::arg("epsilon"));
    m.def("split_nonmanifold_edges", &split_nonmanifold_edges, py::arg("vertices"), py::arg("faces"));
    m.def("mesh_union_of_balls", &mesh_union_of_balls, py::arg("centers"), py::arg("radii"), py::arg("facet_size"),
          py::arg("facet_distance"), py::arg("facet_angle"), py::arg("edge_size"), py::arg("min_size") = 0.0, py::arg("mode") = 3);
    m.def("union_betti_numbers", &union_betti_numbers, py::arg("centers"), py::arg("radii"));
}
