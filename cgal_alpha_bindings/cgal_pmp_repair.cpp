#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef Mesh::Vertex_index Vd;

py::tuple repair_nonmanifold(
    py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        faces) {
  auto vb = vertices.unchecked<2>();
  auto fb = faces.unchecked<2>();

  if (vb.shape(1) != 3)
    throw std::invalid_argument("vertices must be (N,3)");
  if (fb.shape(1) != 3)
    throw std::invalid_argument("faces must be (M,3)");

  const std::size_t nv = vb.shape(0);
  const std::size_t nf = fb.shape(0);

  std::vector<Point_3> points(nv);
  for (std::size_t i = 0; i < nv; ++i)
    points[i] = Point_3(vb(i, 0), vb(i, 1), vb(i, 2));

  std::vector<std::vector<std::size_t>> polygons(nf);
  for (std::size_t i = 0; i < nf; ++i)
    polygons[i] = {static_cast<std::size_t>(fb(i, 0)),
                   static_cast<std::size_t>(fb(i, 1)),
                   static_cast<std::size_t>(fb(i, 2))};

  // Step 1: duplicate vertices at non-manifold edges (soup level)
  PMP::duplicate_non_manifold_edges_in_polygon_soup(points, polygons);

  // Step 2: build Surface_mesh, keeping intentionally duplicated points
  Mesh mesh;
  PMP::polygon_soup_to_polygon_mesh(
      points, polygons, mesh, CGAL::parameters::erase_all_duplicates(false));

  // Step 3: fix remaining non-manifold vertices (mesh level)
  PMP::duplicate_non_manifold_vertices(mesh);

  // Extract
  auto vimap = mesh.add_property_map<Vd, std::size_t>("v:outidx", 0).first;
  std::size_t idx = 0;
  for (auto v : mesh.vertices())
    put(vimap, v, idx++);

  const std::size_t out_nv = idx;
  const std::size_t out_nf = mesh.number_of_faces();

  py::array_t<double> vout({static_cast<py::ssize_t>(out_nv), 3L});
  auto vo = vout.mutable_unchecked<2>();
  for (auto v : mesh.vertices()) {
    const auto &p = mesh.point(v);
    std::size_t i = get(vimap, v);
    vo(i, 0) = p.x();
    vo(i, 1) = p.y();
    vo(i, 2) = p.z();
  }

  py::array_t<std::int64_t> fout({static_cast<py::ssize_t>(out_nf), 3L});
  auto fo = fout.mutable_unchecked<2>();
  std::size_t fi = 0;
  for (auto f : mesh.faces()) {
    int j = 0;
    for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
      fo(fi, j++) = static_cast<std::int64_t>(get(vimap, v));
    ++fi;
  }

  return py::make_tuple(vout, fout);
}

PYBIND11_MODULE(cgal_pmp_repair, m) {
  m.def("repair_nonmanifold", &repair_nonmanifold, py::arg("vertices"),
        py::arg("faces"),
        "Repair non-manifold mesh using CGAL PMP.\n\n"
        "1. duplicate_non_manifold_edges_in_polygon_soup\n"
        "2. polygon_soup_to_polygon_mesh (preserving duplicates)\n"
        "3. duplicate_non_manifold_vertices\n\n"
        "Args:\n"
        "    vertices: (N, 3) float64\n"
        "    faces: (M, 3) int64\n"
        "Returns:\n"
        "    (vertices, faces) as float64 / int64 arrays");
}
