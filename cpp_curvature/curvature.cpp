// Self-contained principal curvature computation using igl's exact
// CurvatureCalculator class. The igl helper functions (adjacency_list,
// vertex_triangle_adjacency, per_face_normals, per_vertex_normals) are
// inlined from the libigl source to avoid requiring igl C++ headers.
//
// The only addition over stock igl: optional custom vertex normals that
// overwrite the internally-computed ones after init().

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <optional>
#include <queue>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define IGL_INLINE inline

// ============================================================
// Inlined igl helper functions (verbatim from libigl)
// ============================================================

namespace igl {

// --- adjacency_list (igl/adjacency_list.cpp) ---

template <typename Index, typename IndexVector>
IGL_INLINE void adjacency_list(const Eigen::MatrixBase<Index> &F,
                               std::vector<std::vector<IndexVector>> &A,
                               bool sorted = false) {
  A.clear();
  A.resize(F.maxCoeff() + 1);

  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < F.cols(); j++) {
      int s = F(i, j);
      int d = F(i, (j + 1) % F.cols());
      A.at(s).push_back(d);
      A.at(d).push_back(s);
    }
  }

  for (int i = 0; i < (int)A.size(); ++i) {
    std::sort(A[i].begin(), A[i].end());
    A[i].erase(std::unique(A[i].begin(), A[i].end()), A[i].end());
  }

  if (sorted) {
    std::vector<std::vector<std::vector<int>>> SR;
    SR.resize(A.size());

    for (int i = 0; i < F.rows(); i++) {
      for (int j = 0; j < F.cols(); j++) {
        int s = F(i, j);
        int d = F(i, (j + 1) % F.cols());
        int v = F(i, (j + 2) % F.cols());

        std::vector<int> e(2);
        e[0] = d;
        e[1] = v;
        SR[s].push_back(e);
      }
    }

    for (int v = 0; v < (int)SR.size(); ++v) {
      std::vector<IndexVector> &vv = A.at(v);
      if (vv.size() == 0)
        continue;
      std::vector<std::vector<int>> &sr = SR[v];
      std::vector<std::vector<int>> pn = sr;

      for (int i = 0; i < (int)sr.size(); ++i) {
        int a = sr[i][0];
        int b = sr[i][1];

        int p = -1;
        for (int j = 0; j < (int)sr.size(); j++)
          if (sr[j][1] == a)
            p = j;
        pn[i][0] = p;

        int n = -1;
        for (int j = 0; j < (int)sr.size(); j++)
          if (sr[j][0] == b)
            n = j;
        pn[i][1] = n;
      }

      int c = 0;
      for (int j = 0; j <= (int)sr.size(); j++)
        if (pn[c][0] != -1)
          c = pn[c][0];

      if (pn[c][0] == -1) {
        for (int j = 0; j < (int)sr.size(); j++) {
          vv[j] = sr[c][0];
          if (pn[c][1] != -1)
            c = pn[c][1];
        }
        vv.back() = sr[c][1];
      } else {
        for (int j = 0; j < (int)sr.size(); j++) {
          vv[j] = sr[c][0];
          c = pn[c][1];
        }
      }
    }
  }
}

// --- vertex_triangle_adjacency (igl/vertex_triangle_adjacency.cpp) ---

template <typename DerivedF, typename VFType, typename VFiType>
IGL_INLINE void
vertex_triangle_adjacency(const typename DerivedF::Scalar n,
                          const Eigen::MatrixBase<DerivedF> &F,
                          std::vector<std::vector<VFType>> &VF,
                          std::vector<std::vector<VFiType>> &VFi) {
  VF.clear();
  VFi.clear();
  VF.resize(n);
  VFi.resize(n);

  typedef typename DerivedF::Index Index;
  for (Index fi = 0; fi < F.rows(); ++fi) {
    for (Index i = 0; i < F.cols(); ++i) {
      VF[F(fi, i)].push_back(fi);
      VFi[F(fi, i)].push_back(i);
    }
  }
}

template <typename DerivedV, typename DerivedF, typename IndexType>
IGL_INLINE void
vertex_triangle_adjacency(const Eigen::MatrixBase<DerivedV> &V,
                          const Eigen::MatrixBase<DerivedF> &F,
                          std::vector<std::vector<IndexType>> &VF,
                          std::vector<std::vector<IndexType>> &VFi) {
  return vertex_triangle_adjacency(V.rows(), F, VF, VFi);
}

// --- per_face_normals (igl/per_face_normals.cpp, serial) ---

template <typename DerivedV, typename DerivedF, typename DerivedZ,
          typename DerivedN>
IGL_INLINE void per_face_normals(const Eigen::MatrixBase<DerivedV> &V,
                                 const Eigen::MatrixBase<DerivedF> &F,
                                 const Eigen::MatrixBase<DerivedZ> &Z,
                                 Eigen::PlainObjectBase<DerivedN> &N) {
  N.resize(F.rows(), 3);
  for (int i = 0; i < F.rows(); i++) {
    const Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v1 =
        V.row(F(i, 1)) - V.row(F(i, 0));
    const Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v2 =
        V.row(F(i, 2)) - V.row(F(i, 0));
    N.row(i) = v1.cross(v2);
    typename DerivedV::Scalar r = N.row(i).norm();
    if (r == 0) {
      N.row(i) = Z;
    } else {
      N.row(i) /= r;
    }
  }
}

template <typename DerivedV, typename DerivedF, typename DerivedN>
IGL_INLINE void per_face_normals(const Eigen::MatrixBase<DerivedV> &V,
                                 const Eigen::MatrixBase<DerivedF> &F,
                                 Eigen::PlainObjectBase<DerivedN> &N) {
  Eigen::Matrix<typename DerivedN::Scalar, 3, 1> Z(0, 0, 0);
  return per_face_normals(V, F, Z, N);
}

// --- per_vertex_normals (igl/per_vertex_normals.cpp, with inlined doublearea)
// ---

enum PerVertexNormalsWeightingType {
  PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM = 0,
  PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA = 1,
  PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE = 2,
  PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT = 3,
  NUM_PER_VERTEX_NORMALS_WEIGHTING_TYPE = 4
};

template <typename DerivedV, typename DerivedF, typename DerivedFN,
          typename DerivedN>
IGL_INLINE void
per_vertex_normals(const Eigen::MatrixBase<DerivedV> &V,
                   const Eigen::MatrixBase<DerivedF> &F,
                   const igl::PerVertexNormalsWeightingType weighting,
                   const Eigen::MatrixBase<DerivedFN> &FN,
                   Eigen::PlainObjectBase<DerivedN> &N) {
  N.setZero(V.rows(), 3);

  Eigen::Matrix<typename DerivedN::Scalar, Eigen::Dynamic, 3> W(F.rows(), 3);
  switch (weighting) {
  case PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM:
    W.setConstant(1.);
    break;
  default:
    assert(false && "Unknown weighting type");
  case PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT:
  case PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA: {
    // Inlined igl::doublearea: 2*triangle area = ||cross product||
    Eigen::Matrix<typename DerivedN::Scalar, Eigen::Dynamic, 1> A(F.rows());
    for (int i = 0; i < F.rows(); i++) {
      Eigen::RowVector3d e1 = V.row(F(i, 1)) - V.row(F(i, 0));
      Eigen::RowVector3d e2 = V.row(F(i, 2)) - V.row(F(i, 0));
      A(i) = e1.cross(e2).norm();
    }
    W = A.replicate(1, 3);
    break;
  }
  case PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE:
    assert(false && "ANGLE weighting not inlined");
    W.setConstant(1.);
    break;
  }

  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      N.row(F(i, j)) += W(i, j) * FN.row(i);
    }
  }

  N.rowwise().normalize();
}

template <typename DerivedV, typename DerivedF, typename DerivedFN,
          typename DerivedN>
IGL_INLINE void per_vertex_normals(const Eigen::MatrixBase<DerivedV> &V,
                                   const Eigen::MatrixBase<DerivedF> &F,
                                   const Eigen::MatrixBase<DerivedFN> &FN,
                                   Eigen::PlainObjectBase<DerivedN> &N) {
  return per_vertex_normals(V, F, PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT, FN,
                            N);
}

} // namespace igl

// ============================================================
// Exact CurvatureCalculator class from igl/principal_curvature.cpp
// ============================================================

typedef enum { SPHERE_SEARCH, K_RING_SEARCH } searchType;

typedef enum { AVERAGE, PROJ_PLANE } normalType;

class CurvatureCalculator {
public:
  std::vector<std::optional<std::array<double, 2>>> curv;
  std::vector<std::optional<std::array<Eigen::Vector3d, 2>>> curvDir;
  bool curvatureComputed;
  class Quadric {
  public:
    IGL_INLINE Quadric() { a() = b() = c() = d() = e() = 1.0; }

    IGL_INLINE Quadric(double av, double bv, double cv, double dv, double ev) {
      a() = av;
      b() = bv;
      c() = cv;
      d() = dv;
      e() = ev;
    }

    IGL_INLINE double &a() { return data[0]; }
    IGL_INLINE double &b() { return data[1]; }
    IGL_INLINE double &c() { return data[2]; }
    IGL_INLINE double &d() { return data[3]; }
    IGL_INLINE double &e() { return data[4]; }

    double data[5];

    IGL_INLINE double evaluate(double u, double v) {
      return a() * u * u + b() * u * v + c() * v * v + d() * u + e() * v;
    }

    IGL_INLINE double du(double u, double v) {
      return 2.0 * a() * u + b() * v + d();
    }

    IGL_INLINE double dv(double u, double v) {
      return 2.0 * c() * v + b() * u + e();
    }

    IGL_INLINE double duv(double, double) { return b(); }

    IGL_INLINE double duu(double, double) { return 2.0 * a(); }

    IGL_INLINE double dvv(double, double) { return 2.0 * c(); }

    IGL_INLINE static Quadric fit(const std::vector<Eigen::Vector3d> &VV) {
      assert(VV.size() >= 5);

      Eigen::MatrixXd A(VV.size(), 5);
      Eigen::MatrixXd b(VV.size(), 1);
      Eigen::MatrixXd sol(5, 1);

      for (unsigned int c = 0; c < VV.size(); ++c) {
        double u = VV[c][0];
        double v = VV[c][1];
        double n = VV[c][2];

        A(c, 0) = u * u;
        A(c, 1) = u * v;
        A(c, 2) = v * v;
        A(c, 3) = u;
        A(c, 4) = v;

        b(c) = n;
      }

      sol = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

      return Quadric(sol(0), sol(1), sol(2), sol(3), sol(4));
    }
  };

public:
  Eigen::MatrixXd vertices;
  Eigen::MatrixXi faces;

  std::vector<std::vector<int>> vertex_to_vertices;
  std::vector<std::vector<int>> vertex_to_faces;
  std::vector<std::vector<int>> vertex_to_faces_index;
  Eigen::MatrixXd face_normals;
  Eigen::MatrixXd vertex_normals;

  /* Size of the neighborhood */
  double sphereRadius;
  int kRing;

  bool localMode;
  bool projectionPlaneCheck;
  bool montecarlo;
  unsigned int montecarloN;

  searchType st;
  normalType nt;

  double lastRadius;
  double scaledRadius;
  std::string lastMeshName;

  bool expStep;
  int step;
  int maxSize;

  IGL_INLINE CurvatureCalculator();
  IGL_INLINE void init(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

  IGL_INLINE void finalEigenStuff(int, const std::array<Eigen::Vector3d, 3> &,
                                  Quadric &);
  IGL_INLINE void fitQuadric(const Eigen::Vector3d &,
                             const std::array<Eigen::Vector3d, 3> &ref,
                             const std::vector<int> &, Quadric *);
  IGL_INLINE void applyProjOnPlane(const Eigen::Vector3d &,
                                   const std::vector<int> &,
                                   std::vector<int> &);
  IGL_INLINE void getSphere(const int, const double, std::vector<int> &,
                            int min);
  IGL_INLINE void getKRing(const int, const double, std::vector<int> &);
  IGL_INLINE Eigen::Vector3d project(const Eigen::Vector3d &,
                                     const Eigen::Vector3d &,
                                     const Eigen::Vector3d &);
  IGL_INLINE void computeReferenceFrame(int, const Eigen::Vector3d &,
                                        std::array<Eigen::Vector3d, 3> &);
  IGL_INLINE void getAverageNormal(int, const std::vector<int> &,
                                   Eigen::Vector3d &);
  IGL_INLINE void getProjPlane(int, const std::vector<int> &,
                               Eigen::Vector3d &);
  IGL_INLINE void applyMontecarlo(const std::vector<int> &, std::vector<int> *);
  IGL_INLINE void computeCurvature();
  IGL_INLINE void printCurvature(const std::string &outpath);
  IGL_INLINE double getAverageEdge();

  IGL_INLINE static int rotateForward(double *v0, double *v1, double *v2) {
    double t;

    if (std::abs(*v2) >= std::abs(*v1) && std::abs(*v2) >= std::abs(*v0))
      return 0;

    t = *v0;
    *v0 = *v2;
    *v2 = *v1;
    *v1 = t;

    return 1 + rotateForward(v0, v1, v2);
  }

  IGL_INLINE static void rotateBackward(int nr, double *v0, double *v1,
                                        double *v2) {
    double t;

    if (nr == 0)
      return;

    t = *v2;
    *v2 = *v0;
    *v0 = *v1;
    *v1 = t;

    rotateBackward(nr - 1, v0, v1, v2);
  }

  IGL_INLINE static Eigen::Vector3d chooseMax(Eigen::Vector3d n,
                                              Eigen::Vector3d abc, double ab) {
    int max_i;
    double max_sp;
    Eigen::Vector3d nt[8];

    n.normalize();
    abc.normalize();

    max_sp = -std::numeric_limits<double>::max();

    for (int i = 0; i < 4; ++i) {
      nt[i] = n;
      if (ab > 0) {
        switch (i) {
        case 0:
          break;

        case 1:
          nt[i][2] = -n[2];
          break;

        case 2:
          nt[i][0] = -n[0];
          nt[i][1] = -n[1];
          break;

        case 3:
          nt[i][0] = -n[0];
          nt[i][1] = -n[1];
          nt[i][2] = -n[2];
          break;
        }
      } else {
        switch (i) {
        case 0:
          nt[i][0] = -n[0];
          break;

        case 1:
          nt[i][1] = -n[1];
          break;

        case 2:
          nt[i][0] = -n[0];
          nt[i][2] = -n[2];
          break;

        case 3:
          nt[i][1] = -n[1];
          nt[i][2] = -n[2];
          break;
        }
      }

      if (nt[i].dot(abc) > max_sp) {
        max_sp = nt[i].dot(abc);
        max_i = i;
      }
    }
    return nt[max_i];
  }
};

class comparer {
public:
  IGL_INLINE bool operator()(const std::pair<int, double> &lhs,
                             const std::pair<int, double> &rhs) const {
    return lhs.second > rhs.second;
  }
};

// --- CurvatureCalculator method implementations (exact from igl) ---

IGL_INLINE CurvatureCalculator::CurvatureCalculator() {
  this->localMode = true;
  this->projectionPlaneCheck = true;
  this->sphereRadius = 5;
  this->st = SPHERE_SEARCH;
  this->nt = AVERAGE;
  this->montecarlo = false;
  this->montecarloN = 0;
  this->kRing = 3;
  this->curvatureComputed = false;
  this->expStep = true;
}

IGL_INLINE void CurvatureCalculator::init(const Eigen::MatrixXd &V,
                                          const Eigen::MatrixXi &F) {
  vertices = V;
  faces = F;
  igl::adjacency_list(F, vertex_to_vertices);
  igl::vertex_triangle_adjacency(V, F, vertex_to_faces, vertex_to_faces_index);
  igl::per_face_normals(V, F, face_normals);
  igl::per_vertex_normals(V, F, face_normals, vertex_normals);
}

IGL_INLINE void
CurvatureCalculator::fitQuadric(const Eigen::Vector3d &v,
                                const std::array<Eigen::Vector3d, 3> &ref,
                                const std::vector<int> &vv, Quadric *q) {
  std::vector<Eigen::Vector3d> points;
  points.reserve(vv.size());

  for (unsigned int i = 0; i < vv.size(); ++i) {
    Eigen::Vector3d cp = vertices.row(vv[i]);
    Eigen::Vector3d vTang = cp - v;

    double x = vTang.dot(ref[0]);
    double y = vTang.dot(ref[1]);
    double z = vTang.dot(ref[2]);
    points.push_back(Eigen::Vector3d(x, y, z));
  }
  if (points.size() < 5) {
    assert(false && "fit function requires at least 5 points");
    *q = Quadric(0, 0, 0, 0, 0);
  } else {
    *q = Quadric::fit(points);
  }
}

IGL_INLINE void CurvatureCalculator::finalEigenStuff(
    int i, const std::array<Eigen::Vector3d, 3> &ref, Quadric &q) {
  const double a = q.a();
  const double b = q.b();
  const double c = q.c();
  const double d = q.d();
  const double e = q.e();

  double E = 1.0 + d * d;
  double F = d * e;
  double G = 1.0 + e * e;

  Eigen::Vector3d n = Eigen::Vector3d(-d, -e, 1.0).normalized();

  double L = 2.0 * a * n[2];
  double M = b * n[2];
  double N = 2 * c * n[2];

  Eigen::Matrix2d m;
  m << L * G - M * F, M * E - L * F, M * E - L * F, N * E - M * F;
  m = m / (E * G - F * F);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(m);

  Eigen::Vector2d c_val = eig.eigenvalues();
  Eigen::Matrix2d c_vec = eig.eigenvectors();

  c_val = -c_val;

  Eigen::Vector3d v1, v2;
  v1[0] = c_vec(0);
  v1[1] = c_vec(1);
  v1[2] = 0;

  v2[0] = c_vec(2);
  v2[1] = c_vec(3);
  v2[2] = 0;

  Eigen::Vector3d v1global = ref[0] * v1[0] + ref[1] * v1[1] + ref[2] * v1[2];
  Eigen::Vector3d v2global = ref[0] * v2[0] + ref[1] * v2[1] + ref[2] * v2[2];

  v1global.normalize();
  v2global.normalize();

  v1global *= c_val(0);
  v2global *= c_val(1);

  if (c_val[0] > c_val[1]) {
    curv[i] = std::array<double, 2>{c_val(0), c_val(1)};
    curvDir[i] = std::array<Eigen::Vector3d, 2>{v1global, v2global};
  } else {
    curv[i] = std::array<double, 2>{c_val(1), c_val(0)};
    curvDir[i] = std::array<Eigen::Vector3d, 2>{v2global, v1global};
  }
}

IGL_INLINE void CurvatureCalculator::getKRing(const int start, const double r,
                                              std::vector<int> &vv) {
  int bufsize = vertices.rows();
  vv.reserve(bufsize);
  std::list<std::pair<int, int>> queue;
  std::vector<bool> visited(bufsize, false);
  queue.push_back(std::pair<int, int>(start, 0));
  visited[start] = true;
  while (!queue.empty()) {
    int toVisit = queue.front().first;
    int distance = queue.front().second;
    queue.pop_front();
    vv.push_back(toVisit);
    if (toVisit < (int)vertex_to_vertices.size()) {
      if (distance < (int)r) {
        for (unsigned int i = 0; i < vertex_to_vertices[toVisit].size(); ++i) {
          int neighbor = vertex_to_vertices[toVisit][i];
          if (!visited[neighbor]) {
            queue.push_back(std::pair<int, int>(neighbor, distance + 1));
            visited[neighbor] = true;
          }
        }
      }
    }
  }
}

IGL_INLINE void CurvatureCalculator::getSphere(const int start, const double r,
                                               std::vector<int> &vv, int min) {
  int bufsize = vertices.rows();
  vv.reserve(bufsize);
  std::list<int> queue;
  std::vector<bool> visited(bufsize, false);
  queue.push_back(start);
  visited[start] = true;
  Eigen::Vector3d me = vertices.row(start);
  std::priority_queue<std::pair<int, double>,
                      std::vector<std::pair<int, double>>, comparer>
      extra_candidates;
  while (!queue.empty()) {
    int toVisit = queue.front();
    queue.pop_front();
    vv.push_back(toVisit);
    for (unsigned int i = 0; i < vertex_to_vertices[toVisit].size(); ++i) {
      int neighbor = vertex_to_vertices[toVisit][i];
      if (!visited[neighbor]) {
        Eigen::Vector3d neigh = vertices.row(neighbor);
        double distance = (me - neigh).norm();
        if (distance < r)
          queue.push_back(neighbor);
        else if ((int)vv.size() < min)
          extra_candidates.push(std::pair<int, double>(neighbor, distance));
        visited[neighbor] = true;
      }
    }
  }
  while (!extra_candidates.empty() && (int)vv.size() < min) {
    std::pair<int, double> cand = extra_candidates.top();
    extra_candidates.pop();
    vv.push_back(cand.first);
    for (unsigned int i = 0; i < vertex_to_vertices[cand.first].size(); ++i) {
      int neighbor = vertex_to_vertices[cand.first][i];
      if (!visited[neighbor]) {
        Eigen::Vector3d neigh = vertices.row(neighbor);
        double distance = (me - neigh).norm();
        extra_candidates.push(std::pair<int, double>(neighbor, distance));
        visited[neighbor] = true;
      }
    }
  }
}

IGL_INLINE Eigen::Vector3d
CurvatureCalculator::project(const Eigen::Vector3d &v,
                             const Eigen::Vector3d &vp,
                             const Eigen::Vector3d &ppn) {
  return (vp - (ppn * ((vp - v).dot(ppn))));
}

IGL_INLINE void CurvatureCalculator::computeReferenceFrame(
    int i, const Eigen::Vector3d &normal, std::array<Eigen::Vector3d, 3> &ref) {
  Eigen::Vector3d longest_v =
      Eigen::Vector3d(vertices.row(vertex_to_vertices[i][0]));

  longest_v = (project(vertices.row(i), longest_v, normal) -
               Eigen::Vector3d(vertices.row(i)))
                  .normalized();

  Eigen::Vector3d y_axis = (normal.cross(longest_v)).normalized();
  ref[0] = longest_v;
  ref[1] = y_axis;
  ref[2] = normal;
}

IGL_INLINE void
CurvatureCalculator::getAverageNormal(int j, const std::vector<int> &vv,
                                      Eigen::Vector3d &normal) {
  normal = (vertex_normals.row(j)).normalized();
  if (localMode)
    return;

  for (unsigned int i = 0; i < vv.size(); ++i) {
    normal += vertex_normals.row(vv[i]).normalized();
  }
  normal.normalize();
}

IGL_INLINE void CurvatureCalculator::getProjPlane(int j,
                                                  const std::vector<int> &vv,
                                                  Eigen::Vector3d &ppn) {
  int nr;
  double a, b, c;
  double nx, ny, nz;
  double abcq;

  a = b = c = 0;

  if (localMode) {
    for (unsigned int i = 0; i < vertex_to_faces.at(j).size(); ++i) {
      Eigen::Vector3d faceNormal =
          face_normals.row(vertex_to_faces.at(j).at(i));
      a += faceNormal[0];
      b += faceNormal[1];
      c += faceNormal[2];
    }
  } else {
    for (unsigned int i = 0; i < vv.size(); ++i) {
      a += vertex_normals.row(vv[i])[0];
      b += vertex_normals.row(vv[i])[1];
      c += vertex_normals.row(vv[i])[2];
    }
  }
  nr = rotateForward(&a, &b, &c);
  abcq = a * a + b * b + c * c;
  nx = sqrt(a * a / abcq);
  ny = sqrt(b * b / abcq);
  nz = sqrt(1 - nx * nx - ny * ny);
  rotateBackward(nr, &a, &b, &c);
  rotateBackward(nr, &nx, &ny, &nz);

  ppn = chooseMax(Eigen::Vector3d(nx, ny, nz), Eigen::Vector3d(a, b, c), a * b);
  ppn.normalize();
}

IGL_INLINE double CurvatureCalculator::getAverageEdge() {
  double sum = 0;
  int count = 0;

  for (int i = 0; i < faces.rows(); ++i) {
    for (short unsigned j = 0; j < 3; ++j) {
      Eigen::Vector3d p1 = vertices.row(faces(i, j));
      Eigen::Vector3d p2 = vertices.row(faces(i, (j + 1) % 3));

      double l = (p1 - p2).norm();

      sum += l;
      ++count;
    }
  }

  return (sum / (double)count);
}

IGL_INLINE void
CurvatureCalculator::applyProjOnPlane(const Eigen::Vector3d &ppn,
                                      const std::vector<int> &vin,
                                      std::vector<int> &vout) {
  for (std::vector<int>::const_iterator vpi = vin.begin(); vpi != vin.end();
       ++vpi)
    if (vertex_normals.row(*vpi) * ppn > 0.0)
      vout.push_back(*vpi);
}

IGL_INLINE void
CurvatureCalculator::applyMontecarlo(const std::vector<int> &vin,
                                     std::vector<int> *vout) {
  if (montecarloN >= vin.size()) {
    *vout = vin;
    return;
  }

  float p = ((float)montecarloN) / (float)vin.size();
  for (std::vector<int>::const_iterator vpi = vin.begin(); vpi != vin.end();
       ++vpi) {
    float r;
    if ((r = ((float)rand() / RAND_MAX)) < p) {
      vout->push_back(*vpi);
    }
  }
}

IGL_INLINE void CurvatureCalculator::computeCurvature() {
  const size_t vertices_count = vertices.rows();

  if (vertices_count == 0)
    return;

  curvDir = std::vector<std::optional<std::array<Eigen::Vector3d, 2>>>(
      vertices_count);
  curv = std::vector<std::optional<std::array<double, 2>>>(vertices_count);

  scaledRadius = getAverageEdge() * sphereRadius;

  std::vector<int> vv;
  std::vector<int> vvtmp;
  Eigen::Vector3d normal;

  for (size_t i = 0; i < vertices_count; ++i) {
    vv.clear();
    vvtmp.clear();
    Eigen::Vector3d me = vertices.row(i);
    switch (st) {
    case SPHERE_SEARCH:
      getSphere(i, scaledRadius, vv, 6);
      break;
    case K_RING_SEARCH:
      getKRing(i, kRing, vv);
      break;
    default:
      fprintf(stderr, "Error: search type not recognized");
      return;
    }

    if (vv.size() < 6) {
      continue;
    }

    if (projectionPlaneCheck) {
      vvtmp.reserve(vv.size());
      applyProjOnPlane(vertex_normals.row(i), vv, vvtmp);
      if (vvtmp.size() >= 6 && vvtmp.size() < vv.size())
        vv = vvtmp;
    }

    switch (nt) {
    case AVERAGE:
      getAverageNormal(i, vv, normal);
      break;
    case PROJ_PLANE:
      getProjPlane(i, vv, normal);
      break;
    default:
      fprintf(stderr, "Error: normal type not recognized");
      return;
    }
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2]) || normal.norm() < 0.5)
      continue;
    if (vv.size() < 6) {
      continue;
    }
    if (montecarlo) {
      if (montecarloN < 6)
        break;
      vvtmp.reserve(vv.size());
      applyMontecarlo(vv, &vvtmp);
      vv = vvtmp;
    }

    if (vv.size() < 6)
      return;
    std::array<Eigen::Vector3d, 3> ref;
    computeReferenceFrame(i, normal, ref);

    Quadric q;
    fitQuadric(me, ref, vv, &q);
    finalEigenStuff(i, ref, q);
  }

  lastRadius = sphereRadius;
  curvatureComputed = true;
}

IGL_INLINE void
CurvatureCalculator::printCurvature(const std::string &outpath) {
  if (!curvatureComputed)
    return;

  std::ofstream of;
  of.open(outpath.c_str());

  if (!of) {
    fprintf(stderr, "Error: could not open output file %s\n", outpath.c_str());
    return;
  }

  int vertices_count = vertices.rows();
  of << vertices_count << std::endl;
  for (int i = 0; i < vertices_count; i++) {
    of << curv[i].value()[0] << " " << curv[i].value()[1] << " "
       << curvDir[i].value()[0][0] << " " << curvDir[i].value()[0][1] << " "
       << curvDir[i].value()[0][2] << " " << curvDir[i].value()[1][0] << " "
       << curvDir[i].value()[1][1] << " " << curvDir[i].value()[1][2]
       << std::endl;
  }

  of.close();
}

// ============================================================
// Pybind11 wrapper
// ============================================================

PYBIND11_MODULE(curvature_ext, m) {
  m.doc() = "Principal curvature via igl's exact CurvatureCalculator";
  m.def(
      "principal_curvature",
      [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
         py::object normals_obj, unsigned int radius,
         bool useKring) -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,
                                      Eigen::VectorXd, Eigen::VectorXd> {
        if (radius < 2)
          radius = 2;

        Eigen::MatrixXd PD1(V.rows(), 3);
        Eigen::MatrixXd PD2(V.rows(), 3);
        Eigen::VectorXd PV1(V.rows());
        Eigen::VectorXd PV2(V.rows());
        PD1.setZero();
        PD2.setZero();
        PV1.setZero();
        PV2.setZero();

        CurvatureCalculator cc;
        cc.init(V, F);

        if (!normals_obj.is_none()) {
          Eigen::MatrixXd N = normals_obj.cast<Eigen::MatrixXd>();
          cc.vertex_normals = N;
        }

        cc.sphereRadius = radius;
        if (useKring) {
          cc.kRing = radius;
          cc.st = K_RING_SEARCH;
        }

        cc.computeCurvature();

        for (unsigned int i = 0; i < V.rows(); ++i) {
          if (cc.curv[i].has_value()) {
            PD1.row(i) << (*cc.curvDir[i])[0][0], (*cc.curvDir[i])[0][1],
                (*cc.curvDir[i])[0][2];
            PD2.row(i) << (*cc.curvDir[i])[1][0], (*cc.curvDir[i])[1][1],
                (*cc.curvDir[i])[1][2];
            PD1.row(i).normalize();
            PD2.row(i).normalize();

            if (std::isnan(PD1(i, 0)) || std::isnan(PD1(i, 1)) ||
                std::isnan(PD1(i, 2)) || std::isnan(PD2(i, 0)) ||
                std::isnan(PD2(i, 1)) || std::isnan(PD2(i, 2))) {
              PD1.row(i).setZero();
              PD2.row(i).setZero();
            }

            PV1(i) = (*cc.curv[i])[0];
            PV2(i) = (*cc.curv[i])[1];

            if (PD1.row(i) * PD2.row(i).transpose() > 10e-6) {
              PD1.row(i).setZero();
              PD2.row(i).setZero();
            }
          } else {
            PV1(i) = 0;
            PV2(i) = 0;
            PD1.row(i).setZero();
            PD2.row(i).setZero();
          }
        }

        return std::make_tuple(PD1, PD2, PV1, PV2);
      },
      py::arg("V"), py::arg("F"), py::arg("normals") = py::none(),
      py::arg("radius") = 5, py::arg("use_kring") = true,
      "Compute principal curvatures using igl's exact CurvatureCalculator.\n\n"
      "Parameters\n----------\n"
      "V : (n, 3) vertex positions\n"
      "F : (m, 3) triangle face indices\n"
      "normals : (n, 3) optional vertex normals. If None, computed from mesh.\n"
      "radius : int, neighborhood size (minimum 2)\n"
      "use_kring : bool, use k-ring search (True) or sphere search (False)\n\n"
      "Returns\n-------\n"
      "PD1, PD2 : (n, 3) principal curvature directions\n"
      "PV1, PV2 : (n,) principal curvature values (PV1=max, PV2=min)");
}
