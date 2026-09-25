#!/usr/bin/env python3
"""
Dirac heat diffusion on several surfaces of one protein, after AtomSurf
supplementary figure 2.

A unit of heat is released on the patch of one atom (by default the atom closest
to the partner chain's centroid, i.e. inside the binding site) and diffused with
the network's spectral operator. With --copy, the source is a single vertex on
one side instead: that tufted copy of the atom, copies ranked by how much their
sheet faces the camera, and on every other mesh the
atom's vertex closest to a point 1.5 A off the atom towards that copy's sheet
(on EDTSurf and NanoShaper, whose vertices carry no atom, the closest vertex).
The untufted vertex, shared by all sheets, then releases heat into every one. The hottest vertices (by temperature) that
together hold 90% of the heat mass are drawn yellow, the rest of each surface in its method's colour from the
surface speed and size figures. Every panel is ray-traced in PyMOL, or with --render mesh drawn
flat-shaded with its triangle edges as in visualize_all_methods.py, from the same
viewpoint: the one passed with --pymol-view, else facing the source atom. --render vector draws the
--render mesh panels as vector triangles, straight into the figure.
EDTSurf and NanoShaper surfaces are built at --grid-scale, or at the scale the kind names
(e.g. nanoshaper@0.4), cleaned as in
training, and each vertex takes the atom whose sphere is closest. The SAS kinds
are the SBL union-of-balls mesh of spectral_comparison, full and decimated. With --ncols and
a single time, the surfaces are laid out in rows of ncols panels.

Outputs:
  <output_dir>/<name>_dirac_diffusion.pdf
  <output_dir>/panels/<name>_<kind>_t<t>.png   (not with --render vector)
"""

import argparse
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree

script_dir = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(script_dir))
FIG_DIR = os.path.join(REPO_ROOT, "plotting", "figures", "spectral")
# spectral_comparison.py builds the meshes; visualize_all_methods.py renders them.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "meshviz"))

from spectral_comparison import build_mesh_set, largest_component, spectra_for  # noqa: E402

from alphasurf.protein.create_surface import mesh_simplification, pdb_to_edtsurf, pdb_to_nanoshaper  # noqa: E402
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402

KIND_LABELS = {
    "alpha": "alpha (no tufting)",
    "alpha_tuft": "alpha (tufting)",
    "msms_dec": "MSMS (simplified)",
    "msms_full": "MSMS",
    "edtsurf": "EDTSurf",
    "nanoshaper": "NanoShaper",
    "sas": "SAS",
    "sas_dec": "SAS (simplified)",
}
GRID_KINDS = ("edtsurf", "nanoshaper")
# The SAS is the union of balls with radii + 1.4 A, meshed from SBL samples as in spectral_comparison.
SAS_KINDS = {"sas": "msms_full", "sas_dec": "msms_dec"}
# Colours of plot_surface_speed.py; grid kinds are keyed by grid scale.
KIND_COLORS = {
    "alpha": "#E41A1C", "alpha_tuft": "#E41A1C", "msms_dec": "#B07CC6", "msms_full": "#6A3D9A",
    ("edtsurf", 0.3): "#9ECAE1", ("edtsurf", 0.4): "#4292C6", ("edtsurf", 0.5): "#08519C",
    ("nanoshaper", 0.4): "#41AB5D", ("nanoshaper", 0.5): "#238B45",
}
BASE_COLOR = (0.12, 0.13, 0.72)
HOT_COLOR = (1.00, 0.88, 0.25)
MASS_FRACTION = 0.9
# Weight of the edge colour in a vector edge; it matches the --render mesh panels, where the translucent
# strokes of the two faces sharing an edge overlap.
VECTOR_EDGE_ALPHA = 0.36


def grid_tool(kind, grid_scale):
    """(tool, grid scale) of a grid kind such as edtsurf or nanoshaper@0.4, bare names taking grid_scale; else None."""
    tool, _, scale = kind.partition("@")
    return (tool, float(scale) if scale else grid_scale) if tool in GRID_KINDS else None


def panel_labels(kinds, grid_scale):
    labels = {}
    for kind in kinds:
        grid = grid_tool(kind, grid_scale)
        labels[kind] = f"{KIND_LABELS[grid[0]]} (gs = {grid[1]:g})" if grid else KIND_LABELS[kind]
    return labels


def source_atom(atom_pos, partner_pdb, meshes):
    """Atom present on every mesh and closest to the partner chain's centroid."""
    shared = set.intersection(*(set(m["atom_idx"].tolist()) for m in meshes.values()))
    shared = np.array(sorted(shared))
    centroid = np.asarray(parse_pdb_path(partner_pdb, use_pqr=False)[5], dtype=float).mean(axis=0)
    return int(shared[np.argmin(np.linalg.norm(atom_pos[shared] - centroid, axis=1))])


def grid_mesh(kind, pdb, atom_pos, atom_radius, grid_scale):
    """EDTSurf or NanoShaper surface, cleaned as in training, with each vertex's closest atom sphere."""
    make = pdb_to_edtsurf if kind == "edtsurf" else pdb_to_nanoshaper
    verts, faces = make(pdb, grid_scale=grid_scale)
    verts, faces, _, _ = mesh_simplification(
        verts=verts, faces=faces, out_ply=None, face_reduction_rate=1.0, surface_method=kind,
        min_vert_number=16, max_vert_number=1000000, use_pymesh=False,
    )
    verts, faces, _ = largest_component(np.asarray(verts, dtype=float), np.asarray(faces))
    dist, idx = cKDTree(atom_pos).query(verts, k=min(16, len(atom_pos)))
    atom_idx = idx[np.arange(len(verts)), (dist - atom_radius[idx]).argmin(axis=1)]
    return dict(verts=verts, faces=faces, atom_idx=atom_idx)


def diffuse(spec, patch, t):
    """Per-vertex heat mass at time t from a unit spread uniformly over the patch."""
    mass = spec["mass"]
    u0 = patch / (mass * patch).sum()
    coef = spec["evecs"].T @ (mass * u0)
    return mass * (spec["evecs"] @ (np.exp(-spec["evals"] * t) * coef))


def sheet_side(verts, faces, v):
    """Unit direction from vertex v towards the centroids of its incident faces."""
    incident = faces[(faces == v).any(axis=1)]
    d = (verts[incident].mean(axis=1) - verts[v]).sum(axis=0)
    return d / np.linalg.norm(d)


def hot_vertices(heat_mass, mass):
    """Hottest vertices, by temperature, that together hold MASS_FRACTION of the heat.

    Ranking by temperature rather than by heat mass keeps small-area vertices
    inside the hot region, so it has no holes on irregular meshes.
    """
    order = np.argsort(heat_mass / mass)[::-1]
    n_hot = int(np.searchsorted(np.cumsum(heat_mass[order]), MASS_FRACTION)) + 1
    hot = np.zeros(len(heat_mass), dtype=bool)
    hot[order[:n_hot]] = True
    return hot


def view_rotation(direction):
    """Rotation taking direction onto +z, towards the PyMOL camera."""
    z = direction / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    return np.stack([x, np.cross(z, x), z])


def visible_faces(verts, faces, hot):
    """One face per set of coincident faces, the one with the most hot vertices.

    Tufted copies of a triangle share their positions on different sheets, and the
    renderer would otherwise show an arbitrary one of them.
    """
    _, position = np.unique(np.round(verts, 4), axis=0, return_inverse=True)
    _, group = np.unique(np.sort(position.ravel()[faces], axis=1), axis=0, return_inverse=True)
    group = group.ravel()
    order = np.lexsort((-hot[faces].sum(axis=1), group))
    return faces[order[np.r_[True, group[order][1:] != group[order][:-1]]]]


def mesh_cgo(verts, faces, hot, color):
    """Flat-shaded triangles; colours are per vertex, so they blend across a face."""
    from pymol.cgo import BEGIN, COLOR, END, NORMAL, TRIANGLES, VERTEX

    faces = visible_faces(verts, faces, hot)
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    obj = [BEGIN, TRIANGLES]
    for face, normal in zip(faces, normals):
        for v in face:
            obj += [COLOR, *(HOT_COLOR if hot[v] else color), NORMAL, *normal, VERTEX, *verts[v]]
    obj.append(END)
    return obj


def render(panels, view_kind, out_dir, name, width, view=None):
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-qc"])
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("two_sided_lighting", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0.15)
    cmd.set("ambient", 0.35)
    cmd.set("orthoscopic", 1)
    cmd.set("depth_cue", 0)
    cmd.set("ray_trace_fog", 0)

    cmd.viewport(width, width)
    cmd.load_cgo(mesh_cgo(*panels[(view_kind, None)]), "frame")
    if view is None:
        cmd.reset()
    else:
        cmd.set_view(view)
    # Keeps the rotation and refits the camera to the mesh.
    cmd.zoom("frame", buffer=5.0, complete=1)
    view = cmd.get_view()
    cmd.delete("frame")

    paths = {}
    for key, panel in panels.items():
        if key[1] is None:
            continue
        kind, t = key
        cmd.load_cgo(mesh_cgo(*panel), "panel")
        cmd.set_view(view)
        path = os.path.join(out_dir, "panels", f"{name}_{kind}_t{t:g}.png")
        cmd.png(path, width=width, height=width, ray=1)
        cmd.delete("panel")
        paths[key] = path
    return paths


def render_mesh(panels, view_kind, out_dir, name, width):
    """Panels drawn as visualize_all_methods.py draws surfaces, seen from +z; each face blends its vertex colours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_all_methods import mesh_collection, mesh_limits

    center, half_width = mesh_limits(panels[(view_kind, None)][0])
    size = 3.3
    paths = {}
    for (kind, t), (verts, faces, hot, color) in panels.items():
        if t is None:
            continue
        faces = visible_faces(verts, faces, hot)
        vertex_rgb = np.where(hot[:, None], HOT_COLOR, color)
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_axes((0, 0, 1, 1), projection="3d")
        ax.add_collection3d(mesh_collection(verts, faces, vertex_rgb[faces].mean(axis=1)))
        for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            set_lim(c - half_width, c + half_width)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=90, azim=-90)
        ax.set_axis_off()
        path = os.path.join(out_dir, "panels", f"{name}_{kind}_t{t:g}.png")
        fig.savefig(path, dpi=width / size, transparent=True)
        plt.close(fig)
        paths[(kind, t)] = path
    return paths


def assemble(paths, labels, times, out_path, ncols=None, crop=(0.0, 1.0, 0.0, 1.0), corner_radius=0.0):
    """Lay the panels out in the order of labels (kind -> title), all cut to one box given as
    (left, right, top, bottom) fractions of the surfaces' extent.

    The box's corners are rounded with a radius of corner_radius times its shorter side.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = {key: plt.imread(path) for key, path in paths.items()}
    # One crop box for every panel, so the surfaces keep a common scale.
    opaque = np.any([img[..., 3] > 0 for img in images.values()], axis=0)
    rows, cols = np.flatnonzero(opaque.any(axis=1)), np.flatnonzero(opaque.any(axis=0))
    height, width = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
    left, right, top, bottom = crop
    crop = (slice(rows[0] + int(top * height), rows[0] + int(bottom * height)),
            slice(cols[0] + int(left * width), cols[0] + int(right * width)))
    aspect = (crop[0].stop - crop[0].start) / (crop[1].stop - crop[1].start)
    images = {key: img[crop].copy() for key, img in images.items()}
    if corner_radius > 0:
        h, w = next(iter(images.values())).shape[:2]
        radius = corner_radius * min(h, w)
        y, x = np.mgrid[:h, :w] + 0.5
        dx = np.maximum(np.maximum(radius - x, x - (w - radius)), 0.0)
        dy = np.maximum(np.maximum(radius - y, y - (h - radius)), 0.0)
        # One pixel of antialiasing along the rounded corners.
        inside = np.clip(radius - np.hypot(dx, dy) + 0.5, 0.0, 1.0)
        for img in images.values():
            img[..., 3] *= inside
    layout(lambda ax, key: ax.imshow(images[key]), aspect, labels, times, out_path, ncols)


def assemble_vector(panels, labels, times, out_path, ncols=None, crop=(0.0, 1.0, 0.0, 1.0), corner_radius=0.0,
                    edge_width=0.16, raster_dpi=None):
    """As assemble, with every panel drawn from +z as flat-shaded vector triangles, edges edge_width points wide.

    With raster_dpi, each panel's triangles are embedded as one image at that resolution; the rest stays vector.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import FancyBboxPatch
    from visualize_all_methods import MESH_EDGE_COLOR, shaded_face_colors

    drawn = {}
    for (kind, t), (verts, faces, hot, color) in panels.items():
        if t is None:
            continue
        faces = visible_faces(verts, faces, hot)
        # Painted from the farthest face to the closest.
        faces = faces[np.argsort(verts[faces, 2].mean(axis=1))]
        vertex_rgb = np.where(hot[:, None], HOT_COLOR, color)
        drawn[(kind, t)] = (verts[faces][:, :, :2], shaded_face_colors(verts, faces, vertex_rgb[faces].mean(axis=1)))
    # One crop box for every panel, so the surfaces keep a common scale.
    xy = np.concatenate([triangles.reshape(-1, 2) for triangles, _ in drawn.values()])
    (x_min, y_min), (x_max, y_max) = xy.min(axis=0), xy.max(axis=0)
    left, right, top, bottom = crop
    x0, x1 = x_min + left * (x_max - x_min), x_min + right * (x_max - x_min)
    y0, y1 = y_max - bottom * (y_max - y_min), y_max - top * (y_max - y_min)
    radius = corner_radius * min(x1 - x0, y1 - y0)
    edge_rgb = np.asarray(MESH_EDGE_COLOR[:3])

    def draw(ax, key):
        triangles, colors = drawn[key]
        inside = ((triangles[..., 0].max(axis=1) > x0) & (triangles[..., 0].min(axis=1) < x1)
                  & (triangles[..., 1].max(axis=1) > y0) & (triangles[..., 1].min(axis=1) < y1))
        # Opaque edges, blended into their face, so no seams show between faces.
        edges = colors[inside, :3] * (1 - VECTOR_EDGE_ALPHA) + edge_rgb * VECTOR_EDGE_ALPHA
        collection = PolyCollection(triangles[inside], facecolors=colors[inside], edgecolors=edges,
                                    linewidths=edge_width, antialiased=True, rasterized=raster_dpi is not None)
        ax.add_collection(collection)
        collection.set_clip_path(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, transform=ax.transData,
                                                boxstyle=f"round,pad=0,rounding_size={radius}"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")

    layout(draw, (y1 - y0) / (x1 - x0), labels, times, out_path, ncols, dpi=raster_dpi or 300)


def layout(draw, aspect, labels, times, out_path, ncols=None, dpi=300):
    """Panels in the order of labels (kind -> title), each drawn by draw(ax, (kind, t)) into a box of this aspect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ncols:
        (t,) = times
        nrows = -(-len(labels) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, (2.4 * aspect + 0.35) * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (kind, label) in zip(axes.flat, labels.items()):
            draw(ax, (kind, t))
            ax.set_title(label, fontsize=10, pad=3)
        fig.tight_layout(pad=0.2, w_pad=1.5, h_pad=1.2)
    else:
        fig, axes = plt.subplots(len(labels), len(times),
                                 figsize=(2.6 * len(times), 2.6 * aspect * len(labels) + 0.4), squeeze=False)
        for r, (kind, label) in enumerate(labels.items()):
            for c, t in enumerate(times):
                ax = axes[r][c]
                draw(ax, (kind, t))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    ax.set_title(f"$t={t:g}$ s", fontsize=12)
                if c == 0:
                    ax.set_ylabel(label, fontsize=11)
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pdb_dir = os.path.join(REPO_ROOT, "data", "pinder-pair", "pdb")
    parser.add_argument("--pdb", default=os.path.join(pdb_dir, "1ycr__A1_Q00987--1ycr__B1_P04637_R.pdb"))
    parser.add_argument("--partner", default=os.path.join(pdb_dir, "1ycr__A1_Q00987--1ycr__B1_P04637_L.pdb"),
                        help="partner chain whose centroid picks the source atom")
    parser.add_argument("--atom", type=int, default=None, help="source atom index; overrides --partner")
    parser.add_argument("--copy", type=int, default=None,
                        help="release on this tufted copy of the source atom only, and on the same side elsewhere; "
                             "copies are ranked by how much their sheet faces the camera, 0 facing it most")
    parser.add_argument("--kinds", default="alpha,alpha_tuft,msms_dec,msms_full")
    parser.add_argument("--times", default="1,5,10,20", help="diffusion times in seconds")
    parser.add_argument("--pymol-view", default=None,
                        help="PyMOL's get_view output for the loaded --pdb, or a file holding it; "
                             "default faces the source atom")
    parser.add_argument("--tag", default="", help="suffix of the output names, e.g. to keep several views")
    parser.add_argument("--k-eig", type=int, default=128)
    parser.add_argument("--msms-reduction", type=float, default=0.1)
    parser.add_argument("--grid-scale", type=float, default=0.5,
                        help="EDTSurf and NanoShaper grid_scale, in grid points per angstrom, for kinds "
                             "named without their own scale as in nanoshaper@0.4")
    parser.add_argument("--ncols", type=int, default=None,
                        help="with a single time, lay the surfaces out in rows of this many panels")
    parser.add_argument("--width", type=int, default=900, help="panel size in pixels")
    parser.add_argument("--render", choices=["pymol", "mesh", "vector"], default="pymol",
                        help="ray-trace the panels in PyMOL, or draw them flat-shaded with triangle edges "
                             "as visualize_all_methods.py does, as images (mesh) or as vector triangles (vector)")
    parser.add_argument("--edge-width", type=float, default=0.16,
                        help="triangle edge width in points on the page, with --render vector")
    parser.add_argument("--raster-dpi", type=int, default=None,
                        help="with --render vector, embed each panel's triangles as one image at this resolution, "
                             "keeping text and layout vector")
    parser.add_argument("--crop", default="0,1,0,1",
                        help="left,right,top,bottom of the box every panel is cut to, as fractions of the "
                             "surfaces' extent from its top-left corner")
    parser.add_argument("--corner-radius", type=float, default=0.0,
                        help="radius of the rounded corners of every panel, as a fraction of its shorter side")
    parser.add_argument("--output-dir", default=FIG_DIR)
    args = parser.parse_args()
    kinds = args.kinds.split(",")
    times = [float(t) for t in args.times.split(",")]
    crop = tuple(float(x) for x in args.crop.split(","))
    if len(crop) != 4:
        parser.error("--crop needs left,right,top,bottom")
    if args.render != "pymol" and args.pymol_view:
        parser.error("--pymol-view needs --render pymol")
    if args.ncols and len(times) != 1:
        parser.error("--ncols needs a single time")
    name = os.path.basename(args.pdb).removesuffix(".pdb")
    if args.tag:
        name += f"_{args.tag}"
    os.makedirs(os.path.join(args.output_dir, "panels"), exist_ok=True)

    grids = {k: grid_tool(k, args.grid_scale) for k in kinds}
    mesh_args = SimpleNamespace(
        pairs=[f"{k}:{k}" for k in kinds if grids[k] is None and k not in SAS_KINDS], alpha_value=0.0,
        msms_density=1.0,
        msms_reduction=args.msms_reduction, min_vert_number=16, max_vert_number=1000000,
        support="patch", msms_radius_offset=0.0, msms_probe=None, surface_engine="msms",
        allow_multiple_components=False,
    )
    meshes, _ = build_mesh_set(args.pdb, mesh_args)
    parsed = parse_pdb_path(args.pdb, use_pqr=False)
    atom_pos = np.asarray(parsed[5], dtype=float)
    for kind in kinds:
        if grids[kind]:
            tool, scale = grids[kind]
            meshes[kind] = grid_mesh(tool, args.pdb, atom_pos, np.asarray(parsed[7], dtype=float), scale)
    sas_kinds = [k for k in kinds if k in SAS_KINDS]
    if sas_kinds:
        sas_args = SimpleNamespace(**{**vars(mesh_args), "pairs": [f"{SAS_KINDS[k]}:{SAS_KINDS[k]}" for k in sas_kinds],
                                      "msms_radius_offset": 1.4, "surface_engine": "sbl", "sbl_epsilon": 1.0})
        sas_meshes, _ = build_mesh_set(args.pdb, sas_args)
        for k in sas_kinds:
            meshes[k] = sas_meshes[SAS_KINDS[k]]
    atom = args.atom if args.atom is not None else source_atom(atom_pos, args.partner, meshes)

    all_verts = np.concatenate([np.asarray(meshes[k]["verts"], dtype=float) for k in kinds])
    centre = all_verts.mean(axis=0)
    view = None
    if args.pymol_view:
        text = open(args.pymol_view).read() if os.path.exists(args.pymol_view) else args.pymol_view
        view = [float(x) for x in re.findall(r"-?\d+\.\d+", text)]
        if len(view) != 18:
            parser.error(f"--pymol-view needs 18 numbers, got {len(view)}")
        # The view is expressed in the PDB frame, so the meshes stay there.
        centre, rot = np.zeros(3), np.eye(3)
    else:
        # Local outward direction, so a source inside a crevice is seen from its opening.
        near = np.linalg.norm(atom_pos - atom_pos[atom], axis=1) < 12.0
        rot = view_rotation(atom_pos[atom] - atom_pos[near].mean(axis=0))

    # PyMOL's view starts with the column-major rotation from model to camera axes.
    toward_camera = rot[2] if view is None else np.array(view[:9]).reshape(3, 3)[:, 2]
    anchor = None
    if args.copy is not None:
        tuft = meshes["alpha_tuft"]
        tuft_verts = np.asarray(tuft["verts"], dtype=float)
        copies = np.flatnonzero(tuft["atom_idx"] == atom)
        if not 0 <= args.copy < len(copies):
            parser.error(f"atom {atom} has {len(copies)} tufted copies")
        sides = np.array([sheet_side(tuft_verts, np.asarray(tuft["faces"]), v) for v in copies])
        # The mesh orders the copies differently from run to run, so they are ranked by how much
        # their sheet faces the camera.
        pick = np.argsort(-(sides @ toward_camera))[args.copy]
        copy_vertex, side = copies[pick], sides[pick]
        anchor = atom_pos[atom] + 1.5 * side
        print(f"copy {args.copy} of {len(copies)}, sheet side {side.round(3)}")

    panels = {}
    print(f"source atom {atom}")
    print(f"{'surface':12s} {'t':>5} {'n_verts':>8} {'hot verts':>10} {'hot area (A^2)':>15}")
    for kind in kinds:
        mesh = meshes[kind]
        color = to_rgb(KIND_COLORS.get(grids[kind] or kind, BASE_COLOR))
        spec = spectra_for(mesh, args.k_eig)
        verts = (np.asarray(mesh["verts"], dtype=float) - centre) @ rot.T
        faces = np.asarray(mesh["faces"])
        patch = (mesh["atom_idx"] == atom).astype(float)
        if anchor is not None:
            own = np.flatnonzero(patch)
            if kind == "alpha_tuft":
                source = copy_vertex
            else:
                pool = np.arange(len(patch)) if grids[kind] else own
                gap = np.linalg.norm(np.asarray(mesh["verts"])[pool] - anchor, axis=1)
                source = pool[np.argmin(gap)]
                print(f"{kind}: source vertex {gap.min():.2f} A from the anchor")
            patch = np.zeros(len(patch))
            patch[source] = 1.0
        panels[(kind, None)] = (verts, faces, np.zeros(len(verts), dtype=bool), color)
        for t in times:
            hot = hot_vertices(diffuse(spec, patch, t), spec["mass"])
            panels[(kind, t)] = (verts, faces, hot, color)
            print(f"{kind:12s} {t:5g} {len(verts):8d} {int(hot.sum()):10d} {spec['mass'][hot].sum():15.1f}")

    view_kind = max(kinds, key=lambda k: np.ptp(panels[(k, None)][0], axis=0).max())
    out_path = os.path.join(args.output_dir, f"{name}_dirac_diffusion.pdf")
    labels = panel_labels(kinds, args.grid_scale)
    if args.render == "vector":
        assemble_vector(panels, labels, times, out_path, args.ncols, crop, args.corner_radius, args.edge_width,
                        args.raster_dpi)
    else:
        if args.render == "mesh":
            paths = render_mesh(panels, view_kind, args.output_dir, name, args.width)
        else:
            paths = render(panels, view_kind, args.output_dir, name, args.width, view)
        assemble(paths, labels, times, out_path, args.ncols, crop, args.corner_radius)
    print(f"figure written to {out_path}")


if __name__ == "__main__":
    main()
