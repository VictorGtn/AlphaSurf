#!/usr/bin/env python3
import argparse
import os
import sys

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

try:
    import open3d.visualization.rendering as rendering
except ImportError:
    pass

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import cgal_alpha  # noqa: E402
except ImportError:
    build_dir = (
        "/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/cgal_alpha_bindings/build"
    )
    sys.path.append(build_dir)
    import cgal_alpha  # noqa: E402

from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402


def compute_meshes(pdb_path, alpha_value=0.0, mode="both"):
    parsed_data = parse_pdb_path(pdb_path, use_pqr=False)
    if parsed_data is None or parsed_data[5] is None:
        raise ValueError(f"Failed to parse PDB: {pdb_path}")

    atom_pos = parsed_data[5].astype(np.float32)
    atom_radius = parsed_data[7].astype(np.float32)

    verts_raw = faces_raw = None
    verts_rep = faces_rep = None

    if mode in ("raw", "both"):
        print("Computing raw mesh (no repair)...")
        verts_raw, faces_raw = cgal_alpha.compute_alpha_complex_from_atoms(
            atom_pos, atom_radius, float(alpha_value), 1.4, "singular+regular"
        )

    if mode in ("repaired", "both", "bridges"):
        print("Computing repaired mesh...")
        verts_rep, faces_rep = cgal_alpha.compute_alpha_complex_from_atoms(
            atom_pos, atom_radius, float(alpha_value), 1.4, "singular+regular"
        )

    return (
        verts_raw,
        faces_raw,
        verts_rep,
        faces_rep,
        atom_pos,
    )


def compute_face_types(faces):
    if faces is None or len(faces) == 0:
        return np.array([], dtype=np.int32)

    edge_counts = defaultdict(int)
    for f in faces:
        for i in range(3):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
            edge_counts[e] += 1

    types = np.zeros(len(faces), dtype=np.int32)
    for idx, f in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
            if edge_counts[e] != 2:
                types[idx] = 1
                break
    return types


def create_mesh(verts, faces, color):
    if faces is None or len(faces) == 0:
        return None
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def wireframe_trace(verts, faces, scene="scene"):
    if faces is None or len(faces) == 0:
        return None
    tri_verts = verts[faces]
    closed = np.concatenate((tri_verts, tri_verts[:, :1, :]), axis=1)
    nans = np.full((len(faces), 1, 3), np.nan)
    lines = np.concatenate((closed, nans), axis=1).reshape(-1, 3)
    xl, yl, zl = lines.T
    return go.Scatter3d(
        x=xl,
        y=yl,
        z=zl,
        mode="lines",
        line=dict(color="rgba(20,20,20,0.8)", width=4),
        showlegend=False,
        hoverinfo="skip",
        scene=scene,
    )


def mesh_trace(verts, faces, name, color, scene="scene"):
    x, y, z = verts.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        name=name,
        lighting=dict(
            ambient=0.5, diffuse=0.5, roughness=0.3, specular=0.3, fresnel=0.1
        ),
        lightposition=dict(x=100, y=200, z=150),
        flatshading=True,
        scene=scene,
    )


def find_bowtie_vertices(faces):
    if faces is None or len(faces) == 0:
        return {}
    vert_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[int(v)].append(fi)

    bowties = {}
    for v, flist in vert_faces.items():
        if len(flist) < 2:
            continue
        link_adj = defaultdict(set)
        face_link_verts = {}
        for fi in flist:
            tri = faces[fi]
            others = [int(u) for u in tri if int(u) != v]
            face_link_verts[fi] = others
            if len(others) == 2:
                link_adj[others[0]].add(others[1])
                link_adj[others[1]].add(others[0])

        all_link_verts = set(link_adj.keys())
        visited, vert_comp, comp_id = set(), {}, 0
        for u in all_link_verts:
            if u in visited:
                continue
            queue = [u]
            visited.add(u)
            while queue:
                cur = queue.pop()
                vert_comp[cur] = comp_id
                for nb in link_adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            comp_id += 1

        if comp_id > 1:
            comps = defaultdict(list)
            for fi in flist:
                others = face_link_verts[fi]
                c = vert_comp.get(others[0], 0) if others else 0
                comps[c].append(fi)
            has_closed = False
            for c, fi_list in comps.items():
                neighbor_count = defaultdict(int)
                for fi in fi_list:
                    for u in face_link_verts[fi]:
                        neighbor_count[u] += 1
                if all(cnt >= 2 for cnt in neighbor_count.values()):
                    has_closed = True
                    break
            kind = "bubble" if has_closed else "real"
            bowties[v] = {"comps": dict(comps), "kind": kind}
    return bowties


BOWTIE_COLORS = ["#FF6600", "#00CCFF", "#FF00FF", "#00FF66", "#FFFF00", "#FF3366"]


def recolor_bowtie_faces(verts, faces, types, scene="scene"):
    bowties = find_bowtie_vertices(faces)
    types_mod = np.copy(types)
    if not bowties:
        return types_mod, []

    real_verts = [v for v, info in bowties.items() if info["kind"] == "real"]
    bubble_verts = [v for v, info in bowties.items() if info["kind"] == "bubble"]
    traces = []
    if real_verts:
        pts = verts[real_verts]
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color="orange",
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
                name=f"Pinch points ({len(real_verts)})",
                hoverinfo="none",
                scene=scene,
            )
        )
        for v in real_verts:
            for c, fi_list in bowties[v]["comps"].items():
                types_mod[fi_list] = 10 + (c % len(BOWTIE_COLORS))
    if bubble_verts:
        pts = verts[bubble_verts]
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color="cyan",
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
                name=f"Bubble bow-ties ({len(bubble_verts)})",
                hoverinfo="none",
                scene=scene,
            )
        )
        for v in bubble_verts:
            for c, fi_list in bowties[v]["comps"].items():
                types_mod[fi_list] = 20 + (c % len(BOWTIE_COLORS))
    return types_mod, traces


CAMERA_SYNC_JS = """
(function() {
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var syncing = false;
    function sync(src, dst) {
        gd.on('plotly_relayout', function(ed) {
            if (syncing) return;
            var cam = ed[src + '.camera'];
            if (!cam) return;
            syncing = true;
            var update = {};
            update[dst + '.camera'] = cam;
            Plotly.relayout(gd, update).then(function() { syncing = false; });
        });
    }
    sync('scene', 'scene2');
    sync('scene2', 'scene');
})();
"""


def create_html_summary(
    output_path,
    verts_raw,
    faces_raw,
    types_raw,
    verts_rep,
    faces_rep,
    types_rep,
    atom_pos=None,
    mode="both",
    bowtie_only=False,
):
    html_path = os.path.splitext(output_path)[0] + "_compare.html"
    COLORS = {0: "#d5d8dc", 1: "#e74c3c", 2: "#3498db", 3: "#00FF00", 4: "#8B00FF"}
    NAMES = {
        0: "Regular",
        1: "Singular",
        2: "Pocket fill",
        3: "Bridges / SE repair",
        4: "Pinch Point Bridges",
    }
    for idx, c in enumerate(BOWTIE_COLORS):
        COLORS[10 + idx] = c
        NAMES[10 + idx] = f"Pinch comp {idx}"
        COLORS[20 + idx] = c
        NAMES[20 + idx] = f"Bubble comp {idx}"

    def add_mesh_traces(verts, faces, types, scene):
        traces = []
        if faces is None or len(faces) == 0:
            return traces
        for t, color in COLORS.items():
            if bowtie_only and t < 10:
                continue
            mask = types == t
            if not mask.any():
                continue
            traces.append(mesh_trace(verts, faces[mask], NAMES[t], color, scene))
        if not bowtie_only:
            tr = wireframe_trace(verts, faces, scene)
            if tr:
                traces.append(tr)
        if atom_pos is not None:
            traces.append(
                go.Scatter3d(
                    x=atom_pos[:, 0],
                    y=atom_pos[:, 1],
                    z=atom_pos[:, 2],
                    mode="markers",
                    marker=dict(size=3, color="black", opacity=0.5),
                    name="Atoms",
                    scene=scene,
                    hoverinfo="none",
                )
            )
        return traces

    titles, specs = [], []
    if mode in ("raw", "both"):
        titles.append(f"Raw — {len(faces_raw)} faces")
        specs.append({"type": "scene"})
    if mode in ("repaired", "both"):
        titles.append(f"Repaired — {len(faces_rep)} faces")
        specs.append({"type": "scene"})

    fig = make_subplots(rows=1, cols=len(specs), specs=[specs], subplot_titles=titles)
    col = 1
    if mode in ("raw", "both"):
        sz = f"scene{col}" if col > 1 else "scene"
        t_mod, pm = recolor_bowtie_faces(verts_raw, faces_raw, types_raw, sz)
        for tr in add_mesh_traces(verts_raw, faces_raw, t_mod, sz):
            fig.add_trace(tr, row=1, col=col)
        for tr in pm:
            fig.add_trace(tr, row=1, col=col)
        col += 1
    if mode in ("repaired", "both"):
        sz = f"scene{col}" if col > 1 else "scene"
        t_mod, pm = recolor_bowtie_faces(verts_rep, faces_rep, types_rep, sz)
        for tr in add_mesh_traces(verts_rep, faces_rep, t_mod, sz):
            fig.add_trace(tr, row=1, col=col)
        for tr in pm:
            fig.add_trace(tr, row=1, col=col)

    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(r=10, l=10, b=10, t=60),
        title=dict(
            text=f"Alpha complex: {os.path.basename(output_path)}", font=dict(size=16)
        ),
    )
    for i in range(1, len(specs) + 1):
        sz = f"scene{i}" if i > 1 else "scene"
        fig.update_layout(
            {
                sz: dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode="data",
                    bgcolor="white",
                )
            }
        )

    with open(html_path, "w") as f:
        f.write(
            fig.to_html(
                include_plotlyjs="cdn",
                post_script=CAMERA_SYNC_JS if mode == "both" else None,
            )
        )
    print(f"Saved interactive HTML: file://{os.path.abspath(html_path)}")
    return html_path


def render_comparison(
    verts_raw,
    faces_raw,
    verts_rep,
    faces_rep,
    output_path,
    mode="both",
    width=1600,
    height=800,
):
    mesh_raw, mesh_rep = None, None
    if mode in ("raw", "both"):
        mesh_raw = create_mesh(verts_raw, faces_raw, [0.76, 0.23, 0.17])
    if mode in ("repaired", "both"):
        mesh_rep = create_mesh(verts_rep, faces_rep, [0.16, 0.50, 0.73])
    if mode == "both" and mesh_raw and mesh_rep:
        offset = [np.ptp(verts_raw, axis=0).max() * 1.3, 0, 0]
        mesh_rep.translate(offset)
    render = rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0.05, 0.05, 0.05, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = 0.4
    if mesh_raw:
        mat.base_color = [0.76, 0.23, 0.17, 1.0]
        render.scene.add_geometry("raw", mesh_raw, mat)
    if mesh_rep:
        mat.base_color = [0.16, 0.50, 0.73, 1.0]
        render.scene.add_geometry("rep", mesh_rep, mat)
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.enable_indirect_light(True)
    bounds = o3d.geometry.AxisAlignedBoundingBox()
    if mesh_raw:
        bounds += mesh_raw.get_axis_aligned_bounding_box()
    if mesh_rep:
        bounds += mesh_rep.get_axis_aligned_bounding_box()
    center, ext = bounds.get_center(), bounds.get_max_extent()
    render.scene.camera.look_at(
        center, center + np.array([0, -1.5, 0.8]) * ext, np.array([0, 0, 1])
    )
    o3d.io.write_image(output_path, render.render_to_image())
    print(f"Saved image: file://{os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize alpha complex")
    parser.add_argument("pdb_file", help="Path to PDB")
    parser.add_argument("--out", default="compare.png", help="Output path")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mode", choices=["raw", "repaired", "both"], default="both")
    parser.add_argument("--bowtie-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.pdb_file):
        print(f"PDB not found: {args.pdb_file}")
        sys.exit(1)

    v_raw, f_raw, v_rep, f_rep, atom_pos = compute_meshes(
        args.pdb_file, args.alpha, args.mode
    )
    t_raw, t_rep = compute_face_types(f_raw), compute_face_types(f_rep)

    create_html_summary(
        args.out,
        v_raw,
        f_raw,
        t_raw,
        v_rep,
        f_rep,
        t_rep,
        atom_pos=atom_pos,
        mode=args.mode,
        bowtie_only=args.bowtie_only,
    )
    if not args.no_render:
        render_comparison(v_raw, f_raw, v_rep, f_rep, args.out, args.mode)


if __name__ == "__main__":
    main()
