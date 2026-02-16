#!/usr/bin/env python3
import argparse
import os
import sys

# Set headless rendering flags BEFORE importing Open3D
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import numpy as np
import open3d as o3d

try:
    import open3d.visualization.rendering as rendering
except ImportError:
    pass  # Older open3d versions might lack this, but usually present.

import plotly.graph_objects as go

# Ensure alphasurf is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
alphasurf_root = os.path.join(current_dir, "alphasurf")
if alphasurf_root not in sys.path:
    sys.path.append(alphasurf_root)

# Try to import cgal_alpha from build dir specifically if not found
try:
    import cgal_alpha
except ImportError:
    # Adjust this path to where cgal_alpha.so is located
    build_dir = (
        "/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/cgal_alpha_bindings/build"
    )
    if os.path.exists(build_dir):
        sys.path.append(build_dir)
        try:
            import cgal_alpha
        except ImportError:
            print(f"Error: Could not import cgal_alpha even after adding {build_dir}")
            sys.exit(1)
    else:
        # Try relative path
        pass

from alphasurf.protein.graphs import parse_pdb_path


def compute_alpha_complex(pdb_path, alpha_value=0.0):
    print(f"Parsing PDB: {pdb_path}")
    parsed_data = parse_pdb_path(pdb_path)
    if parsed_data is None:
        raise ValueError(f"Failed to parse PDB: {pdb_path}")

    atom_pos = parsed_data[5]
    atom_radius = parsed_data[7]

    # 1. Get Singular Faces
    print("Computing Singular Faces...")
    verts_s, faces_s, _, _ = cgal_alpha.compute_alpha_complex_from_atoms(
        atom_pos.astype(np.float32),
        atom_radius.astype(np.float32),
        float(alpha_value),
        1.4,
        "singular",
    )

    # 2. Get Regular Faces
    print("Computing Regular Faces...")
    verts_r, faces_r, _, _ = cgal_alpha.compute_alpha_complex_from_atoms(
        atom_pos.astype(np.float32),
        atom_radius.astype(np.float32),
        float(alpha_value),
        1.4,
        "regular",
    )

    return verts_s, faces_s, verts_r, faces_r


def create_mesh(verts, faces, color):
    if faces is None or len(faces) == 0:
        return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def render_scene(mesh_regular, mesh_singular, output_path, width=1600, height=1200):
    # Setup Open3D Offscreen Renderer
    render = rendering.OffscreenRenderer(width, height)
    # Dark background as requested "like that"
    render.scene.set_background([0.05, 0.05, 0.05, 1.0])

    # Material setup
    mat_reg = rendering.MaterialRecord()
    mat_reg.shader = "defaultLit"
    mat_reg.base_color = [0.7, 0.75, 0.8, 1.0]  # Light Grey
    mat_reg.base_roughness = 0.4
    mat_reg.base_metallic = 0.0

    mat_sing = rendering.MaterialRecord()
    mat_sing.shader = "defaultLit"
    mat_sing.base_color = [0.9, 0.2, 0.2, 1.0]  # Crimson
    mat_sing.base_roughness = 0.2
    mat_sing.base_metallic = 0.0

    mat_wire = rendering.MaterialRecord()
    mat_wire.shader = "unlitLine"
    mat_wire.line_width = 1.5
    mat_wire.base_color = [0.0, 0.0, 0.0, 1.0]  # Black edges

    bounds = o3d.geometry.AxisAlignedBoundingBox()

    if mesh_regular is not None:
        render.scene.add_geometry("regular", mesh_regular, mat_reg)
        bounds += mesh_regular.get_axis_aligned_bounding_box()
        # Create wireframe for regular
        wire_reg = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_regular)
        wire_reg.paint_uniform_color([0.1, 0.1, 0.1])
        render.scene.add_geometry("regular_wire", wire_reg, mat_wire)

    if mesh_singular is not None:
        render.scene.add_geometry("singular", mesh_singular, mat_sing)
        bounds += mesh_singular.get_axis_aligned_bounding_box()
        # Create wireframe for singular
        wire_sing = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_singular)
        wire_sing.paint_uniform_color([0.0, 0.0, 0.0])  # Pure Black edges
        render.scene.add_geometry("singular_wire", wire_sing, mat_wire)

    center = bounds.get_center()
    extent = bounds.get_max_extent()

    if extent == 0:
        print("Error: Empty scene (extent=0)")
        return

    # Lights setup for "Lighting"
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 100000)
    render.scene.scene.enable_indirect_light(True)

    # Camera Setup
    camera = render.scene.camera
    center = bounds.get_center()
    extent = bounds.get_max_extent()
    eye = center + np.array([0, -1.5, 0.8]) * extent
    up = np.array([0, 0, 1])
    render.scene.camera.look_at(center, eye, up)

    print(f"Rendering to {output_path}")
    img = render.render_to_image()
    o3d.io.write_image(output_path, img)

    # Create HTML summary
    html_path = create_html_summary(output_path, mesh_regular, mesh_singular)

    # Print clickable link for VS Code
    abs_path = os.path.abspath(output_path)
    abs_html_path = os.path.abspath(html_path)

    print(f"\nSuccessfully saved image. Click to open:\nfile://{abs_path}")
    print(
        f"Successfully saved INTERACTIVE HTML. Click to open:\nfile://{abs_html_path}"
    )


def create_html_summary(image_path, mesh_regular, mesh_singular):
    base_name = os.path.basename(image_path)
    # Use _interactive.html to distinguish
    html_path = os.path.splitext(image_path)[0] + "_interactive.html"

    print("Generating interactive 3D plot (using Plotly)...")

    data = []

    # Helper to create wireframe lines for Plotly
    def get_wireframe_lines(mesh, color="black"):
        if mesh is None or len(mesh.triangles) == 0:
            return None

        verts = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Optimization: Plotly creates lines between points in sequence.
        # To draw triangles, we need A->B->C->A for each triangle.
        # This is huge for large meshes (Regular), so we skip it there.
        # But for Singular (usually smaller), we can try.
        if len(triangles) > 10000:
            print(f"Skipping wireframe for large mesh ({len(triangles)} faces).")
            return None

        # Prepare line segments (x, y, z with None separators)
        # Vectorized approach:
        # Triangles: [v0, v1, v2]
        # Edges: (v0, v1), (v1, v2), (v2, v0)
        # We can construct a long array [v0, v1, v2, v0, None] for each tri

        tri_indices = np.column_stack((triangles, triangles[:, 0]))  # [v0, v1, v2, v0]
        # Flatten
        flat_indices = tri_indices.flatten()

        # Insert None every 4 vertices? Plotly lines are connected.
        # Actually simpler: Go.Mesh3d doesn't do wireframe easily.
        # Scatter3d with 'lines' mode requires x,y,z arrays.

        # Create x, y, z arrays including None
        # This is tricky without explicit loops which are slow in Python.
        # Fast approximate wireframe: Just points? No, user wants edges.

        # Proper way: extract unique edges.
        # But simpler way for visualization: Just plot lines for all triangle edges.

        # Let's do a simplified wireframe:
        # Just use Scatter3d on vertices? No.

        # Let's try explicit loop but optimize
        x, y, z = [], [], []
        # Pre-allocate?
        # A simple approach for *edges*:
        # x = [x0, x1, None, x1, x2, None, ...]

        # Construct line segments from triangles
        # v0 -> v1, v1 -> v2, v2 -> v0
        # shape (N, 3, 3) -> vertices of each triangle
        vt = verts[triangles]

        # We want to enable wireframe.
        # Let's use a trick: `go.Mesh3d` with `contour` is for surfaces.
        # `go.Scatter3d` is the way.

        # To be fast:
        # Just singular faces?
        return None  # Placeholder, implemented inside add_mesh_trace

    def add_mesh_trace(mesh, name, color, opacity=1.0, wireframe=False):
        if mesh is None or len(mesh.triangles) == 0:
            return

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Compute proper lighting normals if not present
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Plotly uses x, y, z arrays and i, j, k arrays for faces
        x, y, z = verts.T
        i, j, k = faces.T

        trace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            showscale=False,
            # Enabling lighting
            lighting=dict(
                ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.4, fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=150),
            flatshading=False,  # Smooth shading usually looks better with lighting
        )
        data.append(trace)

        if wireframe and len(faces) < 5000:  # Limit wireframe to smaller meshes
            # Create wireframe using Scatter3d
            # Construct lines: (v0, v1), (v1, v2), (v2, v0)
            # Efficiently:
            # We need a sequence of vertices with Nones to break lines
            # [v0, v1, v2, v0, None] for each triangle
            # Using numpy to construct

            tri_verts = verts[faces]  # (N, 3, 3)
            # Add the first vertex to the end to close the loop: (N, 4, 3)
            tri_verts_closed = np.concatenate((tri_verts, tri_verts[:, :1, :]), axis=1)

            # Add a row of NaNs/Nones to separate triangles?
            # Creating a separator array (N, 1, 3) of NaNs
            nans = np.full((len(faces), 1, 3), np.nan)

            # Concatenate: (N, 5, 3)
            tri_verts_with_nan = np.concatenate((tri_verts_closed, nans), axis=1)

            # Flatten to (N*5, 3)
            lines = tri_verts_with_nan.reshape(-1, 3)

            xl, yl, zl = lines.T

            line_trace = go.Scatter3d(
                x=xl,
                y=yl,
                z=zl,
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
                name=f"{name} Edges",
            )
            data.append(line_trace)

    # Add traces
    # Regular: No wireframe (too heavy), Light Grey
    add_mesh_trace(
        mesh_regular, "Regular Faces", "lightgrey", opacity=1.0, wireframe=False
    )
    # Singular: Wireframe YES, Crimson
    add_mesh_trace(
        mesh_singular, "Singular Faces", "crimson", opacity=1.0, wireframe=True
    )

    if not data:
        print("Warning: No mesh data to plot.")
        return None

    # Create layout with Dark Theme as requested via snippet implication, or just clean
    layout = go.Layout(
        title=f"Interactive Singular Face Analysis: {base_name}",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor="rgb(20, 20, 20)",  # Dark background for the 3D scene
        ),
        paper_bgcolor="rgb(20, 20, 20)",  # Dark outer background
        font=dict(color="white"),  # White text
        margin=dict(r=0, l=0, b=0, t=40),
    )

    fig = go.Figure(data=data, layout=layout)

    # Save as standalone HTML
    fig.write_html(html_path)
    print(f"Saved interactive HTML to {html_path}")

    # Return path for printing
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Visualize Singular vs Regular faces")
    parser.add_argument("pdb_file", help="Path to PDB file")
    parser.add_argument("--out", default="singular_vis.png", help="Output image path")
    parser.add_argument("--alpha", type=float, default=0.0, help="Alpha value")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering (useful if headless EGL crashes)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdb_file):
        print(f"PDB file not found: {args.pdb_file}")
        sys.exit(1)

    verts_s, faces_s, verts_r, faces_r = compute_alpha_complex(
        args.pdb_file, args.alpha
    )

    print(f"Singular Faces: {len(faces_s) if faces_s is not None else 0}")
    print(f"Regular Faces:  {len(faces_r) if faces_r is not None else 0}")

    # Create Open3D meshes
    mesh_s = create_mesh(verts_s, faces_s, [0.9, 0.2, 0.2])  # Crimson
    mesh_r = create_mesh(verts_r, faces_r, [0.7, 0.75, 0.8])  # Grey

    if mesh_s:
        o3d.io.write_triangle_mesh(args.out.replace(".png", "_singular.ply"), mesh_s)
        print(f"Saved singular mesh to {args.out.replace('.png', '_singular.ply')}")

    if mesh_r:
        o3d.io.write_triangle_mesh(args.out.replace(".png", "_regular.ply"), mesh_r)
        print(f"Saved regular mesh to {args.out.replace('.png', '_regular.ply')}")

    if not args.no_render:
        try:
            render_scene(mesh_r, mesh_s, args.out)
        except Exception as e:
            print(f"Rendering failed: {e}")
            print(
                "Tip: If this segfaults, use --no-render and view the .ply files locally."
            )
            # Fallback to just HTML generation if PNG render fails (e.g. headless)
            create_html_summary(args.out, mesh_r, mesh_s)
    else:
        print("Skipping rendering as per --no-render.")
        # But still generate HTML!
        create_html_summary(args.out, mesh_r, mesh_s)


if __name__ == "__main__":
    main()
