import torch
import sys
import numpy as np
import open3d as o3d
import os
import argparse
import matplotlib.pyplot as plt

# Add alphasurf to path to load SurfaceObject
sys.path.append("alphasurf")
from alphasurf.protein.surfaces import SurfaceObject


def create_headless_outputs(mesh, wireframe_mesh, ply_path):
    """
    Create alternative visualization outputs when GUI is not available.
    """
    base_name = os.path.splitext(os.path.basename(ply_path))[0]

    print("📊 Generating mesh statistics and visualizations...")

    # Get mesh data
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Basic statistics
    print(f"📈 Mesh Statistics for {base_name}:")
    print(f"   • Vertices: {len(vertices)}")
    print(f"   • Triangles: {len(triangles)}")
    print(f"   • Surface area: {mesh.get_surface_area():.2f}")

    # Volume (only if mesh is watertight)
    try:
        volume = mesh.get_volume()
        print(f"   • Volume: {volume:.2f}")
    except RuntimeError:
        print("   • Volume: N/A (mesh not watertight)")

    # Bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = bbox.min_bound
    bbox_max = bbox.max_bound
    print(
        f"   • Bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to "
        f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]"
    )

    # Create mesh visualization using matplotlib
    fig = plt.figure(figsize=(15, 10))

    # 3D mesh visualization
    ax1 = fig.add_subplot(221, projection="3d")

    # Plot the triangular mesh
    if len(vertices) < 5000:  # Only plot triangles for reasonable mesh sizes
        # Create triangular faces
        faces = np.asarray(mesh.triangles)
        # Use vertex colors if available, otherwise use Z-coordinate for coloring
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            face_colors = colors[faces[:, 0]]  # Use first vertex color for each face
        else:
            # Color by Z-coordinate
            z_colors = vertices[:, 2]
            norm = plt.Normalize(z_colors.min(), z_colors.max())
            face_colors = plt.cm.viridis(norm(z_colors[faces[:, 0]]))

        # Plot each triangle
        for i, face in enumerate(faces):
            verts = vertices[face]
            x, y, z = verts.T
            color = face_colors[i] if mesh.has_vertex_colors() else face_colors[i]
            ax1.plot_trisurf(
                x, y, z, color=color, alpha=0.8, linewidth=0.1, edgecolor="black"
            )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("3D Mesh View")
        ax1.view_init(elev=20, azim=45)
    else:
        # For large meshes, fall back to scatter plot
        ax1.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            c=vertices[:, 2],
            cmap="viridis",
            s=0.5,
            alpha=0.6,
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("3D Point Cloud (large mesh)")

    # XY projection with mesh
    ax2 = fig.add_subplot(222)
    if len(triangles) < 5000:
        # Plot triangular mesh in XY plane
        for face in triangles:
            verts = vertices[face]
            x, y = verts[:, 0], verts[:, 1]
            # Color by Z coordinate
            z_avg = np.mean(verts[:, 2])
            color = plt.cm.viridis(
                (z_avg - vertices[:, 2].min())
                / (vertices[:, 2].max() - vertices[:, 2].min())
            )
            ax2.fill(x, y, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    else:
        # Scatter plot fallback
        scatter2 = ax2.scatter(
            vertices[:, 0],
            vertices[:, 1],
            c=vertices[:, 2],
            cmap="viridis",
            s=1,
            alpha=0.6,
        )
        plt.colorbar(scatter2, ax=ax2, shrink=0.8)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("XY Projection (colored by Z)")
    ax2.set_aspect("equal")

    # XZ projection with mesh
    ax3 = fig.add_subplot(223)
    if len(triangles) < 5000:
        for face in triangles:
            verts = vertices[face]
            x, z = verts[:, 0], verts[:, 2]
            y_avg = np.mean(verts[:, 1])
            color = plt.cm.plasma(
                (y_avg - vertices[:, 1].min())
                / (vertices[:, 1].max() - vertices[:, 1].min())
            )
            ax3.fill(x, z, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    else:
        scatter3 = ax3.scatter(
            vertices[:, 0],
            vertices[:, 2],
            c=vertices[:, 1],
            cmap="plasma",
            s=1,
            alpha=0.6,
        )
        plt.colorbar(scatter3, ax=ax3, shrink=0.8)

    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("XZ Projection (colored by Y)")
    ax3.set_aspect("equal")

    # YZ projection with mesh
    ax4 = fig.add_subplot(224)
    if len(triangles) < 5000:
        for face in triangles:
            verts = vertices[face]
            y, z = verts[:, 1], verts[:, 2]
            x_avg = np.mean(verts[:, 0])
            color = plt.cm.coolwarm(
                (x_avg - vertices[:, 0].min())
                / (vertices[:, 0].max() - vertices[:, 0].min())
            )
            ax4.fill(y, z, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    else:
        scatter4 = ax4.scatter(
            vertices[:, 1],
            vertices[:, 2],
            c=vertices[:, 0],
            cmap="coolwarm",
            s=1,
            alpha=0.6,
        )
        plt.colorbar(scatter4, ax=ax4, shrink=0.8)

    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_title("YZ Projection (colored by X)")
    ax4.set_aspect("equal")

    plt.tight_layout()
    plot_path = f"{base_name}_projections.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Saved 2D projections to {plot_path}")

    # Create HTML summary file
    try:
        volume_str = f"{mesh.get_volume():.2f}"
    except RuntimeError:
        volume_str = "N/A (mesh not watertight)"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Protein Surface Analysis: {base_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .stat-box {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .stat-title {{ font-weight: bold; color: #333; }}
        .stat-value {{ font-size: 1.2em; color: #0066cc; }}
        img {{ max-width: 100%; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <h1>Protein Surface Analysis: {base_name}</h1>

    <div class="stat-box">
        <div class="stat-title">Mesh Statistics</div>
        <div class="stat-value">
            • {len(vertices)} vertices<br>
            • {len(triangles)} triangles<br>
            • Surface area: {mesh.get_surface_area():.2f}<br>
            • Volume: {volume_str}
        </div>
    </div>

    <h2>2D Projections</h2>
    <img src="{base_name}_projections.png" alt="Mesh projections">

    <h2>Files Generated</h2>
    <ul>
        <li><strong>{ply_path}</strong> - Original PLY mesh file</li>
        <li><strong>{plot_path}</strong> - 2D projections plot</li>
        <li><strong>{base_name}_stats.html</strong> - This analysis file</li>
    </ul>

    <p><em>Generated in headless environment - run on GUI-enabled system for interactive 3D visualization</em></p>
</body>
</html>
"""

    html_path = f"{base_name}_stats.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"📄 Saved analysis summary to {html_path}")

    print("✅ Headless visualization complete!")
    print("📁 Generated files:")
    print(f"   • {plot_path} (2D projections)")
    print(f"   • {html_path} (analysis summary)")


def save_mesh_as_ply(file_path, output_path):
    """
    Loads a .pt file containing a mesh and saves it as a .ply file.
    """
    try:
        # The .pt file is a pickled custom object from this project, so it's safe to load.
        # We set `weights_only=False` and ensure the class definition is available.
        data = torch.load(
            file_path, map_location=torch.device("cpu"), weights_only=False
        )
        print(f"Successfully loaded {file_path}")
        print(f"Data type: {type(data)}")

        if isinstance(data, SurfaceObject):
            print("Loaded a SurfaceObject. Attempting to save mesh...")

            if hasattr(data, "verts") and hasattr(data, "faces"):
                vertices = data.verts.cpu().numpy()
                faces = data.faces.cpu().numpy()

                print(f"Vertices shape: {vertices.shape}")
                print(f"Faces shape: {faces.shape}")

                if faces.max() >= len(vertices):
                    print(
                        "Error: Face indices are out of bounds for the number of vertices."
                    )
                    return

                # Create an Open3D TriangleMesh object
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)

                # Optional: compute normals for better visualization
                mesh.compute_vertex_normals()

                # Save the mesh to a .ply file
                print(f"Saving mesh to {output_path}...")
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Successfully saved mesh to {output_path}")

            else:
                print(
                    "Could not find 'verts' and/or 'faces' attributes in the SurfaceObject."
                )
                print("Available attributes (from .keys):", data.keys)

    except Exception as e:
        import traceback

        print(f"Error during processing: {e}")
        traceback.print_exc()


def visualize_ply(ply_path):
    """
    Loads and visualizes a .ply mesh file with nice rendering settings.
    """
    try:
        print(f"Loading {ply_path} for visualization...")

        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(ply_path)

        if not mesh.has_vertices():
            print("Error: Mesh has no vertices")
            return

        print(
            f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles"
        )

        # Ensure normals are computed
        if not mesh.has_vertex_normals():
            print("Computing vertex normals...")
            mesh.compute_vertex_normals()

        # Create a beautiful color gradient based on vertex positions
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 0:
            # Normalize coordinates for color mapping
            # Use z-coordinate for vertical gradient (blue to red)
            z_coords = vertices[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()

            if z_max > z_min:
                # Create color gradient: blue (bottom) -> cyan -> green -> yellow -> red (top)
                normalized_z = (z_coords - z_min) / (z_max - z_min)

                colors = np.zeros((len(vertices), 3))
                for i, z_norm in enumerate(normalized_z):
                    if z_norm < 0.25:
                        # Blue to cyan
                        t = z_norm / 0.25
                        colors[i] = [0, t, 1]  # Blue to cyan
                    elif z_norm < 0.5:
                        # Cyan to green
                        t = (z_norm - 0.25) / 0.25
                        colors[i] = [0, 1, 1 - t]  # Cyan to green
                    elif z_norm < 0.75:
                        # Green to yellow
                        t = (z_norm - 0.5) / 0.25
                        colors[i] = [t, 1, 0]  # Green to yellow
                    else:
                        # Yellow to red
                        t = (z_norm - 0.75) / 0.25
                        colors[i] = [1, 1 - t, 0]  # Yellow to red

                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                print("Applied beautiful color gradient to mesh")
            else:
                # If all z-coords are the same, use a nice solid color
                mesh.paint_uniform_color([0.3, 0.7, 1.0])  # Bright blue
        else:
            mesh.paint_uniform_color([0.3, 0.7, 1.0])  # Bright blue fallback

        # Create a black wireframe version for edges
        wireframe_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe_colors = [
            [0, 0, 0] for _ in range(len(wireframe_mesh.lines))
        ]  # Black edges
        wireframe_mesh.colors = o3d.utility.Vector3dVector(wireframe_colors)

        # Try to create visualization window with nice settings
        vis = o3d.visualization.Visualizer()

        # Try headless rendering first (works in some environments)
        try:
            # Attempt to enable headless rendering
            vis.create_window(
                window_name=f"Protein Surface: {os.path.basename(ply_path)}",
                width=1200,
                height=800,
                visible=False,
            )  # visible=False for headless
        except:
            # If that fails, try regular window creation
            try:
                vis.create_window(
                    window_name=f"Protein Surface: {os.path.basename(ply_path)}",
                    width=1200,
                    height=800,
                )
            except:
                # If both fail, we're definitely in a headless environment
                vis = None

        # Check if window creation was successful
        if vis is None or vis.get_render_option() is None:
            print(
                "❌ Cannot create visualization window (running in headless environment)"
            )
            print("💡 Creating alternative outputs...")

            # Create alternative visualization outputs
            create_headless_outputs(mesh, wireframe_mesh, ply_path)
            return

        # Add the colored mesh
        vis.add_geometry(mesh)

        # Add black wireframe edges on top
        vis.add_geometry(wireframe_mesh)

        # Set up nice rendering options
        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.asarray(
                [0.05, 0.05, 0.05]
            )  # Very dark background
            render_option.light_on = True
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = False  # We handle wireframe manually
            render_option.point_show_normal = False
            render_option.line_width = 2.0  # Thicker wireframe lines

            # Additional material properties for better appearance
            render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
            render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

        # Set up view control for better initial view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)

        print("🎨 Beautiful mesh visualization ready!")
        print(
            "Features: Color gradient (blue→cyan→green→yellow→red) + black wireframe edges"
        )
        print("Controls:")
        print("  - Left mouse + drag: Rotate")
        print("  - Right mouse + drag: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - Ctrl + mouse wheel: Adjust field of view")
        print("  - Press 'H' for help, 'Q' or 'Esc' to quit")

        # Run the visualization
        vis.run()
        vis.destroy_window()

    except Exception as e:
        import traceback

        print(f"Error during visualization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .pt mesh files to .ply and optionally visualize them"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to .pt file (default: hardcoded path)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving .ply file, only visualize if input is .ply",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization after processing",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save .ply file (default: current directory)",
    )

    args = parser.parse_args()

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        # Default path
        input_file = "/Users/gertnervictor/Documents/alphasurf_2/alphasurf/data/masif_ligand/surfaces_full_alpha_complex_1.0_False/1HBI_AB.pt"

    # Check if input is already a .ply file
    if input_file.lower().endswith(".ply"):
        if not args.no_visualize:
            visualize_ply(input_file)
        else:
            print(f"Input is already a .ply file: {input_file}")
            print("Use --no-visualize to skip visualization")
    else:
        # Process .pt file
        # Create an output path for the .ply file
        base_name = os.path.basename(input_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(args.output_dir, f"{file_name_without_ext}.ply")

        if not args.no_save:
            save_mesh_as_ply(input_file, output_file)

        # Visualize if requested and file was saved or exists
        if not args.no_visualize and os.path.exists(output_file):
            visualize_ply(output_file)
