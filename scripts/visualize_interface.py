"""
Visualize protein-protein interfaces for graphs and surfaces.

Usage:
    python visualize_interface.py <pdb_file_1> <pdb_file_2> --mode [graph|surface|both] --threshold 5.0

Outputs:
    - Interactive 3D visualization with red for interface, blue for non-interface
    - HTML file for viewing in browser
"""

import argparse
import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go

# Add alphasurf to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from alphasurf.protein.protein_loader import ProteinLoader
from omegaconf import OmegaConf


def compute_graph_interface(protein_1, protein_2, threshold=5.0, use_full_atom=True):
    """
    Compute graph interface using atom-atom or CA-CA distances.

    Returns:
        interface_residues_1: set of residue indices in protein 1
        interface_residues_2: set of residue indices in protein 2
    """
    if (
        use_full_atom
        and "atom_pos" in protein_1.metadata
        and "atom_pos" in protein_2.metadata
    ):
        # Full atom-atom distances
        atom_pos_1 = protein_1.metadata["atom_pos"]
        atom_pos_2 = protein_2.metadata["atom_pos"]
        res_map_1 = protein_1.metadata["atom_res_map"]
        res_map_2 = protein_2.metadata["atom_res_map"]

        # Compute pairwise distances
        dists = torch.cdist(atom_pos_1, atom_pos_2)
        close_atoms = dists < threshold
        atom_i, atom_j = torch.where(close_atoms)

        # Map to residues
        res_i = res_map_1[atom_i]
        res_j = res_map_2[atom_j]

        interface_residues_1 = set(res_i.numpy())
        interface_residues_2 = set(res_j.numpy())
    else:
        # CA-based distances
        pos_1 = protein_1.graph.node_pos
        pos_2 = protein_2.graph.node_pos

        dists = torch.cdist(pos_1, pos_2)
        close_residues = dists < threshold
        res_i, res_j = torch.where(close_residues)

        interface_residues_1 = set(res_i.numpy())
        interface_residues_2 = set(res_j.numpy())

    return interface_residues_1, interface_residues_2


def compute_surface_interface(protein_1, protein_2, threshold=5.0):
    """
    Compute surface interface using vertex-vertex distances.

    Returns:
        interface_vertices_1: set of vertex indices in protein 1
        interface_vertices_2: set of vertex indices in protein 2
    """
    verts_1 = protein_1.surface.verts
    verts_2 = protein_2.surface.verts

    dists = torch.cdist(verts_1, verts_2)
    close_verts = dists < threshold
    vert_i, vert_j = torch.where(close_verts)

    interface_vertices_1 = set(vert_i.numpy())
    interface_vertices_2 = set(vert_j.numpy())

    return interface_vertices_1, interface_vertices_2


def visualize_graph(
    protein_1, protein_2, interface_res_1, interface_res_2, title="Graph Interface"
):
    """Create 3D scatter plot of residues colored by interface."""
    pos_1 = protein_1.graph.node_pos.numpy()
    pos_2 = protein_2.graph.node_pos.numpy()

    # Create color arrays
    colors_1 = np.array(["blue"] * len(pos_1), dtype=object)
    colors_2 = np.array(["cyan"] * len(pos_2), dtype=object)

    for res_idx in interface_res_1:
        if res_idx < len(colors_1):
            colors_1[res_idx] = "red"
    for res_idx in interface_res_2:
        if res_idx < len(colors_2):
            colors_2[res_idx] = "orange"

    # Create plot
    fig = go.Figure()

    # Protein 1 (blue/red)
    fig.add_trace(
        go.Scatter3d(
            x=pos_1[:, 0],
            y=pos_1[:, 1],
            z=pos_1[:, 2],
            mode="markers",
            marker=dict(size=5, color=colors_1, opacity=0.8),
            name="Protein 1",
            text=[f"Res {i}" for i in range(len(pos_1))],
            hoverinfo="text",
        )
    )

    # Protein 2 (cyan/orange)
    fig.add_trace(
        go.Scatter3d(
            x=pos_2[:, 0],
            y=pos_2[:, 1],
            z=pos_2[:, 2],
            mode="markers",
            marker=dict(size=5, color=colors_2, opacity=0.8),
            name="Protein 2",
            text=[f"Res {i}" for i in range(len(pos_2))],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data",
        ),
        showlegend=True,
    )

    return fig


def visualize_surface(
    protein_1,
    protein_2,
    interface_verts_1,
    interface_verts_2,
    title="Surface Interface",
):
    """Create 3D mesh plot of surfaces colored by interface."""
    verts_1 = protein_1.surface.verts.numpy()
    faces_1 = protein_1.surface.faces.numpy()
    verts_2 = protein_2.surface.verts.numpy()
    faces_2 = protein_2.surface.faces.numpy()

    # Create vertex colors
    colors_1 = np.zeros(len(verts_1))  # 0 = blue
    colors_2 = np.ones(len(verts_2))  # 1 = cyan

    for vert_idx in interface_verts_1:
        if vert_idx < len(colors_1):
            colors_1[vert_idx] = 2  # 2 = red
    for vert_idx in interface_verts_2:
        if vert_idx < len(colors_2):
            colors_2[vert_idx] = 3  # 3 = orange

    # Create colorscale
    colorscale = [
        [0.0, "blue"],  # Protein 1 non-interface
        [0.33, "cyan"],  # Protein 2 non-interface
        [0.66, "red"],  # Protein 1 interface
        [1.0, "orange"],  # Protein 2 interface
    ]

    fig = go.Figure()

    # Protein 1 mesh
    fig.add_trace(
        go.Mesh3d(
            x=verts_1[:, 0],
            y=verts_1[:, 1],
            z=verts_1[:, 2],
            i=faces_1[:, 0],
            j=faces_1[:, 1],
            k=faces_1[:, 2],
            intensity=colors_1,
            colorscale=colorscale,
            cmin=0,
            cmax=3,
            opacity=1.0,
            name="Protein 1",
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Protein 2 mesh
    fig.add_trace(
        go.Mesh3d(
            x=verts_2[:, 0],
            y=verts_2[:, 1],
            z=verts_2[:, 2],
            i=faces_2[:, 0],
            j=faces_2[:, 1],
            k=faces_2[:, 2],
            intensity=colors_2,
            colorscale=colorscale,
            cmin=0,
            cmax=3,
            opacity=1.0,
            name="Protein 2",
            showscale=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data",
        ),
        showlegend=True,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize protein-protein interface")
    parser.add_argument("pdb_file_1", help="Path to first PDB file")
    parser.add_argument("pdb_file_2", help="Path to second PDB file")
    parser.add_argument(
        "--mode",
        choices=["graph", "surface", "both"],
        default="both",
        help="Visualization mode",
    )
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Interface distance threshold (Å)"
    )
    parser.add_argument(
        "--output", default="interface_viz.html", help="Output HTML file"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="Alpha value for alpha complex surface"
    )
    parser.add_argument(
        "--full-atom",
        action="store_true",
        default=True,
        help="Use full atom distances for graph (default: True)",
    )

    args = parser.parse_args()

    # Create protein loader config
    surface_cfg = OmegaConf.create(
        {
            "use_surfaces": True,
            "surface_method": "alpha_complex",
            "alpha_value": args.alpha,
            "min_vert_number": 16,
            "use_whole_surfaces": True,
            "face_reduction_rate": 1.0,
        }
    )

    graph_cfg = OmegaConf.create(
        {
            "use_graphs": True,
        }
    )

    # Load proteins
    print(f"Loading {args.pdb_file_1}...")
    import os

    pdb_dir = os.path.dirname(args.pdb_file_1)
    loader = ProteinLoader(
        pdb_dir=pdb_dir,
        surface_config=surface_cfg,
        graph_config=graph_cfg,
        mode="on_fly",
    )

    protein_1 = loader.load("protein_1", pdb_path=args.pdb_file_1)
    if protein_1 is None:
        print(f"Failed to load {args.pdb_file_1}")
        return

    print(f"Loading {args.pdb_file_2}...")
    protein_2 = loader.load("protein_2", pdb_path=args.pdb_file_2)
    if protein_2 is None:
        print(f"Failed to load {args.pdb_file_2}")
        return

    print(f"\nProtein 1: {len(protein_1.graph.node_pos)} residues")
    if protein_1.surface is not None:
        print(f"           {len(protein_1.surface.verts)} surface vertices")
    print(f"Protein 2: {len(protein_2.graph.node_pos)} residues")
    if protein_2.surface is not None:
        print(f"           {len(protein_2.surface.verts)} surface vertices")

    # Compute interfaces
    figures = []

    if args.mode in ["graph", "both"]:
        print(f"\nComputing graph interface (threshold={args.threshold}Å)...")
        interface_res_1, interface_res_2 = compute_graph_interface(
            protein_1, protein_2, args.threshold, use_full_atom=args.full_atom
        )
        print(f"  Protein 1: {len(interface_res_1)} interface residues")
        print(f"  Protein 2: {len(interface_res_2)} interface residues")

        fig_graph = visualize_graph(
            protein_1,
            protein_2,
            interface_res_1,
            interface_res_2,
            title=f"Graph Interface ({args.threshold}Å)",
        )
        figures.append(("Graph", fig_graph))

    if args.mode in ["surface", "both"]:
        if protein_1.surface is None or protein_2.surface is None:
            print("Warning: Surface not available for one or both proteins")
        else:
            print(f"\nComputing surface interface (threshold={args.threshold}Å)...")
            interface_verts_1, interface_verts_2 = compute_surface_interface(
                protein_1, protein_2, args.threshold
            )
            print(f"  Protein 1: {len(interface_verts_1)} interface vertices")
            print(f"  Protein 2: {len(interface_verts_2)} interface vertices")

            fig_surface = visualize_surface(
                protein_1,
                protein_2,
                interface_verts_1,
                interface_verts_2,
                title=f"Surface Interface ({args.threshold}Å, α={args.alpha})",
            )
            figures.append(("Surface", fig_surface))

    # Create combined visualization
    if len(figures) == 2:
        # Both graph and surface: create subplots
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f[0] for f in figures],
            specs=[[{"type": "scatter3d"}, {"type": "mesh3d"}]],
        )

        # Add graph traces to first subplot
        for trace in figures[0][1].data:
            fig.add_trace(trace, row=1, col=1)

        # Add surface traces to second subplot
        for trace in figures[1][1].data:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title_text=f"Protein-Protein Interface Analysis (threshold={args.threshold}Å)",
            showlegend=True,
            height=600,
        )
    else:
        # Single visualization
        fig = figures[0][1]

    # Save to HTML
    print(f"\nSaving visualization to {args.output}...")
    fig.write_html(args.output)
    print(f"✓ Done! Open {args.output} in your browser to view.")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}Å")
    print(f"Full-atom distances: {args.full_atom}")
    if args.mode in ["surface", "both"]:
        print(f"Alpha value: {args.alpha}")
    print("\nColor legend:")
    print("  Graph:   Blue=Protein 1, Cyan=Protein 2")
    print("           Red=Protein 1 interface, Orange=Protein 2 interface")
    print("  Surface: Same colors as graph")
    print("=" * 60)


if __name__ == "__main__":
    main()
