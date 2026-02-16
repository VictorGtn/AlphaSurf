"""
Unified Protein representation bundling surface and graph.

This module provides a dataclass that encapsulates both surface and graph
representations of a protein, making it reusable across downstream tasks.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from alphasurf.protein.surfaces import SurfaceObject
from torch_geometric.data import Data


@dataclass
class Protein:
    """
    Unified protein representation bundling surface and graph.

    This is the core abstraction that decouples protein loading from
    task-specific logic. A Protein can be loaded from disk or generated
    on-the-fly, and then passed to task-specific datasets.

    Attributes:
        surface: SurfaceObject with vertices, faces, operators, and features
        graph: ResidueGraph (PyG Data) with node positions, edges, and features
        name: Identifier for this protein (e.g., "1ABC_A")
        pdb_path: Optional path to the source PDB file
        metadata: Dictionary for task-specific data (labels, ligand coords, etc.)
    """

    surface: Optional[SurfaceObject] = None
    graph: Optional[Data] = None
    name: str = ""
    pdb_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_data(self) -> Data:
        """
        Convert to PyG Data for batching.

        Returns:
            Data object with surface and graph as attributes
        """
        return Data(
            surface=self.surface,
            graph=self.graph,
            name=self.name,
            **self.metadata,
        )

    def has_surface(self) -> bool:
        """Check if this protein has a valid surface."""
        return self.surface is not None and hasattr(self.surface, "verts")

    def has_graph(self) -> bool:
        """Check if this protein has a valid graph."""
        return self.graph is not None and hasattr(self.graph, "node_pos")

    def validate(self) -> bool:
        """
        Check for NaN values in surface and graph features.

        Returns:
            True if valid, False if NaN detected
        """
        if self.has_surface():
            if hasattr(self.surface, "x") and self.surface.x is not None:
                if not torch.isfinite(self.surface.x).all():
                    return False
            if hasattr(self.surface, "verts") and self.surface.verts is not None:
                if not torch.isfinite(self.surface.verts).all():
                    return False

        if self.has_graph():
            if hasattr(self.graph, "x") and self.graph.x is not None:
                if not torch.isfinite(self.graph.x).all():
                    return False
            if hasattr(self.graph, "node_pos") and self.graph.node_pos is not None:
                if not torch.isfinite(self.graph.node_pos).all():
                    return False

        return True
