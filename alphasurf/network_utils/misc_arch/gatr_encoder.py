"""
GATr wrapper for processing alpha complex / mesh data.

The alpha complex is represented as vertices + faces. We embed:
- Vertices as points (grade-3 multivectors)
- Edges as Plücker lines (grade-2 multivectors) [optional]
- Face normals as oriented planes (grade-1 multivectors) [optional]
"""

import torch
import torch.nn as nn
from gatr import GATr, MLPConfig, SelfAttentionConfig
from gatr.interface import (
    embed_oriented_plane,
    embed_pluecker_ray,
    embed_point,
    extract_scalar,
)


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute unit normals for each face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)


def compute_face_centers(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute centroid of each face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return (v0 + v1 + v2) / 3


def extract_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    """Extract unique edges from face indices.

    Args:
        faces: (M, 3) face indices

    Returns:
        edges: (E, 2) unique edge indices, sorted per edge
    """
    # Each face has 3 edges: (0,1), (1,2), (2,0)
    edges_01 = faces[:, [0, 1]]
    edges_12 = faces[:, [1, 2]]
    edges_20 = faces[:, [2, 0]]
    all_edges = torch.cat([edges_01, edges_12, edges_20], dim=0)

    # Sort each edge so (a,b) and (b,a) are the same
    all_edges = torch.sort(all_edges, dim=1).values

    # Get unique edges
    unique_edges = torch.unique(all_edges, dim=0)
    return unique_edges


def embed_edges_as_pluecker(
    vertices: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Embed edges as Plücker lines.

    Plücker coords: (direction, origin × direction)

    Args:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices

    Returns:
        edge_mv: (E, 16) multivector embeddings
    """
    p0 = vertices[edges[:, 0]]  # (E, 3)
    p1 = vertices[edges[:, 1]]  # (E, 3)

    direction = p1 - p0
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    # Use midpoint as the origin point on the line
    origin = (p0 + p1) / 2
    moment = torch.cross(origin, direction, dim=-1)

    pluecker = torch.cat([direction, moment], dim=-1)  # (E, 6)
    return embed_pluecker_ray(pluecker)


class AlphaComplexGATr(nn.Module):
    """GATr model for alpha complex / mesh processing.

    Input: vertices (points) and faces (triangles)
    Output: per-vertex predictions (configurable output dim)

    Geometric embeddings:
    - Vertices → Points (always)
    - Edges → Plücker lines (optional)
    - Faces → Oriented planes (optional)
    """

    def __init__(
        self,
        out_dim: int = 1,
        hidden_mv_channels: int = 16,
        hidden_s_channels: int = 64,
        num_blocks: int = 8,
        use_edges: bool = False,
        use_face_normals: bool = True,
    ):
        super().__init__()
        self.use_edges = use_edges
        self.use_face_normals = use_face_normals
        self.out_dim = out_dim

        # Input channels: 1 for points, +1 for edges, +1 for faces
        in_mv_channels = 1
        if use_edges:
            in_mv_channels += 1
        if use_face_normals:
            in_mv_channels += 1

        self.gatr = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=out_dim if out_dim > 1 else None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

        if out_dim == 1:
            self.output_head = None
        else:
            self.output_head = nn.Linear(1, out_dim)

    def _build_multivector_input(
        self,
        vertex_mv: torch.Tensor,
        edge_mv: torch.Tensor | None,
        face_mv: torch.Tensor | None,
        in_mv_channels: int,
    ) -> torch.Tensor:
        """Build multi-channel input tensor for a single sample.

        Each token gets `in_mv_channels` channels, but only one is non-zero
        depending on whether it's a vertex/edge/face token.
        """
        num_vertices = vertex_mv.shape[0]
        num_edges = edge_mv.shape[0] if edge_mv is not None else 0
        num_faces = face_mv.shape[0] if face_mv is not None else 0
        total_tokens = num_vertices + num_edges + num_faces

        device = vertex_mv.device
        dtype = vertex_mv.dtype

        # (total_tokens, in_mv_channels, 16)
        inputs = torch.zeros(
            total_tokens, in_mv_channels, 16, device=device, dtype=dtype
        )

        # Vertices go in channel 0
        inputs[:num_vertices, 0, :] = vertex_mv

        channel_idx = 1
        offset = num_vertices

        # Edges go in next channel
        if edge_mv is not None:
            inputs[offset : offset + num_edges, channel_idx, :] = edge_mv
            offset += num_edges
            channel_idx += 1

        # Faces go in next channel
        if face_mv is not None:
            inputs[offset : offset + num_faces, channel_idx, :] = face_mv

        return inputs

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process alpha complex through GATr.

        Args:
            vertices: (batch, N, 3) or (N, 3) vertex positions
            faces: (M, 3) face indices (shared across batch)
            edges: (E, 2) edge indices, or None to extract from faces

        Returns:
            output: (batch, N, out_dim) per-vertex predictions
        """
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)

        batch_size, num_vertices, _ = vertices.shape

        # Extract edges from faces if needed and not provided
        if self.use_edges and edges is None:
            edges = extract_edges_from_faces(faces)

        in_mv_channels = 1 + int(self.use_edges) + int(self.use_face_normals)

        all_inputs = []
        for b in range(batch_size):
            verts_b = vertices[b]

            # Embed vertices
            vertex_mv = embed_point(verts_b)  # (N, 16)

            # Embed edges
            edge_mv = None
            if self.use_edges:
                if edges is None:
                    raise ValueError("Edges must be provided if use_edges is True")
                edge_mv = embed_edges_as_pluecker(verts_b, edges)  # (E, 16)

            # Embed faces
            face_mv = None
            if self.use_face_normals:
                normals = compute_face_normals(verts_b, faces)
                centers = compute_face_centers(verts_b, faces)
                face_mv = embed_oriented_plane(normals, centers)  # (M, 16)

            # Build multi-channel input
            inputs_b = self._build_multivector_input(
                vertex_mv, edge_mv, face_mv, in_mv_channels
            )
            all_inputs.append(inputs_b)

        # Stack batches: (batch, total_tokens, in_mv_channels, 16)
        inputs = torch.stack(all_inputs, dim=0)

        # Run through GATr
        outputs_mv, outputs_s = self.gatr(inputs, scalars=None)

        # Extract vertex outputs only (first N tokens)
        vertex_outputs_mv = outputs_mv[:, :num_vertices]  # (batch, N, 1, 16)

        # Extract scalar predictions
        if outputs_s is not None:
            preds = outputs_s[:, :num_vertices, 0, :]  # (batch, N, out_dim)
        else:
            preds = extract_scalar(vertex_outputs_mv)  # (batch, N, 1, 1)
            preds = preds.squeeze(-1).squeeze(-1)  # (batch, N)
            if self.output_head is not None:
                preds = self.output_head(preds.unsqueeze(-1))
            else:
                preds = preds.unsqueeze(-1)  # (batch, N, 1)

        return preds.squeeze(0) if batch_size == 1 else preds


class AlphaComplexClassifier(nn.Module):
    """GATr classifier for whole-surface classification with global pooling.

    Uses AlphaComplexGATr as backbone, then applies global mean pooling
    over all tokens to get a single representation for classification.
    """

    num_classes: int
    pooling: str

    def __init__(
        self,
        num_classes: int = 7,
        hidden_mv_channels: int = 16,
        hidden_s_channels: int = 64,
        num_blocks: int = 8,
        use_edges: bool = False,
        use_face_normals: bool = True,
        pooling: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling = pooling

        # Input channels: 1 for points, +1 for edges, +1 for faces
        in_mv_channels = 1
        if use_edges:
            in_mv_channels += 1
        if use_face_normals:
            in_mv_channels += 1

        self.use_edges = use_edges
        self.use_face_normals = use_face_normals

        self.gatr = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

        # Classification head: scalar extraction + MLP
        # extract_scalar gives 1 scalar per multivector
        self.classifier = nn.Sequential(
            nn.Linear(16, hidden_s_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_s_channels, num_classes),
        )

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify a single mesh.

        Args:
            vertices: (N, 3) vertex positions
            faces: (M, 3) face indices
            edges: (E, 2) edge indices, or None to extract from faces

        Returns:
            logits: (num_classes,) classification logits
        """
        # Embed vertices
        vertex_mv = embed_point(vertices)  # (N, 16)

        # Embed edges
        edge_mv = None
        if self.use_edges:
            if edges is None:
                edges = extract_edges_from_faces(faces)
            edge_mv = embed_edges_as_pluecker(vertices, edges)

        # Embed faces
        face_mv = None
        if self.use_face_normals:
            normals = compute_face_normals(vertices, faces)
            centers = compute_face_centers(vertices, faces)
            face_mv = embed_oriented_plane(normals, centers)

        # Build input tensor
        in_mv_channels = 1 + int(self.use_edges) + int(self.use_face_normals)

        num_vertices = vertex_mv.shape[0]
        num_edges = edge_mv.shape[0] if edge_mv is not None else 0
        num_faces = face_mv.shape[0] if face_mv is not None else 0
        total_tokens = num_vertices + num_edges + num_faces

        inputs = torch.zeros(
            total_tokens,
            in_mv_channels,
            16,
            device=vertices.device,
            dtype=vertices.dtype,
        )
        inputs[:num_vertices, 0] = vertex_mv

        channel_idx = 1
        offset = num_vertices
        if edge_mv is not None:
            inputs[offset : offset + num_edges, channel_idx] = edge_mv
            offset += num_edges
            channel_idx += 1
        if face_mv is not None:
            inputs[offset : offset + num_faces, channel_idx] = face_mv

        # Add batch dimension: (1, T, C, 16)
        inputs = inputs.unsqueeze(0)

        # Run through GATr
        outputs_mv, _ = self.gatr(inputs, scalars=None)  # (1, T, 1, 16)
        outputs_mv = outputs_mv.squeeze(0).squeeze(1)  # (T, 16)

        # Global pooling over all tokens
        if self.pooling == "mean":
            pooled = outputs_mv.mean(dim=0)  # (16,)
        elif self.pooling == "max":
            pooled = outputs_mv.max(dim=0).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)  # (num_classes,)
        return logits
