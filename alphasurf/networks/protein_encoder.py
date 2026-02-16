import hydra
from torch import nn as nn

from alphasurf.network_utils.communication.surface_graph_comm import (
    SequentialSurfaceGraphCommunication,
)
from alphasurf.network_utils.misc_arch.timing import time_operation


class ProteinEncoder(nn.Module):
    """
    Just piping protein encoder blocks
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        block_list = []
        if cfg is not None:
            for x in cfg.blocks:
                block = hydra.utils.instantiate(x)
                block_list.append(block)
        self.blocks = nn.ModuleList(block_list)

    def forward(self, surface=None, graph=None):
        for block in self.blocks:
            surface, graph = block(surface, graph)
        return surface, graph

    @classmethod
    def from_blocks_list(cls, block_list):
        encoder = cls()
        encoder.blocks = nn.ModuleList(block_list)
        return encoder


class ProteinEncoderBlock(nn.Module):
    def __init__(self, surface_encoder=None, graph_encoder=None, message_passing=None):
        super().__init__()
        self.surface_encoder = surface_encoder
        self.graph_encoder = graph_encoder
        self.message_passing = message_passing

    def forward(self, surface=None, graph=None):
        # Extract point counts from data structures
        def get_point_count(data):
            if data is None:
                return None
            # Try different ways to get point count
            if hasattr(data, "pos") and data.pos is not None:
                return data.pos.shape[0]
            elif hasattr(data, "x") and data.x is not None:
                return data.x.shape[0]
            elif hasattr(data, "num_nodes"):
                return data.num_nodes
            elif hasattr(data, "__len__"):
                return len(data)
            return None

        surface_points = get_point_count(surface)
        graph_points = get_point_count(graph)

        if surface is not None and self.surface_encoder != "None":
            encoder_name = (
                self.surface_encoder.__class__.__name__
                if hasattr(self.surface_encoder, "__class__")
                else "unknown"
            )
            metadata = {"has_surface": True, "has_graph": graph is not None}
            if surface_points is not None:
                metadata["points"] = surface_points

            with time_operation(f"surface_encoder_{encoder_name}", metadata):
                surface = self.surface_encoder(surface)

        if graph is not None and self.graph_encoder != "None":
            encoder_name = (
                self.graph_encoder.__class__.__name__
                if hasattr(self.graph_encoder, "__class__")
                else "unknown"
            )
            metadata = {"has_surface": surface is not None, "has_graph": True}
            if graph_points is not None:
                metadata["points"] = graph_points

            with time_operation(f"graph_encoder_{encoder_name}", metadata):
                graph = self.graph_encoder(graph)

        if self.message_passing != "None":
            mp_name = (
                self.message_passing.__class__.__name__
                if hasattr(self.message_passing, "__class__")
                else "unknown"
            )
            metadata = {
                "has_surface": surface is not None,
                "has_graph": graph is not None,
            }
            # For message passing, use the max of surface/graph points if available
            max_points = max(filter(None, [surface_points, graph_points]), default=None)
            if max_points is not None:
                metadata["points"] = max_points

            with time_operation(f"message_passing_{mp_name}", metadata):
                surface, graph = self.message_passing(surface, graph)

        return surface, graph


class SequentialProteinEncoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.surface_encoder = hydra.utils.instantiate(
            hparams.surface_encoder.instanciate, **hparams.surface_encoder.kwargs
        )
        self.graph_encoder = hydra.utils.instantiate(
            hparams.graph_encoder.instanciate, **hparams.graph_encoder.kwargs
        )
        self.message_passing = hydra.utils.instantiate(
            hparams.communication_block.instanciate,
            **hparams.communication_block.kwargs,
        )

        # check if message_passing is an instance of SequentialSurfaceGraphCommunication
        if not isinstance(self.message_passing, SequentialSurfaceGraphCommunication):
            raise ValueError(
                "message_passing must be an instance of SequentialSurfaceGraphCommunication"
            )

    def forward(self, surface=None, graph=None):
        # Extract point counts from data structures
        def get_point_count(data):
            if data is None:
                return None
            # Try different ways to get point count
            if hasattr(data, "pos") and data.pos is not None:
                return data.pos.shape[0]
            elif hasattr(data, "x") and data.x is not None:
                return data.x.shape[0]
            elif hasattr(data, "num_nodes"):
                return data.num_nodes
            elif hasattr(data, "__len__"):
                return len(data)
            return None

        surface_points = get_point_count(surface)
        graph_points = get_point_count(graph)

        # We always start with surface starting with graph is possible,
        # by adding a layer before that has an identity surface encoder
        if surface is not None:
            encoder_name = (
                self.surface_encoder.__class__.__name__
                if hasattr(self.surface_encoder, "__class__")
                else "unknown"
            )
            metadata = {"has_surface": True, "has_graph": graph is not None}
            if surface_points is not None:
                metadata["points"] = surface_points

            with time_operation(f"sequential_surface_encoder_{encoder_name}", metadata):
                surface = self.surface_encoder(surface)

        mp_name = (
            self.message_passing.__class__.__name__
            if hasattr(self.message_passing, "__class__")
            else "unknown"
        )
        metadata = {"has_surface": surface is not None, "has_graph": graph is not None}
        max_points = max(filter(None, [surface_points, graph_points]), default=None)
        if max_points is not None:
            metadata["points"] = max_points

        with time_operation(f"sequential_mp_first_{mp_name}", metadata):
            surface, graph = self.message_passing(surface, graph, first_pass=True)

        if graph is not None:
            encoder_name = (
                self.graph_encoder.__class__.__name__
                if hasattr(self.graph_encoder, "__class__")
                else "unknown"
            )
            metadata = {"has_surface": surface is not None, "has_graph": True}
            if graph_points is not None:
                metadata["points"] = graph_points

            with time_operation(f"sequential_graph_encoder_{encoder_name}", metadata):
                graph = self.graph_encoder(graph)

        metadata = {"has_surface": surface is not None, "has_graph": graph is not None}
        max_points = max(filter(None, [surface_points, graph_points]), default=None)
        if max_points is not None:
            metadata["points"] = max_points

        with time_operation(f"sequential_mp_second_{mp_name}", metadata):
            surface, graph = self.message_passing(surface, graph, first_pass=False)

        return surface, graph
