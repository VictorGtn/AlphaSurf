from .communication.blocks import (
    ConcurrentCommunication,
    GATCommunicationV1,
    ParallelCommunicationV1,
    SequentialCommunication,
    SequentialCommunicationV1,
)
from .misc_arch.dgcnn import DGCNN, DGCNNLayer
from .misc_arch.graph_blocks import GCNx2Block
from .misc_arch.pointnet import PointNet
from .misc_arch.pronet import ProNet


def __getattr__(name):
    if name == "dMasifWrapper":
        from .misc_arch.dmasif_encoder import dMasifWrapper

        return dMasifWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConcurrentCommunication",
    "SequentialCommunication",
    "GCNx2Block",
    "DGCNN",
    "DGCNNLayer",
    "PointNet",
    "ParallelCommunicationV1",
    "SequentialCommunicationV1",
    "GATCommunicationV1",
    "ProNet",
    "dMasifWrapper",
]
