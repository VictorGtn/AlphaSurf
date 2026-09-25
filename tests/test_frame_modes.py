import h5py
import numpy as np
import pytest
import torch

from alphasurf.tasks.misato_binding_site.dataset import MisatoBindingSiteDataset


def make_dataset(tmp_path, mode, *, index=0, fraction=0.5):
    md_path = tmp_path / "frames.h5"
    trajectory = np.zeros((6, 3, 3), dtype=np.float32)
    for frame in range(6):
        trajectory[frame, :, 0] = frame
    with h5py.File(md_path, "w") as handle:
        handle.create_group("test").create_dataset(
            "trajectory_coordinates", data=trajectory
        )
    return MisatoBindingSiteDataset(
        pdb_ids=["test"],
        data_dir=str(tmp_path),
        md_path=str(md_path),
        surface_cfg=None,
        graph_cfg=None,
        frame_mode=mode,
        frame_index=index,
        frame_fraction=fraction,
    )


def item():
    return {
        "protein_source_index": torch.tensor([0, 1, 2]),
        "ca_atom_index": torch.tensor([0, 2]),
    }


@pytest.mark.parametrize(
    ("mode", "index", "fraction", "expected"),
    [
        ("first", 2, 0.5, 2),
        ("fixed", 4, 0.5, 4),
        ("middle", 0, 0.5, 3),
        ("fraction", 0, 0.0, 0),
        ("fraction", 0, 0.5, 3),
        ("fraction", 0, 1.0, 5),
    ],
)
def test_deterministic_frame_modes(tmp_path, mode, index, fraction, expected):
    dataset = make_dataset(tmp_path, mode, index=index, fraction=fraction)
    atom_pos, ca_pos, frame_idx = dataset._load_frame("test", item())
    assert frame_idx == expected
    assert torch.all(atom_pos[:, 0] == expected)
    assert torch.all(ca_pos[:, 0] == expected)


def test_fraction_must_be_in_unit_interval(tmp_path):
    dataset = make_dataset(tmp_path, "fraction", fraction=1.1)
    with pytest.raises(ValueError, match="frame_fraction"):
        dataset._load_frame("test", item())
