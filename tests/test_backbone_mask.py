from unittest import TestCase

import numpy as np

from alphasurf.protein.protein_loader import (
    ALANINE_CA_CB_LENGTH,
    ALANINE_C_CA_CB_ANGLE_DEG,
    ALANINE_N_CA_CB_ANGLE_DEG,
    ProteinLoader,
)


class BackboneMaskTest(TestCase):
    @staticmethod
    def _arrays():
        atom_names = np.asarray(["N", "CA", "C", "O", "CB", "CG"] * 2, dtype="<U2")
        atom_residue = np.repeat(np.arange(2, dtype=np.int32), 6)
        n_atoms = len(atom_names)
        residue_coords = np.asarray(
            [
                [0.0, 0.0, 0.0],  # N
                [1.46, 0.0, 0.0],  # CA
                [2.00, 1.42, 0.0],  # C
                [1.50, 2.45, 0.0],  # O
                [8.00, 8.00, 8.00],  # deliberately nonphysical native CB
                [9.00, 9.00, 9.00],  # native CG
            ],
            dtype=np.float32,
        )
        atom_pos = np.concatenate(
            [residue_coords, residue_coords + np.asarray([5.0, 0.0, 0.0])],
            axis=0,
        )
        return (
            np.asarray([1, 2], dtype=np.int32),
            np.zeros(n_atoms, dtype=np.int32),
            atom_residue,
            atom_names,
            np.ones(n_atoms, dtype=np.int32),
            atom_pos,
            None,
            np.ones(n_atoms, dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            np.asarray(["A:1", "A:2"], dtype=object),
            np.asarray(
                [
                    f"A:{residue + 1}_{name}"
                    for residue, name in zip(atom_residue, atom_names)
                ],
                dtype=object,
            ),
        )

    def test_s3f_mask_removes_cb_only_at_selected_residue(self):
        masked = ProteinLoader._strip_sidechains_to_ala(
            self._arrays(), [0], keep_cb=False
        )
        selected_names = masked[3][masked[2] == 0].tolist()
        untouched_names = masked[3][masked[2] == 1].tolist()

        self.assertEqual(selected_names, ["N", "CA", "C", "O"])
        self.assertEqual(untouched_names, ["N", "CA", "C", "O", "CB", "CG"])

    @staticmethod
    def _angle_degrees(a, center, b):
        u = a - center
        v = b - center
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def test_alanine_mask_reconstructs_ideal_chiral_cb(self):
        original = self._arrays()
        native_cb = original[5][(original[2] == 0) & (original[3] == "CB")][0]
        masked = ProteinLoader._strip_sidechains_to_ala(self._arrays(), [0])
        selected_names = masked[3][masked[2] == 0].tolist()
        self.assertEqual(selected_names, ["N", "CA", "C", "O", "CB"])

        coords = {
            name: masked[5][i] for i, name in enumerate(masked[3]) if masked[2][i] == 0
        }
        self.assertFalse(np.allclose(coords["CB"], native_cb))
        self.assertAlmostEqual(
            float(np.linalg.norm(coords["CB"] - coords["CA"])),
            ALANINE_CA_CB_LENGTH,
            places=5,
        )
        self.assertAlmostEqual(
            self._angle_degrees(coords["N"], coords["CA"], coords["CB"]),
            ALANINE_N_CA_CB_ANGLE_DEG,
            places=4,
        )
        self.assertAlmostEqual(
            self._angle_degrees(coords["C"], coords["CA"], coords["CB"]),
            ALANINE_C_CA_CB_ANGLE_DEG,
            places=4,
        )
        oriented_volume = np.dot(
            np.cross(coords["C"] - coords["N"], coords["CA"] - coords["N"]),
            coords["CB"] - coords["N"],
        )
        self.assertGreater(oriented_volume, 0.0)

    def test_alanine_mask_adds_the_same_cb_when_native_cb_is_absent(self):
        arrays = list(self._arrays())
        remove_native_sidechain = ~((arrays[2] == 0) & np.isin(arrays[3], ["CB", "CG"]))
        for index in (1, 2, 3, 4, 5, 7, 10):
            arrays[index] = arrays[index][remove_native_sidechain]

        gly_masked = ProteinLoader._strip_sidechains_to_ala(tuple(arrays), [0])
        native_masked = ProteinLoader._strip_sidechains_to_ala(self._arrays(), [0])
        gly_cb = gly_masked[5][(gly_masked[2] == 0) & (gly_masked[3] == "CB")]
        native_cb = native_masked[5][
            (native_masked[2] == 0) & (native_masked[3] == "CB")
        ]

        self.assertEqual(len(gly_cb), 1)
        np.testing.assert_allclose(gly_cb, native_cb, atol=1e-6)
