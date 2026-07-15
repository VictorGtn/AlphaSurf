from unittest import TestCase

import numpy as np

from alphasurf.protein.protein_loader import ProteinLoader


class BackboneMaskTest(TestCase):
    @staticmethod
    def _arrays():
        atom_names = np.asarray(
            ["N", "CA", "C", "O", "CB", "CG"] * 2, dtype="<U2"
        )
        atom_residue = np.repeat(np.arange(2, dtype=np.int32), 6)
        n_atoms = len(atom_names)
        return (
            np.asarray([1, 2], dtype=np.int32),
            np.zeros(n_atoms, dtype=np.int32),
            atom_residue,
            atom_names,
            np.zeros(n_atoms, dtype=np.int32),
            np.zeros((n_atoms, 3), dtype=np.float32),
            None,
            np.ones(n_atoms, dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            np.asarray(["A:1", "A:2"]),
            np.arange(n_atoms),
        )

    def test_s3f_mask_removes_cb_only_at_selected_residue(self):
        masked = ProteinLoader._strip_sidechains_to_ala(
            self._arrays(), [0], keep_cb=False
        )
        selected_names = masked[3][masked[2] == 0].tolist()
        untouched_names = masked[3][masked[2] == 1].tolist()

        self.assertEqual(selected_names, ["N", "CA", "C", "O"])
        self.assertEqual(untouched_names, ["N", "CA", "C", "O", "CB", "CG"])

    def test_existing_alanine_strip_api_still_keeps_cb_by_default(self):
        masked = ProteinLoader._strip_sidechains_to_ala(self._arrays(), [0])
        selected_names = masked[3][masked[2] == 0].tolist()
        self.assertEqual(selected_names, ["N", "CA", "C", "O", "CB"])
