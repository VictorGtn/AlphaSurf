from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from alphasurf.tasks.proteingym.dataset import DMSAssay, Mutant
from alphasurf.tasks.proteingym.scoring import (
    get_optimal_window,
    score_assay_option_f,
)


class S3FWindowTest(TestCase):
    def test_short_sequence_uses_full_length(self):
        self.assertEqual(get_optimal_window(20, 250), (0, 250))

    def test_long_sequence_windows_match_released_s3f_logic(self):
        self.assertEqual(get_optimal_window(10, 2000), (0, 1022))
        self.assertEqual(get_optimal_window(1000, 2000), (489, 1511))
        self.assertEqual(get_optimal_window(1900, 2000), (978, 2000))

    def test_alpha_scoring_rebuilds_geometry_without_cbeta(self):
        mutant = Mutant(
            mutant_str="A6V",
            mutated_sequence="AAAAAVAAAA",
            score=0.0,
            positions=[5],
            wt_aas=["A"],
            mt_aas=["V"],
        )
        assay = DMSAssay(
            assay_id="test",
            uniprot_id="test",
            wt_sequence="AAAAAAAAAA",
            mutants=[mutant],
        )

        class RecordingLoader:
            def __init__(self):
                self.calls = []

            def load(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return None

        loader = RecordingLoader()
        module = SimpleNamespace(model=SimpleNamespace(_esm_loaded=True))
        scores = score_assay_option_f(
            module,
            loader,
            "test.pdb",
            "test",
            assay,
            "cpu",
            structure_length=10,
        )

        self.assertTrue(np.isnan(scores[0]))
        self.assertEqual(len(loader.calls), 1)
        kwargs = loader.calls[0][1]
        self.assertEqual(kwargs["ala_strip_positions"], [5])
        self.assertFalse(kwargs["ala_strip_keep_cb"])
