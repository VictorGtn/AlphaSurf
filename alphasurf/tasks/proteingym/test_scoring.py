from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from alphasurf.tasks.proteingym.dataset import (
    DMSAssay,
    Mutant,
    af2_structure_path,
    find_reference_file,
    list_assay_csvs,
    load_dms_assay,
    load_reference_metadata,
)
from alphasurf.tasks.proteingym.evaluate import resolve_position_offset
from alphasurf.tasks.proteingym.scoring import (
    _scoring_window,
    get_optimal_window,
    score_assay_option_f,
)


class NullProteinLoader:
    def load(self, *args, **kwargs):
        return None


class S3FWindowTest(TestCase):
    def test_short_sequence_uses_full_length(self):
        self.assertEqual(get_optimal_window(20, 250), (0, 250))

    def test_long_sequence_windows_match_released_s3f_logic(self):
        self.assertEqual(get_optimal_window(10, 2000), (0, 1022))
        self.assertEqual(get_optimal_window(1000, 2000), (489, 1511))
        self.assertEqual(get_optimal_window(1900, 2000), (978, 2000))

    def test_special_assay_keeps_absolute_csv_positions(self):
        sequence = "A" * 3423
        with TemporaryDirectory() as directory:
            csv_path = f"{directory}/A0A140D2T1_ZIKV_Sourisseau_2019.csv"
            pd.DataFrame(
                {
                    "mutant": ["I291A"],
                    "mutated_sequence": [sequence],
                    "DMS_score": [0.0],
                }
            ).to_csv(csv_path, index=False)
            assay = load_dms_assay(csv_path)

        self.assertEqual(assay.seq_len, 3423)
        self.assertEqual(assay.uniprot_id, "A0A140D2T1_ZIKV")
        self.assertEqual(assay.mutants[0].positions, [290])
        self.assertEqual(assay.wt_sequence[290], "I")
        self.assertEqual(resolve_position_offset(assay, 3423), 0)
        self.assertEqual(resolve_position_offset(assay, 504), -290)
        self.assertEqual(_scoring_window(assay, 3423, [290]), (290, 794))
        self.assertEqual(_scoring_window(assay, 504, [0]), (0, 504))

    def test_reference_metadata_and_exact_pdb_filename_are_resolved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assay_dir = root / "DMS_ProteinGym_substitutions"
            assay_dir.mkdir()
            (assay_dir / "example.csv").touch()
            reference_file = root / "DMS_substitutions.csv"
            pd.DataFrame(
                {
                    "DMS_id": ["example"],
                    "UniProt_ID": ["A0A140D2T1_ZIKV"],
                    "pdb_file": ["A0A140D2T1_ZIKV.pdb"],
                    "pdb_range": ["291-794"],
                }
            ).to_csv(reference_file, index=False)
            pdb_path = root / "A0A140D2T1_ZIKV.pdb"
            pdb_path.touch()

            self.assertEqual(find_reference_file(assay_dir), reference_file)
            self.assertEqual(list_assay_csvs(root), [assay_dir / "example.csv"])
            metadata = load_reference_metadata(reference_file)
            self.assertEqual(metadata["example"]["pdb_file"], "A0A140D2T1_ZIKV.pdb")
            self.assertEqual(af2_structure_path(root, "A0A140D2T1_ZIKV.pdb"), pdb_path)

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

    def test_geometry_dataloader_runs_with_worker_processes(self):
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
        module = SimpleNamespace(model=SimpleNamespace(_esm_loaded=True))
        scores = score_assay_option_f(
            module,
            NullProteinLoader(),
            "test.pdb",
            "test",
            assay,
            "cpu",
            num_workers=2,
            structure_length=10,
        )
        self.assertTrue(np.isnan(scores[0]))
