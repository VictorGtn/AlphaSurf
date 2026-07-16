from unittest import TestCase

from alphasurf.tasks.proteingym.scoring import get_optimal_window


class S3FWindowTest(TestCase):
    def test_short_sequence_uses_full_length(self):
        self.assertEqual(get_optimal_window(20, 250), (0, 250))

    def test_long_sequence_windows_match_released_s3f_logic(self):
        self.assertEqual(get_optimal_window(10, 2000), (0, 1022))
        self.assertEqual(get_optimal_window(1000, 2000), (489, 1511))
        self.assertEqual(get_optimal_window(1900, 2000), (978, 2000))
