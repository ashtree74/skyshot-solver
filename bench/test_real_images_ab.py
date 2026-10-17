"""Scorer checks with artificial WCS only; run with numpy/astropy available."""
import copy
import unittest

import real_images_ab as bench


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.truth = {"width": 1000, "height": 800, "wcs": {
            "crval1": 30.0, "crval2": 20.0, "crpix1": 500.5, "crpix2": 400.5,
            "cd1_1": -0.001, "cd1_2": 0.0, "cd2_1": 0.0, "cd2_2": 0.001, "sip": None}}

    def row(self, truth):
        header = bench.truth_wcs(truth).to_header(relax=True).tostring(sep="\n")
        return {"solution": {"status": "solved"}, "wcs_header": header}

    def test_identical_wcs_uses_correct_zero_based_pixels(self):
        score = bench.classify(self.row(self.truth), {}, self.truth)
        self.assertEqual(score["classification"], "correct")
        self.assertLess(score["grid_error_px"], 1e-6)

    def test_small_position_error_is_inaccurate_wcs_not_another_field(self):
        candidate = copy.deepcopy(self.truth)
        candidate["wcs"]["crpix1"] += 10
        score = bench.classify(self.row(candidate), {}, self.truth)
        self.assertEqual(score["classification"], "inaccurate_wcs")
        self.assertAlmostEqual(score["center_error_px"], 10, places=6)

    def test_wrong_field_is_rejected(self):
        candidate = copy.deepcopy(self.truth)
        candidate["wcs"]["crval1"] += 10
        self.assertEqual(bench.classify(self.row(candidate), {}, self.truth)["classification"], "wrong_field")

    def test_same_center_mirrored_field_is_rejected(self):
        candidate = copy.deepcopy(self.truth)
        candidate["wcs"]["cd1_1"] *= -1
        score = bench.classify(self.row(candidate), {}, self.truth)
        self.assertLess(score["center_error_px"], 1e-6)
        self.assertEqual(score["classification"], "wrong_field")

    def test_solved_without_reference_stays_unverified(self):
        self.assertEqual(bench.classify(self.row(self.truth), {}, None)["classification"], "unverified_solved")


if __name__ == "__main__":
    unittest.main()
