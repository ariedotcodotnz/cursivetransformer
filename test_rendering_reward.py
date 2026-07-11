import unittest

import numpy as np

from rendering_reward import (
    SequenceDiagnostics,
    TokenSpec,
    analyze_token_sequence,
    compute_rendering_reward,
)


def make_word(slant=0.2, scale=1.0, x_shift=0.0, second_stroke=True):
    points = np.array(
        [
            [0.00, 0.00, 1],
            [0.10 + 0.4 * slant, 0.40, 1],
            [0.20 + 0.8 * slant, 0.80, 1],
            [0.22, 0.75, 0],
            [0.45, 0.00, 1],
            [0.50 + 0.4 * slant, 0.40, 1],
            [0.55 + 0.8 * slant, 0.80, 1],
        ],
        dtype=np.float64,
    )
    if not second_stroke:
        points = points[:4]
    points[:, :2] *= scale
    points[:, 0] += x_shift
    return points


class RenderingRewardTests(unittest.TestCase):
    def setUp(self):
        # Deliberately ragged word lengths exercise the non-rectangular input path.
        self.target = [make_word(), make_word(second_stroke=False)]
        self.complete = SequenceDiagnostics(
            ended=True, valid=True, word_count=2, token_length=40
        )

    def test_identical_render_receives_full_shape_and_style_scores(self):
        result = compute_rendering_reward(
            self.target,
            self.target,
            sequence=self.complete,
            target_token_length=40,
        )

        self.assertAlmostEqual(result.components["shape_similarity"], 1.0, places=6)
        self.assertAlmostEqual(result.components["style_similarity"], 1.0, places=6)
        self.assertEqual(result.diagnostics["out_of_bounds_fraction"], 0.0)
        self.assertFalse(result.diagnostics["hard_failure"])
        self.assertGreater(result.total, 1.5)

    def test_prediction_cannot_choose_its_own_canvas(self):
        tiny = [make_word(scale=0.01), make_word(scale=0.01, second_stroke=False)]
        far = [make_word(x_shift=20.0), make_word(x_shift=20.0, second_stroke=False)]

        tiny_result = compute_rendering_reward(
            tiny, self.target, sequence=self.complete, target_token_length=40
        )
        far_result = compute_rendering_reward(
            far, self.target, sequence=self.complete, target_token_length=40
        )

        self.assertLess(tiny_result.components["shape_similarity"], 0.15)
        self.assertTrue(tiny_result.diagnostics["geometric_collapse"])
        self.assertLessEqual(tiny_result.total, -0.25)
        self.assertGreater(far_result.diagnostics["out_of_bounds_fraction"], 0.95)
        self.assertTrue(far_result.diagnostics["hard_oob"])
        self.assertLessEqual(far_result.total, -0.25)

    def test_blank_truncated_short_and_malformed_are_hard_failures(self):
        cases = [
            ([], self.complete, 40),
            (self.target, SequenceDiagnostics(False, True, 2, 40), 40),
            (self.target, SequenceDiagnostics(True, True, 2, 10), 40),
            (
                [np.array([[np.nan, 0.0, 1.0], [0.0, 1.0, 1.0]])],
                SequenceDiagnostics(True, True, 1, 20),
                20,
            ),
        ]
        for prediction, sequence, target_length in cases:
            with self.subTest(sequence=sequence):
                result = compute_rendering_reward(
                    prediction,
                    self.target,
                    sequence=sequence,
                    target_token_length=target_length,
                )
                self.assertTrue(result.diagnostics["hard_failure"])
                self.assertLessEqual(result.total, -0.25)

    def test_style_statistics_distinguish_opposite_slant(self):
        matching = [make_word(slant=0.45)]
        different = [make_word(slant=-0.45)]
        sequence = SequenceDiagnostics(True, True, 1, 20)
        match_result = compute_rendering_reward(
            matching,
            matching,
            style_reference=matching,
            sequence=sequence,
            target_token_length=20,
        )
        different_result = compute_rendering_reward(
            different,
            different,
            style_reference=matching,
            sequence=sequence,
            target_token_length=20,
        )

        self.assertAlmostEqual(match_result.components["style_similarity"], 1.0, places=6)
        self.assertLess(
            different_result.components["style_slant_similarity"],
            match_result.components["style_slant_similarity"],
        )
        self.assertLess(
            different_result.components["style_similarity"],
            match_result.components["style_similarity"],
        )


class TokenDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.spec = TokenSpec(
            theta_start=10,
            theta_stop=20,
            radius_start=0,
            radius_stop=10,
            word_token=20,
            end_token=21,
            pad_token=22,
        )

    def test_valid_two_word_sequence(self):
        diagnostics = analyze_token_sequence(
            [10, 1, 11, 2, 20, 20, 12, 3, 21, 22, 22], self.spec
        )
        self.assertTrue(diagnostics.ended)
        self.assertTrue(diagnostics.valid)
        self.assertEqual(diagnostics.word_count, 2)
        self.assertEqual(diagnostics.token_length, 9)
        self.assertEqual(diagnostics.reasons, ())

    def test_structure_errors_and_missing_end_are_distinct(self):
        malformed = analyze_token_sequence([10, 1, 20, 12, 3, 21], self.spec)
        truncated = analyze_token_sequence([10, 1], self.spec)

        self.assertTrue(malformed.ended)
        self.assertFalse(malformed.valid)
        self.assertIn("unpaired_word_token", malformed.reasons)
        self.assertFalse(truncated.ended)
        self.assertTrue(truncated.valid)
        self.assertIn("missing_end", truncated.reasons)

    def test_legacy_dataset_with_none_bos_is_supported(self):
        class Dataset:
            cumulative_sizes = [0, 10]
            r_bins = range(10)
            theta_bins = range(5)
            PAD_TOKEN = 15
            END_TOKEN = 16
            WORD_TOKEN = 17
            BOS_TOKEN = None

        spec = TokenSpec.from_dataset(Dataset())
        self.assertIsNone(spec.bos_token)
        self.assertEqual((spec.theta_start, spec.theta_stop), (10, 15))


if __name__ == "__main__":
    unittest.main()
