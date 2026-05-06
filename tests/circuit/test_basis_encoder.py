import unittest

import numpy as np

from nexq.encoder.basis import BasisEncoder


class TestBasisEncoder(unittest.TestCase):
    def test_repeat_false_deduplicates_before_encoding(self):
        encoder = BasisEncoder(repeat=False)
        _, state = encoder.encode([1, 2, 2, 2])

        probs = np.abs(state.to_numpy()) ** 2

        self.assertAlmostEqual(float(probs[1]), 0.5, places=6)
        self.assertAlmostEqual(float(probs[2]), 0.5, places=6)

    def test_repeat_true_preserves_frequency_weights(self):
        encoder = BasisEncoder(repeat=True)
        _, state = encoder.encode([1, 2, 2, 2])

        probs = np.abs(state.to_numpy()) ** 2

        self.assertAlmostEqual(float(probs[1]), 0.25, places=6)
        self.assertAlmostEqual(float(probs[2]), 0.75, places=6)
        self.assertAlmostEqual(float(probs[2] / probs[1]), 3.0, places=6)