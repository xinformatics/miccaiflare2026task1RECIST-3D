from __future__ import annotations

import unittest

import numpy as np

from solution.evaluate import nsd


class EvaluateTest(unittest.TestCase):
    def test_nsd_converts_xyz_spacing_to_array_zyx_once(self):
        reference = np.zeros((5, 5, 5), dtype=bool)
        shifted_z = np.zeros_like(reference)
        shifted_x = np.zeros_like(reference)
        reference[2, 2, 2] = True
        shifted_z[3, 2, 2] = True
        shifted_x[2, 2, 3] = True
        spacing_xyz = np.array([1.0, 1.0, 3.0])

        self.assertEqual(nsd(reference, shifted_z, spacing_xyz), 0.0)
        self.assertEqual(nsd(reference, shifted_x, spacing_xyz), 1.0)


if __name__ == "__main__":
    unittest.main()
