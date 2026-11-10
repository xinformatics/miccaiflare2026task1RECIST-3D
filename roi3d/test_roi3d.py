from __future__ import annotations

import unittest

import numpy as np
import torch

from roi3d.geometry import (
    LINE_CAP_PX, V2_SHAPE_ZYX, recist_geometries, sample_native, scatter_roi,
)
from roi3d.model import RECISTUNet3D, parameter_count
from roi3d.predict import anchored_component
from roi3d.train import boundary_dice, source_group


class ROI3DTest(unittest.TestCase):
    def test_anchored_component_prefers_prompted_component(self):
        mask = np.zeros((5, 8, 8), dtype=bool)
        mask[2, 1:3, 1:3] = True
        mask[1:4, 5:8, 5:8] = True
        prompt = np.zeros_like(mask)
        prompt[2, 1, 1] = True
        selected = anchored_component(mask, prompt)
        self.assertEqual(selected.sum(), 4)
        self.assertTrue(selected[2, 1, 1])

    def test_anchored_component_falls_back_to_largest(self):
        mask = np.zeros((5, 8, 8), dtype=bool)
        mask[2, 1:3, 1:3] = True
        mask[1:4, 5:8, 5:8] = True
        prompt = np.zeros_like(mask)
        prompt[0, 0, 0] = True
        selected = anchored_component(mask, prompt)
        self.assertEqual(selected.sum(), 27)

    def test_pancreatic_dataset_maps_to_pancreas_sampling_group(self):
        self.assertEqual(source_group("Dataset006_PancreaticCancer"), "pancreas")

    def test_documented_model_parameter_count(self):
        self.assertEqual(parameter_count(RECISTUNet3D()), 16_546_883)

    def test_recist_cap_correction_and_v2_spacing(self):
        recist = np.zeros((3, 20, 30), dtype=np.uint8)
        recist[1, 10, 5:16] = 1
        geometry = recist_geometries(recist, np.array([0.8, 0.8, 3.0]))[0]
        self.assertAlmostEqual(geometry.diameter_mm, 8.0, places=5)
        self.assertAlmostEqual(
            geometry.diameter_true_mm, (10 - LINE_CAP_PX) * 0.8, places=5
        )
        self.assertEqual(geometry.roi_spacing_mm, 0.45)

    def test_sample_scatter_roundtrip(self):
        shape = (80, 160, 160)
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        target = ((zz - 40) ** 2 / 8 ** 2 + (yy - 80) ** 2 / 20 ** 2 +
                  (xx - 80) ** 2 / 25 ** 2) <= 1
        recist = np.zeros(shape, dtype=np.uint8)
        recist[40, 80, 55:106] = 1
        spacing = np.array([0.7, 0.7, 2.0])
        geometry = recist_geometries(recist, spacing)[0]
        roi = sample_native(target.astype(np.uint8), geometry, spacing,
                            V2_SHAPE_ZYX, order=0)
        restored = scatter_roi(roi.astype(np.float32), geometry, spacing,
                               shape, order=1) > 0.5
        dsc = 2 * np.count_nonzero(restored & target) / (
            np.count_nonzero(restored) + np.count_nonzero(target)
        )
        self.assertGreater(dsc, 0.995)

    def test_deep_supervision_shapes(self):
        model = RECISTUNet3D(base=2).train()
        outputs = model(torch.zeros(1, 4, 32, 32, 32))
        self.assertEqual([tuple(x.shape[2:]) for x in outputs],
                         [(32, 32, 32), (16, 16, 16), (8, 8, 8)])

    def test_zero_initialized_experts_preserve_generalist(self):
        torch.manual_seed(4)
        generalist = RECISTUNet3D(base=2).eval()
        expert = RECISTUNet3D(base=2, organ_experts=True, expert_rank=2).eval()
        expert.load_state_dict(generalist.state_dict(), strict=False)
        image = torch.randn(2, 4, 32, 32, 32)
        with torch.no_grad():
            expected = generalist(image)
            hard = expert(image, torch.tensor([0, 4]))
            soft = expert(image)
        torch.testing.assert_close(hard, expected)
        torch.testing.assert_close(soft, expected)

    def test_boundary_loss_prefers_aligned_mask(self):
        target = torch.zeros(1, 1, 16, 16, 16)
        target[:, :, 4:12, 4:12, 4:12] = 1
        aligned = (target * 2 - 1) * 8
        shifted_target = torch.roll(target, shifts=2, dims=4)
        shifted = (shifted_target * 2 - 1) * 8
        self.assertLess(boundary_dice(aligned, target), boundary_dice(shifted, target))


if __name__ == "__main__":
    unittest.main()
