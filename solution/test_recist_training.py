#!/usr/bin/env python3
"""Focused tests for the challenge-specific training prompt path."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo
from training.dataset.vos_dataset import tensor_2_PIL
from training.dataset.vos_sampler import RECISTAnchorSampler
from training.dataset.vos_segment_loader import NPZSegmentLoader
from training.model.efficienttam import (
    sample_recist_proxy_boxes,
    sample_recist_proxy_points,
)
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn
from training.loss_fns import boundary_bce_loss
from training.train import apply_resolution_override


class RECISTTrainingTest(unittest.TestCase):
    def test_collate_preserves_actual_recist_prompt_mask(self):
        segment = torch.zeros(8, 8, dtype=torch.uint8)
        segment[1:7, 1:7] = 1
        prompt = torch.zeros_like(segment)
        prompt[4, 2:6] = 1
        datapoint = VideoDatapoint(
            frames=[Frame(
                data=torch.zeros(3, 8, 8),
                objects=[Object(7, 12, segment, prompt)],
            )],
            video_id=3,
            size=(8, 8),
        )
        batch = collate_fn([datapoint], dict_key="all")
        self.assertTrue(torch.equal(batch.prompts[0, 0], prompt.bool()))
        points, labels = sample_recist_proxy_points(
            batch.prompts[0].unsqueeze(1)
        )
        self.assertTrue(torch.all(labels == 1))
        self.assertTrue(torch.all(points[0, :, 1] == 4))
        self.assertTrue(torch.all(points[0, :, 0] >= 2))
        self.assertTrue(torch.all(points[0, :, 0] <= 5))

    def test_uint8_grayscale_frame_is_lazily_converted_to_rgb(self):
        frame = torch.tensor([[0, 64], [128, 255]], dtype=torch.uint8)
        image = tensor_2_PIL(frame)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(np.asarray(image)[1, 1].tolist(), [255, 255, 255])

    def test_float16_255_range_is_not_rescaled_and_saturated(self):
        frame = torch.tensor([[0, 64], [128, 255]], dtype=torch.float16)
        image = tensor_2_PIL(frame)
        self.assertEqual(np.asarray(image)[0, 1].tolist(), [64, 64, 64])
        self.assertEqual(np.asarray(image)[1, 0].tolist(), [128, 128, 128])

    def test_sampler_starts_at_recist_and_tracks_one_direction(self):
        frames = [VOSFrame(i, image_path=None) for i in range(12)]
        recist = np.zeros((12, 8, 8), dtype=np.uint8)
        recist[5, 3, 2:7] = 2
        masks = np.zeros_like(recist)
        masks[3:8, 2:6, 1:7] = 2
        video = VOSVideo("case", 0, frames, recist=recist)

        random.seed(7)
        sample = RECISTAnchorSampler(4, reverse_time_prob=0.5).sample(
            video, NPZSegmentLoader(masks)
        )
        indices = [frame.frame_idx for frame in sample.frames]
        self.assertEqual(indices[0], 5)
        self.assertIn(indices, ([5, 6, 7, 8], [5, 4, 3, 2]))
        self.assertEqual(sample.object_ids, [2])

    def test_three_proxy_points_are_positive_and_ordered_on_diameter(self):
        masks = torch.zeros((1, 1, 16, 24), dtype=torch.bool)
        masks[0, 0, 5:11, 3:21] = True
        points, labels = sample_recist_proxy_points(masks)

        self.assertEqual(tuple(points.shape), (1, 3, 2))
        self.assertTrue(torch.equal(labels, torch.ones_like(labels)))
        xy = points[0].round().long()
        self.assertTrue(all(masks[0, 0, y, x] for x, y in xy.tolist()))
        self.assertGreater(len(torch.unique(xy, dim=0)), 1)

    def test_recist_proxy_box_is_square_with_box_labels(self):
        masks = torch.zeros((1, 1, 40, 40), dtype=torch.bool)
        masks[0, 0, 18:22, 10:30] = True
        points, labels = sample_recist_proxy_boxes(masks, scale=1.6)
        width_height = points[0, 1] - points[0, 0]

        self.assertEqual(tuple(points.shape), (1, 2, 2))
        self.assertEqual(labels.tolist(), [[2, 3]])
        self.assertAlmostEqual(
            float(width_height[0]), float(width_height[1]), places=5
        )

    def test_recist_geometry_stays_fp32_inside_bfloat16_autocast(self):
        masks = torch.zeros((1, 1, 40, 40), dtype=torch.bool)
        masks[0, 0, 18:22, 10:30] = True
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            points, point_labels = sample_recist_proxy_points(masks)
            boxes, box_labels = sample_recist_proxy_boxes(masks, scale=1.0)

        self.assertEqual(points.dtype, torch.float32)
        self.assertEqual(boxes.dtype, torch.float32)
        self.assertEqual(point_labels.dtype, torch.int32)
        self.assertEqual(box_labels.dtype, torch.int32)

    def test_boundary_loss_prefers_aligned_contour_and_has_gradient(self):
        targets = torch.zeros((1, 1, 16, 16))
        targets[:, :, 4:12, 4:12] = 1
        aligned = (targets * 12 - 6).requires_grad_()
        shifted_target = torch.zeros_like(targets)
        shifted_target[:, :, 4:12, 6:14] = 1
        shifted = (shifted_target * 12 - 6).requires_grad_()

        aligned_loss = boundary_bce_loss(aligned, targets, 1)
        shifted_loss = boundary_bce_loss(shifted, targets, 1)
        self.assertEqual(tuple(aligned_loss.shape), (1, 1))
        self.assertLess(
            float(aligned_loss.detach()), float(shifted_loss.detach())
        )
        shifted_loss.sum().backward()
        self.assertIsNotNone(shifted.grad)

    def test_resolution_override_updates_both_rope_grids(self):
        cfg = OmegaConf.create(
            {
                "scratch": {"resolution": 512},
                "trainer": {
                    "model": {
                        "memory_attention": {
                            "layer": {
                                "self_attention": {"feat_sizes": [32, 32]},
                                "cross_attention": {"feat_sizes": [32, 32]},
                            }
                        }
                    }
                },
            }
        )
        apply_resolution_override(cfg, 384)
        self.assertEqual(cfg.scratch.resolution, 384)
        self.assertEqual(list(cfg.trainer.model.memory_attention.layer.self_attention.feat_sizes), [24, 24])
        self.assertEqual(list(cfg.trainer.model.memory_attention.layer.cross_attention.feat_sizes), [24, 24])


if __name__ == "__main__":
    unittest.main()
