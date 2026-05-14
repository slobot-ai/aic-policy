#
# SPDX-License-Identifier: Apache-2.0
#

"""Multiview rectangle fit outputs (two yaw seeds) consumed by task-board control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TaskBoardRectangleCandidate:
    """One multiview rectangle fit starting from ``yaw_seed_rad``."""

    yaw_seed_rad: float
    cx_m: float
    cy_m: float
    yaw_rad: float
    mean_hard_iou: float


@dataclass(frozen=True)
class TaskBoardRectangleEstimationOutput:
    """Two fits: same initializer, distinct yaw seeds (default ``π`` and ``π/2``)."""

    candidates: tuple[TaskBoardRectangleCandidate, TaskBoardRectangleCandidate]


def rectangle_candidate_highest_mean_hard_iou(
    rectangle_out: TaskBoardRectangleEstimationOutput,
) -> tuple[TaskBoardRectangleCandidate, int]:
    """Return ``(candidate, index)`` with the largest ``mean_hard_iou`` (tie → smaller index)."""

    cands = rectangle_out.candidates
    if len(cands) < 1:
        raise ValueError("TaskBoardRectangleEstimationOutput must contain at least one candidate")
    best_i = max(range(len(cands)), key=lambda i: (cands[i].mean_hard_iou, -i))
    return cands[best_i], int(best_i)


def intrinsics_k_for_resolution(
    k_row_major: object,
    *,
    source_shape_hw: tuple[int, int],
    target_shape_hw: tuple[int, int],
) -> NDArray[np.float64]:
    """Map pinhole ``K`` from full BGR resolution to another pixel grid (e.g. mask size)."""

    src_h, src_w = source_shape_hw
    tgt_h, tgt_w = target_shape_hw
    sx = float(tgt_w) / float(src_w)
    sy = float(tgt_h) / float(src_h)
    k = np.asarray(k_row_major, dtype=np.float64).reshape(3, 3).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k
