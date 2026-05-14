#
# SPDX-License-Identifier: Apache-2.0
#

"""Center-estimation result type (initializer merged centroid + per-camera XY)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskBoardCenterEstimationOutput:
    """Estimated board / logo center on the table plane and per-camera mask centroids in ``base_link`` (m)."""

    cx: float
    cy: float
    per_camera_xy_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
