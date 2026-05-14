#
# SPDX-License-Identifier: Apache-2.0
#

"""Fixed multiview ``T_cam_from_base`` matrices for **tests and offline tooling only**.

These values are **not** part of the ``vision`` package. Live :class:`TaskBoardPolicy` / :class:`TaskBoardVision`
resolve wrist extrinsics from ``tf2_ros.Buffer`` only (no static fallback).

Matrices were vendored from the former ``vision.transform.rectangle_forward_projection`` literals
(``tests/data/episode_0`` wrist poses). ``THIRD`` matches ``SECOND`` numerically until a distinct
post–``insert_cable`` wrist calibration is checked in.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_CAMERAS: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")


def _T_from_xyz_quat(
    x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


_T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_FIRST_OBSERVATION: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "left_camera": (
        0.01769838976310112,
        0.38106246660131526,
        0.6454801469467681,
        0.8585819813026575,
        0.49578411729250904,
        0.06529810782630618,
        -0.11300994441727207,
    ),
    "center_camera": (
        0.3717788749871604,
        0.16386991021960184,
        0.5872859442057675,
        0.9914490044008089,
        7.412309353222768e-05,
        1.9798982392258192e-05,
        -0.1304946963918917,
    ),
    "right_camera": (
        0.3540787268312187,
        -0.2409384957968447,
        0.4788138522056634,
        0.8586582407861697,
        -0.49565573102761845,
        -0.06526381480979797,
        -0.11301352266852689,
    ),
}

_T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_SECOND_OBSERVATION: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "left_camera": (
        0.01769838976310112,
        0.38106246660131526,
        0.6454801469467681,
        0.8585819813026575,
        0.4957841172925091,
        0.0652981078263062,
        -0.11300994441727208,
    ),
    "center_camera": (
        0.3717788749871604,
        0.16386991021960184,
        0.5872859442057675,
        0.9914490044008089,
        7.41230935322277e-05,
        1.9798982392258192e-05,
        -0.1304946963918917,
    ),
    "right_camera": (
        0.3540787268312187,
        -0.2409384957968447,
        0.4788138522056634,
        0.8586582407861697,
        -0.49565573102761845,
        -0.06526381480979797,
        -0.11301352266852689,
    ),
}

# Third multiview frame (``insert_cable`` final wrist snapshot): placeholder = second wrist until recalibrated.
_T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_THIRD_OBSERVATION: dict[
    str, tuple[float, float, float, float, float, float, float]
] = {cam: tuple(_T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_SECOND_OBSERVATION[cam]) for cam in _CAMERAS}


T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION: dict[str, np.ndarray] = {
    cam: _T_from_xyz_quat(*vals) for cam, vals in _T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_FIRST_OBSERVATION.items()
}

T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION: dict[str, np.ndarray] = {
    cam: _T_from_xyz_quat(*vals) for cam, vals in _T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_SECOND_OBSERVATION.items()
}

T_CAM_FROM_BASE_BY_CAMERA_THIRD_OBSERVATION: dict[str, np.ndarray] = {
    cam: _T_from_xyz_quat(*vals) for cam, vals in _T_CAM_FROM_BASE_XYZ_QUAT_BY_CAMERA_THIRD_OBSERVATION.items()
}
