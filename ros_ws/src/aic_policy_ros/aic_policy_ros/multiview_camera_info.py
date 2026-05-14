#
# SPDX-License-Identifier: Apache-2.0
#

"""§1.5 — after HSV (§1.3) and task-board segmentation (§1.4), read multiview ``sensor_msgs/CameraInfo``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic_model_interfaces.msg import Observation
from sensor_msgs.msg import CameraInfo


def _float_tuple(seq: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in seq)


@dataclass
class CameraCalibrationSnapshot:
    """One camera: identity, image size, distortion model, ``D``, ``K`` (intrinsics), ``R``, ``P`` (ROS ``CameraInfo``)."""

    camera_name: str
    frame_id: str
    image_width: int
    image_height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]


@dataclass
class MultiviewCameraInfoInput:
    observation: Observation


@dataclass
class MultiviewCameraInfoOutput:
    left: CameraCalibrationSnapshot
    center: CameraCalibrationSnapshot
    right: CameraCalibrationSnapshot


class MultiviewCameraInfo:
    """Extract ``CameraInfo`` for left, center, and right wrists (no image decode)."""

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    @staticmethod
    def _snapshot(camera_name: str, info: CameraInfo) -> CameraCalibrationSnapshot:
        return CameraCalibrationSnapshot(
            camera_name=camera_name,
            frame_id=str(info.header.frame_id),
            image_width=int(info.width),
            image_height=int(info.height),
            distortion_model=str(info.distortion_model),
            d=_float_tuple(info.d),
            k=_float_tuple(info.k),
            r=_float_tuple(info.r),
            p=_float_tuple(info.p),
        )

    def apply(self, inp: MultiviewCameraInfoInput) -> MultiviewCameraInfoOutput:
        obs = inp.observation
        out = MultiviewCameraInfoOutput(
            left=self._snapshot("left_camera", obs.left_camera_info),
            center=self._snapshot("center_camera", obs.center_camera_info),
            right=self._snapshot("right_camera", obs.right_camera_info),
        )
        for snap in (out.left, out.center, out.right):
            self._logger.info(
                "TaskBoardPolicy: camera info "
                f"{snap.camera_name} frame_id={snap.frame_id!r} "
                f"size={snap.image_width}x{snap.image_height} "
                f"distortion_model={snap.distortion_model!r} "
                f"|K|={len(snap.k)} |D|={len(snap.d)} |R|={len(snap.r)} |P|={len(snap.p)}"
            )
        return out
