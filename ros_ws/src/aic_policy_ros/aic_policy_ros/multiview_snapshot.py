#
# SPDX-License-Identifier: Apache-2.0
#

"""§1.1 — one multiview ``Observation``: decode to BGR, dump full-resolution PNGs per camera."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aic_model_interfaces.msg import Observation
from sensor_msgs.msg import Image

_CAM_INPUT_ORDER: tuple[tuple[str, str], ...] = (
    ("left_camera", "left_image"),
    ("center_camera", "center_image"),
    ("right_camera", "right_image"),
)


def episode_dir_name(ep: int) -> str:
    """Directory segment under ``output_root`` for this episode (env overrides for seg tooling)."""

    return (
        os.environ.get("AIC_SEG_EPISODE_ID")
        or os.environ.get("AIC_SEG_TRIAL_ID")
        or f"episode_{ep}"
    )


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    enc = (msg.encoding or "").lower().replace("_", "")
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("invalid image dimensions")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8, count=step * h)
    mat = buf.reshape((h, step))
    if enc in ("rgb8", "bgr8"):
        row = mat[:, : w * 3].reshape((h, w, 3))
        return row if enc == "bgr8" else cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    if enc in ("rgba8", "bgra8"):
        row = mat[:, : w * 4].reshape((h, w, 4))
        if enc == "bgra8":
            return cv2.cvtColor(row, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(row, cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        row = mat[:, :w].reshape((h, w))
        return cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


@dataclass
class MultiviewSnapshotInput:
    observation: Observation
    episode_id: int
    dump_filename: str


@dataclass
class MultiviewSnapshotOutput:
    dump_paths: tuple[Path, Path, Path]
    bgr_left: np.ndarray
    bgr_center: np.ndarray
    bgr_right: np.ndarray


class MultiviewSnapshot:
    """Decode three wrist images once, write BGR PNGs under ``output_root`` for ``episode_id`` / ``dump_filename``."""

    @staticmethod
    def multiview_bgr_dump_paths(
        output_root: Path, episode_id: int, dump_filename: str
    ) -> tuple[Path, Path, Path]:
        """Left, center, right paths under ``output_root`` / episode / ``<camera>/input/``."""

        ep_root = output_root.expanduser() / episode_dir_name(episode_id)
        return tuple(ep_root / cam / "input" / dump_filename for cam, _ in _CAM_INPUT_ORDER)

    def __init__(self, logger: Any, output_root: Path | str) -> None:
        self._logger = logger
        self._output_root = Path(output_root).expanduser()

    def apply(self, inp: MultiviewSnapshotInput) -> MultiviewSnapshotOutput:
        obs = inp.observation

        bgr_dump_paths = self.multiview_bgr_dump_paths(
            self._output_root, inp.episode_id, inp.dump_filename
        )

        bgr_left = image_msg_to_bgr(obs.left_image)
        bgr_center = image_msg_to_bgr(obs.center_image)
        bgr_right = image_msg_to_bgr(obs.right_image)
        bgr_by_attr = {
            "left_image": bgr_left,
            "center_image": bgr_center,
            "right_image": bgr_right,
        }

        paths: list[Path] = []
        for (_cam_id, attr), path in zip(_CAM_INPUT_ORDER, bgr_dump_paths, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), bgr_by_attr[attr])
            resolved = path.resolve()
            paths.append(resolved)
            self._logger.info(f"TaskBoardPolicy: observation dump file://{resolved.as_posix()}")
        self._logger.info(
            "MultiviewSnapshot: BGR sizes L/C/R="
            f"{[bgr_left.shape, bgr_center.shape, bgr_right.shape]} "
            f"dump_paths L/C/R={[p.as_posix() for p in paths]}"
        )
        return MultiviewSnapshotOutput(
            dump_paths=(paths[0], paths[1], paths[2]),
            bgr_left=bgr_left,
            bgr_center=bgr_center,
            bgr_right=bgr_right,
        )
