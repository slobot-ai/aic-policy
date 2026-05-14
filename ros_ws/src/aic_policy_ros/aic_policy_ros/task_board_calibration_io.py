#
# SPDX-License-Identifier: Apache-2.0
#

"""JSON helpers for multiview camera intrinsics / ``T_cam_from_base`` (fixtures + dump script)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aic_model_interfaces.msg import Observation
from numpy.typing import NDArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from aic_policy_ros.multiview_snapshot import episode_dir_name

_CAM_ORDER: tuple[tuple[str, str], ...] = (
    ("left_camera", "left_image"),
    ("center_camera", "center_image"),
    ("right_camera", "right_image"),
)

# Policy dumps and fixtures: ``<episode>/<camera_id>/extrinsics/{intrinsics.json,*_extrinsics.json}``.
_CALIBRATION_JSON_SUBDIR = "extrinsics"


def _per_camera_intrinsics_json_path(root: Path, cam: str) -> Path:
    """Prefer ``<cam>/extrinsics/intrinsics.json``, fall back to legacy ``<cam>/intrinsics.json``."""

    p_new = root / cam / _CALIBRATION_JSON_SUBDIR / "intrinsics.json"
    if p_new.is_file():
        return p_new
    return root / cam / "intrinsics.json"


def pinhole_k_from_horizontal_fov(width: int, height: int, hfov_rad: float) -> list[float]:
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    vfov = 2.0 * math.atan((height / width) * math.tan(hfov_rad / 2.0))
    fy = (height / 2.0) / math.tan(vfov / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def _camera_info(frame_id: str, k: list[float]) -> CameraInfo:
    msg = CameraInfo()
    msg.header = Header(frame_id=frame_id)
    msg.distortion_model = "plumb_bob"
    msg.d = []
    msg.k = k
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def camera_info_sized(frame_id: str, width: int, height: int, k: list[float]) -> CameraInfo:
    msg = _camera_info(frame_id, k)
    msg.width = int(width)
    msg.height = int(height)
    return msg


def bgr_np_to_ros_image(bgr_u8: NDArray[np.uint8]) -> Image:
    if bgr_u8.ndim != 3 or bgr_u8.shape[2] != 3:
        raise ValueError("expected HxWx3 BGR uint8")
    h, w = int(bgr_u8.shape[0]), int(bgr_u8.shape[1])
    msg = Image()
    msg.header = Header()
    msg.height, msg.width = h, w
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = np.asarray(bgr_u8, dtype=np.uint8).tobytes()
    return msg


def intrinsics_dict_from_camera_info(msg: CameraInfo) -> dict[str, Any]:
    return {
        "schema": "aic_policy_ros.camera_intrinsics.v1",
        "header_frame_id": str(msg.header.frame_id),
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
    }


def camera_info_from_intrinsics_dict(d: dict[str, Any]) -> CameraInfo:
    msg = CameraInfo()
    msg.header = Header(frame_id=str(d["header_frame_id"]))
    msg.width = int(d["width"])
    msg.height = int(d["height"])
    msg.distortion_model = str(d["distortion_model"])
    msg.d = [float(x) for x in d["d"]]
    msg.k = [float(x) for x in d["k"]]
    msg.r = [float(x) for x in d["r"]]
    msg.p = [float(x) for x in d["p"]]
    return msg


def extrinsics_dict(
    *,
    camera_id: str,
    T_cam_from_base: NDArray[np.floating],
    source: str,
    observation: str,
) -> dict[str, Any]:
    if observation not in ("first", "second", "third"):
        raise ValueError("observation must be 'first', 'second', or 'third'")
    T = np.asarray(T_cam_from_base, dtype=np.float64).reshape(4, 4)
    return {
        "schema": "aic_policy_ros.camera_extrinsics.v1",
        "camera_id": camera_id,
        "observation": observation,
        "frame_parent": "base_link",
        "convention": "p_camera_homogeneous = T_cam_from_base @ p_base_homogeneous",
        "T_cam_from_base_row_major": [float(x) for x in T.ravel(order="C").tolist()],
        "source": source,
    }


def T_cam_from_base_from_extrinsics_dict(d: dict[str, Any]) -> NDArray[np.float64]:
    flat = d["T_cam_from_base_row_major"]
    return np.asarray(flat, dtype=np.float64).reshape(4, 4)


def observation_from_episode_bgr_pngs(
    episode_root: Path,
    *,
    stem: str = "first_observation",
    hfov_rad: float = 0.8718,
    frame_ids: tuple[str, str, str] = ("left_wrist", "center_wrist", "right_wrist"),
) -> Observation:
    """Load three ``{stem}_bgr.png`` files under ``<camera>/input/`` and build an :class:`Observation`."""

    paths = tuple(episode_root / cam / "input" / f"{stem}_bgr.png" for cam, _ in _CAM_ORDER)
    bgr_views: list[np.ndarray] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode BGR image: {p}")
        bgr_views.append(bgr)

    h0, w0 = bgr_views[0].shape[:2]
    for i, bgr in enumerate(bgr_views):
        if bgr.shape[:2] != (h0, w0):
            raise ValueError(
                f"all three BGR images must share the same resolution; {paths[i]} is {bgr.shape[:2]}, expected {(h0, w0)}"
            )

    width, height = int(w0), int(h0)
    k = pinhole_k_from_horizontal_fov(width, height, hfov_rad)
    obs = Observation()
    obs.left_image = bgr_np_to_ros_image(bgr_views[0])
    obs.center_image = bgr_np_to_ros_image(bgr_views[1])
    obs.right_image = bgr_np_to_ros_image(bgr_views[2])
    obs.left_camera_info = camera_info_sized(frame_ids[0], width, height, k)
    obs.center_camera_info = camera_info_sized(frame_ids[1], width, height, k)
    obs.right_camera_info = camera_info_sized(frame_ids[2], width, height, k)
    return obs


def observation_from_episode_with_intrinsics_json(
    episode_root: Path,
    *,
    stem: str = "first_observation",
    hfov_rad: float = 0.8718,
) -> Observation:
    """Like :func:`observation_from_episode_bgr_pngs` but load per-camera ``extrinsics/intrinsics.json`` when present."""

    intr_paths = [_per_camera_intrinsics_json_path(episode_root, cam) for cam, _ in _CAM_ORDER]
    if not all(p.is_file() for p in intr_paths):
        return observation_from_episode_bgr_pngs(episode_root, stem=stem, hfov_rad=hfov_rad)

    paths = tuple(episode_root / cam / "input" / f"{stem}_bgr.png" for cam, _ in _CAM_ORDER)
    bgr_views: list[np.ndarray] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode BGR image: {p}")
        bgr_views.append(bgr)

    h0, w0 = bgr_views[0].shape[:2]
    infos: list[CameraInfo] = []
    for p_intr, bgr in zip(intr_paths, bgr_views, strict=True):
        data = json.loads(p_intr.read_text(encoding="utf-8"))
        msg = camera_info_from_intrinsics_dict(data)
        if int(msg.width) != int(w0) or int(msg.height) != int(h0):
            raise ValueError(
                f"intrinsics.json size {msg.width}x{msg.height} does not match BGR {w0}x{h0} for {p_intr}"
            )
        infos.append(msg)

    obs = Observation()
    obs.left_image = bgr_np_to_ros_image(bgr_views[0])
    obs.center_image = bgr_np_to_ros_image(bgr_views[1])
    obs.right_image = bgr_np_to_ros_image(bgr_views[2])
    obs.left_camera_info = infos[0]
    obs.center_camera_info = infos[1]
    obs.right_camera_info = infos[2]
    return obs


def observation_from_image_root_and_calibration_fixture(
    image_root: Path,
    calib_fixture_root: Path,
    *,
    stem: str = "first_observation",
    hfov_rad: float = 0.8718,
) -> Observation:
    """Load BGR from ``image_root`` and ``intrinsics.json`` from ``calib_fixture_root`` (per-camera ``extrinsics/``)."""

    intr_paths = [_per_camera_intrinsics_json_path(calib_fixture_root, cam) for cam, _ in _CAM_ORDER]
    if not all(p.is_file() for p in intr_paths):
        return observation_from_episode_bgr_pngs(image_root, stem=stem, hfov_rad=hfov_rad)

    paths = tuple(image_root / cam / "input" / f"{stem}_bgr.png" for cam, _ in _CAM_ORDER)
    bgr_views: list[np.ndarray] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode BGR image: {p}")
        bgr_views.append(bgr)

    h0, w0 = bgr_views[0].shape[:2]
    infos: list[CameraInfo] = []
    for p_intr in intr_paths:
        data = json.loads(p_intr.read_text(encoding="utf-8"))
        msg = camera_info_from_intrinsics_dict(data)
        if int(msg.width) != int(w0) or int(msg.height) != int(h0):
            raise ValueError(
                f"fixture intrinsics size {msg.width}x{msg.height} does not match BGR {w0}x{h0} for {p_intr}"
            )
        infos.append(msg)

    obs = Observation()
    obs.left_image = bgr_np_to_ros_image(bgr_views[0])
    obs.center_image = bgr_np_to_ros_image(bgr_views[1])
    obs.right_image = bgr_np_to_ros_image(bgr_views[2])
    obs.left_camera_info = infos[0]
    obs.center_camera_info = infos[1]
    obs.right_camera_info = infos[2]
    return obs


def dump_multiview_calibration_json(
    out_root: Path,
    observation: Observation,
    t_cam_from_base_by_camera: dict[str, NDArray[np.floating]],
    *,
    observation_phase: str,
    write_intrinsics: bool,
    episode_id: int | None = None,
    extrinsics_matrix_source: str | None = None,
) -> None:
    """Write ``intrinsics.json`` (optional) and ``{first|second|third}_observation_extrinsics.json`` per camera.

    When ``episode_id`` is set, files go under ``out_root`` / :func:`episode_dir_name` / ``<camera_id>`` /
    ``extrinsics/`` (same episode subtree as :class:`aic_policy_ros.multiview_snapshot.MultiviewSnapshot`).
    When ``episode_id`` is ``None``, ``out_root`` is treated as the episode directory itself
    (``<camera_id>/extrinsics/`` directly under ``out_root``), matching fixture layout.

    ``extrinsics_matrix_source`` is stored in each extrinsics JSON ``source`` field; when ``None``,
    defaults to symbolic names under :mod:`aic_policy_ros.wrist_extrinsics_testing_matrices` for the
    observation phase (live dumps from :class:`TaskBoardVision` use TF and store the TF provenance string).
    """

    if observation_phase not in ("first", "second", "third"):
        raise ValueError("observation_phase must be 'first', 'second', or 'third'")

    episode_prefix = Path(episode_dir_name(episode_id)) if episode_id is not None else Path()

    infos = (observation.left_camera_info, observation.center_camera_info, observation.right_camera_info)
    for (cam_id, _), info in zip(_CAM_ORDER, infos, strict=True):
        cam_root = out_root / episode_prefix / cam_id
        calib_dir = cam_root / _CALIBRATION_JSON_SUBDIR
        calib_dir.mkdir(parents=True, exist_ok=True)
        if write_intrinsics:
            intr_path = calib_dir / "intrinsics.json"
            intr_path.write_text(
                json.dumps(intrinsics_dict_from_camera_info(info), indent=2) + "\n",
                encoding="utf-8",
            )
        T = np.asarray(t_cam_from_base_by_camera[cam_id], dtype=np.float64).reshape(4, 4)
        if extrinsics_matrix_source is not None:
            ext_source = extrinsics_matrix_source
        elif observation_phase == "first":
            ext_source = (
                "aic_policy_ros.wrist_extrinsics_testing_matrices.T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION"
            )
        elif observation_phase == "second":
            ext_source = (
                "aic_policy_ros.wrist_extrinsics_testing_matrices.T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION"
            )
        else:
            ext_source = (
                "aic_policy_ros.wrist_extrinsics_testing_matrices.T_CAM_FROM_BASE_BY_CAMERA_THIRD_OBSERVATION"
            )
        ext = extrinsics_dict(
            camera_id=cam_id,
            T_cam_from_base=T,
            source=ext_source,
            observation=observation_phase,
        )
        ext_path = calib_dir / f"{observation_phase}_observation_extrinsics.json"
        ext_path.write_text(json.dumps(ext, indent=2) + "\n", encoding="utf-8")
