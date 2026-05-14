#
# SPDX-License-Identifier: Apache-2.0
#

"""Trial YAML paths and board/table geometry derived from AIC episode config and vision."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from aic_policy_ros.ai_industry_challenge_seg_path import aic_policy_repo_root


def t_base_from_board_from_trial_yaml(trial_yaml: Path | str) -> NDArray[np.float64]:
    """``T_base_from_board`` (4×4) from a generated AIC ``trial_N.yaml`` task_board world pose."""

    from vision.transform.trial_task_board_pose import (
        T_base_from_board_from_world_poses,
        T_world_from_xyz_rpy,
        trial_key_from_generated_config_path,
    )

    path = Path(trial_yaml).expanduser().resolve()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    key = trial_key_from_generated_config_path(path)
    pose = doc["trials"][key]["scene"]["task_board"]["pose"]
    T_world_from_board = T_world_from_xyz_rpy(
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    return T_base_from_board_from_world_poses(T_world_from_board)


def t_cam_from_board_by_camera_from_trial_yaml(trial_yaml: Path | str) -> dict[str, NDArray[np.float64]]:
    """Per-camera ``T_cam_from_board`` (4×4) from ``trial_N.yaml`` and fixed wrist extrinsics."""

    from aic_policy_ros.wrist_extrinsics_testing_matrices import T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION

    T_bb = t_base_from_board_from_trial_yaml(trial_yaml)
    cams: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")
    return {
        cam: np.asarray(T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION[cam], dtype=np.float64) @ T_bb for cam in cams
    }


def _default_trial_yaml_for_episode(episode_id: int) -> Path:
    subdir = os.environ.get("AIC_CONFIG_SUBDIR", "generated_configs")
    return (
        aic_policy_repo_root().parent
        / "aic"
        / "aic_engine"
        / "config"
        / subdir
        / f"trial_{int(episode_id)}.yaml"
    )


def task_board_top_face_center_base_m_from_pose_dict(pose: dict[str, Any]) -> tuple[float, float, float]:
    """URDF top-face AABB center in ``base_link`` (m) for a ``scene.task_board.pose``-style mapping."""

    from vision.rectangle3d_detector import TASK_BOARD_TOP_FACE_AABB
    from vision.transform.trial_task_board_pose import T_world_from_xyz_rpy, T_base_from_board_from_world_poses

    tw = T_world_from_xyz_rpy(
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    t_bb = T_base_from_board_from_world_poses(tw)
    x_min, y_min, x_max, y_max, z_top = TASK_BOARD_TOP_FACE_AABB
    cx_b = 0.5 * float(x_min + x_max)
    cy_b = 0.5 * float(y_min + y_max)
    pt = t_bb @ np.array([cx_b, cy_b, float(z_top), 1.0], dtype=np.float64)
    return float(pt[0]), float(pt[1]), float(pt[2])


def z_face_in_base() -> float:
    """Top-face plane ``z`` (m): :data:`vision.rectangle3d_detector.TASK_BOARD_TOP_FACE_AABB` ``z_top``."""

    from vision.rectangle3d_detector import TASK_BOARD_TOP_FACE_AABB

    return float(TASK_BOARD_TOP_FACE_AABB[4])


def t_cam_from_base_by_camera_dict(
    observation_phase: Literal["first", "second", "third"],
) -> dict[str, NDArray[np.float64]]:
    """Per-camera ``T_cam_from_base`` for **offline tests / tooling** (not used by live ``TaskBoardVision``).

    Live policy resolves extrinsics from ``tf2_ros.Buffer`` only. These dicts mirror the vendored
    episode‑0 wrist matrices in :mod:`aic_policy_ros.wrist_extrinsics_testing_matrices`.
    """

    from aic_policy_ros.wrist_extrinsics_testing_matrices import (
        T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION,
        T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION,
        T_CAM_FROM_BASE_BY_CAMERA_THIRD_OBSERVATION,
    )

    if observation_phase == "first":
        src = T_CAM_FROM_BASE_BY_CAMERA_FIRST_OBSERVATION
    elif observation_phase == "second":
        src = T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION
    else:
        src = T_CAM_FROM_BASE_BY_CAMERA_THIRD_OBSERVATION
    return {cam: np.asarray(src[cam], dtype=np.float64) for cam in ("left_camera", "center_camera", "right_camera")}


def _center_estimation_geometry(
    *, observation_phase: Literal["first", "second"] = "first"
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], float]:
    """Legacy tuple ``(T_left, T_center, T_right), z_face`` — prefer TF + :func:`z_face_in_base`."""

    z_table = z_face_in_base()
    t_by = t_cam_from_base_by_camera_dict(observation_phase)
    return (
        (t_by["left_camera"], t_by["center_camera"], t_by["right_camera"]),
        z_table,
    )
