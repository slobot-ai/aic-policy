#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from aic_policy_ros.task_board_episode_geometry import task_board_top_face_center_base_m_from_pose_dict
from aic_policy_ros.task_board_insert_pose import (
    REFERENCE_TASK_BOARD_XYZ_RPY,
    initial_tcp_pose_from_manifest_row,
    manifest_row_for_task,
    quat_goal_xyzw_with_live_board_yaw_delta,
    tcp_goal_pose_from_manifest_and_board,
)


def _reference_board_top_face_xy_and_pose_z() -> tuple[float, float, float]:
    xr, yr, zr, rr, pr, yw = REFERENCE_TASK_BOARD_XYZ_RPY
    ref = {"x": xr, "y": yr, "z": zr, "roll": rr, "pitch": pr, "yaw": yw}
    cx, cy, _ = task_board_top_face_center_base_m_from_pose_dict(ref)
    return cx, cy, zr


def test_initial_tcp_pose_from_manifest_row() -> None:
    row = {
        "trial_key": "trial_0",
        "initial_tcp_pose": {"xyz_m": [1.0, 2.0, 3.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }
    out = initial_tcp_pose_from_manifest_row(row)
    assert out is not None
    assert out[0] == (1.0, 2.0, 3.0)
    assert out[1] == (0.0, 0.0, 0.0, 1.0)
    assert initial_tcp_pose_from_manifest_row({}) is None
    assert initial_tcp_pose_from_manifest_row({"initial_tcp_pose": "bad"}) is None
    assert initial_tcp_pose_from_manifest_row(None) is None


def test_manifest_row_fallback_trial_key() -> None:
    rows = [
        {"trial_key": "trial_0", "task_kind": "nic", "nic_card_index": 0, "sfp_port_index": 0},
        {"trial_key": "trial_3", "task_kind": "nic", "nic_card_index": 1, "sfp_port_index": 1},
    ]
    empty = SimpleNamespace(
        id="",
        cable_type="",
        cable_name="",
        plug_type="",
        plug_name="",
        port_type="",
        port_name="",
        target_module_name="",
        time_limit=0,
    )
    assert manifest_row_for_task(empty, 0, rows) == rows[0]
    assert manifest_row_for_task(empty, 3, rows) == rows[1]


def test_manifest_row_nic_match() -> None:
    rows = [
        {
            "trial_key": "trial_3",
            "task_kind": "nic",
            "nic_card_index": 1,
            "sfp_port_index": 1,
        },
    ]
    task = SimpleNamespace(
        id="t",
        cable_type="sfp_sc",
        cable_name="c",
        plug_type="sfp",
        plug_name="p",
        port_type="sfp",
        port_name="sfp_port_1",
        target_module_name="nic_card_mount_1",
        time_limit=180,
    )
    assert manifest_row_for_task(task, 99, rows) == rows[0]


def test_manifest_row_sc_match() -> None:
    rows = [
        {"trial_key": "trial_11", "task_kind": "sc", "sc_port_index": 0},
    ]
    task = SimpleNamespace(
        id="t",
        cable_type="sfp_sc",
        cable_name="c",
        plug_type="sc",
        plug_name="sc_tip",
        port_type="sc",
        port_name="sc_port_base",
        target_module_name="sc_port_0",
        time_limit=180,
    )
    assert manifest_row_for_task(task, 99, rows) == rows[0]


def test_tcp_goal_identity_board_matches_episode_delta_plus_z_bump() -> None:
    """When current board pose equals the reference recording pose, output equals p0 + offset (+ z bump)."""

    p0 = np.array([0.35, 0.0, 0.55], dtype=np.float64)
    q0 = Rotation.from_euler("xyz", [0.0, 0.0, 0.1]).as_quat()
    fto = {
        "linear": {"x": 0.01, "y": -0.02, "z": 0.03},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    cx_ref, cy_ref, zr = _reference_board_top_face_xy_and_pose_z()
    pose = tcp_goal_pose_from_manifest_and_board(
        tcp_episode_start_position_m=p0,
        tcp_episode_start_quat_xyzw=q0,
        final_tcp_offset=fto,
        task_kind="nic",
        board_cx_m=float(cx_ref),
        board_cy_m=float(cy_ref),
        board_yaw_rad=float(REFERENCE_TASK_BOARD_XYZ_RPY[5]),
        board_z_m=float(zr),
    )
    exp_p = p0 + np.array([0.01, -0.02, 0.03 + 0.07], dtype=np.float64)
    np.testing.assert_allclose(
        [pose.position.x, pose.position.y, pose.position.z],
        exp_p,
        rtol=0.0,
        atol=1e-6,
    )
    q = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    np.testing.assert_allclose(Rotation.from_quat(q).as_quat(), q0, rtol=0.0, atol=1e-6)


def test_tcp_goal_board_pure_translation_preserves_relative_to_board_origin() -> None:
    """If offset is zero and orientation identity, TCP should move by the same delta as board origin."""

    p0 = np.array([0.35, 0.0, 0.55], dtype=np.float64)
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    fto = {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    cx_ref, cy_ref, zr = _reference_board_top_face_xy_and_pose_z()
    dx, dy = 0.02, -0.03
    pose = tcp_goal_pose_from_manifest_and_board(
        tcp_episode_start_position_m=p0,
        tcp_episode_start_quat_xyzw=q0,
        final_tcp_offset=fto,
        task_kind="sc",
        board_cx_m=float(cx_ref + dx),
        board_cy_m=float(cy_ref + dy),
        board_yaw_rad=float(REFERENCE_TASK_BOARD_XYZ_RPY[5]),
        board_z_m=float(zr),
    )
    # sc adds 2cm z on zero linear z; board and ref same yaw → pure translation of board by (dx,dy)
    exp_p = p0 + np.array([dx, dy, 0.02], dtype=np.float64)
    np.testing.assert_allclose(
        [pose.position.x, pose.position.y, pose.position.z],
        exp_p,
        rtol=0.0,
        atol=1e-6,
    )


def test_tcp_goal_z_extra_sc_vs_nic() -> None:
    p0 = np.zeros(3, dtype=np.float64)
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    fto = {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    cx_ref, cy_ref, zr = _reference_board_top_face_xy_and_pose_z()
    yr_ref = float(REFERENCE_TASK_BOARD_XYZ_RPY[5])
    pose_sc = tcp_goal_pose_from_manifest_and_board(
        tcp_episode_start_position_m=p0,
        tcp_episode_start_quat_xyzw=q0,
        final_tcp_offset=fto,
        task_kind="sc",
        board_cx_m=float(cx_ref),
        board_cy_m=float(cy_ref),
        board_yaw_rad=yr_ref,
        board_z_m=float(zr),
    )
    pose_nic = tcp_goal_pose_from_manifest_and_board(
        tcp_episode_start_position_m=p0,
        tcp_episode_start_quat_xyzw=q0,
        final_tcp_offset=fto,
        task_kind="nic",
        board_cx_m=float(cx_ref),
        board_cy_m=float(cy_ref),
        board_yaw_rad=yr_ref,
        board_z_m=float(zr),
    )
    assert abs(float(pose_nic.position.z) - float(pose_sc.position.z) - 0.05) < 1e-9


def test_quat_goal_world_z_dyaw_left_multiplies_like_tcp_goal_helper() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rec = {"x": 0.15, "y": -0.2, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": 0.2}
    dyaw = 0.45
    out = quat_goal_xyzw_with_live_board_yaw_delta(
        q0,
        live_board_yaw_rad=float(rec["yaw"]) + dyaw,
        recorded_task_board_pose=rec,
    )
    exp = (Rotation.from_euler("xyz", [0.0, 0.0, dyaw]) * Rotation.from_quat(q0)).as_quat()
    np.testing.assert_allclose(out, exp, rtol=0.0, atol=1e-9)


def test_tcp_goal_board_yaw_applies_world_z_rotation() -> None:
    p0 = np.array([0.35, 0.0, 0.55], dtype=np.float64)
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    fto = {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    cx_ref, cy_ref, zr = _reference_board_top_face_xy_and_pose_z()
    yr_ref = float(REFERENCE_TASK_BOARD_XYZ_RPY[5])
    dyaw = 0.5
    pose = tcp_goal_pose_from_manifest_and_board(
        tcp_episode_start_position_m=p0,
        tcp_episode_start_quat_xyzw=q0,
        final_tcp_offset=fto,
        task_kind="sc",
        board_cx_m=float(cx_ref),
        board_cy_m=float(cy_ref),
        board_yaw_rad=float(yr_ref + dyaw),
        board_z_m=float(zr),
    )
    exp_R = Rotation.from_euler("xyz", [0.0, 0.0, dyaw])
    q = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), exp_R.as_matrix(), rtol=0.0, atol=1e-6)


def test_rectangle_candidate_highest_mean_hard_iou_picks_max() -> None:
    from aic_policy_ros.task_board_rectangle_estimation import (
        TaskBoardRectangleCandidate,
        TaskBoardRectangleEstimationOutput,
        rectangle_candidate_highest_mean_hard_iou,
    )

    c0 = TaskBoardRectangleCandidate(
        yaw_seed_rad=3.0, cx_m=0.0, cy_m=0.0, yaw_rad=0.0, mean_hard_iou=0.9
    )
    c1 = TaskBoardRectangleCandidate(
        yaw_seed_rad=1.5, cx_m=1.0, cy_m=1.0, yaw_rad=1.0, mean_hard_iou=0.5
    )
    out = TaskBoardRectangleEstimationOutput(candidates=(c0, c1))
    c, i = rectangle_candidate_highest_mean_hard_iou(out)
    assert i == 0 and c is c0


def test_rectangle_candidate_highest_mean_hard_iou_tie_prefers_lower_index() -> None:
    from aic_policy_ros.task_board_rectangle_estimation import (
        TaskBoardRectangleCandidate,
        TaskBoardRectangleEstimationOutput,
        rectangle_candidate_highest_mean_hard_iou,
    )

    c0 = TaskBoardRectangleCandidate(
        yaw_seed_rad=0.0, cx_m=0.0, cy_m=0.0, yaw_rad=0.0, mean_hard_iou=0.8
    )
    c1 = TaskBoardRectangleCandidate(
        yaw_seed_rad=1.0, cx_m=2.0, cy_m=2.0, yaw_rad=2.0, mean_hard_iou=0.8
    )
    out = TaskBoardRectangleEstimationOutput(candidates=(c0, c1))
    c, i = rectangle_candidate_highest_mean_hard_iou(out)
    assert i == 0 and c is c0
