#
# SPDX-License-Identifier: Apache-2.0
#

"""§3 — pre-insert TCP goal: vision manifest remap (board-local xy + UP yaw +π) then Cartesian settle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from scipy.spatial.transform import Rotation

from aic_model.policy import MoveRobotCallback, Policy

from aic_policy_ros.task_board_insert_pose import (
    default_task_manifest_path,
    initial_tcp_pose_from_manifest_row,
    live_board_yaw_minus_reference_rad,
    load_task_manifest,
    manifest_row_for_task,
    quat_goal_xyzw_with_live_board_yaw_delta,
    tcp_manifest_linear_z_extra_m,
)

from .task_board_center_move import (
    TaskBoardCenterMove,
    TaskBoardPoseMoveInput,
    episode_start_tcp_delta_log_suffix,
    wrap_angle_rad,
)


@dataclass
class TaskPresetMiddlePoseTransformInput:
    """Second-action context: logging, motion, vision board pose, task id (TCP anchor from manifest ``initial_tcp_pose``)."""

    logger: Any
    policy: Policy
    move_robot: MoveRobotCallback
    sim_cycle_sec: float
    task_board_orientation: str
    board_cx_m: float
    board_cy_m: float
    board_yaw_rad: float
    task: Task
    episode_id: int


def _orientation_label_to_enum(label: str) -> Any:
    from vision.task_board_orientation import TaskBoardOrientationKind

    u = (label or "UNKNOWN").strip().upper()
    if u == "UP":
        return TaskBoardOrientationKind.UP
    if u == "DOWN":
        return TaskBoardOrientationKind.DOWN
    if u == "UNKNOWN":
        return TaskBoardOrientationKind.UNKNOWN
    return TaskBoardOrientationKind.UNKNOWN


class TaskPresetMiddlePoseTransform:
    """Load manifest row, remap ``final_tcp_offset`` with :mod:`vision.task_preset_middle_pose_transform`, move TCP."""

    def apply(self, inp: TaskPresetMiddlePoseTransformInput) -> None:
        logger = inp.logger
        logger.info(f"TaskPresetMiddlePoseTransform: task_board_orientation={inp.task_board_orientation!r}")
        rows = load_task_manifest()
        row = manifest_row_for_task(inp.task, inp.episode_id, rows)
        if row is None:
            logger.warning(
                f"TaskPresetMiddlePoseTransform: no manifest row for episode_id={inp.episode_id} "
                f"(task id={getattr(inp.task, 'id', '')!r})"
            )
            return

        logger.info(
            "TaskPresetMiddlePoseTransform: task "
            f"id={getattr(inp.task, 'id', '')!r} cable_type={getattr(inp.task, 'cable_type', '')!r} "
            f"cable_name={getattr(inp.task, 'cable_name', '')!r} plug_type={getattr(inp.task, 'plug_type', '')!r} "
            f"plug_name={getattr(inp.task, 'plug_name', '')!r} port_type={getattr(inp.task, 'port_type', '')!r} "
            f"port_name={getattr(inp.task, 'port_name', '')!r} "
            f"target_module_name={getattr(inp.task, 'target_module_name', '')!r}"
        )
        logger.info(f"TaskPresetMiddlePoseTransform: manifest entry {json.dumps(row, sort_keys=True)}")

        initial_tcp = initial_tcp_pose_from_manifest_row(row)
        if initial_tcp is None:
            logger.warning(
                "TaskPresetMiddlePoseTransform: manifest row missing or invalid initial_tcp_pose "
                "(expected xyz_m[3] and quat_xyzw[4]); skipping second move"
            )
            return
        anchor_pos_m, anchor_quat_xyzw = initial_tcp

        fto = row.get("final_tcp_offset")
        if not isinstance(fto, dict):
            logger.warning("TaskPresetMiddlePoseTransform: manifest row has no final_tcp_offset")
            return

        kind = str(row.get("task_kind", ""))
        trial_key = row.get("trial_key")
        tb = row.get("task_board")
        pose_ok = (
            isinstance(trial_key, str)
            and isinstance(tb, dict)
            and isinstance(tb.get("pose"), dict)
            and all(k in tb["pose"] for k in ("x", "y", "z", "roll", "pitch", "yaw"))
        )

        if not pose_ok:
            logger.warning(
                "TaskPresetMiddlePoseTransform: manifest row missing trial_key or full task_board.pose; "
                "skipping second move"
            )
            return

        from vision.task_preset_middle_pose_transform import (
            InitialTcpPose,
            TaskPresetMiddlePoseTransform as VisionTcpPreset,
            TaskPresetMiddlePoseTransformInput as VisionTcpPresetIn,
            final_tcp_pose_from_initial_and_offset,
        )

        orient = _orientation_label_to_enum(inp.task_board_orientation)
        try:
            mp_out = VisionTcpPreset.apply(
                VisionTcpPresetIn(
                    trial_key=str(trial_key),
                    manifest_path=default_task_manifest_path(),
                    task_board_cx_m=float(inp.board_cx_m),
                    task_board_cy_m=float(inp.board_cy_m),
                    task_board_yaw_rad=float(inp.board_yaw_rad),
                    orientation=orient,
                    initial_tcp_pose=InitialTcpPose(
                        xyz_m=tuple(float(x) for x in anchor_pos_m),
                        quat_xyzw=tuple(float(x) for x in anchor_quat_xyzw),
                    ),
                )
            )
            p_goal, q_goal = final_tcp_pose_from_initial_and_offset(
                np.asarray(anchor_pos_m, dtype=np.float64),
                np.asarray(anchor_quat_xyzw, dtype=np.float64),
                mp_out.tcp_offset.as_manifest_dict(),
            )
            dz_lin = tcp_manifest_linear_z_extra_m(kind)
            if dz_lin != 0.0:
                p_goal = np.asarray(p_goal, dtype=np.float64).copy()
                p_goal[2] += float(dz_lin)
            rec_pose = tb.get("pose") if isinstance(tb, dict) else None
            rec_ok = (
                isinstance(rec_pose, dict)
                and all(k in rec_pose for k in ("x", "y", "z", "roll", "pitch", "yaw"))
            )
            q_before_dyaw = np.asarray(q_goal, dtype=np.float64).copy()
            q_goal = quat_goal_xyzw_with_live_board_yaw_delta(
                q_before_dyaw,
                live_board_yaw_rad=float(inp.board_yaw_rad),
                recorded_task_board_pose=rec_pose if rec_ok else None,
            )
            dyaw_applied = live_board_yaw_minus_reference_rad(
                float(inp.board_yaw_rad),
                rec_pose if rec_ok else None,
            )
            pose = Pose(
                position=Point(x=float(p_goal[0]), y=float(p_goal[1]), z=float(p_goal[2])),
                orientation=Quaternion(
                    x=float(q_goal[0]),
                    y=float(q_goal[1]),
                    z=float(q_goal[2]),
                    w=float(q_goal[3]),
                ),
            )
            ctrl_target: dict[str, Any] = {
                "vision_tcp_preset": True,
                "preset_board_cx_m": mp_out.preset_board_cx_m,
                "preset_board_cy_m": mp_out.preset_board_cy_m,
                "preset_board_yaw_rad": mp_out.preset_board_yaw_rad,
                "fitted_board_yaw_effective_rad": mp_out.fitted_board_yaw_effective_rad,
                "tcp_offset_linear": mp_out.tcp_offset.as_manifest_dict()["linear"],
                "tcp_offset_angular": mp_out.tcp_offset.as_manifest_dict()["angular"],
                "manifest_linear_z_extra_m": float(dz_lin),
                "board_dyaw_applied_rad": float(dyaw_applied),
            }
            logger.info(
                "TaskPresetMiddlePoseTransform: vision manifest remap "
                f"preset_board=({mp_out.preset_board_cx_m:.6f},{mp_out.preset_board_cy_m:.6f},"
                f"yaw={mp_out.preset_board_yaw_rad:.6f}) effective_fitted_yaw={mp_out.fitted_board_yaw_effective_rad:.6f}"
                + (f" manifest_linear_z_extra_m={dz_lin:.6f}" if dz_lin != 0.0 else "")
                + f" board_dyaw_applied_rad={dyaw_applied:.6f}"
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                f"TaskPresetMiddlePoseTransform: vision remap failed ({exc!r}); skipping second move"
            )
            return

        logger.info(f"TaskPresetMiddlePoseTransform: TCP control target {json.dumps(ctrl_target, sort_keys=True)}")
        logger.info(
            "TaskPresetMiddlePoseTransform: commanded TCP pose "
            f"xyz=({pose.position.x:.6f},{pose.position.y:.6f},{pose.position.z:.6f}) "
            f"quat_xyzw=({pose.orientation.x:.6f},{pose.orientation.y:.6f},"
            f"{pose.orientation.z:.6f},{pose.orientation.w:.6f})"
        )

        mover = TaskBoardCenterMove()
        settle_out = mover.apply_target_pose(
            TaskBoardPoseMoveInput(
                policy=inp.policy,
                move_robot=inp.move_robot,
                tf_buffer=inp.policy._parent_node._tf_buffer,
                target_pose=pose,
                sim_cycle_sec=inp.sim_cycle_sec,
            )
        )
        o = settle_out
        target_yaw = float(
            Rotation.from_quat(
                (o.target_quat_x, o.target_quat_y, o.target_quat_z, o.target_quat_w)
            ).as_euler("xyz")[2]
        )
        yaw_err = wrap_angle_rad(o.actual_tcp_yaw_rad - target_yaw)
        offset_suffix = episode_start_tcp_delta_log_suffix(
            actual_tcp_x_m=o.actual_tcp_x_m,
            actual_tcp_y_m=o.actual_tcp_y_m,
            actual_tcp_z_m=o.actual_tcp_z_m,
            actual_tcp_quat_x=o.actual_tcp_quat_x,
            actual_tcp_quat_y=o.actual_tcp_quat_y,
            actual_tcp_quat_z=o.actual_tcp_quat_z,
            actual_tcp_quat_w=o.actual_tcp_quat_w,
            episode_start_tcp_xyz_m=anchor_pos_m,
            episode_start_tcp_quat_xyzw=anchor_quat_xyzw,
        )
        logger.info(
            "TaskBoardPolicy: second action move (pre-insert) "
            f"target_xyz=({o.target_x_m:.6f},{o.target_y_m:.6f},{o.target_z_m:.6f}) "
            f"actual_xyz=({o.actual_tcp_x_m:.6f},{o.actual_tcp_y_m:.6f},{o.actual_tcp_z_m:.6f}) "
            f"actual_quat_xyzw=({o.actual_tcp_quat_x:.6f},{o.actual_tcp_quat_y:.6f},"
            f"{o.actual_tcp_quat_z:.6f},{o.actual_tcp_quat_w:.6f}) "
            f"actual_yaw_rad={o.actual_tcp_yaw_rad:.6f} "
            f"{offset_suffix}"
            f"err_xyz=({o.dx_m:.6f},{o.dy_m:.6f},{o.dz_m:.6f}) yaw_err={yaw_err:.6f} "
            f"orient_err_rad={o.orientation_error_rad:.6f} "
            f"settle_elapsed_s={o.settle_elapsed_sec:.3f} settle_iters={o.settle_iterations}"
        )
