#
# SPDX-License-Identifier: Apache-2.0
#

"""§2.2 — Cartesian move using the rectangle candidate with the highest ``mean_hard_iou``."""

from __future__ import annotations

from typing import Any

from aic_model.policy import MoveRobotCallback, Policy

from aic_policy_ros.task_board_rectangle_estimation import (
    TaskBoardRectangleEstimationOutput,
    rectangle_candidate_highest_mean_hard_iou,
)

from .task_board_center_move import (
    TaskBoardCenterMove,
    TaskBoardCenterMoveInput,
    TaskBoardCenterMoveOutput,
    episode_start_tcp_delta_log_suffix,
)


class CartesianPositionMove:
    """Command the arm toward the best-IoU multiview rectangle candidate."""

    def __init__(self, logger: Any, *, sim_cycle_sec: float = 0.25) -> None:
        self._logger = logger
        self._sim_cycle_sec = sim_cycle_sec

    def apply_rectangle_candidates(
        self,
        *,
        policy: Policy,
        rectangle_out: TaskBoardRectangleEstimationOutput,
        move_robot: MoveRobotCallback,
        initialize_tcp_z_m: float,
        episode_start_tcp_xyz_m: tuple[float, float, float] | None = None,
        episode_start_tcp_quat_xyzw: tuple[float, float, float, float] | None = None,
    ) -> tuple[TaskBoardCenterMoveOutput, ...]:
        mover = TaskBoardCenterMove()
        buf = policy._parent_node._tf_buffer
        if len(rectangle_out.candidates) < 2:
            raise ValueError("TaskBoardRectangleEstimationOutput must contain two candidates")
        cand, cand_idx = rectangle_candidate_highest_mean_hard_iou(rectangle_out)
        out = mover.apply(
            TaskBoardCenterMoveInput(
                policy=policy,
                move_robot=move_robot,
                tf_buffer=buf,
                target_cx_m=cand.cx_m,
                target_cy_m=cand.cy_m,
                target_yaw_rad=cand.yaw_rad,
                initialize_tcp_z_m=initialize_tcp_z_m,
                sim_cycle_sec=self._sim_cycle_sec,
            )
        )
        offset_suffix = episode_start_tcp_delta_log_suffix(
            actual_tcp_x_m=out.actual_tcp_x_m,
            actual_tcp_y_m=out.actual_tcp_y_m,
            actual_tcp_z_m=out.actual_tcp_z_m,
            actual_tcp_quat_x=out.actual_tcp_quat_x,
            actual_tcp_quat_y=out.actual_tcp_quat_y,
            actual_tcp_quat_z=out.actual_tcp_quat_z,
            actual_tcp_quat_w=out.actual_tcp_quat_w,
            episode_start_tcp_xyz_m=episode_start_tcp_xyz_m,
            episode_start_tcp_quat_xyzw=episode_start_tcp_quat_xyzw,
        )
        self._logger.info(
            "TaskBoardPolicy: center move (best mean_hard_iou) "
            f"candidate[{cand_idx}] mean_hard_iou={cand.mean_hard_iou:.6f} "
            f"seed_y={cand.yaw_seed_rad:.6f} target_yaw_rad={out.target_yaw_rad:.6f} "
            f"target=({out.target_cx_m:.6f},{out.target_cy_m:.6f}) "
            f"actual_xyz=({out.actual_tcp_x_m:.6f},{out.actual_tcp_y_m:.6f},{out.actual_tcp_z_m:.6f}) "
            f"actual_quat_xyzw=({out.actual_tcp_quat_x:.6f},{out.actual_tcp_quat_y:.6f},"
            f"{out.actual_tcp_quat_z:.6f},{out.actual_tcp_quat_w:.6f}) "
            f"actual_yaw_rad={out.actual_tcp_yaw_rad:.6f} "
            f"{offset_suffix}"
            f"err_xyz=({out.dx_m:.6f},{out.dy_m:.6f},{out.dz_m:.6f}) yaw_err={out.yaw_error_rad:.6f} "
            f"settle_elapsed_s={out.settle_elapsed_sec:.3f} settle_iters={out.settle_iterations}"
        )
        return (out,)
