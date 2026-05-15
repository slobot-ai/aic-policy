#
# SPDX-License-Identifier: Apache-2.0
#

"""Section 2 task-board control: ``TaskBoardControl`` runs TCP lookup then rectangle centering."""

from __future__ import annotations

from typing import Any

from aic_task_interfaces.msg import Task
from scipy.spatial.transform import Rotation

from aic_model.policy import MoveRobotCallback, Policy

from aic_policy_ros.task_board_rectangle_estimation import TaskBoardRectangleEstimationOutput

from .cartesian_position_move import CartesianPositionMove
from .link_transform import LinkTransform, LinkTransformInput
from .task_board_center_move import (
    TaskBoardCenterMove,
    TaskBoardCenterMoveInput,
    TaskBoardCenterMoveOutput,
    read_gripper_tcp_pose_base_link,
)
from .task_preset_middle_pose_transform import (
    TaskPresetMiddlePoseTransform,
    TaskPresetMiddlePoseTransformInput,
)


def task_board_center(logger: Any, *, sim_cycle_sec: float = 0.25) -> CartesianPositionMove:
    """Construct the Cartesian move helper used after rectangle estimation (2.2)."""

    return CartesianPositionMove(logger, sim_cycle_sec=sim_cycle_sec)


class TaskBoardControl:
    """Delegates to ``CartesianPositionMove`` for rectangle-based centering after TCP lookup."""

    @property
    def tcp_episode_start_pos_m(self) -> tuple[float, float, float] | None:
        return self._tcp_episode_start_pos_m

    @property
    def tcp_episode_start_quat_xyzw(self) -> tuple[float, float, float, float] | None:
        return self._tcp_episode_start_quat_xyzw

    def __init__(
        self,
        logger: Any,
        move_robot: MoveRobotCallback,
        policy: Policy,
        *,
        sim_cycle_sec: float = 0.25,
    ) -> None:
        self._cartesian = task_board_center(logger, sim_cycle_sec=sim_cycle_sec)
        self._move_robot = move_robot
        self._policy = policy
        self._tcp_episode_start_pos_m: tuple[float, float, float] | None = None
        self._tcp_episode_start_quat_xyzw: tuple[float, float, float, float] | None = None

    def first_action(
        self,
        *,
        rectangle_out: TaskBoardRectangleEstimationOutput,
    ) -> tuple[TaskBoardCenterMoveOutput, ...]:
        tcp0 = LinkTransform().apply(LinkTransformInput(tf_buffer=self._policy._parent_node._tf_buffer))
        self._tcp_episode_start_pos_m = (
            float(tcp0.translation_x_m),
            float(tcp0.translation_y_m),
            float(tcp0.translation_z_m),
        )
        self._tcp_episode_start_quat_xyzw = (
            float(tcp0.rotation_x),
            float(tcp0.rotation_y),
            float(tcp0.rotation_z),
            float(tcp0.rotation_w),
        )
        r_roll, r_pitch, r_yaw = Rotation.from_quat(
            (
                tcp0.rotation_x,
                tcp0.rotation_y,
                tcp0.rotation_z,
                tcp0.rotation_w,
            )
        ).as_euler("xyz")
        self._cartesian._logger.info(
            "TaskBoardControl.first_action: initial TCP pose in base_link (gripper/tcp) "
            f"xyz_m=({tcp0.translation_x_m:.6f},{tcp0.translation_y_m:.6f},{tcp0.translation_z_m:.6f}) "
            f"quat_xyzw=({tcp0.rotation_x:.6f},{tcp0.rotation_y:.6f},{tcp0.rotation_z:.6f},{tcp0.rotation_w:.6f}) "
            f"rpy_rad_xyz=({r_roll:.6f},{r_pitch:.6f},{r_yaw:.6f})"
        )
        return self.apply_rectangle_candidates(
            rectangle_out=rectangle_out,
            initialize_tcp_z_m=tcp0.translation_z_m,
            episode_start_tcp_xyz_m=self._tcp_episode_start_pos_m,
            episode_start_tcp_quat_xyzw=self._tcp_episode_start_quat_xyzw,
        )

    def apply_rectangle_candidates(
        self,
        *,
        rectangle_out: TaskBoardRectangleEstimationOutput,
        initialize_tcp_z_m: float,
        episode_start_tcp_xyz_m: tuple[float, float, float] | None = None,
        episode_start_tcp_quat_xyzw: tuple[float, float, float, float] | None = None,
    ) -> tuple[TaskBoardCenterMoveOutput, ...]:
        return self._cartesian.apply_rectangle_candidates(
            policy=self._policy,
            rectangle_out=rectangle_out,
            move_robot=self._move_robot,
            initialize_tcp_z_m=initialize_tcp_z_m,
            episode_start_tcp_xyz_m=episode_start_tcp_xyz_m,
            episode_start_tcp_quat_xyzw=episode_start_tcp_quat_xyzw,
        )

    def second_action(
        self,
        *,
        task_board_orientation: str,
        board_cx_m: float,
        board_cy_m: float,
        board_yaw_rad: float,
        task: Task,
        episode_id: int,
    ) -> None:
        """Delegates to :class:`TaskPresetMiddlePoseTransform` (manifest pre-insert pose vs board + episode TCP)."""

        TaskPresetMiddlePoseTransform().apply(
            TaskPresetMiddlePoseTransformInput(
                logger=self._cartesian._logger,
                policy=self._policy,
                move_robot=self._move_robot,
                sim_cycle_sec=self._cartesian._sim_cycle_sec,
                task_board_orientation=task_board_orientation,
                board_cx_m=board_cx_m,
                board_cy_m=board_cy_m,
                board_yaw_rad=board_yaw_rad,
                task=task,
                episode_id=episode_id,
            )
        )

    def third_action(self, *, port_cx_m: float, port_cy_m: float) -> TaskBoardCenterMoveOutput:
        """Move TCP over the SC port centroid in ``xy`` at the current height; keep current planar yaw."""

        buf = self._policy._parent_node._tf_buffer
        _x, _y, az, _qx, _qy, _qz, _qw, ayaw = read_gripper_tcp_pose_base_link(buf)
        mover = TaskBoardCenterMove()
        out = mover.apply(
            TaskBoardCenterMoveInput(
                policy=self._policy,
                move_robot=self._move_robot,
                tf_buffer=buf,
                target_cx_m=float(port_cx_m),
                target_cy_m=float(port_cy_m),
                target_yaw_rad=float(ayaw),
                initialize_tcp_z_m=float(az),
                sim_cycle_sec=self._cartesian._sim_cycle_sec,
            )
        )
        self._cartesian._logger.info(
            "TaskBoardControl.third_action: SC port merged centroid xy move "
            f"target=({port_cx_m:.6f},{port_cy_m:.6f}) "
            f"actual_xyz=({out.actual_tcp_x_m:.6f},{out.actual_tcp_y_m:.6f},{out.actual_tcp_z_m:.6f}) "
            f"err_xyz=({out.dx_m:.6f},{out.dy_m:.6f},{out.dz_m:.6f})"
        )
        return out
