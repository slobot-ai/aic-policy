#
# SPDX-License-Identifier: Apache-2.0
#

"""Vision policy for ``insert_cable``: first observation, rectangle fit, centering move, second observation."""

from __future__ import annotations

import math

from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_task_interfaces.msg import Task

from aic_policy_ros.control import TaskBoardControl
from aic_policy_ros.task_board_rectangle_estimation import rectangle_candidate_highest_mean_hard_iou
from aic_policy_ros.task_board_vision import TaskBoardVision


class TaskBoardPolicy(Policy):
    def __init__(self, parent_node) -> None:
        super().__init__(parent_node)
        self._parent_node.declare_parameter("episode_id", 0)
        self.last_task_board_orientation: str | None = None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        del kwargs, send_feedback

        episode_id = int(self._parent_node.get_parameter("episode_id").value)

        tf_buf = getattr(self._parent_node, "_tf_buffer", None)
        task_board_vision = TaskBoardVision(
            self,
            get_observation,
            episode_id,
            tf_buffer=tf_buf,
        )
        rectangle_out = task_board_vision.first_observation()

        task_board_control = TaskBoardControl(self.get_logger(), move_robot, self)
        task_board_control.first_action(
            rectangle_out=rectangle_out,
        )

        task_board_vision.second_observation(
            task=task,
        )
        self.last_task_board_orientation = task_board_vision.last_task_board_orientation

        if (
            task_board_vision.last_planar_board_cx_m is not None
            and task_board_vision.last_planar_board_cy_m is not None
            and task_board_vision.last_planar_board_yaw_rad is not None
        ):
            task_board_control.second_action(
                task_board_orientation=self.last_task_board_orientation or "UNKNOWN",
                board_cx_m=task_board_vision.last_planar_board_cx_m,
                board_cy_m=task_board_vision.last_planar_board_cy_m,
                board_yaw_rad=task_board_vision.last_planar_board_yaw_rad,
                task=task,
                episode_id=episode_id,
            )
        else:
            self.get_logger().warning("TaskBoardPolicy: skipping second_action (no planar board pose)")
        task_board_vision.third_observation()
        sc_rect = task_board_vision.last_sc_port_rectangle_estimation
        if sc_rect is not None:
            best, _idx = rectangle_candidate_highest_mean_hard_iou(sc_rect)
            if math.isfinite(best.cx_m) and math.isfinite(best.cy_m):
                task_board_control.third_action(port_cx_m=best.cx_m, port_cy_m=best.cy_m)
                return True
        self.get_logger().warning("TaskBoardPolicy: skipping third_action (no finite SC port rectangle fit)")
        return True
