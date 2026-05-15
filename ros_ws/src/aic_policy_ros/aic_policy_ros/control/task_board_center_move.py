#
# SPDX-License-Identifier: Apache-2.0
#

"""§2.2 — move gripper TCP to vision rectangle pose at initial height; poll pose error until within tolerance or timeout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from scipy.spatial.transform import Rotation, Slerp

from aic_model.policy import MoveRobotCallback, Policy


def _wrap_angle_rad(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def wrap_angle_rad(a: float) -> float:
    """Public alias for :func:`_wrap_angle_rad` (second-action settle log vs target yaw)."""

    return _wrap_angle_rad(a)


def _position_error_norm_m(dx_m: float, dy_m: float, dz_m: float) -> float:
    return math.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)


def _pose_within_tolerance(
    *,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    yaw_error_rad: float,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> bool:
    return (
        _position_error_norm_m(dx_m, dy_m, dz_m) <= position_tolerance_m
        and abs(yaw_error_rad) <= yaw_tolerance_rad
    )


def _pose_position_and_orientation_within_tolerance(
    *,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    orientation_error_rad: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> bool:
    return (
        _position_error_norm_m(dx_m, dy_m, dz_m) <= position_tolerance_m
        and abs(orientation_error_rad) <= orientation_tolerance_rad
    )


def episode_start_tcp_delta_log_suffix(
    *,
    actual_tcp_x_m: float,
    actual_tcp_y_m: float,
    actual_tcp_z_m: float,
    actual_tcp_quat_x: float,
    actual_tcp_quat_y: float,
    actual_tcp_quat_z: float,
    actual_tcp_quat_w: float,
    episode_start_tcp_xyz_m: tuple[float, float, float] | None,
    episode_start_tcp_quat_xyzw: tuple[float, float, float, float] | None,
) -> str:
    """Suffix used by centering and second-action settle logs (TCP delta vs episode-start pose)."""

    if episode_start_tcp_xyz_m is None or episode_start_tcp_quat_xyzw is None:
        return ""
    sx, sy, sz = episode_start_tcp_xyz_m
    sqx, sqy, sqz, sqw = episode_start_tcp_quat_xyzw
    ddx = actual_tcp_x_m - sx
    ddy = actual_tcp_y_m - sy
    ddz = actual_tcp_z_m - sz
    r0 = Rotation.from_quat((sqx, sqy, sqz, sqw))
    r1 = Rotation.from_quat(
        (actual_tcp_quat_x, actual_tcp_quat_y, actual_tcp_quat_z, actual_tcp_quat_w)
    )
    r_rel = r1 * r0.inv()
    qrx, qry, qrz, qrw = r_rel.as_quat()
    d_ang = float(r_rel.magnitude())
    return (
        f"tcp_delta_vs_episode_start_m=({ddx:.6f},{ddy:.6f},{ddz:.6f}) "
        f"tcp_delta_rot_quat_xyzw=({qrx:.6f},{qry:.6f},{qrz:.6f},{qrw:.6f}) "
        f"tcp_delta_rot_angle_rad={d_ang:.6f} "
    )


def _gripper_tcp_pose_base_link(
    tf_buffer: Any,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return ``x, y, z, qx, qy, qz, qw, yaw_rad`` for ``gripper/tcp`` in ``base_link`` (yaw = euler-z ``xyz``)."""

    tf = tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
    t = tf.transform.translation
    r = tf.transform.rotation
    qx = float(r.x)
    qy = float(r.y)
    qz = float(r.z)
    qw = float(r.w)
    yaw = float(Rotation.from_quat((qx, qy, qz, qw)).as_euler("xyz")[2])
    return (float(t.x), float(t.y), float(t.z), qx, qy, qz, qw, yaw)


def read_gripper_tcp_pose_base_link(
    tf_buffer: Any,
) -> tuple[float, float, float, float, float, float, float, float]:
    """TCP pose in ``base_link`` for ``gripper/tcp`` (same tuple as :func:`_gripper_tcp_pose_base_link`)."""

    return _gripper_tcp_pose_base_link(tf_buffer)


def _blend_pose_toward_target(
    cur_xyz: tuple[float, float, float],
    cur_quat_xyzw: tuple[float, float, float, float],
    target: Pose,
    alpha: float,
) -> Pose:
    """Linear blend in position and SLERP in orientation (``alpha`` in ``[0, 1]``).

    Same convention as ``ground_truth/ros/GroundTruth.py`` ``calc_gripper_pose``:
    ``position_fraction`` / ``slerp_fraction`` toward a fixed target from the current TCP.
    """

    a = min(1.0, max(0.0, float(alpha)))
    tx = float(target.position.x)
    ty = float(target.position.y)
    tz = float(target.position.z)
    tqx = float(target.orientation.x)
    tqy = float(target.orientation.y)
    tqz = float(target.orientation.z)
    tqw = float(target.orientation.w)
    p0 = np.asarray(cur_xyz, dtype=np.float64)
    p1 = np.array([tx, ty, tz], dtype=np.float64)
    p = (1.0 - a) * p0 + a * p1
    r0 = Rotation.from_quat(np.asarray(cur_quat_xyzw, dtype=np.float64))
    r1 = Rotation.from_quat(np.array([tqx, tqy, tqz, tqw], dtype=np.float64))
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
    q = slerp([a]).as_quat()[0]
    return Pose(
        position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
        orientation=Quaternion(
            x=float(q[0]),
            y=float(q[1]),
            z=float(q[2]),
            w=float(q[3]),
        ),
    )


def _pose_translate_azimuth(
    tf_buffer: Any,
    *,
    cx_m: float,
    cy_m: float,
    z_m: float,
    azimuth_rad: float,
) -> Pose:
    """TCP target: translate to ``(cx, cy, z)``, keep current roll/pitch, set yaw to azimuth (horizontal plane)."""

    tf = tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
    q = (
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    )
    roll, pitch, _ = Rotation.from_quat(q).as_euler("xyz")
    q_cmd = Rotation.from_euler(
        "xyz",
        [float(roll), float(pitch), float(azimuth_rad)],
        degrees=False,
    ).as_quat()
    return Pose(
        position=Point(x=float(cx_m), y=float(cy_m), z=float(z_m)),
        orientation=Quaternion(
            x=float(q_cmd[0]),
            y=float(q_cmd[1]),
            z=float(q_cmd[2]),
            w=float(q_cmd[3]),
        ),
    )


@dataclass
class TaskBoardCenterMoveInput:
    """Target rectangle pose in ``base_link`` and motion context."""

    policy: Policy
    move_robot: MoveRobotCallback
    tf_buffer: Any
    target_cx_m: float
    target_cy_m: float
    target_yaw_rad: float
    initialize_tcp_z_m: float
    sim_cycle_sec: float
    max_settle_sec: float = 5.0
    position_tolerance_m: float = 0.01
    yaw_tolerance_rad: float = 0.05


@dataclass(frozen=True)
class TaskBoardCenterMoveOutput:
    """TCP pose error in ``base_link`` after polling (see ``settle_*``)."""

    target_cx_m: float
    target_cy_m: float
    target_yaw_rad: float
    actual_tcp_x_m: float
    actual_tcp_y_m: float
    actual_tcp_z_m: float
    actual_tcp_quat_x: float
    actual_tcp_quat_y: float
    actual_tcp_quat_z: float
    actual_tcp_quat_w: float
    actual_tcp_yaw_rad: float
    dx_m: float
    dy_m: float
    dz_m: float
    yaw_error_rad: float
    settle_elapsed_sec: float = 0.0
    settle_iterations: int = 0


@dataclass
class TaskBoardPoseMoveInput:
    """Full TCP target in ``base_link`` and motion context (second-action style pre-insert pose)."""

    policy: Policy
    move_robot: MoveRobotCallback
    tf_buffer: Any
    target_pose: Pose
    sim_cycle_sec: float
    max_settle_sec: float = 5.0
    position_tolerance_m: float = 0.01
    orientation_tolerance_rad: float = 0.05


@dataclass(frozen=True)
class TaskBoardPoseMoveOutput:
    """TCP vs commanded pose in ``base_link`` after polling (same settle fields as :class:`TaskBoardCenterMoveOutput`)."""

    target_x_m: float
    target_y_m: float
    target_z_m: float
    target_quat_x: float
    target_quat_y: float
    target_quat_z: float
    target_quat_w: float
    actual_tcp_x_m: float
    actual_tcp_y_m: float
    actual_tcp_z_m: float
    actual_tcp_quat_x: float
    actual_tcp_quat_y: float
    actual_tcp_quat_z: float
    actual_tcp_quat_w: float
    actual_tcp_yaw_rad: float
    dx_m: float
    dy_m: float
    dz_m: float
    orientation_error_rad: float
    settle_elapsed_sec: float = 0.0
    settle_iterations: int = 0


class TaskBoardCenterMove:
    """Send Cartesian pose target for TCP at ``(cx, cy, z_init)`` with vision azimuth; poll until within tolerance."""

    def __init__(
        self,
        *,
        stiffness: Sequence[float] = (360.0, 360.0, 360.0, 200.0, 200.0, 200.0),
        damping: Sequence[float] = (100.0, 100.0, 100.0, 40.0, 40.0, 40.0),
        approach_interp_steps: int = 100,
        approach_interp_sleep_sec: float = 0.05,
    ) -> None:
        # Match ``GroundTruth`` admittance defaults in ``ai-industry-challenge`` / ``ground_truth/ros/GroundTruth.py``.
        self._stiffness = list(stiffness)
        self._damping = list(damping)
        # Match ``GroundTruth.insert_cable`` approach ramp: 100 commands × 0.05 s (see ``calc_gripper_pose``).
        self._approach_interp_steps = int(approach_interp_steps)
        self._approach_interp_sleep_sec = float(approach_interp_sleep_sec)

    def _approach_pose_target(
        self,
        policy: Policy,
        move_robot: MoveRobotCallback,
        tf_buffer: Any,
        pose_target: Pose,
    ) -> None:
        """Blend from live TCP toward ``pose_target`` (LERP xyz + SLERP quat), like ``GroundTruth``."""

        n = self._approach_interp_steps
        if n <= 0:
            policy.set_pose_target(
                move_robot,
                pose_target,
                stiffness=self._stiffness,
                damping=self._damping,
            )
            return
        sleep_s = self._approach_interp_sleep_sec
        for step in range(n):
            alpha = float(step + 1) / float(n)
            ax, ay, az, aqx, aqy, aqz, aqw, _ = _gripper_tcp_pose_base_link(tf_buffer)
            cmd = _blend_pose_toward_target(
                (ax, ay, az),
                (aqx, aqy, aqz, aqw),
                pose_target,
                alpha,
            )
            policy.set_pose_target(
                move_robot,
                cmd,
                stiffness=self._stiffness,
                damping=self._damping,
            )
            if sleep_s > 0.0:
                policy.sleep_for(sleep_s)

    def apply(self, inp: TaskBoardCenterMoveInput) -> TaskBoardCenterMoveOutput:
        pose = _pose_translate_azimuth(
            inp.tf_buffer,
            cx_m=inp.target_cx_m,
            cy_m=inp.target_cy_m,
            z_m=inp.initialize_tcp_z_m,
            azimuth_rad=inp.target_yaw_rad,
        )
        self._approach_pose_target(inp.policy, inp.move_robot, inp.tf_buffer, pose)

        dt = float(inp.sim_cycle_sec)
        if dt <= 0.0:
            raise ValueError("TaskBoardCenterMoveInput.sim_cycle_sec must be positive")
        max_t = float(inp.max_settle_sec)
        if max_t <= 0.0:
            raise ValueError("TaskBoardCenterMoveInput.max_settle_sec must be positive")

        elapsed = 0.0
        iterations = 0
        ax = ay = az = float("nan")
        aqx = aqy = aqz = aqw = ayaw = float("nan")
        yaw_err = float("nan")
        dx = dy = dz = float("nan")

        while elapsed < max_t - 1e-9:
            step = min(dt, max_t - elapsed)
            inp.policy.sleep_for(step)
            elapsed += step
            iterations += 1
            ax, ay, az, aqx, aqy, aqz, aqw, ayaw = _gripper_tcp_pose_base_link(inp.tf_buffer)

            yaw_err = _wrap_angle_rad(ayaw - inp.target_yaw_rad)
            dx = ax - inp.target_cx_m
            dy = ay - inp.target_cy_m
            dz = az - inp.initialize_tcp_z_m
            if _pose_within_tolerance(
                dx_m=dx,
                dy_m=dy,
                dz_m=dz,
                yaw_error_rad=yaw_err,
                position_tolerance_m=inp.position_tolerance_m,
                yaw_tolerance_rad=inp.yaw_tolerance_rad,
            ):
                break

        return TaskBoardCenterMoveOutput(
            target_cx_m=inp.target_cx_m,
            target_cy_m=inp.target_cy_m,
            target_yaw_rad=inp.target_yaw_rad,
            actual_tcp_x_m=ax,
            actual_tcp_y_m=ay,
            actual_tcp_z_m=az,
            actual_tcp_quat_x=aqx,
            actual_tcp_quat_y=aqy,
            actual_tcp_quat_z=aqz,
            actual_tcp_quat_w=aqw,
            actual_tcp_yaw_rad=ayaw,
            dx_m=dx,
            dy_m=dy,
            dz_m=dz,
            yaw_error_rad=yaw_err,
            settle_elapsed_sec=elapsed,
            settle_iterations=iterations,
        )

    def apply_target_pose(self, inp: TaskBoardPoseMoveInput) -> TaskBoardPoseMoveOutput:
        """Send ``target_pose`` for TCP, then poll TF until position and orientation errors are within tolerance."""

        tp = inp.target_pose
        tx = float(tp.position.x)
        ty = float(tp.position.y)
        tz = float(tp.position.z)
        tqx = float(tp.orientation.x)
        tqy = float(tp.orientation.y)
        tqz = float(tp.orientation.z)
        tqw = float(tp.orientation.w)
        r_tgt = Rotation.from_quat((tqx, tqy, tqz, tqw))

        self._approach_pose_target(inp.policy, inp.move_robot, inp.tf_buffer, tp)

        dt = float(inp.sim_cycle_sec)
        if dt <= 0.0:
            raise ValueError("TaskBoardPoseMoveInput.sim_cycle_sec must be positive")
        max_t = float(inp.max_settle_sec)
        if max_t <= 0.0:
            raise ValueError("TaskBoardPoseMoveInput.max_settle_sec must be positive")

        elapsed = 0.0
        iterations = 0
        ax = ay = az = aqx = aqy = aqz = aqw = ayaw = float("nan")
        orient_err = float("nan")
        dx = dy = dz = float("nan")

        while elapsed < max_t - 1e-9:
            step = min(dt, max_t - elapsed)
            inp.policy.sleep_for(step)
            elapsed += step
            iterations += 1
            ax, ay, az, aqx, aqy, aqz, aqw, ayaw = _gripper_tcp_pose_base_link(inp.tf_buffer)
            r_act = Rotation.from_quat((aqx, aqy, aqz, aqw))
            orient_err = float((r_act * r_tgt.inv()).magnitude())
            dx = ax - tx
            dy = ay - ty
            dz = az - tz
            if _pose_position_and_orientation_within_tolerance(
                dx_m=dx,
                dy_m=dy,
                dz_m=dz,
                orientation_error_rad=orient_err,
                position_tolerance_m=inp.position_tolerance_m,
                orientation_tolerance_rad=inp.orientation_tolerance_rad,
            ):
                break

        return TaskBoardPoseMoveOutput(
            target_x_m=tx,
            target_y_m=ty,
            target_z_m=tz,
            target_quat_x=tqx,
            target_quat_y=tqy,
            target_quat_z=tqz,
            target_quat_w=tqw,
            actual_tcp_x_m=ax,
            actual_tcp_y_m=ay,
            actual_tcp_z_m=az,
            actual_tcp_quat_x=aqx,
            actual_tcp_quat_y=aqy,
            actual_tcp_quat_z=aqz,
            actual_tcp_quat_w=aqw,
            actual_tcp_yaw_rad=ayaw,
            dx_m=dx,
            dy_m=dy,
            dz_m=dz,
            orientation_error_rad=orient_err,
            settle_elapsed_sec=elapsed,
            settle_iterations=iterations,
        )
