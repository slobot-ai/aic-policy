#
# SPDX-License-Identifier: Apache-2.0
#

"""Multiview ``T_cam_from_base`` from ``tf2_ros.Buffer`` and :class:`aic_model_interfaces.msg.Observation`."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as ScipyRotation

_CAM_KEYS: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")


def _lookup_time_from_observation(observation: Any) -> Any:
    from rclpy.time import Time

    st = observation.left_image.header.stamp
    if int(st.sec) == 0 and int(st.nanosec) == 0:
        return Time()
    return Time.from_msg(st)


def transform_stamped_to_T_target_from_source(tf_msg: Any) -> NDArray[np.float64]:
    """Homogeneous ``(4, 4)`` with ``p_target = T @ p_source`` (``tf2`` :meth:`lookup_transform` convention)."""

    t = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    Rm = ScipyRotation.from_quat((float(r.x), float(r.y), float(r.z), float(r.w))).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[0, 3] = float(t.x)
    T[1, 3] = float(t.y)
    T[2, 3] = float(t.z)
    return T


def transform_stamped_from_homogeneous(
    *,
    target_frame: str,
    source_frame: str,
    T: NDArray[np.floating],
) -> Any:
    """Build ``geometry_msgs/TransformStamped`` for ``p_target = T @ p_source`` (tests, tooling)."""

    from geometry_msgs.msg import TransformStamped

    Tm = np.asarray(T, dtype=np.float64).reshape(4, 4)
    Rm = Tm[:3, :3]
    tvec = Tm[:3, 3]
    quat = ScipyRotation.from_matrix(Rm).as_quat()
    ts = TransformStamped()
    ts.header.frame_id = str(target_frame)
    ts.child_frame_id = str(source_frame)
    ts.transform.translation.x = float(tvec[0])
    ts.transform.translation.y = float(tvec[1])
    ts.transform.translation.z = float(tvec[2])
    ts.transform.rotation.x = float(quat[0])
    ts.transform.rotation.y = float(quat[1])
    ts.transform.rotation.z = float(quat[2])
    ts.transform.rotation.w = float(quat[3])
    return ts


def t_cam_from_base_by_camera_from_tf_buffer(
    tf_buffer: Any,
    observation: Any,
    logger: Any,
    *,
    base_frame: str = "base_link",
    tf_lookup_timeout_sec: float = 0.5,
) -> dict[str, NDArray[np.float64]] | None:
    """Return ``{left_camera, center_camera, right_camera} -> T_cam_from_base`` from TF, or ``None`` on failure.

    Uses each ``sensor_msgs/CameraInfo.header.frame_id`` as the camera / optical link resolved in TF
    (same convention as image projection). ``lookup_transform(camera_frame, base_frame, …)`` yields
    ``T_cam_from_base`` with ``p_cam_homogeneous = T @ p_base_homogeneous``.
    """

    import tf2_ros
    from rclpy.duration import Duration

    infos = (observation.left_camera_info, observation.center_camera_info, observation.right_camera_info)
    frames: list[str] = []
    for info in infos:
        fid = str(info.header.frame_id).strip()
        if not fid:
            logger.warning(
                "TaskBoardVision: empty CameraInfo.header.frame_id for one camera; "
                "cannot resolve TF extrinsics"
            )
            return None
        frames.append(fid)

    lookup_time = _lookup_time_from_observation(observation)
    timeout = Duration(seconds=float(tf_lookup_timeout_sec))
    out: dict[str, NDArray[np.float64]] = {}
    for cam_key, cam_frame in zip(_CAM_KEYS, frames, strict=True):
        try:
            tf_msg = tf_buffer.lookup_transform(cam_frame, base_frame, lookup_time, timeout=timeout)
        except tf2_ros.TransformException as exc:
            logger.warning(
                f"TaskBoardVision: TF lookup_transform({cam_frame!r}, {base_frame!r}) failed ({exc}); "
                "using static extrinsics fallback"
            )
            return None
        out[cam_key] = transform_stamped_to_T_target_from_source(tf_msg)
    logger.info("TaskBoardVision: multiview T_cam_from_base resolved from tf2_ros.Buffer")
    return out
