#
# SPDX-License-Identifier: Apache-2.0
#

"""Manifest lookup and TCP goal pose for post-vision insertion (board pose + tcp_offset)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from aic_policy_ros.task_board_episode_geometry import task_board_top_face_center_base_m_from_pose_dict

# All manifest episodes were recorded with this task_board world pose (map grid generator).
REFERENCE_TASK_BOARD_XYZ_RPY = (0.15, -0.2, 1.14, 0.0, 0.0, 0.0)

_Z_EXTRA_SC_M = 0.02
# Extra +z on manifest linear offset for NIC / SFP pre-insert (lifts TCP vs raw dataset offset).
_Z_EXTRA_SFP_M = 0.07


def default_task_manifest_path() -> Path:
    """``share/aic_policy_ros/config/task_manifest.json`` when sourced; else package ``config/`` in-tree."""

    fallback = Path(__file__).resolve().parent.parent / "config" / "task_manifest.json"
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return fallback
    try:
        return Path(get_package_share_directory("aic_policy_ros")) / "config" / "task_manifest.json"
    except (LookupError, ValueError):
        return fallback


def load_task_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or default_task_manifest_path()
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_trailing_int(s: str, pattern: str) -> int | None:
    m = re.search(pattern, s, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def manifest_row_for_task(task: Any, episode_id: int, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match ``Task`` fields to a manifest row; fall back to ``trial_{episode_id}``."""

    plug = (getattr(task, "plug_type", None) or "").strip().lower()
    port_type = (getattr(task, "port_type", None) or "").strip().lower()
    port_name = (getattr(task, "port_name", None) or "").strip().lower()
    target = (getattr(task, "target_module_name", None) or "").strip().lower()

    def _match_sc(row: dict[str, Any]) -> bool:
        if row.get("task_kind") != "sc":
            return False
        si = row.get("sc_port_index")
        if si is None:
            return False
        idx = _parse_trailing_int(target, r"sc_port_(\d+)")
        if idx is None:
            return False
        return int(si) == idx

    def _match_nic(row: dict[str, Any]) -> bool:
        if row.get("task_kind") != "nic":
            return False
        ni = row.get("nic_card_index")
        pi = row.get("sfp_port_index")
        if ni is None or pi is None:
            return False
        card = _parse_trailing_int(target, r"nic_card(?:_mount)?_(\d+)")
        sfp = _parse_trailing_int(port_name, r"sfp_port_(\d+)")
        if card is None or sfp is None:
            return False
        return int(ni) == card and int(pi) == sfp

    is_sc = plug == "sc" or port_type == "sc" or "sc_port" in target
    matchers = (_match_sc,) if is_sc else (_match_nic,)
    for row in rows:
        for m in matchers:
            if m(row):
                return row

    key = f"trial_{int(episode_id)}"
    for row in rows:
        if row.get("trial_key") == key:
            return row
    return None


def initial_tcp_pose_from_manifest_row(
    row: dict[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Return ``(xyz_m, quat_xyzw)`` from ``row['initial_tcp_pose']`` if present and well-formed."""

    if row is None:
        return None
    block = row.get("initial_tcp_pose")
    if not isinstance(block, dict):
        return None
    xyz = block.get("xyz_m")
    quat = block.get("quat_xyzw")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        return None
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        return None
    try:
        pos = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        q = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    except (TypeError, ValueError):
        return None
    return pos, q


def _reference_board_pose_dict(recorded_task_board_pose: dict[str, Any] | None) -> dict[str, float]:
    if recorded_task_board_pose and all(
        k in recorded_task_board_pose for k in ("x", "y", "z", "roll", "pitch", "yaw")
    ):
        return {k: float(recorded_task_board_pose[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    xr, yr, zr, rr, pr, yw = REFERENCE_TASK_BOARD_XYZ_RPY
    return {"x": xr, "y": yr, "z": zr, "roll": rr, "pitch": pr, "yaw": yw}


def _wrap_angle_rad(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def tcp_manifest_linear_z_extra_m(task_kind: str) -> float:
    """Additive ``linear.z`` bump for ``final_tcp_offset`` (same rule as :func:`tcp_goal_pose_from_manifest_and_board`)."""

    k = (task_kind or "").strip().lower()
    if k == "sc":
        return float(_Z_EXTRA_SC_M)
    if k == "nic":
        return float(_Z_EXTRA_SFP_M)
    return 0.0


def live_board_yaw_minus_reference_rad(
    live_board_yaw_rad: float,
    recorded_task_board_pose: dict[str, Any] | None,
) -> float:
    """``wrap(live_board_yaw_rad - ref.yaw)`` using the same reference pose as the TCP goal helpers."""

    ref = _reference_board_pose_dict(recorded_task_board_pose)
    return _wrap_angle_rad(float(live_board_yaw_rad) - float(ref["yaw"]))


def quat_goal_xyzw_with_live_board_yaw_delta(
    quat_before_dyaw_xyzw: NDArray[np.float64],
    *,
    live_board_yaw_rad: float,
    recorded_task_board_pose: dict[str, Any] | None,
) -> NDArray[np.float64]:
    """Apply the same ``R_z(dyaw)`` as the legacy planar TCP goal in :func:`tcp_goal_pose_from_manifest_and_board`.

    The vision manifest xy remap does not rotate the TCP when the task board's **world yaw** changes
    vs the recording; this left-multiplies ``base_link`` z by ``dyaw = live_yaw - ref_yaw`` so the
    final gripper orientation tracks the fitted board like the legacy planar helper.
    """

    q = np.asarray(quat_before_dyaw_xyzw, dtype=np.float64).reshape(4)
    dyaw = live_board_yaw_minus_reference_rad(live_board_yaw_rad, recorded_task_board_pose)
    if abs(dyaw) < 1e-14:
        return q
    r_naive = Rotation.from_quat(q)
    r_goal = Rotation.from_euler("xyz", [0.0, 0.0, dyaw]) * r_naive
    return r_goal.as_quat()


def tcp_goal_pose_from_manifest_and_board(
    *,
    tcp_episode_start_position_m: NDArray[np.float64],
    tcp_episode_start_quat_xyzw: NDArray[np.float64],
    final_tcp_offset: dict[str, Any],
    task_kind: str,
    board_cx_m: float,
    board_cy_m: float,
    board_yaw_rad: float,
    board_z_m: float | None = None,
    recorded_task_board_pose: dict[str, Any] | None = None,
) -> Pose:
    """Apply dataset ``tcp_offset`` vs episode start, z lift by task kind, then **planar** board correction.

    The manifest was recorded with the board at :data:`REFERENCE_TASK_BOARD_XYZ_RPY` unless
    ``recorded_task_board_pose`` supplies an alternate ``scene.task_board.pose``. We first form the
    TCP pose in ``base_link`` as at recording (offset from episode-start TCP). We then **translate** that
    goal by the vision-measured board **top-face center** delta vs the pose-implied reference center, and
    **left-multiply** a world-frame yaw about ``base_link`` z by ``Δyaw`` vs the reference pose yaw.
    """

    lin = final_tcp_offset["linear"]
    ang = final_tcp_offset["angular"]
    lx0 = float(lin["x"])
    ly0 = float(lin["y"])
    lz0 = float(lin["z"])
    lx, ly, lz = lx0, ly0, lz0
    kind = (task_kind or "").strip().lower()
    if kind == "sc":
        lz += _Z_EXTRA_SC_M
    elif kind == "nic":
        lz += _Z_EXTRA_SFP_M

    ax = float(ang["x"])
    ay = float(ang["y"])
    az = float(ang["z"])

    p0 = np.asarray(tcp_episode_start_position_m, dtype=np.float64).reshape(3)
    q0 = np.asarray(tcp_episode_start_quat_xyzw, dtype=np.float64).reshape(4)
    R0 = Rotation.from_quat(q0)
    delta_p = np.array([lx, ly, lz], dtype=np.float64)
    R_delta = Rotation.from_rotvec(np.array([ax, ay, az], dtype=np.float64))
    p_naive = p0 + delta_p
    R_naive = R0 * R_delta

    ref_pose = _reference_board_pose_dict(recorded_task_board_pose)
    cx_ref, cy_ref, _ = task_board_top_face_center_base_m_from_pose_dict(ref_pose)
    zr_pose = float(ref_pose["z"])
    z_board = float(board_z_m) if board_z_m is not None else zr_pose
    d_board = np.array(
        [
            float(board_cx_m) - float(cx_ref),
            float(board_cy_m) - float(cy_ref),
            z_board - zr_pose,
        ],
        dtype=np.float64,
    )
    dyaw = _wrap_angle_rad(float(board_yaw_rad) - float(ref_pose["yaw"]))

    p_goal = p_naive + d_board
    R_goal = Rotation.from_euler("xyz", [0.0, 0.0, dyaw]) * R_naive
    quat = R_goal.as_quat()
    return Pose(
        position=Point(x=float(p_goal[0]), y=float(p_goal[1]), z=float(p_goal[2])),
        orientation=Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        ),
    )
