#
# SPDX-License-Identifier: Apache-2.0
#

"""Integration test for ``TaskBoardPolicy.insert_cable`` using saved wrist images."""

from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rclpy
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation as SciRotation

from aic_policy_ros.ai_industry_challenge_seg_path import aic_policy_repo_root
from aic_policy_ros.multiview_camera_extrinsics_tf import transform_stamped_from_homogeneous
from aic_policy_ros.multiview_snapshot import image_msg_to_bgr
from aic_policy_ros.ros.TaskBoardPolicy import TaskBoardPolicy
from aic_policy_ros.task_board_calibration_io import (
    T_cam_from_base_from_extrinsics_dict,
    bgr_np_to_ros_image,
    observation_from_image_root_and_calibration_fixture,
)
from aic_policy_ros.task_board_vision import TaskBoardVision

"""BGR frames and calibration JSON live under ``tests/fixtures/episode_0`` (vendored)."""

_EPISODE_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "episode_0"
_CAMERAS: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")
_WRIST_TO_CAM: dict[str, str] = {
    "left_wrist": "left_camera",
    "center_wrist": "center_camera",
    "right_wrist": "right_camera",
}


class _MockTfBuffer:
    """Returns a stable TCP pose until ``last_cmd_pose`` is set by ``move_robot``, then echoes it."""

    def __init__(self) -> None:
        self.last_cmd_pose: Any | None = None

    def lookup_transform(self, target_frame: str, source_frame: str, time: Any, **kwargs: Any) -> TransformStamped:
        del kwargs
        del time
        if source_frame == "base_link" and target_frame in _WRIST_TO_CAM:
            cam = _WRIST_TO_CAM[target_frame]
            path = _EPISODE_FIXTURE_ROOT / cam / "extrinsics" / "first_observation_extrinsics.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            T = T_cam_from_base_from_extrinsics_dict(doc)
            return transform_stamped_from_homogeneous(
                target_frame=target_frame,
                source_frame="base_link",
                T=T,
            )
        ts = TransformStamped()
        pose = self.last_cmd_pose
        if pose is None:
            ts.transform.translation = Vector3(x=0.35, y=0.0, z=0.55)
            q = SciRotation.from_euler("xyz", [0.0, 0.0, 0.1]).as_quat()
        else:
            ts.transform.translation = pose.position
            q = (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        ts.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return ts


@pytest.fixture
def ros_init() -> None:
    rclpy.init()
    yield
    rclpy.shutdown()


def _trial_yaml_path() -> Path:
    challenge_trial = (
        aic_policy_repo_root().parent
        / "ai-industry-challenge"
        / "tests"
        / "data"
        / "trial_configs"
        / "trial_0.yaml"
    )
    if challenge_trial.is_file():
        return challenge_trial.resolve()
    alt = aic_policy_repo_root().parent / "aic" / "aic_engine" / "config" / "generated_configs" / "trial_0.yaml"
    if alt.is_file():
        return alt.resolve()
    pytest.skip(f"missing trial yaml: tried {trial} and {alt}")


@pytest.fixture
def fast_rectangle_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    import vision.rectangle_classifier as rc

    _real = rc.RectangleClassifier

    def _fast(**kwargs: Any) -> Any:
        kwargs.setdefault("steps", 600)
        kwargs["steps"] = min(int(kwargs["steps"]), 48)
        kwargs.setdefault("seed", 0)
        return _real(**kwargs)

    monkeypatch.setattr(rc, "RectangleClassifier", _fast)


def _assert_extrinsics_fixtures_match_vision() -> None:
    from vision.transform.rectangle_forward_projection import (
        T_CAM_FROM_BASE_BY_CAMERA,
        T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION,
    )

    for cam in _CAMERAS:
        for phase in ("first", "second"):
            path = _EPISODE_FIXTURE_ROOT / cam / "extrinsics" / f"{phase}_observation_extrinsics.json"
            assert path.is_file(), f"missing fixture {path}"
            doc = json.loads(path.read_text(encoding="utf-8"))
            T_json = T_cam_from_base_from_extrinsics_dict(doc)
            src = T_CAM_FROM_BASE_BY_CAMERA if phase == "first" else T_CAM_FROM_BASE_BY_CAMERA_SECOND_OBSERVATION
            T_ref = np.asarray(src[cam], dtype=np.float64)
            np.testing.assert_allclose(T_json, T_ref, rtol=0.0, atol=1e-9)


def _build_observations() -> tuple[Observation, Observation]:
    root = _EPISODE_FIXTURE_ROOT
    required = [
        root / cam / "input" / f"{phase}_observation_bgr.png"
        for cam in _CAMERAS
        for phase in ("first", "second")
    ]
    if not all(p.is_file() for p in required):
        pytest.skip(
            f"missing vendored BGR fixtures under {root} (expected six *_observation_bgr.png under each camera input/)"
        )

    intr_fixtures = [root / cam / "extrinsics" / "intrinsics.json" for cam in _CAMERAS]
    if not all(p.is_file() for p in intr_fixtures):
        pytest.skip(
            f"missing intrinsics fixtures under {root} — add each camera's extrinsics/intrinsics.json "
            "(same paths as dump_multiview_calibration_json writes under an episode root)."
        )

    obs_first = observation_from_image_root_and_calibration_fixture(
        root,
        root,
        stem="first_observation",
    )
    obs_second = observation_from_image_root_and_calibration_fixture(
        root,
        root,
        stem="second_observation",
    )
    return obs_first, obs_second


def _observation_one_pixel_perturb_each_camera(obs: Observation) -> Observation:
    o = copy.deepcopy(obs)
    for attr in ("left_image", "center_image", "right_image"):
        msg = getattr(o, attr)
        bgr = image_msg_to_bgr(msg).copy()
        bgr[0, 0, 0] = (int(bgr[0, 0, 0]) + 1) % 256
        new_msg = bgr_np_to_ros_image(bgr)
        new_msg.header = msg.header
        setattr(o, attr, new_msg)
    return o


def test_third_observation_writes_bgr_and_md5_retry(
    ros_init: None,
    fast_rectangle_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    obs_first, obs_second = _build_observations()
    _assert_extrinsics_fixtures_match_vision()

    trial_yaml = _trial_yaml_path()
    monkeypatch.setattr(
        "aic_policy_ros.task_board_episode_geometry._default_trial_yaml_for_episode",
        lambda _episode_id: trial_yaml,
    )

    node = Node("test_third_observation")
    buf = _MockTfBuffer()
    node._tf_buffer = buf  # type: ignore[attr-defined]

    policy = TaskBoardPolicy(node)
    node.set_parameters([Parameter("episode_id", Parameter.Type.INTEGER, 0)])
    monkeypatch.setattr(policy, "sleep_for", lambda _d: None)

    obs_third = _observation_one_pixel_perturb_each_camera(obs_second)
    # ``second_observation`` uses ``obs_first`` then up to two ``obs_second`` (MD5 vs first); keep one
    # ``obs_second`` before ``obs_third`` so ``third_observation`` can retry once on MD5 vs second.
    obs_queue: deque[Observation] = deque([obs_first, obs_second, obs_second, obs_second, obs_third])

    def get_observation() -> Observation:
        return obs_queue.popleft()

    out_root = tmp_path / "out"
    vision = TaskBoardVision(policy, get_observation, 0, output_root=out_root)
    vision.first_observation()
    vision.second_observation()
    vision.third_observation()

    for cam in _CAMERAS:
        png = out_root / "episode_0" / cam / "input" / "third_observation_bgr.png"
        assert png.is_file(), f"expected dump {png}"

    node.destroy_node()


def test_insert_cable_episode_0_bgr(ros_init: None, fast_rectangle_classifier: None, monkeypatch: pytest.MonkeyPatch) -> None:
    obs_first, obs_second = _build_observations()
    _assert_extrinsics_fixtures_match_vision()

    trial_yaml = _trial_yaml_path()
    monkeypatch.setattr(
        "aic_policy_ros.task_board_episode_geometry._default_trial_yaml_for_episode",
        lambda _episode_id: trial_yaml,
    )

    node = Node("test_task_board_policy_insert_cable")
    buf = _MockTfBuffer()
    node._tf_buffer = buf  # type: ignore[attr-defined]

    policy = TaskBoardPolicy(node)
    node.set_parameters(
        [
            Parameter("episode_id", Parameter.Type.INTEGER, 0),
        ]
    )

    def noop_sleep(_duration_sec: float) -> None:
        return None

    monkeypatch.setattr(policy, "sleep_for", noop_sleep)

    def move_robot(motion_update: Any = None, joint_motion_update: Any = None) -> None:
        del joint_motion_update
        if motion_update is not None:
            buf.last_cmd_pose = motion_update.pose

    # ``second_observation`` may call ``get_observation`` twice when any camera
    # BGR still matches the first snapshot (fixtures reuse identical L/R PNGs).
    # ``third_observation`` runs after ``insert_cable`` and needs at least one more frame.
    obs_third = _observation_one_pixel_perturb_each_camera(obs_second)
    obs_iter = iter((obs_first, obs_second, obs_second, obs_third))

    def get_observation() -> Observation:
        return next(obs_iter)

    ok = policy.insert_cable(Task(), get_observation, move_robot, lambda _s: None)
    assert ok is True
    assert policy.last_task_board_orientation in ("UP", "DOWN", "UNKNOWN")

    node.destroy_node()
