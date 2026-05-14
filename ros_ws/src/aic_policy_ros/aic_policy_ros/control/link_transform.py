#
# SPDX-License-Identifier: Apache-2.0
#

"""Lookup ``parent_frame`` → ``child_frame`` from a TF2 buffer (§2.1 TCP pose)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rclpy.time import Time


@dataclass
class LinkTransformInput:
    """Frames and buffer for a single transform lookup."""

    tf_buffer: Any
    parent_frame: str = "base_link"
    child_frame: str = "gripper/tcp"


@dataclass(frozen=True)
class LinkTransformOutput:
    """Rigid transform from TF (translation in meters, unit quaternion rotation)."""

    translation_x_m: float
    translation_y_m: float
    translation_z_m: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    rotation_w: float


class LinkTransform:
    """Read ``geometry_msgs/TransformStamped`` via ``tf2_ros.BufferInterface.lookup_transform``."""

    def apply(self, inp: LinkTransformInput) -> LinkTransformOutput:
        tf = inp.tf_buffer.lookup_transform(inp.parent_frame, inp.child_frame, Time())
        t = tf.transform.translation
        r = tf.transform.rotation
        return LinkTransformOutput(
            translation_x_m=float(t.x),
            translation_y_m=float(t.y),
            translation_z_m=float(t.z),
            rotation_x=float(r.x),
            rotation_y=float(r.y),
            rotation_z=float(r.z),
            rotation_w=float(r.w),
        )
