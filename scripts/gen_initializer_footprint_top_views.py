#!/usr/bin/env python3
"""Write ``multiview/<obs_id>/xy_top_view_initializer_footprints_only.png`` for a dumped episode tree.

Uses segmentation masks under ``<camera>/<obs_id>/yaw_pi/`` and extrinsics JSON from each camera
tree (same convention as a live ``TaskBoardPolicy`` dump). Requires ``PYTHONPATH`` to include
``ai-industry-challenge/src`` and ``aic-policy/ros_ws/src/aic_policy_ros`` (same as policy tests).

Example::

  cd aic-policy && source pixi_env_setup.sh \\
    && PYTHONPATH=../ai-industry-challenge/src:ros_ws/src/aic_policy_ros \\
    python scripts/gen_initializer_footprint_top_views.py doc/img/episode_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_CAMERAS: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")


def _T_cam_from_base_from_extrinsics_json(path: Path) -> np.ndarray:
    doc = json.loads(path.read_text(encoding="utf-8"))
    flat = doc["T_cam_from_base_row_major"]
    return np.asarray(flat, dtype=np.float64).reshape(4, 4)


def _z_for_observation(obs_id: int) -> float:
    from aic_policy_ros.task_board_episode_geometry import z_face_in_base

    if obs_id in (1, 2, 3):
        return float(z_face_in_base())
    raise ValueError(f"unsupported observation id {obs_id}")


def _extrinsics_stem(obs_id: int) -> str:
    return {1: "first_observation", 2: "second_observation", 3: "third_observation"}[obs_id]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "episode_root",
        type=Path,
        help="Episode dump root (e.g. doc/img/episode_1 or /tmp/aic-policy/episode_1)",
    )
    ap.add_argument(
        "--obs-ids",
        type=str,
        default="1,2,3",
        help="Comma-separated observation dump ids (default: 1,2,3)",
    )
    args = ap.parse_args()
    episode_root = args.episode_root.expanduser().resolve()
    if not episode_root.is_dir():
        print(f"error: not a directory: {episode_root}", file=sys.stderr)
        return 1

    from vision.rectangle3d_detector import nominal_basler_K_for_image_wh
    from vision.rectangle_initializer import RectangleInitializer
    from vision.rectangle_vision_dump import _write_xy_top_view_initializer_footprints_only

    obs_ids = [int(x.strip()) for x in args.obs_ids.split(",") if x.strip()]
    for obs_id in obs_ids:
        z_m = _z_for_observation(obs_id)
        stem = _extrinsics_stem(obs_id)
        masks: list[np.ndarray] = []
        K_list: list[np.ndarray] = []
        T_list: list[np.ndarray] = []
        for cam in _CAMERAS:
            mask_path = episode_root / cam / str(obs_id) / "yaw_pi" / f"segmentation_mask_{cam}.png"
            if not mask_path.is_file():
                print(f"warning: skip obs {obs_id}: missing {mask_path}", file=sys.stderr)
                break
            ext_path = episode_root / cam / "extrinsics" / f"{stem}_extrinsics.json"
            if not ext_path.is_file():
                print(f"warning: skip obs {obs_id}: missing {ext_path}", file=sys.stderr)
                break
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f"warning: skip obs {obs_id}: could not read {mask_path}", file=sys.stderr)
                break
            masks.append(m)
            h, w = int(m.shape[0]), int(m.shape[1])
            K_list.append(nominal_basler_K_for_image_wh(w, h))
            T_list.append(_T_cam_from_base_from_extrinsics_json(ext_path))
        else:
            out_path = (
                episode_root / "multiview" / str(obs_id) / "xy_top_view_initializer_footprints_only.png"
            )
            bool_masks = tuple(np.asarray(m, dtype=np.uint8) > 0 for m in masks)
            init = RectangleInitializer.candidate_xy_from_masks_on_z_table(
                bool_masks,
                tuple(K_list),
                tuple(T_list),
                z_m,
            )
            cx_m, cy_m = float(init.merged_centroid_x_m), float(init.merged_centroid_y_m)
            _write_xy_top_view_initializer_footprints_only(
                out_path=out_path,
                z_face_m=z_m,
                masks_u8_downscaled=tuple(masks),
                K_downscaled_list=tuple(K_list),
                T_cam_from_base_list=tuple(T_list),
                camera_names=_CAMERAS,
                centroid_x_m=float(cx_m),
                centroid_y_m=float(cy_m),
            )
            print(f"wrote {out_path} (z_face_m={z_m:.9f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
