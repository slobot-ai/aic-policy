#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
"""Write third-observation segmentation masks from saved ``third_observation_hsv.png`` dumps.

Policy code saves HSV with ``cv2.imwrite`` on an ``H×W×3`` array in OpenCV HSV order
(``[...,0]`` = H, ``[...,1]`` = S, ``[...,2]`` = V). ``cv2.imread`` returns a 3-channel
matrix whose channels are still H, S, V in that order (same layout as in-memory HSV).

Uses the current :data:`vision.rectangle_segmentation.BLUE_SC_PORT_FILTER` and
:class:`vision.rectangle_segmentation.RectangleSegmentation` so previews match the live
third-observation path.

Example::

    cd aic-policy && pixi run python scripts/third_observation_masks_from_hsv_pngs.py \\
        --episode-root doc/img/episode_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_CAMERAS: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episode-root",
        type=Path,
        default=Path("doc/img/episode_1"),
        help="Episode directory containing <cam>/input/third_observation_hsv.png",
    )
    p.add_argument(
        "--output-suffix",
        default="_segmentation_mask_from_hsv_preview",
        help="Stem suffix before .png (default writes third_observation<suffix>.png)",
    )
    args = p.parse_args()
    root: Path = args.episode_root.expanduser().resolve()

    _repo = Path(__file__).resolve().parents[1]
    _aic_policy_ros_src = _repo / "ros_ws" / "src" / "aic_policy_ros"
    if _aic_policy_ros_src.is_dir():
        rs = str(_aic_policy_ros_src.resolve())
        if rs not in sys.path:
            sys.path.insert(0, rs)

    from vision.rectangle_segmentation import (
        RectangleSegmentation,
        RectangleSegmentationInput,
        SegmentationMode,
    )

    seg = RectangleSegmentation()
    for cam in _CAMERAS:
        hsv_path = root / cam / "input" / "third_observation_hsv.png"
        if not hsv_path.is_file():
            raise SystemExit(f"missing HSV dump: {hsv_path}")
        bgr_layout = cv2.imread(str(hsv_path), cv2.IMREAD_COLOR)
        if bgr_layout is None:
            raise SystemExit(f"failed to read: {hsv_path}")
        hsv_u8 = bgr_layout
        out = seg.run(
            RectangleSegmentationInput(hsv_u8=hsv_u8, mode=SegmentationMode.BLUE_SC_PORT)
        ).mask_u8
        out_path = hsv_path.parent / f"third_observation{args.output_suffix}.png"
        if not cv2.imwrite(str(out_path), out):
            raise SystemExit(f"failed to write {out_path}")
        n_on = int((out > 0).sum())
        print(f"wrote {out_path}  foreground_pixels={n_on}")


if __name__ == "__main__":
    main()
