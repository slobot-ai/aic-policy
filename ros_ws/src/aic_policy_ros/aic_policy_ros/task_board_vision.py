#
# SPDX-License-Identifier: Apache-2.0
#

"""Multiview task-board policy vision: snapshots, dumps, :class:`vision.rectangle_vision.RectangleVision`."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from aic_policy_ros.multiview_camera_extrinsics_tf import t_cam_from_base_by_camera_from_tf_buffer
from aic_policy_ros.multiview_snapshot import (
    MultiviewSnapshot,
    MultiviewSnapshotInput,
    MultiviewSnapshotOutput,
    episode_dir_name,
    image_msg_to_bgr,
)
from aic_policy_ros.task_board_calibration_io import dump_multiview_calibration_json
from aic_policy_ros.task_board_center_estimation import TaskBoardCenterEstimationOutput
from aic_policy_ros.task_board_episode_geometry import z_face_in_base
from aic_policy_ros.task_board_rectangle_estimation import (
    TaskBoardRectangleCandidate,
    TaskBoardRectangleEstimationOutput,
    rectangle_candidate_highest_mean_hard_iou,
)

_YAW_SEEDS_RAD: tuple[float, float] = (math.pi, math.pi / 2.0)
_BGR_QUARTER_SCALE = 0.25
_HSV_DUMP_SCALE = 0.25

_EXTRINSICS_SOURCE_TF2 = (
    "tf2_ros.Buffer.lookup_transform(CameraInfo.header.frame_id, base_link) "
    "→ T_cam_from_base (p_cam = T @ p_base)"
)


def _bgr_quarter_views(snap_out: MultiviewSnapshotOutput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downscale full snapshot BGR before HSV + segmentation (same scale as HSV preview dumps)."""

    from vision.image_scaler import ImageBuffer, ImageScaler

    scaler = ImageScaler(scale=_BGR_QUARTER_SCALE)
    return tuple(
        scaler.scale(ImageBuffer(image=np.asarray(x, dtype=np.uint8))).image
        for x in (snap_out.bgr_left, snap_out.bgr_center, snap_out.bgr_right)
    )


def _bgr_snap_md5_hex_triple(snap_out: MultiviewSnapshotOutput) -> tuple[str, str, str]:
    return tuple(
        hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
        for x in (snap_out.bgr_left, snap_out.bgr_center, snap_out.bgr_right)
    )


def _observation_bgr_md5_hex_triple(observation: Any) -> tuple[str, str, str]:
    return tuple(
        hashlib.md5(np.ascontiguousarray(image_msg_to_bgr(img)).tobytes()).hexdigest()
        for img in (observation.left_image, observation.center_image, observation.right_image)
    )


def _center_estimation_output_from_initializer(init: Any) -> TaskBoardCenterEstimationOutput:
    pcs = init.per_camera_centroid_xy_m
    if len(pcs) != 3:
        raise ValueError(f"expected three camera centroids, got {len(pcs)}")
    return TaskBoardCenterEstimationOutput(
        cx=float(init.merged_centroid_x_m),
        cy=float(init.merged_centroid_y_m),
        per_camera_xy_m=(tuple(map(float, pcs[0])), tuple(map(float, pcs[1])), tuple(map(float, pcs[2]))),
    )


def _task_summary_for_plot(task: Any) -> str:
    if task is None:
        return "Task: (none)"
    return (
        "Task: "
        f"id={getattr(task, 'id', '')!r} "
        f"target_module_name={getattr(task, 'target_module_name', '')!r} "
        f"plug_type={getattr(task, 'plug_type', '')!r} "
        f"port_type={getattr(task, 'port_type', '')!r} "
        f"port_name={getattr(task, 'port_name', '')!r}"
    )


class TaskBoardVision:
    """Runs first/second/third observation chains using ``RectangleVision`` from ``ai-industry-challenge``.

    ``policy`` must provide ``get_logger()`` and ``sleep_for(duration_sec)`` (e.g. :class:`TaskBoardPolicy`).
    """

    def __init__(
        self,
        policy: Any,
        get_observation: Callable[[], Any],
        episode_id: int,
        output_root: Path | str = "/tmp/aic-policy",
        *,
        intrinsics_logo_rectangle_half_extent_x_m: float = 0.05,
        intrinsics_logo_rectangle_half_extent_y_m: float = 0.05,
        tf_buffer: Any | None = None,
        base_link_frame: str = "base_link",
        tf_lookup_timeout_sec: float = 0.5,
    ) -> None:
        self._policy = policy
        self._logger = policy.get_logger()
        self._get_observation = get_observation
        self._episode_id = int(episode_id)
        self._output_root = Path(output_root).expanduser()
        self._intrinsics_logo_half_extent_x_m = float(intrinsics_logo_rectangle_half_extent_x_m)
        self._intrinsics_logo_half_extent_y_m = float(intrinsics_logo_rectangle_half_extent_y_m)
        self._tf_buffer = tf_buffer
        self._base_link_frame = str(base_link_frame)
        self._tf_lookup_timeout_sec = float(tf_lookup_timeout_sec)
        self._first_obs_bgr_md5: tuple[str, str, str] | None = None
        self._second_obs_bgr_md5: tuple[str, str, str] | None = None
        self._multiview_snapshot = MultiviewSnapshot(self._logger, self._output_root)
        self._rectangle_vision: Any | None = None
        self.last_task_board_center_estimation: TaskBoardCenterEstimationOutput | None = None
        self.last_intrinsics_logo_center_estimation: TaskBoardCenterEstimationOutput | None = None
        self.last_intrinsics_logo_rectangle_estimation: TaskBoardRectangleEstimationOutput | None = None
        self._board_rectangle_vision_input: Any | None = None
        self._board_rectangle_vision_outputs: tuple[Any, Any] | None = None
        self.last_task_board_orientation: str | None = None
        self._first_observation_second_candidate: TaskBoardRectangleCandidate | None = None
        self.last_planar_board_cx_m: float | None = None
        self.last_planar_board_cy_m: float | None = None
        self.last_planar_board_yaw_rad: float | None = None

    def _episode_tree_root(self) -> Path:
        """``output_root`` / ``episode_<id>`` (same basename as :class:`MultiviewSnapshot` dumps)."""

        return self._output_root / episode_dir_name(self._episode_id)

    def _rv(self) -> Any:
        if self._rectangle_vision is None:
            from vision.rectangle_vision import RectangleVision

            self._rectangle_vision = RectangleVision()
        return self._rectangle_vision

    def _resolve_multiview_cam_extrinsics(
        self, observation: Any, observation_phase: Literal["first", "second", "third"]
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, np.ndarray], str]:
        """``(T_left, T_center, T_right), dict for dumps, extrinsics JSON source string)``."""

        if self._tf_buffer is None:
            raise RuntimeError(
                "TaskBoardVision requires the ROS node's tf2_ros.Buffer as _tf_buffer "
                "(static wrist extrinsics fallback was removed)."
            )
        t_dict = t_cam_from_base_by_camera_from_tf_buffer(
            self._tf_buffer,
            observation,
            self._logger,
            base_frame=self._base_link_frame,
            tf_lookup_timeout_sec=self._tf_lookup_timeout_sec,
        )
        if t_dict is None:
            raise RuntimeError(
                "TaskBoardVision: could not resolve multiview T_cam_from_base from TF "
                "(check CameraInfo.header.frame_id and TF connectivity)."
            )
        t_tuple = (t_dict["left_camera"], t_dict["center_camera"], t_dict["right_camera"])
        return t_tuple, t_dict, _EXTRINSICS_SOURCE_TF2

    def _dump_scaled_hsv_turbo_multiview(self, snap_out: MultiviewSnapshotOutput) -> None:
        """Quarter-res BGR → HSV; writes raw HSV and **V** channel mapped with Turbo (dark-board cue)."""

        from vision.bgr_to_hsv_converter import BgrToHsvConverter
        from vision.image_scaler import ImageBuffer, ImageScaler

        scaler = ImageScaler(scale=_HSV_DUMP_SCALE)
        conv = BgrToHsvConverter()
        for bgr, bgr_dump in zip(
            (snap_out.bgr_left, snap_out.bgr_center, snap_out.bgr_right),
            snap_out.dump_paths,
            strict=True,
        ):
            hsv_arr = conv.convert(ImageBuffer(image=scaler.scale(ImageBuffer(image=bgr)).image)).image
            stem = bgr_dump.stem
            base = stem.replace("_bgr", "", 1) if "_bgr" in stem else stem
            hsv_path = bgr_dump.parent / f"{base}_hsv.png"
            turbo_path = bgr_dump.parent / f"{base}_hsv_turbo.png"
            v = hsv_arr[:, :, 2]
            turbo_bgr = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
            if not cv2.imwrite(str(hsv_path), hsv_arr):
                self._logger.error(f"BgrToHsv: failed to write HSV dump {hsv_path}")
            else:
                self._logger.info(f"BgrToHsv: HSV dump file://{hsv_path.resolve().as_posix()}")
            if not cv2.imwrite(str(turbo_path), turbo_bgr):
                self._logger.error(f"BgrToHsv: failed to write HSV turbo preview {turbo_path}")
            else:
                self._logger.info(f"BgrToHsv: HSV turbo preview file://{turbo_path.resolve().as_posix()}")

    def _dump_scaled_hsv_h_turbo_multiview(self, snap_out: MultiviewSnapshotOutput) -> None:
        """Quarter-res BGR → HSV; writes raw HSV and **H** channel mapped with Turbo (purple-logo cue)."""

        from vision.bgr_to_hsv_converter import BgrToHsvConverter
        from vision.image_scaler import ImageBuffer, ImageScaler

        scaler = ImageScaler(scale=_HSV_DUMP_SCALE)
        conv = BgrToHsvConverter()
        for bgr, bgr_dump in zip(
            (snap_out.bgr_left, snap_out.bgr_center, snap_out.bgr_right),
            snap_out.dump_paths,
            strict=True,
        ):
            hsv_arr = conv.convert(ImageBuffer(image=scaler.scale(ImageBuffer(image=bgr)).image)).image
            stem = bgr_dump.stem
            base = stem.replace("_bgr", "", 1) if "_bgr" in stem else stem
            hsv_path = bgr_dump.parent / f"{base}_hsv.png"
            turbo_path = bgr_dump.parent / f"{base}_hsv_h_turbo.png"
            h = hsv_arr[:, :, 0].astype(np.float32) * (255.0 / 179.0)
            h_u8 = np.clip(h, 0.0, 255.0).astype(np.uint8)
            turbo_bgr = cv2.applyColorMap(h_u8, cv2.COLORMAP_TURBO)
            if not cv2.imwrite(str(hsv_path), hsv_arr):
                self._logger.error(f"BgrToHsv: failed to write HSV dump {hsv_path}")
            else:
                self._logger.info(f"BgrToHsv: HSV dump file://{hsv_path.resolve().as_posix()}")
            if not cv2.imwrite(str(turbo_path), turbo_bgr):
                self._logger.error(f"BgrToHsv: failed to write HSV H turbo preview {turbo_path}")
            else:
                self._logger.info(f"BgrToHsv: HSV H turbo preview file://{turbo_path.resolve().as_posix()}")

    def _write_mask_pngs(self, masks: tuple[np.ndarray, ...], paths: tuple[Path, Path, Path], dump_kind: str) -> None:
        for m, mask_path in zip(masks, paths, strict=True):
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), m)
            resolved = mask_path.resolve()
            self._logger.info(f"TaskBoardPolicy: {dump_kind} dump file://{resolved.as_posix()}")

    def _fit_two_candidates(
        self,
        *,
        bgr_left: np.ndarray,
        bgr_center: np.ndarray,
        bgr_right: np.ndarray,
        segmentation_mode: Any,
        z_table: float,
        t_cam_from_base: tuple[np.ndarray, np.ndarray, np.ndarray],
        mask_dump_paths: tuple[Path, Path, Path],
        mask_dump_kind: str,
        vision_subject: str,
        rectangle_half_extent_x_m: float | None = None,
        rectangle_half_extent_y_m: float | None = None,
        image_scale: float = 1.0,
        observation_dump_id: int | None = None,
        classifier_camera_indices: tuple[int, ...] | None = None,
        initializer_camera_indices: tuple[int, ...] | None = None,
    ) -> tuple[TaskBoardRectangleEstimationOutput, tuple[Any, Any], Any]:
        from vision.rectangle_vision import RectangleVisionInput

        bgr_list = (
            np.asarray(bgr_left, dtype=np.uint8),
            np.asarray(bgr_center, dtype=np.uint8),
            np.asarray(bgr_right, dtype=np.uint8),
        )
        t_list = tuple(np.asarray(t, dtype=np.float64).reshape(4, 4) for t in t_cam_from_base)
        dump_episode_root: Path | None = None
        dump_obs_id: str | None = None
        dump_layout = "flat"
        if observation_dump_id is not None:
            dump_episode_root = self._episode_tree_root()
            dump_obs_id = str(int(observation_dump_id))
            dump_layout = "episode_cameras"
        cam_names = ("left_camera", "center_camera", "right_camera")
        if initializer_camera_indices is not None:
            sub_i = ", ".join(cam_names[i] for i in initializer_camera_indices)
            self._logger.info(
                f"TaskBoardPolicy: RectangleVision[{vision_subject}] "
                f"initializer uses subset only: {sub_i} (indices={initializer_camera_indices!r})"
            )
        if classifier_camera_indices is not None:
            sub = ", ".join(cam_names[i] for i in classifier_camera_indices)
            self._logger.info(
                f"TaskBoardPolicy: RectangleVision[{vision_subject}] "
                f"classifier uses subset only: {sub} (indices={classifier_camera_indices!r})"
            )
        rv = self._rv()
        candidates: list[TaskBoardRectangleCandidate] = []
        rv_outs: list[Any] = []
        inp_board_template: Any | None = None
        for i, yaw_seed in enumerate(_YAW_SEEDS_RAD):
            inp = RectangleVisionInput(
                bgr_u8_list=bgr_list,
                segmentation_mode=segmentation_mode,
                z_face_in_base=float(z_table),
                T_cam_from_base=t_list,
                image_scale=float(image_scale),
                init_yaw_rad=float(yaw_seed),
                rectangle_half_extent_x_m=rectangle_half_extent_x_m,
                rectangle_half_extent_y_m=rectangle_half_extent_y_m,
                dump_dir=dump_episode_root,
                dump_layout=dump_layout,
                dump_observation_id=dump_obs_id,
                camera_names=cam_names,
                classifier_camera_indices=classifier_camera_indices,
                initializer_camera_indices=initializer_camera_indices,
            )
            if i == 0:
                inp_board_template = inp
            t0 = time.perf_counter()
            out = rv.run(inp)
            elapsed_s = time.perf_counter() - t0
            self._logger.info(
                f"TaskBoardPolicy: RectangleVision[{vision_subject}] "
                f"run[{i}] seed_y_rad={yaw_seed:.6f} runtime_sec={elapsed_s:.6f}"
            )
            rv_outs.append(out)
            if i == 0:
                self._write_mask_pngs(out.masks_u8_downscaled, mask_dump_paths, mask_dump_kind)
            candidates.append(
                TaskBoardRectangleCandidate(
                    yaw_seed_rad=float(yaw_seed),
                    cx_m=float(out.cx_m),
                    cy_m=float(out.cy_m),
                    yaw_rad=float(out.yaw_rad),
                    mean_hard_iou=float(out.mean_hard_iou),
                )
            )
        ys = ", ".join(
            f"seed_y={c.yaw_seed_rad:.6f}→cx={c.cx_m:.9f} cy={c.cy_m:.9f} yaw={c.yaw_rad:.9f} mean_hard_iou={c.mean_hard_iou:.6f}"
            for c in candidates
        )
        self._logger.info(
            f"TaskBoardPolicy: RectangleVision[{vision_subject}] z_face_in_base={z_table:.9f} candidates: {ys}"
        )
        for idx, c in enumerate(candidates):
            self._logger.info(
                f"TaskBoardPolicy: RectangleVision[{vision_subject}] candidate[{idx}] "
                f"cx_m={c.cx_m:.9f} cy_m={c.cy_m:.9f} yaw_rad={c.yaw_rad:.9f} "
                f"mean_hard_iou={c.mean_hard_iou:.6f} yaw_seed_rad={c.yaw_seed_rad:.6f}"
            )
        assert inp_board_template is not None
        return (
            TaskBoardRectangleEstimationOutput(candidates=(candidates[0], candidates[1])),
            (rv_outs[0], rv_outs[1]),
            inp_board_template,
        )

    def _dump_multiview_calibration_json(
        self,
        observation: Any,
        observation_phase: str,
        *,
        write_intrinsics: bool,
        t_cam_from_base_by_camera: dict[str, np.ndarray],
        extrinsics_matrix_source: str,
    ) -> None:
        if observation_phase not in ("first", "second", "third"):
            raise ValueError("observation_phase must be 'first', 'second', or 'third'")
        dump_multiview_calibration_json(
            self._output_root,
            observation,
            t_cam_from_base_by_camera,
            observation_phase=observation_phase,
            write_intrinsics=write_intrinsics,
            episode_id=self._episode_id,
            extrinsics_matrix_source=extrinsics_matrix_source,
        )

    def first_observation(self) -> TaskBoardRectangleEstimationOutput:
        observation = self._get_observation()
        t_cam_from_base, t_by, matrix_src = self._resolve_multiview_cam_extrinsics(observation, "first")
        self._dump_multiview_calibration_json(
            observation,
            "first",
            write_intrinsics=True,
            t_cam_from_base_by_camera=t_by,
            extrinsics_matrix_source=matrix_src,
        )
        snap_out = self._multiview_snapshot.apply(
            MultiviewSnapshotInput(
                observation=observation,
                episode_id=self._episode_id,
                dump_filename="first_observation_bgr.png",
            )
        )
        self._first_obs_bgr_md5 = _bgr_snap_md5_hex_triple(snap_out)
        self._dump_scaled_hsv_turbo_multiview(snap_out)
        z_table = z_face_in_base()
        self._logger.info(f"TaskBoardVision: center-estimation geometry z_table={z_table:.9f}")
        from vision.rectangle_segmentation import SegmentationMode

        bql, bqc, bqr = _bgr_quarter_views(snap_out)
        mask_paths = MultiviewSnapshot.multiview_bgr_dump_paths(
            self._output_root, self._episode_id, "first_observation_task_board_mask.png"
        )
        out, board_rv_outs, inp_board = self._fit_two_candidates(
            bgr_left=bql,
            bgr_center=bqc,
            bgr_right=bqr,
            segmentation_mode=SegmentationMode.DARK_TASK_BOARD,
            z_table=z_table,
            t_cam_from_base=t_cam_from_base,
            mask_dump_paths=mask_paths,
            mask_dump_kind="task-board mask",
            image_scale=1.0,
            vision_subject="task_board",
            observation_dump_id=1,
        )
        self._board_rectangle_vision_input = inp_board
        self._board_rectangle_vision_outputs = board_rv_outs
        init = self._rv().last_output.initializer
        self.last_task_board_center_estimation = _center_estimation_output_from_initializer(init)
        chosen, _ = rectangle_candidate_highest_mean_hard_iou(out)
        self._first_observation_second_candidate = chosen
        return out

    def _observation_after_stale_full_res_bgr_retry(
        self,
        observation: Any,
        *,
        ref_bgr_md5: tuple[str, str, str] | None,
        ref_snapshot_name: str,
        phase_name: str,
    ) -> Any:
        """If ``ref_bgr_md5`` is set, retry once when any camera still matches that snapshot (MD5 of decoded BGR)."""

        if ref_bgr_md5 is None:
            return observation
        for attempt in range(2):
            md5_triple = _observation_bgr_md5_hex_triple(observation)
            stale_idxs = [i for i, (a, b) in enumerate(zip(md5_triple, ref_bgr_md5, strict=True)) if a == b]
            if not stale_idxs:
                break
            cam_names = ("left", "center", "right")
            stale_names = ", ".join(cam_names[i] for i in stale_idxs)
            self._logger.warning(
                f"TaskBoardVision: {phase_name} full-res BGR matches {ref_snapshot_name} for "
                f"{stale_names} (MD5); yielding for a fresh multiview frame"
            )
            if attempt == 0:
                self._policy.sleep_for(0.05)
                observation = self._get_observation()
                continue
            self._logger.warning(
                f"TaskBoardVision: {phase_name} still matches {ref_snapshot_name} after retry; "
                "proceeding with current images"
            )
            break
        return observation

    def second_observation(
        self,
        *,
        task: Any | None = None,
    ) -> TaskBoardRectangleEstimationOutput:
        observation = self._observation_after_stale_full_res_bgr_retry(
            self._get_observation(),
            ref_bgr_md5=self._first_obs_bgr_md5,
            ref_snapshot_name="first",
            phase_name="second observation",
        )

        t_cam_from_base, t_by, matrix_src = self._resolve_multiview_cam_extrinsics(observation, "second")
        self._dump_multiview_calibration_json(
            observation,
            "second",
            write_intrinsics=False,
            t_cam_from_base_by_camera=t_by,
            extrinsics_matrix_source=matrix_src,
        )
        snap_out = self._multiview_snapshot.apply(
            MultiviewSnapshotInput(
                observation=observation,
                episode_id=self._episode_id,
                dump_filename="second_observation_bgr.png",
            )
        )
        self._second_obs_bgr_md5 = _bgr_snap_md5_hex_triple(snap_out)
        self._dump_scaled_hsv_h_turbo_multiview(snap_out)
        z_table = z_face_in_base()
        self._logger.info(f"TaskBoardVision: center-estimation geometry z_table={z_table:.9f}")
        from vision.rectangle_segmentation import SegmentationMode

        bql, bqc, bqr = _bgr_quarter_views(snap_out)
        mask_paths = MultiviewSnapshot.multiview_bgr_dump_paths(
            self._output_root, self._episode_id, "second_observation_intrinsics_logo_mask.png"
        )
        rectangle_logo, logo_rv_outs, _ = self._fit_two_candidates(
            bgr_left=bql,
            bgr_center=bqc,
            bgr_right=bqr,
            segmentation_mode=SegmentationMode.PURPLE_INTRINSICS_LOGO,
            z_table=z_table,
            t_cam_from_base=t_cam_from_base,
            mask_dump_paths=mask_paths,
            mask_dump_kind="intrinsics-logo mask",
            rectangle_half_extent_x_m=self._intrinsics_logo_half_extent_x_m,
            rectangle_half_extent_y_m=self._intrinsics_logo_half_extent_y_m,
            image_scale=1.0,
            vision_subject="intrinsics_logo",
            observation_dump_id=2,
        )
        init = self._rv().last_output.initializer
        self.last_intrinsics_logo_center_estimation = _center_estimation_output_from_initializer(init)
        self.last_intrinsics_logo_rectangle_estimation = rectangle_logo
        self.last_planar_board_cx_m = None
        self.last_planar_board_cy_m = None
        self.last_planar_board_yaw_rad = None
        orient_res = None
        if self._board_rectangle_vision_input is not None and self._board_rectangle_vision_outputs is not None:
            from vision.task_board_orientation import TaskBoardOrientation

            xy_top_path = self._episode_tree_root() / "second_observation_task_board_xy_top_view.png"
            orient_res = TaskBoardOrientation.classify_from_fit_pairs(
                self._board_rectangle_vision_input,
                self._board_rectangle_vision_outputs,
                logo_rv_outs,
                out_path=None,
            )
            self.last_task_board_orientation = orient_res.kind.value
            self._logger.info(f"TaskBoardPolicy: TaskBoardOrientation={orient_res.kind.value}")

            ib = orient_res.board_candidate_index
            il = orient_res.logo_candidate_index
            if ib is not None and il is not None:
                board_out = self._board_rectangle_vision_outputs[ib]
                logo_out = logo_rv_outs[il]
                title_extra = ""
                overlay_xy: tuple[float, float] | None = None
                if (
                    task is not None
                    and orient_res.task_board_cx_m is not None
                    and orient_res.task_board_cy_m is not None
                    and orient_res.task_board_yaw_rad is not None
                ):
                    from aic_policy_ros.task_board_insert_pose import (
                        default_task_manifest_path,
                        initial_tcp_pose_from_manifest_row,
                        load_task_manifest,
                        manifest_row_for_task,
                        tcp_manifest_linear_z_extra_m,
                    )
                    from vision.task_preset_middle_pose_transform import (
                        InitialTcpPose,
                        TaskPresetMiddlePoseTransform as VisionTcpPreset,
                        TaskPresetMiddlePoseTransformInput as VisionTcpPresetIn,
                        final_tcp_pose_from_initial_and_offset,
                    )

                    rows = load_task_manifest()
                    row = manifest_row_for_task(task, self._episode_id, rows)
                    initial_tcp = initial_tcp_pose_from_manifest_row(row) if row is not None else None
                    pose_ok = (
                        row is not None
                        and isinstance(row.get("trial_key"), str)
                        and isinstance(row.get("task_board"), dict)
                        and isinstance(row["task_board"].get("pose"), dict)
                        and all(k in row["task_board"]["pose"] for k in ("x", "y", "z", "roll", "pitch", "yaw"))
                    )
                    if row is not None and initial_tcp is not None and pose_ok:
                        anchor_pos_m, anchor_quat_xyzw = initial_tcp
                        try:
                            mp_out = VisionTcpPreset.apply(
                                VisionTcpPresetIn(
                                    trial_key=str(row["trial_key"]),
                                    manifest_path=default_task_manifest_path(),
                                    task_board_cx_m=float(orient_res.task_board_cx_m),
                                    task_board_cy_m=float(orient_res.task_board_cy_m),
                                    task_board_yaw_rad=float(orient_res.task_board_yaw_rad),
                                    orientation=orient_res.kind,
                                    initial_tcp_pose=InitialTcpPose(
                                        xyz_m=tuple(float(x) for x in anchor_pos_m),
                                        quat_xyzw=tuple(float(x) for x in anchor_quat_xyzw),
                                    ),
                                )
                            )
                            p_goal, _q_goal = final_tcp_pose_from_initial_and_offset(
                                np.asarray(anchor_pos_m, dtype=np.float64),
                                np.asarray(anchor_quat_xyzw, dtype=np.float64),
                                mp_out.tcp_offset.as_manifest_dict(),
                            )
                            dz_lin = tcp_manifest_linear_z_extra_m(str(row.get("task_kind", "")))
                            p_plot = np.asarray(p_goal, dtype=np.float64).copy()
                            if dz_lin != 0.0:
                                p_plot[2] += float(dz_lin)
                            overlay_xy = (float(p_plot[0]), float(p_plot[1]))
                            title_extra = (
                                f"\n{_task_summary_for_plot(task)}"
                                "\nPreset board (manifest task_board top-face, base_link xy): "
                                f"cx_preset={mp_out.preset_board_cx_m:.4f}, cy_preset={mp_out.preset_board_cy_m:.4f}, "
                                f"yaw_preset={mp_out.preset_board_yaw_rad:.4f}"
                                "\nLive fitted board: "
                                f"cx_live={float(orient_res.task_board_cx_m):.4f}, cy_live={float(orient_res.task_board_cy_m):.4f}, "
                                f"yaw_live={float(orient_res.task_board_yaw_rad):.4f}"
                                f"\nEffective fitted yaw (TCP remap): {mp_out.fitted_board_yaw_effective_rad:.4f}"
                                f"\nTarget TCP xy (remapped offset, base_link): ({overlay_xy[0]:.4f}, {overlay_xy[1]:.4f})"
                                + (
                                    f"\nManifest linear z extra (task_kind): +{dz_lin:.4f} m → TCP z={float(p_plot[2]):.4f}"
                                    if dz_lin != 0.0
                                    else ""
                                )
                            )
                        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
                            title_extra = f"\nTCP overlay skipped: {exc!r}"
                    elif row is None:
                        title_extra = "\nTCP overlay skipped: no manifest row for task/episode_id"
                    elif initial_tcp is None:
                        title_extra = "\nTCP overlay skipped: manifest row missing or invalid initial_tcp_pose"
                    else:
                        title_extra = (
                            "\nTCP overlay skipped: manifest row missing trial_key or full task_board.pose"
                        )

                base_title = TaskBoardOrientation.figure_title_for_result(orient_res)
                TaskBoardOrientation.write_xy_top_view_task_board_and_logo(
                    out_path=xy_top_path,
                    board_out=board_out,
                    logo_out=logo_out,
                    orientation_result=orient_res,
                    figure_title=base_title + title_extra,
                    overlay_xy_m=overlay_xy,
                    overlay_label="TCP goal xy",
                    overlay_markersize=14.0,
                    logo_style="cross",
                )
                if xy_top_path.is_file():
                    self._logger.info(f"TaskBoardPolicy: xy top view file://{xy_top_path.resolve().as_posix()}")
            else:
                self._logger.warning(
                    "TaskBoardPolicy: second_observation_task_board_xy_top_view.png not written "
                    "(missing board/logo candidate indices)"
                )
        else:
            self.last_task_board_orientation = None
            self._logger.warning("TaskBoardPolicy: TaskBoardOrientation skipped (missing board RectangleVision state)")

        if orient_res is not None and orient_res.task_board_cx_m is not None:
            self.last_planar_board_cx_m = float(orient_res.task_board_cx_m)
            self.last_planar_board_cy_m = float(orient_res.task_board_cy_m or 0.0)
            self.last_planar_board_yaw_rad = float(orient_res.task_board_yaw_rad or 0.0)
        elif self._first_observation_second_candidate is not None:
            c = self._first_observation_second_candidate
            self.last_planar_board_cx_m = float(c.cx_m)
            self.last_planar_board_cy_m = float(c.cy_m)
            self.last_planar_board_yaw_rad = float(c.yaw_rad)
        return rectangle_logo

    def third_observation(self) -> None:
        """Decode multiview BGR, HSV H preview, blue-port H segmentation, and ``RectangleVision`` (dump id 3).

        If any camera's full-res BGR still matches the second observation (MD5 of decoded
        ``sensor_msgs/Image`` bytes), yields once and retries ``get_observation()`` —
        same stale-frame pattern as :meth:`second_observation` vs the first snapshot,
        but compared against the second multiview snapshot.

        Uses the same TF-only extrinsics resolution as observations **1** and **2** (see
        :meth:`_resolve_multiview_cam_extrinsics`); matrices are written as
        ``third_observation_extrinsics.json`` (same numeric convention as the second pose).
        """

        observation = self._get_observation()
        if self._second_obs_bgr_md5 is None:
            self._logger.warning(
                "TaskBoardVision: third observation MD5 check skipped (no second snapshot yet)"
            )
        else:
            observation = self._observation_after_stale_full_res_bgr_retry(
                observation,
                ref_bgr_md5=self._second_obs_bgr_md5,
                ref_snapshot_name="second",
                phase_name="third observation",
            )

        t_cam_from_base, t_by, matrix_src = self._resolve_multiview_cam_extrinsics(observation, "third")
        self._dump_multiview_calibration_json(
            observation,
            "third",
            write_intrinsics=False,
            t_cam_from_base_by_camera=t_by,
            extrinsics_matrix_source=matrix_src,
        )

        snap_out = self._multiview_snapshot.apply(
            MultiviewSnapshotInput(
                observation=observation,
                episode_id=self._episode_id,
                dump_filename="third_observation_bgr.png",
            )
        )
        self._dump_scaled_hsv_h_turbo_multiview(snap_out)
        z_table = z_face_in_base()
        self._logger.info(f"TaskBoardVision: third observation z_face_in_base={z_table:.9f}")
        from vision.rectangle3d_detector import sc_port_rectangle_half_extents_xy_m
        from vision.rectangle_segmentation import SegmentationMode

        bql, bqc, bqr = _bgr_quarter_views(snap_out)
        mask_paths = MultiviewSnapshot.multiview_bgr_dump_paths(
            self._output_root, self._episode_id, "third_observation_blue_port_mask.png"
        )
        hx_sc, hy_sc = sc_port_rectangle_half_extents_xy_m()
        self._logger.info(
            "TaskBoardVision: third observation blue SC port geometry "
            f"RectangleVision z_face_in_base=z_face_in_base()={z_table:.9f} (task-board top; no SC mesh z stack) "
            f"rectangle_half_extent_xy_m=({hx_sc:.9f}, {hy_sc:.9f})"
        )
        self._fit_two_candidates(
            bgr_left=bql,
            bgr_center=bqc,
            bgr_right=bqr,
            segmentation_mode=SegmentationMode.BLUE_SC_PORT,
            z_table=z_table,
            t_cam_from_base=t_cam_from_base,
            mask_dump_paths=mask_paths,
            mask_dump_kind="blue-port mask",
            rectangle_half_extent_x_m=hx_sc,
            rectangle_half_extent_y_m=hy_sc,
            image_scale=1.0,
            vision_subject="blue_sc_port",
            observation_dump_id=3,
            initializer_camera_indices=(1,),
            classifier_camera_indices=(1,),
        )
