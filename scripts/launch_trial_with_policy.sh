#!/usr/bin/env bash
# Start AIC engine in ``aic_eval`` (``docker run``, not compose), wait for engine handoff, then run this repo's
# ``TaskBoardPolicy`` policy via ``pixi run aic-policy-model`` (same pattern as ``aic-act/scripts/launch_trial_with_policy.sh``).
# Wait for ``insert_cable() returned`` in the policy log, then scoring in the engine log, teardown, MCAP append.
# By default ``KEEP_MCAP_BAG=1`` keeps ``bag_trial_<N>_*/`` on disk after append; set ``KEEP_MCAP_BAG=0`` to delete and prune.
# Single attempt (no retries). Uses ``${AIC_CONFIG_SUBDIR}/trial_<N>.yaml`` on the sibling ``aic`` repo (default: generated_configs).
#
# Usage:
#   launch_trial_with_policy.sh <TRIAL_ID>
set -euo pipefail

TRIAL_ID="${1:?usage: $0 <TRIAL_ID> (e.g. 0 for generated_configs/trial_0.yaml on sibling aic)}"
if [[ -n "${2:-}" ]]; then
  echo "error: unexpected argument '$2'; this script only runs TaskBoardPolicy (no backend selector). See env EPISODE_ID, …" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_POLICY_ROOT="${AIC_POLICY_ROOT:-${AIC_VISION_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
# Optional: absolute path to ai-industry-challenge (required only for MCAP → LeRobot append at end of trial).
AIC_ROOT="${AIC_ROOT:-$(cd "${SCRIPT_DIR}/../../aic" && pwd)}"
AIC_CONFIG_SUBDIR="${AIC_CONFIG_SUBDIR:-generated_configs}"
CONFIG="${AIC_ROOT}/aic_engine/config/${AIC_CONFIG_SUBDIR}/trial_${TRIAL_ID}.yaml"
AIC_INSTALL_PREFIX_CONTAINER="${AIC_INSTALL_PREFIX_CONTAINER:-/ws_aic/install}"
AIC_CONTAINER_WS="${AIC_CONTAINER_WS:-/ws_aic}"
CONFIG_CONTAINER="${AIC_INSTALL_PREFIX_CONTAINER}/share/aic_engine/config/${AIC_CONFIG_SUBDIR}/trial_${TRIAL_ID}.yaml"
CONTAINER="${AIC_CONTAINER:-aic_eval}"
AIC_EVAL_IMAGE="${AIC_EVAL_IMAGE:-aic_eval}"
AIC_DOCKER_EXEC_USER="${AIC_DOCKER_EXEC_USER:-}"
AIC_REMOTE_GPU="${AIC_REMOTE_GPU:-false}"
AIC_EGL_VENDOR_JSON_HOST="${AIC_EGL_VENDOR_JSON_HOST:-${AIC_POLICY_ROOT}/docker/10_nvidia.json}"
AIC_EGL_VENDOR_JSON_CONTAINER="${AIC_EGL_VENDOR_JSON_CONTAINER:-/etc/glvnd/egl_vendor.d/10_nvidia.json}"
LOG_DIR="${LOG_DIR:-/tmp/aic/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/aic_engine_trial_${TRIAL_ID}.log}"
_default_policy_log="${LOG_DIR}/aic_policy_vision_trial_${TRIAL_ID}.log"
POLICY_LOG="${POLICY_LOG:-${AIC_POLICY_LOG:-${_default_policy_log}}}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
# Alias: ``RVIZ=false`` (or ``true``) overrides ``LAUNCH_RVIZ`` when set.
if [[ -n "${RVIZ:-}" ]]; then
  LAUNCH_RVIZ="${RVIZ}"
fi
ON_SHUTDOWN_MARKER="${ON_SHUTDOWN_MARKER:-insert_cable() returned}"
ON_SHUTDOWN_TIMEOUT_SECS="${ON_SHUTDOWN_TIMEOUT_SECS:-1800}"
ON_SHUTDOWN_POLL_SECS="${ON_SHUTDOWN_POLL_SECS:-5}"
AIC_ENGINE_SCORING_MARKER="${AIC_ENGINE_SCORING_MARKER:-Complete Scoring Results}"
POST_SCORING_COMPLETE_SLEEP_SECS="${POST_SCORING_COMPLETE_SLEEP_SECS:-2}"
SCORING_RESULTS_TIMEOUT_SECS="${SCORING_RESULTS_TIMEOUT_SECS:-180}"
SCORING_RESULTS_POLL_SECS="${SCORING_RESULTS_POLL_SECS:-1}"
AIC_POLICY_HANDOFF_MARKER="${AIC_POLICY_HANDOFF_MARKER:-${AIC_GYM_HANDOFF_MARKER:-No node with name 'aic_model' found. Retrying}}"

AIC_RESULTS_ROOT="${AIC_RESULTS_ROOT:-${HOME}/aic_results}"
AIC_RESULTS_DIR_CONTAINER="${AIC_RESULTS_DIR_CONTAINER:-/aic_results}"
# When ``KEEP_MCAP_BAG`` is ``1``, ``true``, or ``yes`` (case-insensitive, default), skip deleting
# ``bag_trial_<N>_*/`` under ``AIC_RESULTS_ROOT`` after MCAP→LeRobot and skip pruning older bags
# for the same trial id (so reruns accumulate instead of removing the previous bag).
KEEP_MCAP_BAG="${KEEP_MCAP_BAG:-1}"

# TaskBoardPolicy ROS parameters (passed to ``aic_model`` after ``pixi run aic-policy-model``).
EPISODE_ID="${EPISODE_ID:-${TRIAL_ID}}"
# LeRobot dataset id for MCAP append (local root below). Default is eval-only — not the training set ``slobot/aic``.
# ``mcap-to-lerobot`` never uploads to the Hub (writes under ``LEROBOT_DATASET_ROOT_SUCCESS`` only; see ai-industry-challenge ``doc/mcap_to_lerobot.md``).
LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-slobot/aic-eval}"
LEROBOT_HF_HOME="${LEROBOT_HF_HOME:-/tmp/aic_eval}"
LEROBOT_DATASET_ROOT_SUCCESS="${LEROBOT_DATASET_ROOT_SUCCESS:-${LEROBOT_HF_HOME}/${LEROBOT_REPO_ID}}"
LEROBOT_DATASET_ROOT_FAILED="${LEROBOT_DATASET_ROOT_FAILED:-/tmp/aic_lerobot_failed}"

if [[ ! -f "$CONFIG" ]]; then
  echo "error: trial config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -d "$AIC_POLICY_ROOT" ]]; then
  echo "error: aic-policy root not found: $AIC_POLICY_ROOT (set AIC_POLICY_ROOT or AIC_VISION_ROOT)" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required" >&2
  exit 1
fi

mkdir -p "$AIC_RESULTS_ROOT"

cleanup_trial_services() {
  docker exec "${CONTAINER}" pkill -INT -f 'ros2 launch aic_bringup' 2>/dev/null || true
  docker exec "${CONTAINER}" pkill -INT -f 'rmw_zenohd' 2>/dev/null || true
  pkill -INT -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -INT -f '/lib/aic_model/aic_model' 2>/dev/null || true
  pkill -INT -f 'pixi run aic-policy-model' 2>/dev/null || true
  sleep 1
  pkill -KILL -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -KILL -f '/lib/aic_model/aic_model' 2>/dev/null || true
  pkill -KILL -f 'pixi run aic-policy-model' 2>/dev/null || true
  docker rm -f "${CONTAINER}" 2>/dev/null || true
}

rm_rf_maybe_sudo() {
  local path="$1"
  if rm -rf "$path" 2>/dev/null; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "Removing (requires sudo; root-owned files): $path"
    sudo rm -rf "$path" || echo "warning: could not remove $path" >&2
  else
    echo "warning: cannot remove root-owned path without sudo: $path" >&2
  fi
}

prune_old_bags_for_trial() {
  [[ -d "${AIC_RESULTS_ROOT}" ]] || return 0
  local -a dirs=()
  mapfile -t dirs < <(find "${AIC_RESULTS_ROOT}" -maxdepth 1 -type d -name "bag_trial_${TRIAL_ID}_*" 2>/dev/null | LC_ALL=C sort || true)
  [[ ${#dirs[@]} -le 1 ]] && return 0
  local newest="${dirs[${#dirs[@]}-1]}"
  [[ -n "$newest" ]] || return 0
  for d in "${dirs[@]}"; do
    if [[ "$d" == "$newest" ]]; then
      continue
    fi
    echo "Removing older bag directory for trial ${TRIAL_ID}: $d"
    rm_rf_maybe_sudo "$d"
  done
}

_resolve_ai_challenge_root() {
  if [[ -n "${AI_CHALLENGE_ROOT:-}" && -d "${AI_CHALLENGE_ROOT}" ]]; then
    echo "${AI_CHALLENGE_ROOT}"
    return 0
  fi
  local cand
  cand="$(cd "${SCRIPT_DIR}/../../ai-industry-challenge" 2>/dev/null && pwd)" || true
  if [[ -n "$cand" && -d "$cand" ]]; then
    echo "$cand"
    return 0
  fi
  echo ""
}

append_mcap_to_lerobot() {
  local dataset_root="$1"
  local ch_root
  ch_root="$(_resolve_ai_challenge_root)"
  if [[ -z "$ch_root" ]]; then
    echo "warning: skipping MCAP → LeRobot append (set AI_CHALLENGE_ROOT to ai-industry-challenge or place it as ../../ai-industry-challenge from this script)" >&2
    return 0
  fi
  (
    cd "$ch_root"
    pixi run mcap-to-lerobot \
      --config "$(dirname "$CONFIG")" \
      --input-folder "$AIC_RESULTS_ROOT" \
      --trial-id "trial_${TRIAL_ID}" \
      --repo-id "$LEROBOT_REPO_ID" \
      --dataset-root "$dataset_root"
  )
}

remove_mcap_bag_dir_for_trial() {
  local remove_all="${1:-false}"
  local trial_key="trial_${TRIAL_ID}"
  local prefix="bag_${trial_key}_"
  local -a bag_dirs=()
  mapfile -t bag_dirs < <(find "${AIC_RESULTS_ROOT}" -maxdepth 1 -type d -name "${prefix}*" 2>/dev/null | LC_ALL=C sort || true)
  if [[ ${#bag_dirs[@]} -eq 0 ]]; then
    echo "warning: no bag directory matching ${prefix}* under ${AIC_RESULTS_ROOT}; skipping MCAP folder removal" >&2
    return 0
  fi
  if [[ "${remove_all}" == "true" ]]; then
    for bag_dir in "${bag_dirs[@]}"; do
      echo "Removing MCAP bag directory: $bag_dir"
      rm_rf_maybe_sudo "$bag_dir"
    done
    return 0
  fi
  local bag_dir="${bag_dirs[${#bag_dirs[@]}-1]}"
  echo "Removing MCAP bag directory: $bag_dir"
  rm_rf_maybe_sudo "$bag_dir"
}

docker_rm_container() {
  docker rm -f "${CONTAINER}" 2>/dev/null || true
}

docker_rm_container

cleanup_host_policy_processes() {
  echo "Stopping any leftover host aic_model / aic-policy-model processes..."
  pkill -INT -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -INT -f '/lib/aic_model/aic_model' 2>/dev/null || true
  pkill -INT -f 'pixi run aic-policy-model' 2>/dev/null || true
  sleep "${STARTUP_POLICY_KILL_SLEEP_SECS:-3}"
  pkill -KILL -f 'ros2 run aic_model aic_model' 2>/dev/null || true
  pkill -KILL -f '/lib/aic_model/aic_model' 2>/dev/null || true
  pkill -KILL -f 'pixi run aic-policy-model' 2>/dev/null || true
  sleep 1
}

cleanup_host_policy_processes

if [[ -n "${AIC_DOCKER_EXEC_USER}" ]]; then
  _container_user="$AIC_DOCKER_EXEC_USER"
  _container_fake_home=0
  [[ "${AIC_DOCKER_EXEC_USER}" =~ ^[0-9]+:[0-9]+$ ]] && _container_fake_home=1
else
  _container_user="$(id -u):$(id -g)"
  _container_fake_home=1
fi

_aic_video_gid=""
_aic_render_gid=""
_docker_run_extra=()
if [[ "${LAUNCH_RVIZ}" == "true" ]]; then
  _docker_run_extra+=(-e "DISPLAY=${DISPLAY:-:0}")
  if [[ -d /tmp/.X11-unix ]]; then
    _docker_run_extra+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  fi
  _xauth_host="${XAUTHORITY:-${HOME}/.Xauthority}"
  if [[ -f "$_xauth_host" ]]; then
    _docker_run_extra+=(-e XAUTHORITY=/tmp/.host_xauthority -v "${_xauth_host}:/tmp/.host_xauthority:ro")
  fi
fi
if [[ "${AIC_REMOTE_GPU}" == "1" || "${AIC_REMOTE_GPU}" == "true" ]]; then
  _docker_run_extra+=(--gpus all)
  _docker_run_extra+=(-e NVIDIA_VISIBLE_DEVICES=all)
  _docker_run_extra+=(-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
  _docker_run_extra+=(-e "__EGL_VENDOR_LIBRARY_FILENAMES=${AIC_EGL_VENDOR_JSON_CONTAINER}")
  if [[ -f "${AIC_EGL_VENDOR_JSON_HOST}" ]]; then
    _docker_run_extra+=(-v "${AIC_EGL_VENDOR_JSON_HOST}:${AIC_EGL_VENDOR_JSON_CONTAINER}:ro")
  else
    echo "warning: AIC_REMOTE_GPU enabled but EGL vendor json not found: ${AIC_EGL_VENDOR_JSON_HOST}" >&2
  fi
fi
shopt -s nullglob
for _aic_dri in /dev/dri/card* /dev/dri/renderD*; do
  [[ -c "$_aic_dri" ]] && _docker_run_extra+=(--device="${_aic_dri}")
done
shopt -u nullglob
if getent group video >/dev/null 2>&1; then
  _aic_video_gid="$(getent group video | cut -d: -f3)"
  _docker_run_extra+=(--group-add "$_aic_video_gid")
fi
if getent group render >/dev/null 2>&1; then
  _aic_render_gid="$(getent group render | cut -d: -f3)"
  _docker_run_extra+=(--group-add "$_aic_render_gid")
fi

_dri_supp_groups=""
[[ -n "${_aic_video_gid:-}" ]] && _dri_supp_groups="${_aic_video_gid}"
[[ -n "${_aic_render_gid:-}" ]] && _dri_supp_groups="${_dri_supp_groups:+${_dri_supp_groups},}${_aic_render_gid}"

if ! docker run -d --name "${CONTAINER}" -p 7447:7447 \
  "${_docker_run_extra[@]}" \
  -v "${AIC_RESULTS_ROOT}:${AIC_RESULTS_DIR_CONTAINER}:rw" \
  -v "${AIC_ROOT}/aic_engine/config:${AIC_INSTALL_PREFIX_CONTAINER}/share/aic_engine/config:ro" \
  --entrypoint /bin/bash "${AIC_EVAL_IMAGE}" -c 'exec sleep infinity'; then
  echo "error: docker run failed for ${CONTAINER} (image ${AIC_EVAL_IMAGE})" >&2
  cleanup_trial_services
  exit 1
fi

_rot_ready=0
for _ in $(seq 1 60); do
  if docker exec "${CONTAINER}" true >/dev/null 2>&1; then
    _rot_ready=1
    break
  fi
  sleep 1
done
if [[ "${_rot_ready}" -ne 1 ]]; then
  echo "error: ${CONTAINER} did not become ready for docker exec" >&2
  cleanup_trial_services
  exit 1
fi

trap "cleanup_trial_services" EXIT

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$POLICY_LOG")"
: >"$LOG_FILE"
: >"$POLICY_LOG"
echo "Logging engine to $LOG_FILE"
echo "Logging policy (aic-policy-model / TaskBoardPolicy) to $POLICY_LOG"
echo "Config: $CONFIG (container: $CONFIG_CONTAINER)"
echo "Vision policy params: episode_id=${EPISODE_ID}"

_engine_uid="$(id -u)"
_engine_gid="$(id -g)"
if [[ -n "${AIC_DOCKER_EXEC_USER}" ]]; then
  if [[ "${AIC_DOCKER_EXEC_USER}" =~ ^([0-9]+):([0-9]+)$ ]]; then
    _engine_uid="${BASH_REMATCH[1]}"
    _engine_gid="${BASH_REMATCH[2]}"
  else
    _engine_uid="$(docker exec "$CONTAINER" id -u "$AIC_DOCKER_EXEC_USER")"
    _engine_gid="$(docker exec "$CONTAINER" id -g "$AIC_DOCKER_EXEC_USER")"
  fi
fi

_docker_exec_env=(
  -e "AIC_RESULTS_DIR=${AIC_RESULTS_DIR_CONTAINER}"
  -e "DISPLAY=${DISPLAY:-:0}"
)
if [[ "$_container_fake_home" -eq 1 ]]; then
  _docker_exec_env+=(-e "HOME=/tmp")
fi
if [[ "${LAUNCH_RVIZ}" == "true" ]] && [[ -f "${XAUTHORITY:-${HOME}/.Xauthority}" ]]; then
  _docker_exec_env+=(-e XAUTHORITY=/tmp/.host_xauthority)
elif [[ -n "${XAUTHORITY:-}" ]]; then
  _docker_exec_env+=(-e "XAUTHORITY=${XAUTHORITY}")
fi

_engine_priv=()
if docker exec "$CONTAINER" sh -c 'command -v setpriv >/dev/null 2>&1'; then
  _engine_priv=(setpriv)
  if [[ -n "${_dri_supp_groups}" ]]; then
    _engine_priv+=(--groups "${_dri_supp_groups}")
  else
    _engine_priv+=(--clear-groups)
  fi
  _engine_priv+=(--reuid="${_engine_uid}" --regid="${_engine_gid}" --)
else
  echo "warning: setpriv not in image; using docker exec -u (GPU groups from docker run may be missing — RViz can be slow)." >&2
fi

if [[ ${#_engine_priv[@]} -gt 0 ]]; then
  docker exec \
    "${_docker_exec_env[@]}" \
    -w "${AIC_CONTAINER_WS}" \
    "$CONTAINER" \
    "${_engine_priv[@]}" \
    /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    aic_engine_config_file:="${CONFIG_CONTAINER}" \
    launch_rviz:="${LAUNCH_RVIZ}" \
    gazebo_gui:=false >>"$LOG_FILE" 2>&1 &
else
  docker exec -u "$_container_user" \
    "${_docker_exec_env[@]}" \
    -w "${AIC_CONTAINER_WS}" \
    "$CONTAINER" \
    /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    aic_engine_config_file:="${CONFIG_CONTAINER}" \
    launch_rviz:="${LAUNCH_RVIZ}" \
    gazebo_gui:=false >>"$LOG_FILE" 2>&1 &
fi
ENGINE_PID=$!
echo "Engine docker exec pid: $ENGINE_PID"

for i in $(seq 1 300); do
  if grep -Fq "$AIC_POLICY_HANDOFF_MARKER" "$LOG_FILE" 2>/dev/null; then
    echo "Engine handoff marker matched after ${i}s; starting aic-policy-model (TaskBoardPolicy)"
    break
  fi
  sleep 1
done

if ! grep -Fq "$AIC_POLICY_HANDOFF_MARKER" "$LOG_FILE" 2>/dev/null; then
  echo "error: timeout waiting for engine handoff marker in $LOG_FILE (expected: ${AIC_POLICY_HANDOFF_MARKER})" >&2
  trap - EXIT
  cleanup_trial_services
  exit 1
fi

sleep "${POLICY_START_DELAY_SECS:-3}"

(
  cd "$AIC_POLICY_ROOT"
  export AIC_ZENOH_ROUTER_ADDR="${AIC_ZENOH_ROUTER_ADDR:-127.0.0.1:7447}"
  export ZENOH_ROUTER_CHECK_ATTEMPTS="${ZENOH_ROUTER_CHECK_ATTEMPTS:--1}"
  export AIC_CONFIG_SUBDIR="${AIC_CONFIG_SUBDIR:-generated_configs}"
  # shellcheck source=/dev/null
  source "${AIC_POLICY_ROOT}/pixi_env_setup.sh"
  exec pixi run aic-policy-model -- \
    --ros-args \
    -p "episode_id:=${EPISODE_ID}" >>"$POLICY_LOG" 2>&1
) &
echo "aic-policy-model started (logs: $POLICY_LOG); engine still running (pid $ENGINE_PID)"

echo "Waiting up to ${ON_SHUTDOWN_TIMEOUT_SECS}s (polling every ${ON_SHUTDOWN_POLL_SECS}s) for policy log line (completion marker): ${ON_SHUTDOWN_MARKER}"
_rot_shutdown_deadline=$(( $(date +%s) + ON_SHUTDOWN_TIMEOUT_SECS ))
while true; do
  if grep -Fq "$ON_SHUTDOWN_MARKER" "$POLICY_LOG" 2>/dev/null; then
    break
  fi
  if [[ $(date +%s) -ge ${_rot_shutdown_deadline} ]]; then
    break
  fi
  sleep "$ON_SHUTDOWN_POLL_SECS"
done
if ! grep -Fq "$ON_SHUTDOWN_MARKER" "$POLICY_LOG" 2>/dev/null; then
  echo "error: timeout waiting for policy completion (${ON_SHUTDOWN_MARKER}) in $POLICY_LOG" >&2
  trap - EXIT
  cleanup_trial_services
  exit 1
fi

echo "Policy completion marker matched; waiting for engine log line: ${AIC_ENGINE_SCORING_MARKER}"
_scoring_deadline=$(( $(date +%s) + SCORING_RESULTS_TIMEOUT_SECS ))
while true; do
  if grep -Fq "$AIC_ENGINE_SCORING_MARKER" "$LOG_FILE" 2>/dev/null; then
    echo "Engine scoring marker matched; sleeping ${POST_SCORING_COMPLETE_SLEEP_SECS}s before stopping engine"
    sleep "${POST_SCORING_COMPLETE_SLEEP_SECS}"
    break
  fi
  if [[ $(date +%s) -ge ${_scoring_deadline} ]]; then
    echo "warning: timeout waiting for ${AIC_ENGINE_SCORING_MARKER} in $LOG_FILE; stopping engine anyway" >&2
    break
  fi
  sleep "$SCORING_RESULTS_POLL_SECS"
done

echo "Stopping engine and policy"
docker exec "$CONTAINER" pkill -INT -f 'ros2 launch aic_bringup' 2>/dev/null || true
docker exec "$CONTAINER" pkill -INT -f 'rmw_zenohd' 2>/dev/null || true
pkill -INT -f 'ros2 run aic_model aic_model' 2>/dev/null || true
pkill -INT -f 'pixi run aic-policy-model' 2>/dev/null || true
echo "Engine and policy stop signaled."

trap - EXIT
docker_rm_container

find "$AIC_ROOT" -maxdepth 1 -type f \( -name 'core' -o -name 'core.*' \) -delete 2>/dev/null || true

mkdir -p "${LEROBOT_DATASET_ROOT_SUCCESS}"
echo "Appending MCAP to LeRobot dataset ${LEROBOT_REPO_ID} at ${LEROBOT_DATASET_ROOT_SUCCESS}"
append_mcap_to_lerobot "$LEROBOT_DATASET_ROOT_SUCCESS"
_keep_mcap=false
case "${KEEP_MCAP_BAG,,}" in
  1|true|yes) _keep_mcap=true ;;
esac
if [[ "${_keep_mcap}" == "true" ]]; then
  echo "KEEP_MCAP_BAG set: leaving bag directories under ${AIC_RESULTS_ROOT} intact (no removal or prune)."
else
  remove_mcap_bag_dir_for_trial true
  prune_old_bags_for_trial
fi
echo "=== Trial ${TRIAL_ID} (TaskBoardPolicy / aic-policy) finished ==="
if grep -q 'score:' "$LOG_FILE" 2>/dev/null; then
  echo "--- Engine scoring summary (grep) ---"
  grep -E 'tier_|score:|message:' "$LOG_FILE" | tail -40 || true
fi
exit 0
