#!/usr/bin/env bash
# Build aic_policy_ros, then run aic_model with TaskBoardPolicy policy.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIC_WS="${AIC_WS:-$(cd "${ROOT}/../aic" && pwd)}"
SETUP="${AIC_WS}/install/setup.bash"

if [[ ! -f "$SETUP" ]]; then
  echo "error: AIC workspace install not found: $SETUP" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$SETUP"

_pixi_bin="${ROOT}/.pixi/envs/default/bin"
if [[ -d "${_pixi_bin}" ]]; then
  export PATH="${_pixi_bin}:${PATH}"
fi

cd "${ROOT}/ros_ws"
rm -rf "${ROOT}/ros_ws/build/aic_policy_ros" "${ROOT}/ros_ws/install/aic_policy_ros"
colcon build --packages-select aic_policy_ros
# shellcheck source=/dev/null
source "${ROOT}/ros_ws/install/setup.bash"

# shellcheck source=/dev/null
source "${ROOT}/pixi_env_setup.sh"

exec ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_policy_ros.ros.TaskBoardPolicy \
  "$@"
