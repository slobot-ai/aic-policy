#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
_base_zenoh="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"
if [[ -n "${AIC_ZENOH_ROUTER_ADDR:-}" ]]; then
  export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/${AIC_ZENOH_ROUTER_ADDR}\"];${_base_zenoh}"
else
    export ZENOH_CONFIG_OVERRIDE="${_base_zenoh}"
fi

# ``colcon``, ``ros2``, etc. live in the Pixi env ``bin``. Some tooling invokes shells
# without Pixi's full activation; prepend so those CLIs resolve after ``pixi install``.
_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_env_bin="${_repo_root}/.pixi/envs/default/bin"
if [[ -d "${_env_bin}" ]]; then
    case ":${PATH}:" in
        *:"${_env_bin}":*) ;;
        *) export PATH="${_env_bin}:${PATH}" ;;
    esac
fi
unset _repo_root _env_bin
