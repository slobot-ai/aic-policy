#!/usr/bin/bash
# Sourced by Docker launch script (paths fixed in-image).
set -e
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
_base_zenoh="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"
if [[ -n "${AIC_ROUTER_ADDR:-}" ]]; then
  export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"tcp/${AIC_ROUTER_ADDR}\"];${_base_zenoh}"
else
  export ZENOH_CONFIG_OVERRIDE="${_base_zenoh}"
fi
