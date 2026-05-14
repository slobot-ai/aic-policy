# syntax=docker/dockerfile:1.7
# Build with BuildKit additional contexts:
#   ``aic`` → ``aic`` repo source (``aic_model``, ``aic_interfaces``).
#   ``aic_challenge`` → ``ai-industry-challenge`` repo root (``src/vision``; git uses a symlink there).
#
# Example (cwd = ``aic-policy``; sibling repos next to this checkout)::
#
#   docker build -f docker/Dockerfile . \\
#     --build-context aic=../aic \\
#     --build-context aic_challenge=../ai-industry-challenge
#
# Colcon builds ``aic_model`` + ``aic_interfaces`` + ``aic_policy_ros`` inside the image.

FROM docker.io/library/ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /aic-vision

COPY pixi.toml pixi.lock pixi_env_setup.sh /aic-vision/
COPY scripts /aic-vision/scripts
COPY docker/pixi_env_setup.docker.sh /aic-vision/docker/pixi_env_setup.docker.sh

WORKDIR /ws
COPY --from=aic aic_model /ws/src/aic_model
COPY --from=aic aic_interfaces /ws/src/aic_interfaces
COPY ros_ws/src/aic_policy_ros /ws/src/aic_policy_ros
# Git tree uses ``vision`` → ``ai-industry-challenge`` symlink; materialize for the image.
RUN rm -rf /ws/src/aic_policy_ros/vision
COPY --from=aic_challenge src/vision /ws/src/aic_policy_ros/vision

WORKDIR /aic-vision

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

RUN --mount=type=cache,target=/root/.cache/rattler/cache \
    --mount=type=cache,target=/aic-vision/.pixi/build \
    cd /aic-vision && pixi install --locked && \
    (find .pixi/build -mindepth 1 -delete 2>/dev/null || true) && \
    find .pixi/envs/default -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

RUN cd /aic-vision && pixi run bash -eo pipefail -c '\
    cd /ws && \
    colcon build \
      --packages-select \
      aic_control_interfaces aic_model_interfaces aic_task_interfaces aic_model aic_policy_ros && \
    rm -rf /ws/build /ws/log'

FROM docker.io/library/ros:kilted-ros-core

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /aic-vision

COPY --from=builder /aic-vision/.pixi /aic-vision/.pixi
COPY --from=builder /aic-vision/pixi.toml /aic-vision/pixi.toml
COPY --from=builder /aic-vision/pixi.lock /aic-vision/pixi.lock
COPY --from=builder /aic-vision/pixi_env_setup.sh /aic-vision/pixi_env_setup.sh
COPY --from=builder /aic-vision/docker/pixi_env_setup.docker.sh /aic-vision/docker/pixi_env_setup.docker.sh
COPY --from=builder /ws/install /ws_aic/install

COPY --chmod=755 <<"EOF" /launch_model.sh
#!/bin/bash
set -eo pipefail
# shellcheck source=/dev/null
source /aic-vision/docker/pixi_env_setup.docker.sh
# shellcheck source=/dev/null
source /ws_aic/install/setup.bash
export PATH="/aic-vision/.pixi/envs/default/bin:${PATH}"
export RCUTILS_LOGGING_USE_STDOUT=1
cd /aic-vision
exec ros2 run aic_model aic_model "$@"
EOF

COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "${AIC_ROUTER_ADDR:-}" ]]; then
  echo "AIC_ROUTER_ADDR must be set (e.g. eval:7447 in compose)"
  exit 1
fi

should_enable_acl() {
  [[ "${AIC_ENABLE_ACL}" == "true" || "${AIC_ENABLE_ACL}" == "1" ]]
}

if should_enable_acl; then
  if [[ -z "${AIC_MODEL_PASSWD:-}" ]]; then
    echo "AIC_MODEL_PASSWD must be set when AIC_ENABLE_ACL is true"
    exit 1
  fi
  echo "model:${AIC_MODEL_PASSWD}" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"${AIC_ROUTER_ADDR}"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"${AIC_MODEL_PASSWD}"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE}"

cd /aic-vision
exec pixi run --as-is /launch_model.sh "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "use_sim_time:=true", "-p", "policy:=aic_policy_ros.ros.TaskBoardPolicy"]
