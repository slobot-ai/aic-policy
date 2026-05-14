#
# SPDX-License-Identifier: Apache-2.0
#

"""Resolve ``aic-policy`` checkout root (module name kept for stable imports)."""

from __future__ import annotations

from pathlib import Path


def aic_policy_repo_root() -> Path:
    """``aic-policy`` checkout root (parent of ``ros_ws``).

    Works for source and symlink installs by walking parents for ``ros_ws/src/aic_policy_ros``.
    Plain installs under ``site-packages`` have no ``ros_ws`` on disk — parents[4] is the historical fallback.
    """

    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "ros_ws" / "src" / "aic_policy_ros").is_dir():
            return p
    return here.parents[4]
