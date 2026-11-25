"""Find source prefixes from a single recursive cloud directory listing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import rclone
from .settings import DATA_DIRS, is_data_dir, validate_prefix


def list_sources(binary: Path, config_dir: Path, remote_root: str) -> list[str]:
    """List sorted unique prefixes directly containing a result or Database tree.

    Use the verified rclone binary and config_dir/rclone.conf to read remote_root.
    Do not return the cloud root itself or any candidate inside a data tree.
    Raise CalledProcessError on rclone failure or ValueError on invalid output.
    """
    # Keep each tree's directory in the listing, but prune its child directories.
    prune = "- {" + ",".join(DATA_DIRS) + "}/*/"
    result = rclone.run_cli(
        binary,
        config_dir,
        [
            "lsjson",
            remote_root,
            "--dirs-only",
            "--recursive",
            "--fast-list",
            "--filter",
            prune,
        ],
        capture_output=True,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    value: object = json.loads(result.stdout)
    if not isinstance(value, list):
        raise ValueError("rclone returned a non-list directory listing.")
    prefixes: set[str] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("Path"), str)
            or item.get("IsDir") is not True
        ):
            raise ValueError("rclone returned an invalid directory listing item.")
        prefix, _, leaf = item["Path"].rpartition("/")
        if (
            prefix
            and leaf in DATA_DIRS
            and not any(map(is_data_dir, prefix.split("/")))
        ):
            prefixes.add(validate_prefix(prefix))
    return sorted(prefixes)
