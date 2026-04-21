from __future__ import annotations

from pathlib import Path

from typing_extensions import Literal


def get_bitfile(version: Literal["v1", "v2"]) -> str:
    version_dict = {
        "v1": "qick_216.bit",
        "v2": "qick_216_v2.bit",
    }
    if version not in version_dict:
        raise ValueError(f"Invalid version {version}")
    return str(Path(__file__).resolve().parent / version_dict[version])
