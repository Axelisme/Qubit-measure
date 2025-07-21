import os
from typing import Literal


def get_bitfile(version: Literal["v1", "v2"]) -> str:
    version_dict = {
        "v1": "qick_216.bit",
        "v2": "qick_216_v2.bit",
    }
    if version not in version_dict:
        raise ValueError(f"Invalid version {version}")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), version_dict[version]
    )
