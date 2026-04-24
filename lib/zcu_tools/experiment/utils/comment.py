from __future__ import annotations

import json
import time
from typing import Any, Optional

from zcu_tools.config import ConfigBase
from zcu_tools.utils import format_obj


def make_comment(cfg: ConfigBase, comment: Optional[str] = None) -> str:
    """
    Generate a formatted comment string from a configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary to be converted to a string.
        prepend (str, optional): Additional string to prepend to the comment. Defaults to "".

    Returns:
        str: A formatted comment string.
    """
    dump_dict = {}

    dump_dict["cfg"] = format_obj(cfg.to_dict())
    if comment is not None:
        dump_dict["comment"] = comment

    dump_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return json.dumps(dump_dict, indent=2)


def parse_comment(
    comment: str,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    try:
        dump_dict = json.loads(comment)
    except json.JSONDecodeError:
        return None, None, None

    return (
        dump_dict.get("cfg"),
        dump_dict.get("comment"),
        dump_dict.get("timestamp"),
    )
