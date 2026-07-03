from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path

RequiredModule = tuple[str, str]


def bootstrap_standalone_server(
    server_file: str | Path,
    *,
    required_modules: Iterable[RequiredModule] = (),
) -> Path:
    """Prepare a standalone MCP server script for absolute package imports.

    MCP entry servers are launched as files, so Python starts with the server's
    package directory on ``sys.path`` instead of the repo ``lib`` directory. This
    helper inserts ``lib`` and performs dependency preflight before the entry
    imports any ``zcu_tools.*`` modules.
    """

    lib_dir = Path(server_file).resolve().parents[3]
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    for module_name, message in required_modules:
        if importlib.util.find_spec(module_name) is None:
            sys.stderr.write(message.format(module=module_name))
            raise SystemExit(1)

    return lib_dir
