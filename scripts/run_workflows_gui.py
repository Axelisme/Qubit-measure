#!/usr/bin/env python3
"""Launch the concrete T1 workflows GUI through the shared runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zcu_tools.gui.launcher import add_runtime_cli_options, runtime_options_from_args

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Parse runtime options and launch an EMPTY workflows session.

    argv=None reads command-line arguments. Returns the shared Qt exit code.
    No entry, input draft or hardware connection is restored automatically.
    """
    parser = argparse.ArgumentParser(description="Launch the T1 workflows GUI")
    add_runtime_cli_options(
        parser,
        control_port_help="Control port; default uses an OS-assigned ephemeral port.",
        allow_external=True,
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from zcu_tools.gui.runtime import launch_gui_runtime
    from zcu_tools.resources.entry import component_registry

    from zcu_lab.components import register_all
    from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior

    register_all(component_registry)
    return launch_gui_runtime(
        WorkflowsGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        project_root=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
