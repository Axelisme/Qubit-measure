#!/usr/bin/env python3
"""Compose lab registrations and launch the generic workflows GUI runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
    from zcu_tools.gui.catalog_loader import SourceCatalogLoader

    from zcu_lab.workflows.common.calibration import Calibrated

from zcu_tools.gui.launcher import add_runtime_cli_options, runtime_options_from_args

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_catalog() -> WorkflowCatalog[Calibrated]:
    """Read current lab registrations, validating unique names and nonempty order.

    Import at each call so source reload does not retain an old builder or step.
    Import/declaration failures propagate without constructing live resources.
    """
    from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog

    from zcu_lab.workflows.registry import build_registry

    return WorkflowCatalog(build_registry())


def build_source_loader() -> SourceCatalogLoader[WorkflowCatalog[Calibrated]]:
    """Monitor fixed framework/lab sources and reload only lab workflows.

    Returns an owner-thread bounded loader over the current lab registry. No
    reload occurs yet. Framework and component changes require process restart;
    workflow/forms/registry changes are loaded together. Source errors propagate.
    """
    import zcu_tools
    from zcu_tools.gui.catalog_loader import SourceCatalogLoader, SourcePackage

    import zcu_lab

    return SourceCatalogLoader(
        sources=(
            SourcePackage("zcu_tools", Path(zcu_tools.__file__).parent),
            SourcePackage("zcu_lab", Path(zcu_lab.__file__).parent),
        ),
        reload_modules=("zcu_lab.workflows",),
        preserved_modules=(),
        catalog_module="zcu_lab.workflows.registry",
        catalog_builder=build_catalog,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse runtime options and launch an EMPTY workflows session.

    argv=None reads command-line arguments. Returns the shared Qt exit code.
    Parsed plan/knobs input is restored unless --clean is supplied. No entry,
    native run or hardware connection is restored automatically.
    """
    parser = argparse.ArgumentParser(description="Launch the workflows GUI")
    add_runtime_cli_options(
        parser,
        control_port_help="Control port; default uses an OS-assigned ephemeral port.",
        allow_external=True,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Skip reading saved workflow input; normal close still saves current input.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
    from zcu_tools.gui.runtime import launch_gui_runtime
    from zcu_tools.resources.entry import component_registry

    from zcu_lab import load_alias_presets
    from zcu_lab.components import register_all
    from zcu_lab.workflows.registry import build_registry

    register_all(component_registry)
    return launch_gui_runtime(
        WorkflowsGuiBehavior,
        runtime_options_from_args(args, log_root=PROJECT_ROOT),
        extra_logging_namespaces=("zcu_lab",),
        registrations=build_registry(),
        loader=build_source_loader(),
        alias_presets=load_alias_presets(),
        project_root=str(PROJECT_ROOT),
        clean=args.clean,
    )


if __name__ == "__main__":
    raise SystemExit(main())
