"""Composition root for autofluxdep-gui (skeleton, hardware-free).

Assembles State + EventBus + Controller. The Qt MainWindow and the
RemoteControlAdapter are Phase C/D — this skeleton stops at the domain core so
the Node dependency model can be exercised (see ``demo_dry_run``) without a UI
or hardware.

Unlike fluxdep/dispersive this app is *control type*: Phase B wires the soc /
device connection and a run worker (the heavy half is aligned with measure-gui,
not the analysis GUIs). The full ``run_app`` that opens a window lands in
Phase C.
"""

from __future__ import annotations

from typing_extensions import Optional

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState, ProjectInfo


def build_core(project: Optional[ProjectInfo] = None) -> Controller:
    """Wire State + EventBus + Controller — the testable domain core.

    Phase C: a ``run_app`` will call this, then build the MainWindow and (when
    control opts are given) the RemoteControlAdapter, and run the Qt loop.
    """
    state = AutoFluxDepState(project=project)
    return Controller(state, EventBus())
