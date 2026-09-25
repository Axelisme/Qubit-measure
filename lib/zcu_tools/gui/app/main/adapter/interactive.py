"""Experiment-facing declaration for the interactive plugin's Qt-free half.

The Qt frontend factory belongs to the GUI frontend contract, not to the
committed-state core. Concrete interactive adapters supply both at composition.
"""

from __future__ import annotations

from typing import Any, Protocol

from zcu_tools.gui.app.main.interactive import PluginDefinition

from .types import AnalyzeRequest


class InteractivePluginProvider(Protocol):
    """Supply one captured-input plugin definition per analysis operation.

    Heterogeneous adapters erase their concrete state/result type at the app
    boundary; each concrete plugin's actions and result builder stay typed.
    """

    def make_interactive_plugin(
        self, request: AnalyzeRequest[Any, Any]
    ) -> PluginDefinition[Any, Any]: ...
