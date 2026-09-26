"""Experiment-facing declaration for the interactive plugin's Qt-free half.

The Qt frontend factory belongs to the GUI frontend contract, not to the
committed-state core. Concrete interactive adapters supply both at composition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from zcu_tools.gui.app.main.interactive import PluginDefinition, Session

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )

from .types import AnalyzeRequest


class InteractivePluginProvider(Protocol):
    """Supply one captured-input plugin definition per analysis operation.

    Heterogeneous adapters erase their concrete state/result type at the app
    boundary; each concrete plugin's actions and result builder stay typed.
    """

    def make_interactive_plugin(
        self, request: AnalyzeRequest[Any, Any]
    ) -> PluginDefinition[Any, Any]: ...

    def make_interactive_frontend(
        self,
        plugin: PluginDefinition[Any, Any],
        session: Session[Any],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[Figure], bool],
        request_cancel: Callable[[], bool],
    ) -> InteractiveFrontend: ...
