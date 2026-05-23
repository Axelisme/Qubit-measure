from __future__ import annotations

import os
import sys

BACKEND_NAME = "module://zcu_tools.gui.mpl_backend"
_CONFIGURED_ENV = "ZCU_TOOLS_GUI_MPL_BACKEND"


def configure_gui_matplotlib_backend() -> None:
    if os.environ.get(_CONFIGURED_ENV) == BACKEND_NAME:
        return

    if "matplotlib.pyplot" in sys.modules:
        raise RuntimeError(
            "matplotlib.pyplot is already imported; GUI backend must be configured earlier"
        )

    import matplotlib

    matplotlib.use(BACKEND_NAME)
    os.environ[_CONFIGURED_ENV] = BACKEND_NAME
