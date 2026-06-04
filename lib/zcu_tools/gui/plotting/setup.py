"""Select the shared GUI matplotlib backend.

Import-clean by design: importing this module must never pull in matplotlib, so
an entry script can always do ``from zcu_tools.gui.plotting.setup import
configure_matplotlib_backend`` first — before importing anything that uses
matplotlib — and be sure the import itself did not load pyplot too early. This
module therefore imports only ``sys`` at module scope; matplotlib is imported
inside the function body, and ``backend``/``host``/``container``/``routing`` are
never imported here.
"""

from __future__ import annotations

import sys

BACKEND_NAME = "module://zcu_tools.gui.plotting.backend"

_configured = False


def configure_matplotlib_backend() -> None:
    """Select the shared GUI custom matplotlib backend process-wide.

    Both GUI apps (measure-gui, fluxdep) call this from their entry script before
    any pyplot import. matplotlib only honours a backend selection made before
    ``matplotlib.pyplot`` is first imported.

    Idempotent within a process. Fast-fails if called too late — when pyplot is
    already imported the custom backend can no longer take effect, so a silent
    no-op would route figures to detached windows instead of the Qt canvas.
    """
    global _configured
    if _configured:
        return

    if "matplotlib.pyplot" in sys.modules:
        raise RuntimeError(
            "matplotlib.pyplot is already imported; the GUI backend must be "
            "configured before any pyplot import. Call "
            "configure_matplotlib_backend() at the top of the entry script, "
            "before importing modules that use matplotlib."
        )

    import matplotlib

    matplotlib.use(BACKEND_NAME)
    _configured = True
