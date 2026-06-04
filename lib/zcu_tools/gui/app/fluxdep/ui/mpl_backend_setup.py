"""Select fluxdep's custom matplotlib backend process-wide.

Import-clean by design: importing this module must NOT pull in matplotlib, so an
entry script can call ``configure_fluxdep_matplotlib_backend()`` first — before
importing anything that uses matplotlib — and be sure the import itself did not
load pyplot too early.
"""

from __future__ import annotations

import sys

BACKEND_NAME = "module://zcu_tools.gui.app.fluxdep.ui.mpl_backend"

_configured = False


def configure_fluxdep_matplotlib_backend() -> None:
    """Select fluxdep's embedded backend process-wide (before any pyplot import).

    matplotlib only honours a backend selection made before ``matplotlib.pyplot``
    is first imported, so the entry point must call this before importing any
    module that (directly or transitively) imports pyplot. Idempotent; fast-fails
    if called too late (a silent no-op would route figures to detached windows).
    """
    global _configured
    if _configured:
        return

    if "matplotlib.pyplot" in sys.modules:
        raise RuntimeError(
            "matplotlib.pyplot is already imported; the fluxdep backend must be "
            "configured before any pyplot import. Call "
            "configure_fluxdep_matplotlib_backend() at the top of the entry "
            "script, before importing modules that use matplotlib."
        )

    import matplotlib

    matplotlib.use(BACKEND_NAME)
    _configured = True
