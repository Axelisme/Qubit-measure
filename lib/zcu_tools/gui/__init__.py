"""GUI applications namespace.

Each GUI app lives under ``app/`` (e.g. ``app/measure`` = the measure-gui). This
parent package re-exports nothing and stays import-clean: importing
``zcu_tools.gui`` pulls in neither Qt nor matplotlib. Import a concrete app
explicitly, e.g. ``from zcu_tools.gui.app.measure import State``.
"""
