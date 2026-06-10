"""dispersive-fit-gui domain services — headless, Qt-free analysis steps that
read from and write to ``DispersiveState`` on the Qt main thread.

The heavy numerics are reused from ``zcu_tools.notebook`` / ``zcu_tools.simulate``
/ ``zcu_tools.utils`` (loading, signal preprocessing, dispersive simulation, the
scipy fit); the plotly result figures are rewritten as matplotlib here (``viz``).
"""
