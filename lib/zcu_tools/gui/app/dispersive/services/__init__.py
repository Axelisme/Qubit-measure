"""dispersive-fit-gui domain services — headless, Qt-free analysis steps that
read from and write to ``DispersiveState`` on the Qt main thread.

The numerical preprocessing and candidate search live in
``zcu_tools.analysis.dispersive``; simulation lives in ``zcu_tools.simulate``.
The app renders analysis results as matplotlib figures in ``viz``.
"""
