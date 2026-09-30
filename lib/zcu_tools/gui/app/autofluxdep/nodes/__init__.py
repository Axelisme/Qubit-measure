"""Execution contracts described by ADR-0062 for autofluxdep providers.

``builder`` owns Builder / Node / RunEnv / PlacedNode, ``io`` owns Snapshot /
Patch, and ``spec`` owns dependency declarations. ``predictor`` is the
pure-compute Service on this same execution seam. User-editable measurement
definitions and their shared mechanics live in
``zcu_tools.experiment.v2_gui.autofluxdep``.
"""
