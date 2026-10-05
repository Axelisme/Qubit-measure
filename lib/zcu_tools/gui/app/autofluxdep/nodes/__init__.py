"""Execution contracts described by ADR-0062 for autofluxdep providers.

``builder`` owns Builder / Node / RunEnv / PlacedNode, ``io`` owns Snapshot /
Patch, and ``spec`` owns dependency declarations. ``predictor`` is the
pure-compute Service on this same execution seam. Callers inject user-editable
measurement declarations through the app catalog; concrete definitions and
shared acquisition mechanics remain outside these execution contracts.
"""
