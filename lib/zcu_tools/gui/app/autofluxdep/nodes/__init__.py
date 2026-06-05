"""Node specs for autofluxdep-gui.

``spec`` holds the dependency model; ``io`` holds the Snapshot (read-only
projection in) / Patch (produced container out) contract a Node runs against.
Each other module is one experiment's NodeSpec. Phase C adds lenrabi / ro_opt /
t1 / t2ramsey / t2echo / mist alongside the ``qubit_freq`` worked example.
"""
