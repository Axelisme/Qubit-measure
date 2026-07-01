"""Node providers for autofluxdep-gui.

``builder`` holds the execution abstraction — ``Builder`` (the stateless kind of
provider, one subclass per experiment) / ``Node`` (its per-flux-point unit with
the environment curried in) / ``PlacedNode`` (a placed Builder + user params).
``spec`` holds the dependency declarations (``Dependency`` / ``ModuleDep``);
``io`` holds the Snapshot (read-only projection in) / Patch (produced container
out) contract a Node runs against; ``result`` holds the sweep-lived flux-aware
Results. Each experiment module is one Builder; ``predictor`` is a Service Builder
(pure-compute Node). Measurement builders include qubit_freq, ro_optimize,
lenrabi, t1, t2ramsey, t2echo, and mist.
"""
