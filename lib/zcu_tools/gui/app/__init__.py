"""GUI app composition roots.

One subpackage per standalone GUI app: ``main`` (the measure-gui), ``fluxdep``
(flux-dependence analysis), ``dispersive`` (dispersive-shift fitting), and
``autofluxdep`` (automated multi-task flux sweeps). The apps are independent; the
only extracted shared layers so far are ``gui/remote`` and ``gui/plotting`` —
each app otherwise carries its own copy of the common machinery (decision D2:
don't abstract prematurely; converge once requirements stabilize).
"""
