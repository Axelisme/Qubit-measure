"""Shared experiment implementation mechanics.

This private package contains only helpers used by at least two concrete
measurement experiments.  It may depend on the stable ``nodes`` contracts, but
must never import a concrete experiment or the experiment catalog.
"""
