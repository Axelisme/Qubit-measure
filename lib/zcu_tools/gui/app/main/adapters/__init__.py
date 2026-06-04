"""Driven adapters — concrete infrastructure that implements a service port.

Hexagonal/DDD: application services depend on a ``Protocol`` port (see
``services/ports.py``); the Qt / OS / hardware implementation lives here so the
service stays free of that dependency. See ``docs/adr/0008`` § "Driven Adapter".
"""
