"""sim — physical simulation sub-package for program/v2.

Turns a "mock soc carrying SimParams" running the real experiment path into
physically-realistic data, via a TLS Bloch-equation core plus a dispersive
readout model.  The public surface is:

  - :class:`SimParams` — the physical parameter container.
  - :class:`SimEngine` — assembles lowering + bloch + readout into QICK raw
    accumulated I/Q (driven by ``MyProgramV2.acquire`` on a sim soc).

The remaining sub-modules (``bloch``, ``lowering``, ``readout``) are the physics
layers the engine delegates to; import them directly when needed (the
analytic-limit / lowering tests do).
"""

from .engine import SimEngine
from .params import SimParams

__all__ = ["SimParams", "SimEngine"]
