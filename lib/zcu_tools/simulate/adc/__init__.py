"""Independent finite-pulse ADC readout in physical ns/GHz and ADC sample units.

This tool accepts explicitly ordered classical populations, not densities. It
models deterministic state-conditioned moments, initial joint-state draws and
receiver noise; QICK representation/normalization belongs to the adapter.
"""

from .model import ReadoutModel, Receiver
from .request import (
    ADCSamples,
    ConditionalResponse,
    PopulationInput,
    ReadoutPulse,
    SampleGrid,
)
from .sampling import ADC
