"""Independent finite-pulse ADC readout in physical ns/GHz and ADC sample units.

This tool accepts explicitly ordered classical populations, not densities. It
models deterministic state-conditioned moments, initial joint-state draws and
receiver noise; QICK representation/normalization belongs to the adapter.
"""

from .model import ReadoutModel, Receiver
from .populations import POPULATION_TOLERANCE, normalize_populations
from .readout_batch import (
    ChannelReadout,
    ConditionalSums,
    ReadoutBatch,
    round_statistics,
    sample_traces,
)
from .request import (
    ADCSamples,
    ConditionalResponse,
    PopulationInput,
    ReadoutPulse,
    SampleGrid,
)
from .response import ConditionalReadout
from .sampling import ADC
