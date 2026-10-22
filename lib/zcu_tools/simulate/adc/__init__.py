"""Independent finite-pulse ADC readout in physical ns/GHz and ADC sample units.

This tool accepts explicitly ordered classical populations, not densities. It
models deterministic state-conditioned moments, continuous-time shot paths and
receiver noise. ADC owns complete integrated raw sums and numeric layout;
acquisition normalization belongs to program.
"""

from .. import _numba_cache  # Bootstrap before any dispatcher is created.
from .cache import ReadoutCache
from .cavity import (
    CavityAverageCache,
    CavityFieldAverage,
    cavity_field_average,
    cavity_fields,
)
from .model import ReadoutModel, Receiver
from .populations import (
    POPULATION_TOLERANCE,
    PopulationValidationError,
    normalize_populations,
)
from .raw import ADCRawShots, PreparedRawShots
from .readout_batch import (
    ChannelReadout,
    ConditionalSums,
    PreparedStatistics,
    ReadoutBatch,
    round_statistics,
)
from .request import (
    ADCSamples,
    ConditionalResponse,
    PiecewiseLinearEnvelope,
    PopulationInput,
    ReadoutPulse,
    SampleGrid,
)
from .response import ConditionalReadout
from .sampling import ADC, sample_traces
