"""Independent ns/GHz evolution, explicit bases and population conversion.

Engine consumes ModelSource and TimelineBatch and returns EvolutionResult.
NumericModel is a separate fixed-coordinate record, not an Engine input.
No QICK, environment, device, resource or YAML dependencies. Import public
records and tools here. Kernel, dispatch, scheduler and cache modules are internal
implementation details, not a second integration interface.
"""

from .. import _numba_cache  # Bootstrap before any dispatcher is created.
from ._inspection import DispatchCandidate, DispatchInspection, DispatchRow, RowKey
from .controls import ControlPulse, ControlSequence, Envelope, PhaseEvent, quantize_ns
from .engine import (
    SEGMENT_NUMERIC_VERSION,
    Cancelled,
    DriveTrack,
    Engine,
    EnvelopeSpec,
    EvolutionResult,
    PathDiagnostics,
    TimelineBatch,
    merge_path_diagnostics,
    segment_spans,
)
from .exceptions import NoValidKernel
from .joint_fock import (
    JointFockDiagnostic,
    JointFockRequest,
    JointFockResult,
    joint_fock_spectrum,
)
from .kernels._rwa_numeric import (
    RWACoefficients,
    interaction_hamiltonian,
    rwa_coefficients,
)
from .model import DriveSource, NumericModel
from .routing_history import RoutingHistory
from .secular import (
    FluxExpansion,
    FluxLevel,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    SecularDrive,
    SecularModel,
    SegmentShift,
)
from .state import Basis, DensityState, Populations, batch_populations, populations
from .telemetry import FamilyRouting
