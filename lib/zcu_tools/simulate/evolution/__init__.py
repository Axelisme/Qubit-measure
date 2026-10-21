"""Independent ns/GHz evolution, explicit bases and population conversion.

Engine consumes ModelSource and TimelineBatch and returns EvolutionResult.
NumericModel is a separate fixed-coordinate record, not an Engine input.
No QICK, environment, device, resource or YAML dependencies. Import public
records and tools here. Kernel, dispatch, scheduler and cache modules are internal
implementation details, not a second integration interface.
"""

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
from .joint_fock import JointFockRequest, JointFockResult, joint_fock_spectrum
from .model import DriveSource, NumericModel
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
