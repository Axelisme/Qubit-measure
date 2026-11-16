"""Independent ns/GHz evolution, explicit bases and population conversion.

Engine consumes ModelSource and TimelineBatch and returns EvolutionResult.
NumericModel is a separate fixed-coordinate record, not an Engine input.
No QICK, environment, device, resource or YAML dependencies. Import public
records and tools here. Kernel, dispatch, scheduler and cache modules are internal
implementation details, not a second integration interface.
"""

from .. import _numba_cache  # Bootstrap before any dispatcher is created.
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
    RWAGuardMetric,
    RWAUncomputed,
    interaction_hamiltonian,
    rwa_coefficients,
)
from .kernels.common_frame import CommonFrameDiagnostic, common_frame_diagnostic
from .kernels.multitone_rwa import MultitoneRWADiagnostic, multitone_rwa_diagnostic
from .kernels.single_tone_rwa import SingleToneRWADiagnostic, single_tone_rwa_diagnostic
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
