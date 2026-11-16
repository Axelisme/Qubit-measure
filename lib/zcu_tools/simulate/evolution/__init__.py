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
from .frozen import (
    FROZEN_PHYSICAL_VERSION,
    FrozenInput,
    FrozenRecord,
    canonical_json,
    digest,
    physical_input,
    read_frozen_input,
    read_frozen_manifest,
    read_physical_input,
    write_frozen_input,
    write_frozen_manifest,
)
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
from .reference_cache import (
    ReferenceResult,
    SolverTolerances,
    load_reference,
    reference_key,
    write_reference,
)
from .routing_history import RoutingHistory
from .scene_cache import (
    SCENE_CACHE_VERSION,
    FrozenIndexEntry,
    read_frozen_index,
    read_scene_records,
    write_frozen_index,
    write_scene_record,
)
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
