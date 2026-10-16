"""Independent ns/GHz evolution, explicit bases and population conversion.

No QICK, environment, device, resource or YAML dependencies. Import public
records and tools here; submodule layout is not a second integration interface.
"""

from .batch import CPUReference, PlateauFamily
from .controls import ControlPulse, ControlSequence, Envelope, PhaseEvent, quantize_ns
from .engine import (
    Cancelled,
    DriveTrack,
    Engine,
    EnvelopeSpec,
    EvolutionResult,
    PathDiagnostics,
    TimelineBatch,
)
from .intel import IntelOpenCL
from .library import BuildPlan, EvolutionLibrary, LibrarySnapshot, RecordReference
from .model import DriveSource, NumericModel, SolverSettings
from .reuse import NUMERIC_VERSION, EvaluationReceipt
from .secular import (
    FluxExpansion,
    FluxLevel,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from .solver import Evolution, propagator_power
from .state import Basis, DensityState, Populations, populations
from .validation import ValidationEvidence, ValidationLimits
