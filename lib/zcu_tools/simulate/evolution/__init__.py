"""Independent ns/GHz evolution, explicit bases and population conversion.

No QICK, environment, device, resource or YAML dependencies. Import public
records and tools here; submodule layout is not a second integration interface.
"""

from .batch import CPUReference, PlateauFamily
from .controls import ControlPulse, ControlSequence, Envelope, PhaseEvent
from .intel import IntelOpenCL
from .library import BuildPlan, EvolutionLibrary, LibrarySnapshot
from .model import DriveSource, NumericModel, SolverSettings
from .reuse import NUMERIC_VERSION, EvaluationReceipt
from .solver import Evolution, propagator_power
from .state import Basis, DensityState, Populations, populations
from .validation import ValidationEvidence, ValidationLimits
