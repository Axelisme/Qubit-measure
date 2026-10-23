"""Internal algorithm registry, resolved by stable Kernel.name."""

from ..kernel import Kernel
from .adiabatic_map import AdiabaticMapKernel
from .common_frame_kernel import CommonFrameKernel
from .dop853 import DOP853Kernel
from .dyson import DysonKernel
from .floquet import FloquetKernel
from .floquet_period import FloquetPeriodKernel
from .flux_dop853 import FluxDOP853Kernel
from .gate_cache import GateCacheKernel
from .multitone_gate import MultitoneGateKernel
from .multitone_rwa import MultitoneRWAKernel
from .period_table import PeriodTableKernel
from .shaped_sambe import ShapedSambeKernel
from .single_tone_gate import SingleToneGateKernel
from .single_tone_rwa import SingleToneRWAKernel
from .slow_envelope import SlowEnvelopeKernel
from .t4 import T4
from .t6 import T6Kernel

REGISTRY: tuple[Kernel, ...] = (
    T4(),
    AdiabaticMapKernel(),
    CommonFrameKernel(),
    T6Kernel(),
    SlowEnvelopeKernel(),
    FloquetPeriodKernel(),
    DysonKernel(),
    DOP853Kernel(),
    FluxDOP853Kernel(),
    GateCacheKernel(),
    PeriodTableKernel(),
    ShapedSambeKernel(),
    FloquetKernel(),
    MultitoneRWAKernel(),
    MultitoneGateKernel(),
    SingleToneGateKernel(),
    SingleToneRWAKernel(),
)
