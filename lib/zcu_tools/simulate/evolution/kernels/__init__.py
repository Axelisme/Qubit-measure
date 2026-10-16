"""Internal algorithm registry, resolved by stable Kernel.name."""

from ..kernel import Kernel
from .dyson import DysonKernel
from .gate_cache import GateCacheKernel
from .period_table import PeriodTableKernel
from .rwa import RWAKernel
from .t4 import T4
from .t6 import T6Kernel

REGISTRY: tuple[Kernel, ...] = (
    T4(),
    T6Kernel(),
    DysonKernel(),
    RWAKernel(),
    GateCacheKernel(),
    PeriodTableKernel(),
)
