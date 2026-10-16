"""Internal algorithm registry, resolved by stable Kernel.name."""

from ..kernel import Kernel
from .gate_cache import GateCacheKernel
from .period_table import PeriodTableKernel
from .t4 import T4
from .t6 import T6Kernel

REGISTRY: tuple[Kernel, ...] = (T4(), T6Kernel(), GateCacheKernel(), PeriodTableKernel())
