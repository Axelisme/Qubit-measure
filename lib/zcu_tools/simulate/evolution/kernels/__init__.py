"""Internal numerical kernel registry used by the segment engine."""

from ..kernel import Kernel
from .gate_cache import GateCacheKernel
from .period_table import PeriodTableKernel
from .t6 import T6Kernel

REGISTRY: tuple[Kernel, ...] = (T6Kernel(), GateCacheKernel(), PeriodTableKernel())
