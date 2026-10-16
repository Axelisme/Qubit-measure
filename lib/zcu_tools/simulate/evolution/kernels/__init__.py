"""Internal numerical kernel registry used by the segment engine."""

from ..kernel import Kernel
from .t6 import T6Kernel

REGISTRY: tuple[Kernel, ...] = (T6Kernel(),)
