"""Static register access summaries for legal parallel Module composition."""

from collections.abc import Sequence

from ..modules.base import Module
from ..modules.computed_pulse import ComputedPulse
from ..modules.control import Branch, Repeat
from ..modules.delay import DelayAuto, Join
from ..modules.dmem import LoadValue, LoadWord, ScanWith
from ..modules.pulse import TableLengthPulse
from ..modules.readout import PulseReadout, TablePulseReadout


def register_access(modules: Sequence[Module]) -> tuple[set[str], set[str]]:
    """Return registers read/written by modules, including nested paths.

    Names are semantic program registers, not lowered hardware addresses. This
    conservative summary includes all Branch candidates. It does not initialize
    registers or execute modules; Join uses it to reject cross-branch dependency.
    """
    reads: set[str] = set()
    writes: set[str] = set()
    for module in modules:
        nested: list[Module] = []
        if isinstance(module, (LoadValue, LoadWord)):
            reads.add(module.idx_reg)
            writes.add(module.val_reg)
        elif isinstance(module, ComputedPulse):
            reads.add(module.val_reg)
        elif isinstance(module, (TableLengthPulse, TablePulseReadout)):
            reads.add(module.idx_reg)
        elif isinstance(module, PulseReadout):
            reads.update(
                r
                for r in (module.freq_val, module.ro_freq_val, module.gain_val)
                if r is not None
            )
        elif isinstance(module, DelayAuto) and isinstance(module.t, str):
            reads.add(module.t)
        elif isinstance(module, Branch):
            reads.add(module.compare_reg)
            nested = [child for branch in module.branches for child in branch]
        elif isinstance(module, Repeat):
            if isinstance(module.n, str):
                reads.add(module.n)
            writes.add(module.counter_reg)
            nested = module.sub_modules
        elif isinstance(module, ScanWith):
            nested = [module.repeat_mod]
        elif isinstance(module, Join):
            nested = [child for branch in module.join_modules for child in branch]
        child_reads, child_writes = (
            register_access(nested) if nested else (set(), set())
        )
        reads.update(child_reads)
        writes.update(child_writes)
    return reads, writes
