"""Static register access summaries for legal parallel Module composition."""

from collections.abc import Sequence

from ..modules.base import Module
from ..modules.computed_pulse import ComputedPulse
from ..modules.control import Branch, Repeat
from ..modules.delay import Delay, DelayAuto, Join, SoftDelay
from ..modules.dmem import LoadValue, LoadWord, ScanWith
from ..modules.pulse import Pulse, TableLengthPulse
from ..modules.readout import PulseReadout, TablePulseReadout
from ..modules.reset import BathReset, NoneReset, PulseReset, TwoPulseReset


def validate_modules(
    modules: Sequence[Module], path: str = "body"
) -> list[TablePulseReadout]:
    """Reject unknown semantic operations even in unexecuted branches.

    Return all table readouts whose hardware loop hooks run independently of
    whether their branch played a pulse. Errors include concrete type/name and
    enclosing module path. No program, backend or mutable state is modified.
    """
    tables: list[TablePulseReadout] = []
    supported = (
        Pulse,
        PulseReadout,
        TablePulseReadout,
        TableLengthPulse,
        ComputedPulse,
        Delay,
        SoftDelay,
        DelayAuto,
        Join,
        LoadValue,
        LoadWord,
        Repeat,
        Branch,
        ScanWith,
        NoneReset,
        PulseReset,
        TwoPulseReset,
        BathReset,
    )
    for module in modules:
        location = f"{path}/{type(module).__name__}({module.name!r})"
        if type(module) not in supported:
            raise ValueError(f"{location} simulation unsupported")
        if isinstance(module, TablePulseReadout):
            tables.append(module)
        tables.extend(validate_modules(_children(module), location))
    return tables


def _children(module: Module) -> list[Module]:
    if isinstance(module, Repeat):
        return module.sub_modules
    if isinstance(module, Branch):
        return [child for branch in module.branches for child in branch]
    if isinstance(module, Join):
        return [child for branch in module.join_modules for child in branch]
    if isinstance(module, ScanWith):
        return [module.repeat_mod]
    return []


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
