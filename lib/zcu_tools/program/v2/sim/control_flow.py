"""Resolve the control-flow modules of a timeline at one sweep point.

``Branch``, ``Repeat`` and ``ComputedPulse`` pick or repeat their content from
registers.  When a sweep-loop counter or a LoadValue table determines those
registers, the content each one plays at a sweep point is known statically, so
the lowering replaces it by the plain modules it plays.
"""

from __future__ import annotations

from collections.abc import Sequence

from qick.asm_v2 import QickParam

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse
from zcu_tools.program.v2.modules.readout import AbsReadout

from .dmem import DmemTable, load_value
from .errors import UnsupportedModuleError


def select_branch(branch: Branch, point: dict[str, int]) -> list[Module]:
    """Pick the active sub-sequence of a deterministic ``Branch`` at one point.

    ``Branch`` (control.py) does not create its own loop: it selects branch *i*
    from the value of ``compare_reg``, which is the counter of an *external* sweep
    loop registered via ``add_loop``.  Lowering already holds that counter as the
    ``point[compare_reg]`` index, so the branch taken at this sweep point is fully
    determined statically — no measurement feedback is involved.

    Fast-fail (per CLAUDE.md): if ``compare_reg`` is not a sweep axis in ``point``
    the selector is not a registered loop counter (e.g. a measurement-conditional
    branch), which this lowering cannot resolve; raise instead of guessing.  An
    out-of-range index also raises.
    """

    if branch.compare_reg not in point:
        raise UnsupportedModuleError(
            f"Branch {branch.name!r} selects on register {branch.compare_reg!r}, "
            "which is not a sweep axis at this point; only deterministic branches "
            "driven by a registered sweep-loop counter can be lowered"
        )
    idx = point[branch.compare_reg]
    if not 0 <= idx < len(branch.branches):
        raise UnsupportedModuleError(
            f"Branch {branch.name!r} index {idx} is out of range "
            f"(it has {len(branch.branches)} branches)"
        )
    return branch.branches[idx]


def _static_us(value: float | QickParam, *, consumer: str) -> float:
    if isinstance(value, QickParam):
        raise UnsupportedModuleError(f"{consumer} has a swept timing parameter")
    return float(value)


def _computed_pulse_modules(
    module: ComputedPulse, point: dict[str, int], dmem_tables: dict[str, DmemTable]
) -> list[Module]:
    """Pick the candidate a ``ComputedPulse`` plays at one point.

    The hardware reads the candidate index from ``val_reg`` and advances time by
    the longest candidate, so a shorter candidate is followed by idle padding.
    """

    consumer = f"ComputedPulse {module.name!r}"
    idx = load_value(module.val_reg, dmem_tables, point, consumer=consumer)
    if not 0 <= idx < len(module.pulse_modules):
        raise UnsupportedModuleError(
            f"{consumer} index {idx} is out of range "
            f"(it has {len(module.pulse_modules)} candidates)"
        )

    def total_us(pulse: Pulse) -> float:
        cfg = pulse.cfg
        assert cfg is not None
        return sum(
            _static_us(value, consumer=consumer)
            for value in (cfg.pre_delay, cfg.waveform.length, cfg.post_delay)
        )

    chosen = module.pulse_modules[idx]
    padding = max(total_us(p) for p in module.pulse_modules) - total_us(chosen)
    if padding <= 0.0:
        return [chosen]
    return [chosen, SoftDelay(f"{module.name}_pad", padding)]


def unroll(
    module: Module, point: dict[str, int], dmem_tables: dict[str, DmemTable]
) -> list[Module]:
    """Expand ``Repeat`` and ``ComputedPulse`` into the modules they play.

    A ``Repeat`` count comes from its int ``n`` or from the LoadValue table
    behind its ``n`` register; its body plays back to back.  A readout or a
    ``Branch`` inside a ``Repeat`` fast-fails.  Every other module passes
    through unchanged.
    """

    if isinstance(module, ComputedPulse):
        return _computed_pulse_modules(module, point, dmem_tables)
    if not isinstance(module, Repeat):
        return [module]

    if isinstance(module.n, int):
        count = module.n
    else:
        count = load_value(
            module.n, dmem_tables, point, consumer=f"Repeat {module.name!r}"
        )
    body: list[Module] = []
    for sub in module.sub_modules:
        if isinstance(sub, (AbsReadout, Branch)):
            raise UnsupportedModuleError(
                f"Repeat {module.name!r} contains {type(sub).__name__} "
                f"{sub.name!r}; only evolution modules can repeat"
            )
        body.extend(unroll(sub, point, dmem_tables))
    return body * count


def iter_evolution_modules(
    modules: Sequence[Module],
    point: dict[str, int],
    dmem_tables: dict[str, DmemTable],
) -> list[Module]:
    """Flatten the timeline at one point into the modules it plays.

    A ``Branch`` is replaced by the sub-sequence chosen by ``point``, and
    ``Repeat`` / ``ComputedPulse`` by their expansion (see ``unroll``), so a
    qubit ``Pulse`` inside any of them defines the frame.  Nested branches are
    not flattened here; they fast-fail when lowered.
    """

    flat: list[Module] = []
    for module in modules:
        if isinstance(module, Branch):
            for sub in select_branch(module, point):
                flat.extend(unroll(sub, point, dmem_tables))
        else:
            flat.extend(unroll(module, point, dmem_tables))
    return flat
