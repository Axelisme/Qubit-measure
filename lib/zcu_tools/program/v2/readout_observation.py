"""Observe source readout provenance and triggers without exposing compiler wrappers."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qick.asm_v2 import Macro, Trigger

from ._timeline_macros import (
    ModuleEmission,
    PreparedTimelineMacro,
    RepeatLoweringEnter,
    RepeatLoweringExit,
    TimelineBranch,
    TimeProducer,
)
from .macro.readout import ReadoutEmission, readout_control_scope

if TYPE_CHECKING:
    from .timeline import TimelineProgramV2


@dataclass(frozen=True)
class ReadoutObservation:
    """One borrowed readout source macro and its trigger admission context.

    source is a ReadoutEmission provenance marker or an ADC Trigger, not a
    linked instruction or an executed event. It retains the original object
    identity; callers must not mutate it or its owner.
    control_scope is None for provenance markers and ordinary triggers. For
    controlled triggers it is "Repeat", "Repeat <name repr>" or
    "Branch <name repr>", using the innermost timeline scope when present,
    otherwise the native QICK readout control scope.
    """

    source: ReadoutEmission | Trigger
    control_scope: str | None


def observe_readouts(prog: TimelineProgramV2) -> Iterator[ReadoutObservation]:
    """Yield source readout markers and every trigger of a compiled program.

    prog must be a successfully compiled TimelineProgramV2, unmodified while
    iterating. Results borrow its source macros without changing the program.
    Compiler wrappers are hidden. Source order is preserved, branch cases are
    visited in stored case order (including all alternatives), and each repeat
    body is visited once, not multiplied by its execution count. Empty programs
    and empty cases contribute no observations. This is static observation, not
    a runtime trace, path selection, or linked-instruction/timing analysis.
    Trigger control_scope preserves the existing native QICK readout admission
    rules as well as timeline Repeat/Branch context. No label, owner or read-count
    validation is performed here; collection consumers retain those failures.
    """
    prefix: list[Macro] = []
    for source, scope in _source_macros(prog.macro_list):
        if isinstance(source, ReadoutEmission):
            yield ReadoutObservation(source, None)
        elif isinstance(source, Trigger):
            yield ReadoutObservation(source, scope or readout_control_scope(prefix))
        prefix.append(source)


def _source_macros(
    macros: Sequence[Macro], scope: str | None = None
) -> Iterator[tuple[Macro, str | None]]:
    scopes = [scope]
    for macro in macros:
        source = macro.source if isinstance(macro, PreparedTimelineMacro) else macro
        if isinstance(source, RepeatLoweringEnter):
            scopes.append("Repeat")
        elif isinstance(source, RepeatLoweringExit):
            scopes.pop()
        elif isinstance(source, TimelineBranch):
            for case in source.cases:
                yield from _source_macros(case, f"Branch {source.name!r}")
        elif isinstance(source, (ModuleEmission, TimeProducer)):
            yield from _source_macros(source.macros, scopes[-1])
        else:
            yield source, scopes[-1]
