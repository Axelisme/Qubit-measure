"""Observe source readout provenance and triggers without exposing compiler wrappers."""

from __future__ import annotations

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


@dataclass(frozen=True)
class ReadoutPath:
    """One feasible static Branch path and its borrowed readout observations.

    cases lists (selector loop name, zero-based case index) pairs in first-visit
    order. Repeated branches on the same selector share its chosen index.
    observations preserves source order along that path, including unconditional
    emissions. Repeat bodies appear once and retain a Repeat control_scope;
    timeline Branch scope is represented by cases, not control_scope. Native
    QICK control scopes remain unchanged. Empty paths have empty observations.
    All fields are immutable containers; source macros remain borrowed and must
    not be mutated. This is provenance, not a runtime or timing trace.
    """

    cases: tuple[tuple[str, int], ...]
    observations: tuple[ReadoutObservation, ...]


def observe_readout_paths(prog: TimelineProgramV2) -> Iterator[ReadoutPath]:
    """Yield feasible readout paths of an unmodified, compiled timeline program.

    prog must have compiled successfully. Enumerate each combination of timeline
    Branch choices, including implicit empty cases, in zero-based case order.
    Repeated selectors keep the same index. Each result borrows original source
    macros and includes Repeat bodies once, irrespective of their count.
    No labels, ADC settings, read counts or times are validated, and no program
    state is changed. Programs without Branch yield one path, even when empty.
    Enumeration scales with the number of feasible Branch choice combinations.
    """
    for cases, sources in _source_paths(prog.macro_list, ()):
        observations: list[ReadoutObservation] = []
        prefix: list[Macro] = []
        for source, scope in sources:
            if isinstance(source, ReadoutEmission):
                observations.append(ReadoutObservation(source, None))
            elif isinstance(source, Trigger):
                observations.append(
                    ReadoutObservation(source, scope or readout_control_scope(prefix))
                )
            prefix.append(source)
        yield ReadoutPath(cases, tuple(observations))


type _Sources = list[tuple[Macro, str | None]]
type _Choices = tuple[tuple[str, int], ...]


def _source_paths(
    macros: Sequence[Macro], choices: _Choices, scope: str | None = None
) -> Iterator[tuple[_Choices, _Sources]]:
    paths: list[tuple[_Choices, _Sources]] = [(choices, [])]
    scopes = [scope]
    for macro in macros:
        source = macro.source if isinstance(macro, PreparedTimelineMacro) else macro
        if isinstance(source, RepeatLoweringEnter):
            scopes.append("Repeat")
        elif isinstance(source, RepeatLoweringExit):
            scopes.pop()
        elif isinstance(source, (TimelineBranch, ModuleEmission, TimeProducer)):
            expanded: list[tuple[_Choices, _Sources]] = []
            for selected, prefix in paths:
                if isinstance(source, TimelineBranch):
                    for index, case in enumerate(source.cases):
                        previous = dict(selected).get(source.by)
                        if previous is not None and previous != index:
                            continue
                        next_choices = (
                            (*selected, (source.by, index))
                            if previous is None
                            else selected
                        )
                        expanded.extend(
                            (cases, prefix + body)
                            for cases, body in _source_paths(
                                case, next_choices, scopes[-1]
                            )
                        )
                else:
                    expanded.extend(
                        (cases, prefix + body)
                        for cases, body in _source_paths(
                            source.macros, selected, scopes[-1]
                        )
                    )
            paths = expanded
        else:
            for _, body in paths:
                body.append((source, scopes[-1]))
    yield from paths


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
