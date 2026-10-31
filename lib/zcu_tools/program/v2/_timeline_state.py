"""Relative timeline state, native channel tails and repeat scope bookkeeping."""

import warnings
from dataclasses import dataclass, field, replace

from qick.asm_v2 import QickParam, WriteReg

from ._timeline_expression import (
    CycleVariable,
    LinearTime,
    TimeExpression,
    add_time,
    choose_time,
    exact_bounds,
    expression_axes,
    multiply_time,
    projected_expression_bounds,
    reduce_loop_time,
    scale_time,
    select_time,
    slack_variables,
)
from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds
from ._timeline_runtime import (
    RuntimeEnd,
    RuntimeMath,
    Time,
    bounded_source_expression,
    lookup_expression,
    native_event_expression,
    projected_bounds,
    source_expression,
    time_bounds,
    time_expression,
)
from .macro.table import TableSelection
from .macro.write_reg import WriteRegOp


@dataclass(frozen=True)
class RepeatFrameSnapshot:
    """Repeat path effects saved while Branch prepares each case.

    advance is reference movement in us; variable marks path-dependent movement.
    written_gens and written_ros contain channel IDs touched on the path.
    pulse_guards holds independent per-generator playback guards for that path.
    """

    advance: float | QickParam
    variable: bool
    written_gens: set[int]
    written_ros: set[int]
    pulse_guards: dict[int, _PulseGuard]
    expression_advance: TimeExpression


@dataclass
class RepeatFrame:
    parent: str
    external: dict[str, Time]
    waiting: set[str]
    native_gens: dict[int, float | QickParam]
    native_ros: dict[int, float | QickParam]
    advance: float | QickParam = 0.0
    variable: bool = False
    written_gens: set[int] = field(default_factory=set)
    written_ros: set[int] = field(default_factory=set)
    pulse_guards: dict[int, _PulseGuard] = field(default_factory=dict)
    defined: set[str] = field(default_factory=set)
    initial_external: dict[str, Time] = field(default_factory=dict)
    entry_math: RuntimeMath | None = None
    debits: list[tuple[RuntimeMath, Time]] = field(default_factory=list)
    tail_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    tail_ros: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_ros: dict[int, RuntimeEnd] = field(default_factory=dict)
    expression_advance: TimeExpression = field(default_factory=LinearTime)
    initial_native_expressions: dict[tuple[bool, int], TimeExpression] = field(
        default_factory=dict
    )
    initial_source_expressions: dict[str, TimeExpression] = field(default_factory=dict)

    def restore_expressions(
        self,
        prog,
        count: TimeExpression,
        generated_gens: dict[int, Time],
        generated_ros: dict[int, Time],
        *,
        axis: str,
    ) -> None:
        """Restore total Repeat debit provenance without changing register plans.

        External tails lose count times the iteration's actual advance. Events
        generated inside the loop instead keep their last-iteration remainder.
        """
        total = (
            reduce_loop_time(self.expression_advance, axis)
            if axis in expression_axes(self.expression_advance)
            else multiply_time(self.expression_advance, count)
        )
        book = prog._timeline_book
        for outer in book.repeats:
            outer.expression_advance = add_time(
                outer.expression_advance,
                add_time(total, scale_time(self.expression_advance, -1)),
            )
        for name, initial in self.initial_external.items():
            current = book.ends[name]
            if isinstance(current, RuntimeEnd):
                expression = choose_time(
                    LinearTime(),
                    add_time(time_expression(prog, initial), scale_time(total, -1)),
                )
                book.ends[name] = replace(current, expression=expression)
        for native, initial_runtime, current, generated, written in (
            (
                self.native_gens,
                self.initial_runtime_gens,
                book.runtime_gens,
                generated_gens,
                self.written_gens,
            ),
            (
                self.native_ros,
                self.initial_runtime_ros,
                book.runtime_ros,
                generated_ros,
                self.written_ros,
            ),
        ):
            for ch, end in current.items():
                initial = initial_runtime.get(ch, native[ch])
                expression = (
                    select_time(
                        choose_time(LinearTime(1), count, maximum=False),
                        (
                            time_expression(prog, initial),
                            time_expression(prog, generated[ch]),
                        ),
                    )
                    if ch in written
                    else choose_time(
                        LinearTime(),
                        add_time(time_expression(prog, initial), scale_time(total, -1)),
                    )
                )
                current[ch] = replace(end, expression=expression)
        for key, initial in self.initial_native_expressions.items():
            generator, ch = key
            written = self.written_gens if generator else self.written_ros
            if ch in written:
                generated = book.native_expressions[key]
                if axis in expression_axes(generated):
                    generated = reduce_loop_time(generated, axis, last=True)
                expression = select_time(
                    choose_time(LinearTime(1), count, maximum=False),
                    (initial, generated),
                )
            else:
                expression = choose_time(
                    LinearTime(), add_time(initial, scale_time(total, -1))
                )
            book.native_expressions[key] = expression
        self.restore_sources(book, count, axis=axis)
        book.source_expressions[axis] = LinearTime(0)

    def restore_sources(
        self, book: "TimelineBook", count: TimeExpression, *, axis: str
    ) -> None:
        """Keep entry values at zero iterations, otherwise the last body write."""
        limits = projected_expression_bounds(count)
        if limits.maximum == 0:
            book.source_expressions = self.initial_source_expressions.copy()
            return
        executed = choose_time(LinearTime(1), count, maximum=False)
        restored = {}
        for name, expression in book.source_expressions.items():
            if axis in expression_axes(expression):
                expression = reduce_loop_time(expression, axis, last=True)
            if limits.minimum == 0:
                if name not in self.initial_source_expressions:
                    # A body-only definition is not available on the zero path.
                    continue
                expression = select_time(
                    executed, (self.initial_source_expressions[name], expression)
                )
            restored[name] = expression
        book.source_expressions = restored

    def merge_expression_advance(
        self, prog, selector: str, snapshots: list[RepeatFrameSnapshot]
    ) -> None:
        """Retain the branch-selected iteration advance on the selector's axis."""
        self.expression_advance = select_time(
            source_expression(prog, selector),
            tuple(s.expression_advance for s in snapshots),
        )

    def promote(
        self, prog, generated_gens: dict[int, Time], generated_ros: dict[int, Time]
    ) -> None:
        """Materialize unknown total advances at entry and debit their exact paths.

        Preprocessing records reference transfers at their owning macro plans.
        Only an unknown count/path requests these entry stores and transfers.
        Native tails share a store with a matching named end when possible.
        """
        if self.entry_math is None:
            raise RuntimeError("repeat entry has no arithmetic plan")
        for name, end in self.external.items():
            if time_bounds(prog, end)[1] > 1e-12:
                self.external[name] = self._store(prog, end)
        self.tail_gens = self._native_stores(
            prog, self.native_gens, self.initial_runtime_gens, self.written_gens
        )
        self.tail_ros = self._native_stores(
            prog, self.native_ros, self.initial_runtime_ros, self.written_ros
        )
        stores = {
            end.register: end
            for end in [
                *self.external.values(),
                *self.tail_gens.values(),
                *self.tail_ros.values(),
            ]
            if isinstance(end, RuntimeEnd)
        }
        for math, advance in self.debits:
            for end in stores.values():
                remaining = math.cycles(math.floor(end, advance))
                math.operations.append(
                    WriteReg(dst=end.register, src=remaining.register)
                )
        for tails, generated, written in (
            (self.tail_gens, generated_gens, self.written_gens),
            (self.tail_ros, generated_ros, self.written_ros),
        ):
            for ch in written:
                operand = prog._timeline_book.math.cycles(generated[ch])
                prog._timeline_book.math.operations.append(
                    WriteReg(dst=tails[ch].register, src=operand.register)
                )
                tail = tails[ch]
                tails[ch] = RuntimeEnd(
                    tail.register, 0, max(tail.maximum, operand.maximum)
                )

    def _store(self, prog, end: Time) -> RuntimeEnd:
        if self.entry_math is None:
            raise RuntimeError("repeat entry has no arithmetic plan")
        original = self.entry_math.cycles(end)
        register = prog.add_reg()
        self.entry_math.operations.append(WriteReg(dst=register, src=original.register))
        return RuntimeEnd(register, 0, original.maximum)

    def _native_stores(
        self,
        prog,
        initial: dict[int, float | QickParam],
        runtime: dict[int, RuntimeEnd],
        written: set[int],
    ) -> dict[int, RuntimeEnd]:
        result: dict[int, RuntimeEnd] = {}
        for ch, value in initial.items():
            end = runtime.get(ch, value)
            if time_bounds(prog, end)[1] <= 1e-12 and ch not in written:
                continue
            match = next(
                (
                    name
                    for name, candidate in self.initial_external.items()
                    if candidate is end
                    or (
                        not isinstance(candidate, RuntimeEnd)
                        and not isinstance(end, RuntimeEnd)
                        and all(abs(v) < 1e-12 for v in bounds(candidate - end))
                    )
                ),
                None,
            )
            candidate = (
                self.external[match]
                if match is not None and ch not in written
                else None
            )
            result[ch] = (
                candidate
                if isinstance(candidate, RuntimeEnd)
                else self._store(prog, end)
            )
        return result

    def rebase_native(self, prog, count: int) -> None:
        """Debit external timestamps, preserving tails generated in the last iteration."""
        generated_gens = {ch: prog.get_timestamp(gen_ch=ch) for ch in self.written_gens}
        generated_ros = {ch: prog.get_timestamp(ro_ch=ch) for ch in self.written_ros}
        for ch, end in self.native_gens.items():
            prog.set_timestamp(end, gen_ch=ch)
        for ch, end in self.native_ros.items():
            prog.set_timestamp(end, ro_ch=ch)
        prog.decrement_timestamps(self.advance * count)
        for ch, end in generated_gens.items():
            # With a zero minimum the body may not execute. Retain the later
            # native timestamp; QICK owns native timestamp comparison policy.
            value = end if count else max(end, prog.get_timestamp(gen_ch=ch))
            prog.set_timestamp(value, gen_ch=ch)
        for ch, end in generated_ros.items():
            value = end if count else max(end, prog.get_timestamp(ro_ch=ch))
            prog.set_timestamp(value, ro_ch=ch)


@dataclass(frozen=True)
class BranchProvenance:
    """Register and channel expressions captured at the same branch position.

    sources maps register names to integer tProc-cycle expressions. native maps
    (is_generator, channel_id) to relative integer-cycle tails. Both mappings
    are detached snapshots, so preparing another case cannot mutate this path.
    """

    sources: dict[str, TimeExpression]
    native: dict[tuple[bool, int], TimeExpression]


@dataclass
class BranchProvenanceFrame:
    """One active branch's entry snapshot and completed case snapshots.

    entry restores both provenance domains before each case. paths stores one
    paired snapshot per completed case, in selector order. A single stack frame
    owns their common begin/restore/record/finish lifetime, including nesting.
    """

    entry: BranchProvenance
    paths: list[BranchProvenance] = field(default_factory=list)


class TimelineBook:
    """Store only relative ends, including the ordinary default timeline."""

    def __init__(self):
        self.ends: dict[str, Time] = {"default": 0.0}
        self.shell_wait_bound: int | None = None
        self.cycle_axes: dict[str, CycleVariable] = {}
        self.source_expressions: dict[str, TimeExpression] = {}
        self.branch_provenance: list[BranchProvenanceFrame] = []
        self.math: RuntimeMath
        self.preparing = 0
        self.pinned: set[str] = set()
        self.runtime_gens: dict[int, RuntimeEnd] = {}
        self.runtime_ros: dict[int, RuntimeEnd] = {}
        # Compile-only event tails. Runtime stores and QICK timestamps keep
        # their existing scheduling roles; these expressions never emit code.
        # Keys are (is_generator, channel_id).
        self.native_expressions: dict[tuple[bool, int], TimeExpression] = {}
        self.waiting: set[str] = set()
        self.repeats: list[RepeatFrame] = []
        self.native_only = False
        self.used: set[str] | None = None
        self.case_lineage: list[tuple[dict[str, str], set[str]]] = []

    def repeat_count(
        self, prog, iterations: int | str, minimum: int, maximum: int
    ) -> TimeExpression:
        """Describe count points without requiring a bounds-only count producer."""
        return (
            bounded_source_expression(prog, iterations, minimum, maximum)
            if isinstance(iterations, str)
            else LinearTime(iterations)
        )

    def observe_selection(
        self, prog, register: str, selection: TableSelection | None
    ) -> None:
        """Capture ordinary/selected table values after a valid producer occurrence.

        Invalid or undefined inputs remain the producer audit's deferred error;
        non-time lookups must not gain a new timing initialization requirement.
        """
        if selection is None or selection.register != register:
            return
        if prog._time_bounds.input_errors.get(register):
            self.source_expressions.pop(register, None)
            return
        self.source_expressions[register] = lookup_expression(
            prog, register, selection.table.values, selection.selector
        )

    def selected_expression(
        self, prog, selector: str, values: list[Time]
    ) -> TimeExpression:
        """Select path ends at their common sweep index without interval union."""
        return select_time(
            source_expression(prog, selector),
            tuple(time_expression(prog, v) for v in values),
        )

    def begin_branch_sources(self, prog) -> None:
        """Save incoming source/channel expressions before collecting branch cases."""
        self.branch_provenance.append(
            BranchProvenanceFrame(self.snapshot_branch_provenance(prog))
        )

    def restore_branch_sources(self) -> None:
        """Start a case with the same incoming source values as its siblings."""
        entry = self.branch_provenance[-1].entry
        self.source_expressions = entry.sources.copy()
        self.native_expressions = entry.native.copy()

    def record_branch_sources(self, prog) -> None:
        """Capture source/channel values after this case's local timeline closure."""
        self.branch_provenance[-1].paths.append(self.snapshot_branch_provenance(prog))

    def snapshot_branch_provenance(self, prog) -> BranchProvenance:
        """Copy current register sources and all declared native channel tails."""
        return BranchProvenance(
            self.source_expressions.copy(), self.snapshot_native_expressions(prog)
        )

    def finish_branch_sources(self, prog, selector: str) -> None:
        """Select channel tails and register sources defined on every branch path."""
        frame = self.branch_provenance.pop()
        self.native_expressions = {
            key: select_time(
                source_expression(prog, selector),
                tuple(path.native[key] for path in frame.paths),
            )
            for key in frame.paths[0].native
        }
        self.source_expressions = {
            name: select_time(
                source_expression(prog, selector),
                tuple(path.sources[name] for path in frame.paths),
            )
            for name in set.intersection(*(set(path.sources) for path in frame.paths))
        }

    def observe_write(self, prog, macro) -> None:
        """Keep current native constant/copy/add/subtract point metadata in sync."""
        if type(macro) is WriteReg:
            if isinstance(macro.src, int):
                self.source_expressions[macro.dst] = LinearTime(macro.src)
            elif isinstance(macro.src, str) and self._known_source(prog, macro.src):
                self.source_expressions[macro.dst] = source_expression(prog, macro.src)
            else:
                self.source_expressions.pop(macro.dst, None)
        elif type(macro) is WriteRegOp:
            self._observe_alu(prog, macro)

    def _known_source(self, prog, name: str) -> bool:
        register = prog.reg_dict.get(name)
        return (
            name in self.source_expressions
            or name in prog.loop_dict
            or (register is not None and register.init is not None)
        )

    def _observe_alu(self, prog, macro: WriteRegOp) -> None:
        if macro.op not in ("+", "-") or not self._known_source(prog, macro.lhs):
            self.source_expressions.pop(macro.dst, None)
            return
        right = macro.rhs
        if isinstance(right, int):
            rhs = LinearTime(right)
        elif isinstance(right, str) and self._known_source(prog, right):
            rhs = source_expression(prog, right)
        else:
            self.source_expressions.pop(macro.dst, None)
            return
        self.source_expressions[macro.dst] = add_time(
            source_expression(prog, macro.lhs),
            rhs if macro.op == "+" else scale_time(rhs, -1),
        )

    def set_end(self, name: str, end: Time) -> None:
        self.math.check_end(end)
        self.ends[name] = end

    def require(self, name: str) -> None:
        """Reject unknown names without marking a timeline used or changing its end."""
        if name not in self.ends:
            raise ValueError(f"{name}: unknown timeline; declare it with fork in _body")

    def end(self, name: str, *, placement: bool = False) -> Time:
        self.require(name)
        if self.used is not None:
            self.used.add(name)
        if placement and name in self.waiting:
            raise ValueError(
                f"{name}: timeline can only wait after register repeat; "
                "sync it or use join/fork to establish a conservative scheduling point"
            )
        return self.ends[name]

    def decrement(self, t: Time, *, observe_pulses: bool = True) -> None:
        if self.native_only:
            return
        if observe_pulses:
            advance = time_expression(self.math.prog, t)
            self.native_expressions = {
                key: choose_time(LinearTime(), add_time(end, scale_time(advance, -1)))
                for key, end in self.native_expressions.items()
            }
        consumed = self._debit_native(t) if observe_pulses else set()
        self._debit_repeats(t, observe_pulses=observe_pulses, consumed=consumed)
        self.ends = {name: self.math.floor(end, t) for name, end in self.ends.items()}
        self.waiting = {
            name
            for name in self.waiting
            if time_bounds(self.math.prog, self.ends[name])[1] > 1e-12
        }

    def _debit_native(self, t: Time) -> set[int]:
        consumed: set[int] = set()
        for channels, generator in (
            (self.runtime_gens, True),
            (self.runtime_ros, False),
        ):
            for ch, end in list(channels.items()):
                remaining = self.math.floor(end, t)
                if isinstance(remaining, RuntimeEnd):
                    channels[ch] = remaining
                else:
                    del channels[ch]
                    if generator and time_bounds(self.math.prog, remaining)[1] <= 1e-12:
                        consumed.add(ch)
                self.native_timestamp(
                    self.math.prog,
                    projected_bounds(self.math.prog, remaining).maximum,
                    gen_ch=ch if generator else None,
                    ro_ch=None if generator else ch,
                )

        return consumed

    def _debit_repeats(
        self, t: Time, *, observe_pulses: bool, consumed: set[int]
    ) -> None:
        if self.repeats:
            if observe_pulses:
                for active in self.repeats:
                    active.debits.append((self.math, t))
                    active.expression_advance = add_time(
                        active.expression_advance, time_expression(self.math.prog, t)
                    )
            frame = self.repeats[-1]
            proof_advance = (
                time_bounds(self.math.prog, t)[0] if isinstance(t, RuntimeEnd) else t
            )
            frame.variable = frame.variable or isinstance(t, RuntimeEnd)
            frame.advance += proof_advance
            if observe_pulses:
                for ch, guard in frame.pulse_guards.items():
                    # A consumed runtime pulse end proves a constant-zero
                    # transfer, preserving correlation between start and end.
                    shifts = () if ch in consumed else (-proof_advance,)
                    guard.then(_PulseGuard(ch, shifts=shifts, floors=(0.0,)))

    def native_timestamp(
        self,
        prog,
        value: float | QickParam,
        *,
        gen_ch: int | None = None,
        ro_ch: int | None = None,
    ) -> None:
        """Project a compiler bound without recording a new hardware event."""
        previous = self.native_only
        self.native_only = True
        try:
            prog.set_timestamp(value, gen_ch=gen_ch, ro_ch=ro_ch)
        finally:
            self.native_only = previous

    def snapshot_native_expressions(
        self, prog
    ) -> dict[tuple[bool, int], TimeExpression]:
        """Capture all declared channel tails for a Repeat or Branch entry/path."""
        return {
            (generator, ch): self.native_expression(prog, ch, generator=generator)
            for generator, channels in ((True, prog.gen_chs), (False, prog.ro_chs))
            for ch in channels
        }

    def record_native(self, prog, ch: int, end: Time, *, generator: bool) -> None:
        """Capture an actual event end in cycles, without changing native storage."""
        self.native_expressions[generator, ch] = native_event_expression(prog, end)

    def native_expression(self, prog, ch: int, *, generator: bool) -> TimeExpression:
        """Return the compile-only tail, or the initial native end before any event."""
        key = (generator, ch)
        if key in self.native_expressions:
            return self.native_expressions[key]
        return native_event_expression(
            prog, self.channel_end(prog, ch, generator=generator)
        )

    def channel_end(self, prog, ch: int, *, generator: bool) -> Time:
        """Return a declared native channel's exact runtime end or static timestamp."""
        runtime = self.runtime_gens if generator else self.runtime_ros
        if ch in runtime:
            return runtime[ch]
        return prog.get_timestamp(
            gen_ch=ch if generator else None, ro_ch=None if generator else ch
        )

    def channel_maximum(self, prog, *, gens: bool, ros: bool) -> Time | None:
        """Return selected native channel ends, retaining any runtime value.

        None means neither selection contains a declared channel, matching QICK
        auto-delay behavior. Static channels keep their native timestamps.
        """
        result: Time | None = None
        for enabled, channels, generator in (
            (gens, prog.gen_chs, True),
            (ros, prog.ro_chs, False),
        ):
            if enabled:
                for ch in channels:
                    value = self.channel_end(prog, ch, generator=generator)
                    result = (
                        value if result is None else self.math.maximum(result, value)
                    )
        return result

    def check_shell_wait(self, prog) -> None:
        """Project final ADC WAIT and diagnose slack after all body cancellation.

        Known producers use exact legal-point extrema. A bounds-only unknown
        emits an explicitly conservative warning and never triggers the relax
        guard. This check emits no runtime instructions.
        """
        if not prog.ro_chs:
            return
        expression = choose_time(
            *(self.native_expression(prog, ch, generator=False) for ch in prog.ro_chs)
        )
        exact = exact_bounds(expression)
        limits = exact or projected_expression_bounds(expression)
        # Keep native timestamps untouched: final_delay must still use its
        # original pointwise reference advance, not the scalar shell WAIT.
        self.shell_wait_bound = limits.maximum
        slack = limits.maximum - limits.minimum
        if slack > 0:
            variables = slack_variables(expression)
            message = (
                f"Shell WAIT static bound: variables={variables}, slack={slack} cycles"
            )
            relax = prog.us2cycles(prog.relax_delay)
            if exact is None:
                message += " (slack upper bound, not actual slack)"
            elif slack > relax:
                raise ValueError(f"{message}, relax={relax} cycles")
            warnings.warn(message, UserWarning, stacklevel=2)

    def maximum(self) -> Time:
        end: Time = 0.0
        for candidate in self.ends.values():
            end = self.math.maximum(end, candidate)
        return end
