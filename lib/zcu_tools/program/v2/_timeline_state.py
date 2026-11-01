"""Relative timeline state, native channel tails and repeat scope bookkeeping."""

import warnings
from dataclasses import dataclass, field, replace

from qick.asm_v2 import QickParam, WriteReg

from ._timeline_axis import CycleAxis, CycleProducer, TimeVariable
from ._timeline_expression import (
    TimeExpression,
    add_time,
    choose_time,
    cycle_axis,
    difference_limits,
    exact_bounds,
    expression_axes,
    multiply_time,
    projected_expression_bounds,
    reduce_loop_time,
    scale_time,
    select_time,
    slack_variables,
    substitute_time,
    unbounded_time,
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
    optional_time_bounds,
    projected_bounds,
    runtime_expression,
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
    native_gens: dict[int, float | QickParam]
    native_ros: dict[int, float | QickParam]
    advance: float | QickParam = 0.0
    variable: bool = False
    written_gens: set[int] = field(default_factory=set)
    written_ros: set[int] = field(default_factory=set)
    pulse_guards: dict[int, _PulseGuard] = field(default_factory=dict)
    # (timeline, ADC channel, entry-relative start cycles, branch selector/index path).
    adc_starts: list[tuple[str, int, TimeExpression, tuple[tuple[str, int], ...]]] = (
        field(default_factory=list)
    )
    defined: set[str] = field(default_factory=set)
    initial_external: dict[str, Time] = field(default_factory=dict)
    entry_math: RuntimeMath | None = None
    debits: list[tuple[RuntimeMath, Time]] = field(default_factory=list)
    tail_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    tail_ros: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_ros: dict[int, RuntimeEnd] = field(default_factory=dict)
    expression_advance: TimeExpression = field(default_factory=TimeVariable)
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
                    TimeVariable(),
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
                        choose_time(TimeVariable(1), count, maximum=False),
                        (
                            time_expression(prog, initial),
                            time_expression(prog, generated[ch]),
                        ),
                    )
                    if ch in written
                    else choose_time(
                        TimeVariable(),
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
                    choose_time(TimeVariable(1), count, maximum=False),
                    (initial, generated),
                )
            else:
                expression = choose_time(
                    TimeVariable(), add_time(initial, scale_time(total, -1))
                )
            book.native_expressions[key] = expression
        self.restore_sources(book, count, axis=axis)
        book.source_expressions[axis] = TimeVariable(0)

    def restore_sources(
        self, book: "TimelineBook", count: TimeExpression, *, axis: str
    ) -> None:
        """Keep entry values at zero iterations, otherwise the last body write."""
        limits = None if unbounded_time(count) else projected_expression_bounds(count)
        if limits is not None and limits.maximum == 0:
            book.source_expressions = self.initial_source_expressions.copy()
            return
        executed = choose_time(TimeVariable(1), count, maximum=False)
        restored = {}
        for name, expression in book.source_expressions.items():
            if axis in expression_axes(expression):
                expression = reduce_loop_time(expression, axis, last=True)
            if limits is None or limits.minimum == 0:
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
            limits = optional_time_bounds(prog, end)
            if limits is None or limits[1] > 1e-12:
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
                tails[ch] = (
                    RuntimeEnd(tail.register, None, None)
                    if tail.maximum is None or operand.maximum is None
                    else RuntimeEnd(
                        tail.register, 0, max(tail.maximum, operand.maximum)
                    )
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
            limits = optional_time_bounds(prog, end)
            if limits is not None and limits[1] <= 1e-12 and ch not in written:
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


@dataclass
class LoopTimelineFrame:
    """One Repeat accumulator, isolated namespace and explicit reference debits.

    parent is the visible loop timeline. external restores outer names.
    entry is its incoming end; origin adds an opaque iteration-progress identity.
    initial_channels/native retain incoming physical tails and their provenance.
    entry_math initializes shot-local storage before the loop. pinned restores
    register liveness at exit. starts records managed generator starts and branch
    paths for the backedge proof, in relative integer cycles. adc_starts records
    (timeline, ADC channel, start cycles, branch selector/index path) for the same
    proof on independent ADC tails. advance is cumulative
    reference movement in cycles; debits retain its execution-position plans.
    written_channels distinguishes generated tails from untouched outer tails.
    body_math starts empty and retains the prepared RuntimeMath plans emitted
    in this frame's body, in preparation order. Loop exit may rewrite these plans
    to s14-only pointer updates when their event offsets and duration are proven
    constant. recording starts False and becomes True at the body boundary,
    after the iteration-origin capture. Only the active innermost frame records
    plans; popping the frame at exit ends recording before exit plans prepare.
    """

    parent: str
    external: dict[str, Time]
    entry: Time
    origin: RuntimeEnd
    progress: CycleAxis
    initial_channels: dict[tuple[bool, int], Time]
    initial_native: dict[tuple[bool, int], TimeExpression]
    entry_math: RuntimeMath
    pinned: set[str]
    starts: list[tuple[int, TimeExpression, tuple[tuple[str, int], ...]]] = field(
        default_factory=list
    )
    adc_starts: list[tuple[str, int, TimeExpression, tuple[tuple[str, int], ...]]] = (
        field(default_factory=list)
    )
    advance: TimeExpression = field(default_factory=TimeVariable)
    debits: list[tuple[RuntimeMath, Time]] = field(default_factory=list)
    written_channels: set[tuple[bool, int]] = field(default_factory=set)
    body_math: list[RuntimeMath] = field(default_factory=list)
    recording: bool = False


@dataclass(frozen=True)
class BranchProvenance:
    """Register and channel expressions captured at the same branch position.

    sources maps register names to integer tProc-cycle expressions. native maps
    (is_generator, channel_id) to relative integer-cycle tails. Both mappings
    are detached snapshots, so preparing another case cannot mutate this path.
    unknown_adcs contains channels last triggered with native t=None; those tails
    cannot prove ordering for a later managed trigger and do not alter closure.
    """

    sources: dict[str, TimeExpression]
    native: dict[tuple[bool, int], TimeExpression]
    unknown_adcs: frozenset[int]


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
        self.cycle_axes: dict[str, CycleAxis] = {}
        # Native length sweeps are immutable program parameters. Share their
        # QICK initialization registers across physical-end and post plans.
        self.native_length_sources: dict[
            tuple[int, tuple[tuple[str, int], ...]], RuntimeEnd
        ] = {}
        self.source_expressions: dict[str, TimeExpression] = {}
        self.branch_provenance: list[BranchProvenanceFrame] = []
        self.math: RuntimeMath
        self.preparing = 0
        self.output_time: TimeExpression | None = None
        self.pinned: set[str] = set()
        self.runtime_gens: dict[int, RuntimeEnd] = {}
        self.runtime_ros: dict[int, RuntimeEnd] = {}
        self.unknown_native_adcs: set[int] = set()
        # Compile-only event tails. Runtime stores and QICK timestamps keep
        # their existing scheduling roles; these expressions never emit code.
        # Keys are (is_generator, channel_id).
        self.native_expressions: dict[tuple[bool, int], TimeExpression] = {}
        self.repeats: list[RepeatFrame] = []
        self.loop_timelines: list[LoopTimelineFrame] = []
        self.case_path: list[tuple[str, int]] = []
        self.native_only = False
        self.reference_guards = 0
        self.used: set[str] | None = None

    def repeat_count(
        self, prog, iterations: int | str, minimum: int, maximum: int | None
    ) -> TimeExpression:
        """Describe count points without requiring a bounds-only count producer."""
        if isinstance(iterations, str) and maximum is None:
            return self.source_expressions.get(
                iterations,
                TimeVariable(terms=((cycle_axis(iterations, None, None), 1),)),
            )
        return (
            bounded_source_expression(prog, iterations, minimum, maximum)
            if isinstance(iterations, str) and maximum is not None
            else TimeVariable(iterations if isinstance(iterations, int) else minimum)
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
        self.unknown_native_adcs = set(entry.unknown_adcs)

    def record_branch_sources(self, prog) -> None:
        """Capture source/channel values after this case's local timeline closure."""
        self.branch_provenance[-1].paths.append(self.snapshot_branch_provenance(prog))

    def snapshot_branch_provenance(self, prog) -> BranchProvenance:
        """Copy current register sources and all declared native channel tails."""
        return BranchProvenance(
            self.source_expressions.copy(),
            self.snapshot_native_expressions(prog),
            frozenset(self.unknown_native_adcs),
        )

    def finish_branch_sources(self, prog, selector: str) -> None:
        """Select channel tails and register sources defined on every branch path."""
        frame = self.branch_provenance.pop()
        self.unknown_native_adcs = set().union(
            *(path.unknown_adcs for path in frame.paths)
        )
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
        """Keep current producer identities and known arithmetic points in sync."""
        if type(macro) is WriteRegOp:
            self._observe_alu(prog, macro)
            return
        if (
            type(macro) is not WriteReg
            and isinstance(getattr(macro, "dst", None), str)
            and macro.dst in prog._time_bounds.author_writes
        ):
            self.source_expressions[macro.dst] = TimeVariable(
                terms=((cycle_axis(macro.dst, None, None), 1),)
            )
        if type(macro) is WriteReg:
            if isinstance(macro.src, int):
                self.source_expressions[macro.dst] = TimeVariable(macro.src)
            elif isinstance(macro.src, str) and self._known_source(prog, macro.src):
                self.source_expressions[macro.dst] = source_expression(prog, macro.src)
            else:
                self.source_expressions.pop(macro.dst, None)

    def _known_source(self, prog, name: str) -> bool:
        register = prog.reg_dict.get(name)
        return (
            name in self.source_expressions
            or name in prog.loop_dict
            or (register is not None and register.init is not None)
        )

    def _observe_alu(self, prog, macro: WriteRegOp) -> None:
        right = macro.rhs
        rhs = (
            TimeVariable(right)
            if isinstance(right, int)
            else source_expression(prog, right)
            if isinstance(right, str) and self._known_source(prog, right)
            else None
        )
        if (
            macro.op not in ("+", "-")
            or not self._known_source(prog, macro.lhs)
            or rhs is None
        ):
            if macro.dst in prog._time_bounds.allowed.get(id(macro), ()):
                # Unsupported ALU operations retain the owner's certified range.
                self.source_expressions.pop(macro.dst, None)
            else:
                self.source_expressions[macro.dst] = TimeVariable(
                    terms=((cycle_axis(macro.dst, None, None), 1),)
                )
            return
        expression = add_time(
            source_expression(prog, macro.lhs),
            rhs if macro.op == "+" else scale_time(rhs, -1),
        )
        if macro.dst not in prog._time_bounds.allowed.get(id(macro), ()):
            # Author arithmetic is a runtime recipe, not a certified loop range.
            expression = TimeVariable(
                terms=(
                    (
                        cycle_axis(
                            macro.dst,
                            None,
                            None,
                            CycleProducer(reconstruction=expression),
                        ),
                        1,
                    ),
                )
            )
        self.source_expressions[macro.dst] = expression

    def set_end(self, name: str, end: Time) -> None:
        self.math.check_end(end)
        self.ends[name] = end

    def require(self, name: str) -> None:
        """Reject unknown names without marking a timeline used or changing its end."""
        if name not in self.ends:
            raise ValueError(f"{name}: unknown timeline; declare it with fork in _body")

    def end(self, name: str) -> Time:
        self.require(name)
        if self.used is not None:
            self.used.add(name)
        return self.ends[name]

    def check_loop_pulse(self, prog, channel: int, start: Time) -> None:
        """Prove managed starts against physical tails, cancelling loop origins."""
        if not self.loop_timelines:
            return
        expression = time_expression(prog, start)
        gap = add_time(
            expression,
            scale_time(self.native_expression(prog, channel, generator=True), -1),
        )
        gap = substitute_time(
            gap, {frame.progress: TimeVariable() for frame in self.loop_timelines}
        )
        if not unbounded_time(gap) and projected_expression_bounds(gap).minimum < 0:
            raise ValueError(
                f"channel {channel} pulse overlap inside repeat; use timeline_inc"
            )
        self.loop_timelines[-1].starts.append(
            (channel, expression, tuple(self.case_path))
        )

    def check_adc(self, prog, timeline: str, channel: int, start: Time) -> None:
        """Reject a trigger unless its integer start covers the ceiled ADC tail.

        Diagnostics report the worst overlap in tProc cycles, or an unprovable
        runtime ordering when the input has no sufficient bounds.
        """
        if channel in self.unknown_native_adcs:
            raise ValueError(
                f"{timeline}: channel {channel} ADC overlap; "
                "cannot locate previous native trigger (t=None); overlap cycles unknown"
            )
        expression = time_expression(prog, start)
        previous = self.native_expression(prog, channel, generator=False)
        gap = add_time(expression, scale_time(previous, -1))
        gap = substitute_time(
            gap, {frame.progress: TimeVariable() for frame in self.loop_timelines}
        )
        lower, _ = difference_limits(gap, TimeVariable())
        if lower is None or lower < 0:
            detail = (
                "cannot prove runtime start >= previous end (overlap cycles unknown)"
                if lower is None
                else f"{-lower} cycles; cannot prove start >= previous end"
            )
            raise ValueError(f"{timeline}: channel {channel} ADC overlap; {detail}")
        if self.loop_timelines:
            self.loop_timelines[-1].adc_starts.append(
                (timeline, channel, expression, tuple(self.case_path))
            )
        for frame in self.repeats:
            first = substitute_time(
                expression,
                {loop.progress: TimeVariable() for loop in self.loop_timelines},
            )
            frame.adc_starts.append(
                (
                    timeline,
                    channel,
                    add_time(first, frame.expression_advance),
                    tuple(self.case_path),
                )
            )

    def check_adc_backedge(
        self,
        prog,
        timeline: str,
        channel: int,
        gap: TimeExpression,
        path: tuple[tuple[str, int], ...],
    ) -> None:
        """Prove an ADC reentry gap without changing state or emitting delays.

        gap is next-iteration start minus the previous ADC window end in tProc
        cycles, expressed in the same relative coordinate system with loop
        progress removed. path lists enclosing (sweep selector, case index)
        pairs from outermost to innermost; an empty path means unconditional.
        prog supplies selector point counts and their count expressions.
        Only the selected path must have a nonnegative gap; zero is safe.
        Raise ValueError with timeline, ADC channel and overlap cycles if the
        lower bound is negative, or an unknown-overlap diagnostic if unprovable.
        """
        for selector, index in reversed(path):
            gap = select_time(
                self.repeat_count(prog, selector, 0, prog.loop_dict[selector] - 1),
                tuple(
                    gap if i == index else TimeVariable()
                    for i in range(prog.loop_dict[selector])
                ),
            )
        lower, _ = difference_limits(gap, TimeVariable())
        if lower is None or lower < 0:
            detail = "unknown overlap cycles" if lower is None else f"{-lower} cycles"
            raise ValueError(
                f"{timeline}: channel {channel} ADC overlap across repeat backedge; "
                f"{detail}; cannot prove runtime start >= previous end"
            )

    def check_reference_adc_backedge(self, prog, frame: RepeatFrame) -> None:
        """Prove reference-loop reentry against the final relative ADC tails."""
        for timeline, channel, start, path in frame.adc_starts:
            gap = add_time(
                start,
                scale_time(self.native_expression(prog, channel, generator=False), -1),
            )
            self.check_adc_backedge(prog, timeline, channel, gap, path)

    def require_reference_advance(self) -> None:
        """Reject reference operations inside branch cases or guarded lowering."""
        if self.used is not None or self.reference_guards:
            raise ValueError("cannot advance reference; use timeline_inc instead")

    def decrement(self, t: Time, *, observe_pulses: bool = True) -> None:
        self.require_reference_advance()
        if self.native_only:
            return
        if observe_pulses:
            advance = time_expression(self.math.prog, t)
            for frame in self.loop_timelines:
                frame.advance = add_time(frame.advance, advance)
                frame.debits.append((self.math, t))
            self.native_expressions = {
                key: choose_time(TimeVariable(), add_time(end, scale_time(advance, -1)))
                for key, end in self.native_expressions.items()
            }
        consumed = self._debit_native(t) if observe_pulses else set()
        self._debit_repeats(t, observe_pulses=observe_pulses, consumed=consumed)
        self.ends = {name: self.math.floor(end, t) for name, end in self.ends.items()}

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
                if not isinstance(remaining, RuntimeEnd):
                    self.native_expressions[generator, ch] = native_event_expression(
                        self.math.prog, remaining
                    )
                self.native_timestamp(
                    self.math.prog,
                    remaining,
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
            limits = optional_time_bounds(self.math.prog, t)
            # Sync advances are nonnegative by construction. Zero is only the
            # guaranteed minimum, never the actual runtime reference debit.
            proof_advance = (
                (0.0 if limits is None else limits[0])
                if isinstance(t, RuntimeEnd)
                else t
            )
            if limits is None:
                for guard in frame.pulse_guards.values():
                    guard.unbounded = True
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
        value: Time,
        *,
        gen_ch: int | None = None,
        ro_ch: int | None = None,
    ) -> None:
        """Project a finite bound without claiming an unbounded hardware time.

        Runtime channel maps retain exact storage when no finite native
        timestamp can be projected. This does not record a new event.
        """
        if isinstance(value, RuntimeEnd):
            if value.maximum is None or unbounded_time(runtime_expression(value)):
                return
            value = projected_bounds(prog, value).maximum
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
        if not generator:
            self.unknown_native_adcs.discard(ch)

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
        values = [
            self.channel_end(prog, ch, generator=generator)
            for enabled, channels, generator in (
                (gens, prog.gen_chs, True),
                (ros, prog.ro_chs, False),
            )
            if enabled
            for ch in channels
        ]
        return self.math.join(values) if values else None

    def check_shell_wait(self, prog) -> None:
        """Project final ADC WAIT and diagnose slack after all body cancellation.

        Known producers use exact legal-point extrema. A bounds-only unknown
        emits an explicitly conservative warning and never triggers the relax
        guard. Body closure has already consumed unbounded ADC tails at runtime.
        This check emits no runtime instructions.
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

    def maximum(self, *, close_adc: bool = False) -> Time:
        """Return the maximum selected relative end as a Time value.

        With close_adc=False, select only the current timeline ends. With
        close_adc=True, also select actual native ADC ends whose expressions
        have no compile-time upper bound. True is only for the existing
        all-timeline shot closure; finite ADC tails stay on the constant shell
        WAIT and slack-proof path rather than being added as extra candidates.

        Return a float or QickParam when the maximum is statically representable,
        otherwise a RuntimeEnd for the actual runtime maximum. An empty selection
        returns zero. This method does not advance reference or debit ends; the
        caller consumes the returned end when synchronizing.
        """
        values = list(self.ends.values())
        if close_adc:
            prog = self.math.prog
            values.extend(
                self.channel_end(prog, ch, generator=False)
                for ch in prog.ro_chs
                if unbounded_time(self.native_expression(prog, ch, generator=False))
            )
        return self.math.join(values or [0.0])
