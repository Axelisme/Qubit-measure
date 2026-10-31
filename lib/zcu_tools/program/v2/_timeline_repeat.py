"""Shot-local Repeat accumulators and explicit reference-debit bookkeeping."""

from dataclasses import dataclass, replace

from qick.asm_v2 import AsmInst, Macro, WriteReg

from ._timeline_axis import TimeVariable
from ._timeline_expression import (
    TimeExpression,
    add_time,
    choose_time,
    cycle_axis,
    multiply_time,
    projected_expression_bounds,
    scale_time,
    select_time,
    substitute_time,
    time_variables,
    unbounded_time,
)
from ._timeline_guard import PulseGuard
from ._timeline_runtime import RuntimeEnd, Time, _EventTime, runtime_expression, time_expression
from ._timeline_expression import difference_limits
from .macro.write_reg import WriteRegOp
from ._timeline_state import LoopTimelineFrame, TimelineBook
from .modules.util import get_fclk


def _end(
    register: str, expression: TimeExpression, covers: tuple[Time, ...] = ()
) -> RuntimeEnd:
    if unbounded_time(expression):
        return RuntimeEnd(register, 0, None, covers=covers, expression=expression)
    limits = projected_expression_bounds(expression)
    return RuntimeEnd(
        register, limits.minimum, limits.maximum, covers=covers, expression=expression
    )


@dataclass(frozen=True)
class _ReferenceSummary:
    """Relative end, total reference advance and final entry, in tProc cycles."""

    end: TimeExpression
    advance: TimeExpression
    last_entry: TimeExpression


class LoopTimelineEnter(Macro):
    """Initialize one loop accumulator at its execution position, not prologue."""

    def __repr__(self) -> str:
        # QICK eagerly formats macros even when debug logging is disabled.
        return f"{type(self).__name__}()"

    def __init__(self, book: TimelineBook, parent: str, counter: str | None):
        self.book = book
        self.parent = parent
        self.counter = counter
        self.simple = False
        self.post_operations: list[Macro] = []

    def preprocess(self, prog) -> None:
        book = self.book
        entry = book.end(self.parent, placement=True)
        if self.counter is not None:
            prog.add_reg(self.counter)
        storage = book.math.snapshot(book.math.cycles(entry))
        book.math.operations.append(WriteReg(dst="s_out_time", src=storage.register))
        progress = cycle_axis(storage.register, 0, None)
        origin = RuntimeEnd(
            storage.register,
            0,
            None,
            expression=add_time(
                time_expression(prog, entry), TimeVariable(terms=((progress, 1),))
            ),
        )
        channels = {
            (generator, ch): book.channel_end(prog, ch, generator=generator)
            for generator, ids in ((True, prog.gen_chs), (False, prog.ro_chs))
            for ch in ids
        }
        external = book.ends.copy()
        del external[self.parent]
        self.frame = LoopTimelineFrame(
            self.parent,
            external,
            book.waiting.copy(),
            entry,
            origin,
            progress,
            channels,
            book.snapshot_native_expressions(prog),
            book.math,
            book.pinned.copy(),
        )
        book.loop_timelines.append(self.frame)
        book.pinned.update(prog._timeline_live_registers())
        book.pinned.add(storage.register)
        book.ends = {self.parent: origin}
        book.waiting.clear()

    def translate(self, prog) -> None:
        pass


class LoopTimelineExit(Macro):
    """Store the iteration end and expose the whole loop's aggregate provenance."""

    def __repr__(self) -> str:
        # QICK eagerly formats macros even when debug logging is disabled.
        return f"{type(self).__name__}()"

    def __init__(
        self,
        entry: LoopTimelineEnter,
        iterations: int | str,
        minimum: int,
        maximum: int | None,
    ):
        self.entry = entry
        self.iterations = iterations
        self.minimum = minimum
        self.maximum = maximum

    def preprocess(self, prog) -> None:
        book = self.entry.book
        frame = book.loop_timelines.pop()
        finish = book.maximum()
        if frame.debits:
            self._reference_exit(prog, frame, finish)
            return
        duration = add_time(
            time_expression(prog, finish),
            scale_time(runtime_expression(frame.origin), -1),
        )
        count = book.repeat_count(prog, self.iterations, self.minimum, self.maximum)
        total = multiply_time(duration, count)
        result_expression = add_time(time_expression(prog, frame.entry), total)
        self._check_backedge(prog, frame, duration)
        if book.loop_timelines:
            for ch, start, path in frame.starts:
                book.loop_timelines[-1].starts.append(
                    (ch, substitute_time(start, {frame.progress: TimeVariable()}), path)
                )
        result = _end(frame.origin.register, result_expression)
        book.math.check_end(result)
        operand = book.math.cycles(finish)
        book.math.operations.append(WriteReg(dst="s_out_time", src=operand.register))
        simple = self._pointer_body(prog, frame, finish)
        for (generator, ch), initial in frame.initial_channels.items():
            current = book.channel_end(prog, ch, generator=generator)
            native = book.native_expression(prog, ch, generator=generator)
            if native == frame.initial_native[generator, ch]:
                continue
            # Each tail has storage initialized before the loop, including n=0.
            tail = frame.entry_math.snapshot(frame.entry_math.cycles(initial))
            last = book.math.cycles(current)
            if not simple:
                book.math.operations.append(WriteReg(dst=tail.register, src=last.register))
            relative = add_time(
                native, scale_time(runtime_expression(frame.origin), -1)
            )
            last_expression = add_time(
                result_expression, add_time(relative, scale_time(duration, -1))
            )
            if self.minimum == 0:
                active = choose_time(count, TimeVariable(1), maximum=False)
                last_expression = add_time(
                    multiply_time(
                        frame.initial_native[generator, ch],
                        add_time(TimeVariable(1), scale_time(active, -1)),
                    ),
                    multiply_time(last_expression, active),
                )
            if simple:
                self._capture_channel(prog, tail, result, native, time_expression(prog, finish))
            tail = _end(tail.register, last_expression)
            channels = book.runtime_gens if generator else book.runtime_ros
            channels[ch] = tail
            book.native_expressions[generator, ch] = last_expression
            book.native_timestamp(
                prog,
                tail,
                gen_ch=ch if generator else None,
                ro_ch=None if generator else ch,
            )
            gap = add_time(duration, scale_time(relative, -1))
            if (
                not unbounded_time(gap)
                and projected_expression_bounds(gap).minimum >= 0
            ):
                result = replace(result, covers=(*result.covers, tail))
        if book.repeats and not book.loop_timelines:
            self._publish_reference_guards(prog, frame)
        book.ends = {**frame.external, frame.parent: result}
        book.waiting = frame.waiting
        book.pinned = frame.pinned

    def _pointer_body(self, prog, frame: LoopTimelineFrame, finish: Time) -> bool:
        """Lower already-prepared straight-line plans without per-iteration storage."""
        if not self.entry.simple:
            return False
        origin = runtime_expression(frame.origin)
        end = time_expression(prog, finish)
        duration = difference_limits(end, origin)
        if duration[0] is None or duration[0] != duration[1]:
            return False
        events = [op for math in frame.body_math for op in math.operations if isinstance(op, _EventTime)]
        offsets = [difference_limits(event.expression, origin) for event in events]
        if any(lo is None or lo != hi for lo, hi in offsets):
            return False
        for generator, ch in frame.initial_channels:
            native = self.entry.book.native_expression(prog, ch, generator=generator)
            if native != frame.initial_native[generator, ch]:
                lo, hi = difference_limits(native, end)
                if lo is None or lo != hi:
                    return False
        for math in frame.body_math:
            math.operations = [op for op in math.operations if isinstance(op, _EventTime)]
        for event, (offset, _) in zip(events, offsets):
            if offset is None:
                raise RuntimeError("pointer event lost its constant offset")
            event.base = RuntimeEnd("s_out_time", expression=origin)
            event.offset = int(offset)
        last = events[-1].expression if events else origin
        advance, _ = difference_limits(end, last)
        if advance is None:
            raise RuntimeError("pointer end lost its constant increment")
        self.entry.book.math.operations = [WriteRegOp(dst="s_out_time", lhs="s_out_time", op="+", rhs=int(advance))]
        return True

    def _capture_channel(self, prog, tail: RuntimeEnd, result: RuntimeEnd, native: TimeExpression, finish: TimeExpression) -> None:
        """Derive a final physical tail from the pointer, preserving n=0 tails."""
        offset, _ = difference_limits(native, finish)
        if offset is None:
            raise RuntimeError("pointer channel lost its constant final offset")
        operations = self.entry.post_operations
        if self.minimum == 0 and isinstance(self.iterations, str):
            operations.append(AsmInst(inst={"CMD": "TEST", "OP": prog._get_reg(self.iterations), "UF": "1"}, addr_inc=1))
            operations.append(AsmInst(inst={"CMD": "REG_WR", "DST": prog._get_reg(tail.register), "SRC": "op", "OP": f"{prog._get_reg(result.register)} + #{int(offset)}", "IF": "NZ"}, addr_inc=1))
        else:
            operations.append(WriteRegOp(dst=tail.register, lhs=result.register, op="+", rhs=int(offset)))

    def _reference_exit(self, prog, frame: LoopTimelineFrame, finish: Time) -> None:
        """Store exact runtime debits, then restore loop-relative provenance."""
        book = self.entry.book
        summary = self._reference_summary(prog, frame, finish)
        result = _end(frame.origin.register, summary.end)
        book.math.check_end(result)
        operand = book.math.cycles(finish)
        book.math.operations.append(WriteReg(dst="s_out_time", src=operand.register))
        external = self._reference_external(prog, frame, summary.advance)
        result = self._reference_channels(prog, frame, result, summary)
        self._check_backedge(
            prog, frame, TimeVariable(), next_entry=time_expression(prog, finish)
        )
        extra_advance = add_time(summary.advance, scale_time(frame.advance, -1))
        for outer in book.loop_timelines:
            outer.advance = add_time(outer.advance, extra_advance)
            outer.starts.extend(
                (ch, substitute_time(start, {frame.progress: TimeVariable()}), path)
                for ch, start, path in frame.starts
            )
        for outer in book.repeats:
            outer.expression_advance = add_time(outer.expression_advance, extra_advance)
        if book.repeats and not book.loop_timelines:
            self._publish_reference_guards(prog, frame)
        book.ends = {**external, frame.parent: result}
        book.waiting = frame.waiting
        book.pinned = frame.pinned

    def _reference_summary(
        self, prog, frame: LoopTimelineFrame, finish: Time
    ) -> "_ReferenceSummary":
        """Fold compile-only provenance, without choosing or replaying lowering."""
        book = self.entry.book
        incoming = time_expression(prog, frame.entry)
        transfer = time_expression(prog, finish)
        count = book.repeat_count(prog, self.iterations, self.minimum, self.maximum)

        def apply(expression: TimeExpression, value: TimeExpression) -> TimeExpression:
            return substitute_time(
                expression, {frame.progress: add_time(value, scale_time(incoming, -1))}
            )

        def fold(n: int) -> tuple[TimeExpression, TimeExpression]:
            value, advance = incoming, TimeVariable()
            step, debit = transfer, frame.advance
            while n:
                if n & 1:
                    advance = add_time(advance, apply(debit, value))
                    value = apply(step, value)
                n >>= 1
                if n:
                    debit = add_time(debit, apply(debit, step))
                    step = apply(step, step)
            return value, advance

        active = choose_time(count, TimeVariable(1), maximum=False)
        if isinstance(self.iterations, int):
            result_expression, total_advance = fold(self.iterations)
            last_entry, _ = fold(max(0, self.iterations - 1))
        elif frame.progress not in time_variables(transfer):
            first_advance = apply(frame.advance, incoming)
            later_advance = apply(frame.advance, transfer)
            result_expression = select_time(active, (incoming, transfer))
            total_advance = select_time(
                active,
                (
                    TimeVariable(),
                    add_time(
                        first_advance,
                        multiply_time(add_time(count, TimeVariable(-1)), later_advance),
                    ),
                ),
            )
            last_entry = select_time(
                choose_time(count, TimeVariable(2), maximum=False),
                (incoming, incoming, transfer),
            )
        elif frame.progress not in time_variables(
            add_time(transfer, scale_time(runtime_expression(frame.origin), -1))
        ) and frame.progress not in time_variables(frame.advance):
            delta = add_time(transfer, scale_time(runtime_expression(frame.origin), -1))
            result_expression = add_time(incoming, multiply_time(count, delta))
            total_advance = multiply_time(count, frame.advance)
            last_entry = add_time(
                incoming, multiply_time(add_time(count, TimeVariable(-1)), delta)
            )
        elif self.maximum is not None:
            states = [fold(i) for i in range(self.maximum + 1)]
            result_expression = select_time(count, tuple(value for value, _ in states))
            total_advance = select_time(count, tuple(advance for _, advance in states))
            last_entry = select_time(
                count, tuple(states[max(0, i - 1)][0] for i in range(len(states)))
            )
        else:
            result_expression = TimeVariable(
                terms=((cycle_axis("repeat relative end", 0, None), 1),)
            )
            total_advance = TimeVariable(
                terms=((cycle_axis("repeat reference advance", 0, None), 1),)
            )
            last_entry = TimeVariable(
                terms=((cycle_axis("repeat final entry", 0, None), 1),)
            )

        return _ReferenceSummary(result_expression, total_advance, last_entry)

    def _reference_store(
        self,
        prog,
        frame: LoopTimelineFrame,
        initial: Time,
        expression: TimeExpression,
        *,
        generated: Time | None = None,
    ) -> RuntimeEnd:
        book = self.entry.book
        register = prog.add_reg()
        book.pinned.add(register)
        initial_operand = frame.entry_math.cycles(initial)
        frame.entry_math.operations.append(
            WriteReg(dst=register, src=initial_operand.register)
        )
        storage = _end(register, time_expression(prog, initial))
        if generated is None:
            for math, advance in frame.debits:
                remainder = math.cycles(math.floor(storage, advance))
                math.operations.append(WriteReg(dst=register, src=remainder.register))
        else:
            tail = book.math.cycles(generated)
            book.math.operations.append(WriteReg(dst=register, src=tail.register))
        return _end(register, expression)

    def _reference_external(
        self, prog, frame: LoopTimelineFrame, advance: TimeExpression
    ) -> dict[str, Time]:
        """Debit each outer name at the reference operation's execution position."""
        external = {}
        for name, initial in frame.external.items():
            expression = choose_time(
                TimeVariable(),
                add_time(time_expression(prog, initial), scale_time(advance, -1)),
            )
            if time_expression(prog, initial) == TimeVariable():
                external[name] = initial
            else:
                external[name] = self._reference_store(prog, frame, initial, expression)
                frame.waiting.add(name)

        return external

    def _reference_channels(
        self,
        prog,
        frame: LoopTimelineFrame,
        result: RuntimeEnd,
        summary: "_ReferenceSummary",
    ) -> RuntimeEnd:
        """Keep last-generated tails distinct from repeatedly debited outer tails."""
        book = self.entry.book
        count = book.repeat_count(prog, self.iterations, self.minimum, self.maximum)
        active = choose_time(count, TimeVariable(1), maximum=False)
        for (generator, ch), initial in frame.initial_channels.items():
            current = book.channel_end(prog, ch, generator=generator)
            if (generator, ch) in frame.written_channels:
                last = substitute_time(
                    book.native_expression(prog, ch, generator=generator),
                    {
                        frame.progress: add_time(
                            summary.last_entry,
                            scale_time(time_expression(prog, frame.entry), -1),
                        )
                    },
                )
                expression = select_time(
                    active, (frame.initial_native[generator, ch], last)
                )
                tail = self._reference_store(
                    prog, frame, initial, expression, generated=current
                )
            else:
                expression = choose_time(
                    TimeVariable(),
                    add_time(
                        frame.initial_native[generator, ch],
                        scale_time(summary.advance, -1),
                    ),
                )
                if expression == TimeVariable():
                    continue
                tail = self._reference_store(prog, frame, initial, expression)
            channels = book.runtime_gens if generator else book.runtime_ros
            channels[ch] = tail
            book.native_expressions[generator, ch] = expression
            book.native_timestamp(
                prog,
                tail,
                gen_ch=ch if generator else None,
                ro_ch=None if generator else ch,
            )
            if expression == summary.end:
                result = replace(result, covers=(*result.covers, tail))

        return result

    def _publish_reference_guards(self, prog, frame: LoopTimelineFrame) -> None:
        book = self.entry.book
        for ch in prog.gen_chs:
            starts = [
                substitute_time(start, {frame.progress: TimeVariable()})
                for channel, start, _ in frame.starts
                if channel == ch
            ]
            if not starts:
                continue
            first = choose_time(*starts, maximum=False)
            tail = book.native_expression(prog, ch, generator=True)
            guard = PulseGuard(
                ch, unbounded=unbounded_time(first) or unbounded_time(tail)
            )
            if not guard.unbounded:
                try:
                    lower = book.math.time_parameter(_end("s0", first))
                    upper = book.math.time_parameter(_end("s0", tail))
                except RuntimeError:
                    lower = projected_expression_bounds(first).minimum / get_fclk(prog)
                    upper = projected_expression_bounds(tail).maximum / get_fclk(prog)
                guard = PulseGuard(ch, shifts=(), floors=(upper,), limits=(lower,))
            book.repeats[-1].pulse_guards[ch].then(guard)

    def _check_backedge(
        self,
        prog,
        frame: LoopTimelineFrame,
        duration: TimeExpression,
        *,
        next_entry: TimeExpression | None = None,
    ) -> None:
        if self.maximum is None or self.maximum < 2:
            return
        book = self.entry.book
        for ch, start, path in frame.starts:
            next_start = (
                add_time(start, duration)
                if next_entry is None
                else substitute_time(
                    start,
                    {
                        frame.progress: add_time(
                            next_entry,
                            scale_time(time_expression(prog, frame.entry), -1),
                        )
                    },
                )
            )
            gap = add_time(
                next_start,
                scale_time(book.native_expression(prog, ch, generator=True), -1),
            )
            gap = substitute_time(
                gap, {f.progress: TimeVariable() for f in [*book.loop_timelines, frame]}
            )
            for selector, index in reversed(path):
                gap = select_time(
                    book.repeat_count(prog, selector, 0, prog.loop_dict[selector] - 1),
                    tuple(
                        gap if i == index else TimeVariable()
                        for i in range(prog.loop_dict[selector])
                    ),
                )
            if not unbounded_time(gap) and projected_expression_bounds(gap).minimum < 0:
                raise ValueError(
                    f"repeat: channel {ch} pulse overlap across backedge; use delay_timeline"
                )

    def translate(self, prog) -> None:
        pass
