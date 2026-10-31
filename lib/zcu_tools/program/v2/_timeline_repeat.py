"""Shot-local Repeat accumulators and explicit reference-debit bookkeeping."""

from dataclasses import replace

from qick.asm_v2 import Macro, WriteReg

from ._timeline_expression import (
    CycleVariable,
    LinearTime,
    TimeExpression,
    add_time,
    choose_time,
    multiply_time,
    projected_expression_bounds,
    scale_time,
    select_time,
    substitute_time,
    time_variables,
    unbounded_time,
)
from ._timeline_guard import PulseGuard
from ._timeline_runtime import RuntimeEnd, Time, runtime_expression, time_expression
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


class LoopTimelineEnter(Macro):
    """Initialize one loop accumulator at its execution position, not prologue."""

    def __init__(self, book: TimelineBook, parent: str, counter: str | None):
        self.book = book
        self.parent = parent
        self.counter = counter

    def preprocess(self, prog) -> None:
        book = self.book
        entry = book.end(self.parent, placement=True)
        if self.counter is not None:
            prog.add_reg(self.counter)
        storage = book.math.snapshot(book.math.cycles(entry))
        progress = CycleVariable(storage.register, 0, None)
        origin = RuntimeEnd(
            storage.register,
            0,
            None,
            expression=add_time(
                time_expression(prog, entry), LinearTime(terms=((progress, 1),))
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
                    (ch, substitute_time(start, {frame.progress: LinearTime()}), path)
                )
        result = _end(frame.origin.register, result_expression)
        book.math.check_end(result)
        operand = book.math.cycles(finish)
        book.math.operations.append(WriteReg(dst=result.register, src=operand.register))
        for (generator, ch), initial in frame.initial_channels.items():
            current = book.channel_end(prog, ch, generator=generator)
            native = book.native_expression(prog, ch, generator=generator)
            if native == frame.initial_native[generator, ch]:
                continue
            # Each tail has storage initialized before the loop, including n=0.
            tail = frame.entry_math.snapshot(frame.entry_math.cycles(initial))
            last = book.math.cycles(current)
            book.math.operations.append(WriteReg(dst=tail.register, src=last.register))
            relative = add_time(
                native, scale_time(runtime_expression(frame.origin), -1)
            )
            last_expression = add_time(
                result_expression, add_time(relative, scale_time(duration, -1))
            )
            if self.minimum == 0:
                active = choose_time(count, LinearTime(1), maximum=False)
                last_expression = add_time(
                    multiply_time(
                        frame.initial_native[generator, ch],
                        add_time(LinearTime(1), scale_time(active, -1)),
                    ),
                    multiply_time(last_expression, active),
                )
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

    def _reference_exit(self, prog, frame: LoopTimelineFrame, finish: Time) -> None:
        """Carry the relative end while debiting actual advances on every path.

        Folding here is compile-only provenance, never a lowering decision or
        a replay of macros. Hardware keeps one accumulator and the recorded
        reference operations, including zero-count and shot-local entry stores.
        """
        book = self.entry.book
        incoming = time_expression(prog, frame.entry)
        transfer = time_expression(prog, finish)
        count = book.repeat_count(prog, self.iterations, self.minimum, self.maximum)

        def apply(expression: TimeExpression, value: TimeExpression) -> TimeExpression:
            return substitute_time(expression, {frame.progress: add_time(value, scale_time(incoming, -1))})

        def fold(n: int) -> tuple[TimeExpression, TimeExpression]:
            value, advance = incoming, LinearTime()
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

        active = choose_time(count, LinearTime(1), maximum=False)
        if isinstance(self.iterations, int):
            result_expression, total_advance = fold(self.iterations)
            last_entry, _ = fold(max(0, self.iterations - 1))
        elif frame.progress not in time_variables(transfer):
            first_advance = apply(frame.advance, incoming)
            later_advance = apply(frame.advance, transfer)
            result_expression = select_time(active, (incoming, transfer))
            total_advance = select_time(active, (LinearTime(), add_time(first_advance, multiply_time(add_time(count, LinearTime(-1)), later_advance))))
            last_entry = select_time(choose_time(count, LinearTime(2), maximum=False), (incoming, incoming, transfer))
        elif frame.progress not in time_variables(add_time(transfer, scale_time(runtime_expression(frame.origin), -1))) and frame.progress not in time_variables(frame.advance):
            delta = add_time(transfer, scale_time(runtime_expression(frame.origin), -1))
            result_expression = add_time(incoming, multiply_time(count, delta))
            total_advance = multiply_time(count, frame.advance)
            last_entry = add_time(incoming, multiply_time(add_time(count, LinearTime(-1)), delta))
        elif self.maximum is not None:
            states = [fold(i) for i in range(self.maximum + 1)]
            result_expression = select_time(count, tuple(value for value, _ in states))
            total_advance = select_time(count, tuple(advance for _, advance in states))
            last_entry = select_time(count, tuple(states[max(0, i - 1)][0] for i in range(len(states))))
        else:
            result_expression = LinearTime(terms=((CycleVariable("repeat relative end", 0, None), 1),))
            total_advance = LinearTime(terms=((CycleVariable("repeat reference advance", 0, None), 1),))
            last_entry = LinearTime(terms=((CycleVariable("repeat final entry", 0, None), 1),))

        result = _end(frame.origin.register, result_expression)
        book.math.check_end(result)
        operand = book.math.cycles(finish)
        book.math.operations.append(WriteReg(dst=result.register, src=operand.register))

        def store(initial: Time, expression: TimeExpression, *, generated: Time | None = None) -> RuntimeEnd:
            register = prog.add_reg()
            book.pinned.add(register)
            initial_operand = frame.entry_math.cycles(initial)
            frame.entry_math.operations.append(WriteReg(dst=register, src=initial_operand.register))
            storage = _end(register, time_expression(prog, initial))
            if generated is None:
                for math, advance in frame.debits:
                    remainder = math.cycles(math.floor(storage, advance))
                    math.operations.append(WriteReg(dst=register, src=remainder.register))
            else:
                tail = book.math.cycles(generated)
                book.math.operations.append(WriteReg(dst=register, src=tail.register))
            return _end(register, expression)

        external = {}
        for name, initial in frame.external.items():
            expression = choose_time(LinearTime(), add_time(time_expression(prog, initial), scale_time(total_advance, -1)))
            if time_expression(prog, initial) == LinearTime():
                external[name] = initial
            else:
                external[name] = store(initial, expression)
                frame.waiting.add(name)

        for (generator, ch), initial in frame.initial_channels.items():
            current = book.channel_end(prog, ch, generator=generator)
            if (generator, ch) in frame.written_channels:
                last = apply(book.native_expression(prog, ch, generator=generator), last_entry)
                expression = select_time(active, (frame.initial_native[generator, ch], last))
                tail = store(initial, expression, generated=current)
            else:
                expression = choose_time(LinearTime(), add_time(frame.initial_native[generator, ch], scale_time(total_advance, -1)))
                if expression == LinearTime():
                    continue
                tail = store(initial, expression)
            channels = book.runtime_gens if generator else book.runtime_ros
            channels[ch] = tail
            book.native_expressions[generator, ch] = expression
            book.native_timestamp(prog, tail, gen_ch=ch if generator else None, ro_ch=None if generator else ch)
            if expression == result_expression:
                result = replace(result, covers=(*result.covers, tail))

        self._check_backedge(prog, frame, LinearTime(), next_entry=transfer)
        extra_advance = add_time(total_advance, scale_time(frame.advance, -1))
        for outer in book.loop_timelines:
            outer.advance = add_time(outer.advance, extra_advance)
            outer.starts.extend((ch, apply(start, incoming), path) for ch, start, path in frame.starts)
        for outer in book.repeats:
            outer.expression_advance = add_time(outer.expression_advance, extra_advance)
        if book.repeats and not book.loop_timelines:
            self._publish_reference_guards(prog, frame)
        book.ends = {**external, frame.parent: result}
        book.waiting = frame.waiting
        book.pinned = frame.pinned

    def _publish_reference_guards(self, prog, frame: LoopTimelineFrame) -> None:
        book = self.entry.book
        for ch in prog.gen_chs:
            starts = [
                substitute_time(start, {frame.progress: LinearTime()})
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
        self, prog, frame: LoopTimelineFrame, duration: TimeExpression, *, next_entry: TimeExpression | None = None
    ) -> None:
        if self.maximum is None or self.maximum < 2:
            return
        book = self.entry.book
        for ch, start, path in frame.starts:
            next_start = add_time(start, duration) if next_entry is None else substitute_time(start, {frame.progress: add_time(next_entry, scale_time(time_expression(prog, frame.entry), -1))})
            gap = add_time(
                next_start,
                scale_time(book.native_expression(prog, ch, generator=True), -1),
            )
            gap = substitute_time(
                gap, {f.progress: LinearTime() for f in [*book.loop_timelines, frame]}
            )
            for selector, index in reversed(path):
                gap = select_time(
                    book.repeat_count(prog, selector, 0, prog.loop_dict[selector] - 1),
                    tuple(
                        gap if i == index else LinearTime()
                        for i in range(prog.loop_dict[selector])
                    ),
                )
            if not unbounded_time(gap) and projected_expression_bounds(gap).minimum < 0:
                raise ValueError(
                    f"repeat: channel {ch} pulse overlap across backedge; use delay_timeline"
                )

    def translate(self, prog) -> None:
        pass
