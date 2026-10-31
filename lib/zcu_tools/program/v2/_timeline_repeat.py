"""Shot-local accumulator lowering for reference-preserving Repeat scopes."""

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
    unbounded_time,
)
from ._timeline_runtime import RuntimeEnd, Time, runtime_expression, time_expression
from ._timeline_state import LoopTimelineFrame, TimelineBook


def _end(register: str, expression: TimeExpression, covers: tuple[Time, ...] = ()) -> RuntimeEnd:
    if unbounded_time(expression):
        return RuntimeEnd(register, 0, None, covers=covers, expression=expression)
    limits = projected_expression_bounds(expression)
    return RuntimeEnd(register, limits.minimum, limits.maximum, covers=covers, expression=expression)


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
            storage.register, 0, None,
            expression=add_time(time_expression(prog, entry), LinearTime(terms=((progress, 1),))),
        )
        channels = {
            (generator, ch): book.channel_end(prog, ch, generator=generator)
            for generator, ids in ((True, prog.gen_chs), (False, prog.ro_chs))
            for ch in ids
        }
        external = book.ends.copy()
        del external[self.parent]
        self.frame = LoopTimelineFrame(
            self.parent, external, book.waiting.copy(), entry, origin, progress,
            channels, book.snapshot_native_expressions(prog), book.math, book.pinned.copy(),
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

    def __init__(self, entry: LoopTimelineEnter, iterations: int | str, minimum: int, maximum: int | None):
        self.entry = entry
        self.iterations = iterations
        self.minimum = minimum
        self.maximum = maximum

    def preprocess(self, prog) -> None:
        book = self.entry.book
        frame = book.loop_timelines.pop()
        finish = book.maximum()
        duration = add_time(time_expression(prog, finish), scale_time(runtime_expression(frame.origin), -1))
        count = book.repeat_count(prog, self.iterations, self.minimum, self.maximum)
        total = multiply_time(duration, count)
        result_expression = add_time(time_expression(prog, frame.entry), total)
        self._check_backedge(prog, frame, duration)
        if book.loop_timelines:
            for ch, start, path in frame.starts:
                book.loop_timelines[-1].starts.append((ch, substitute_time(start, {frame.progress: LinearTime()}), path))
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
            relative = add_time(native, scale_time(runtime_expression(frame.origin), -1))
            last_expression = add_time(result_expression, add_time(relative, scale_time(duration, -1)))
            if self.minimum == 0:
                active = choose_time(count, LinearTime(1), maximum=False)
                last_expression = add_time(
                    multiply_time(frame.initial_native[generator, ch], add_time(LinearTime(1), scale_time(active, -1))),
                    multiply_time(last_expression, active),
                )
            tail = _end(tail.register, last_expression)
            channels = book.runtime_gens if generator else book.runtime_ros
            channels[ch] = tail
            book.native_expressions[generator, ch] = last_expression
            book.native_timestamp(prog, tail, gen_ch=ch if generator else None, ro_ch=None if generator else ch)
            gap = add_time(duration, scale_time(relative, -1))
            if not unbounded_time(gap) and projected_expression_bounds(gap).minimum >= 0:
                result = replace(result, covers=(*result.covers, tail))
        book.ends = {**frame.external, frame.parent: result}
        book.waiting = frame.waiting
        book.pinned = frame.pinned

    def _check_backedge(self, prog, frame: LoopTimelineFrame, duration: TimeExpression) -> None:
        if self.maximum is None or self.maximum < 2:
            return
        book = self.entry.book
        for ch, start, path in frame.starts:
            gap = add_time(add_time(start, duration), scale_time(book.native_expression(prog, ch, generator=True), -1))
            gap = substitute_time(gap, {f.progress: LinearTime() for f in [*book.loop_timelines, frame]})
            for selector, index in reversed(path):
                gap = select_time(book.repeat_count(prog, selector, 0, prog.loop_dict[selector] - 1), tuple(gap if i == index else LinearTime() for i in range(prog.loop_dict[selector])))
            if not unbounded_time(gap) and projected_expression_bounds(gap).minimum < 0:
                raise ValueError(f"repeat: channel {ch} pulse overlap across backedge; use delay_timeline")

    def translate(self, prog) -> None:
        pass
