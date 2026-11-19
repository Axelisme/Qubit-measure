"""Author-label identity, forward transfers and path joins for the timeline body."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from qick.asm_v2 import CondJump, Jump, Label, Macro

from ._timeline_expression import choose_time
from ._timeline_runtime import RuntimeEnd, RuntimeMath, Time
from ._timeline_state import PathState, TimelineBook
from .ir.labels import PSEUDO_LABELS, make_label

if TYPE_CHECKING:
    from .timeline import TimelineProgramV2


@dataclass(frozen=True, eq=False)
class _Target:
    # QICK expansion carries this identity verbatim. Native labels are strings;
    # allocate final machine names only after every native label is known.
    name: str


def _capture(prog: TimelineProgramV2, book: TimelineBook) -> PathState:
    return replace(
        book.capture(prog), native_expressions=book.snapshot_native_expressions(prog)
    )


class TimelineJump(Macro):
    def __init__(self, flow: TimelineJumps, target: _Target, native: Jump | CondJump):
        self.flow = flow
        self.target = target
        self.native = native
        self.kind = "jump" if isinstance(native, Jump) else "cond_jump"

    def preprocess(self, prog: TimelineProgramV2) -> None:
        self.state = _capture(prog, self.flow.book)
        self.flow.live.setdefault(self.target, set()).update(
            value.register
            for value in (
                *self.state.ends.values(),
                *self.state.channel_ends().values(),
            )
            if isinstance(value, RuntimeEnd)
        )

    def translate(self, prog: TimelineProgramV2) -> None:
        self.native.translate(prog)


class TimelineLabel(Label):
    def __init__(
        self,
        flow: TimelineJumps,
        target: _Target,
        incoming: list[TimelineJump],
        *,
        fallthrough: bool,
    ):
        super().__init__(label=target)
        self.flow = flow
        self.book = flow.book
        self.target = target
        self.incoming = incoming
        self.fallthrough = fallthrough

    def preprocess(self, prog: TimelineProgramV2) -> None:
        self.math = RuntimeMath(prog)
        states = [jump.state for jump in self.incoming]
        if self.fallthrough:
            states.append(_capture(prog, self.book))
        if states:
            self.book.restore(prog, states[0])
            if len(states) > 1:
                self._merge(prog, states)
        self.flow.live.pop(self.target, None)

    def _check_states(
        self, states: list[PathState], edges: list[str], where: str
    ) -> None:
        first = states[0]
        for field in ("used", "repeat_frames", "pinned"):
            if any(
                getattr(state, field) != getattr(first, field) for state in states[1:]
            ):
                raise ValueError(
                    f"{where}: incompatible {field} across incoming edges ({', '.join(edges)})"
                )
        for ch in first.read_counts:
            counts = [state.read_counts[ch] for state in states]
            if len(set(counts)) != 1:
                shown = ", ".join(
                    f"{edge} -> {count}"
                    for edge, count in zip(edges, counts, strict=True)
                )
                raise ValueError(
                    f"{where}: ADC {ch} reads differ across incoming edges: {shown}"
                )

    def _merge(self, prog: TimelineProgramV2, states: list[PathState]) -> None:
        edges = [f"{jump.kind} #{i}" for i, jump in enumerate(self.incoming, 1)]
        if self.fallthrough:
            edges.append("fall-through")
        where = f"label {self.target.name!r}"
        self._check_states(states, edges, where)

        def end(name: str, values: list[Time]) -> Time:
            if any(isinstance(value, RuntimeEnd) for value in values):
                if all(value is values[0] or value == values[0] for value in values):
                    return values[0]
                raise ValueError(
                    f"{where}: {name} has different runtime ends on incoming edges ({', '.join(edges)}); "
                    "sync the paths or make their ends equal before the jump"
                )
            if all(isinstance(value, float | int) for value in values):
                return max(value for value in values if isinstance(value, float | int))
            return self.math.join(values)

        self.book.ends = {
            name: end(f"timeline {name!r}", [state.ends[name] for state in states])
            for name in states[0].ends
            if all(name in state.ends for state in states[1:])
        }
        channels = [state.channel_ends() for state in states]
        channel_ends = {
            (gen, ch): end(
                f"{'generator' if gen else 'ADC'} {ch}",
                [path[gen, ch] for path in channels],
            )
            for gen, ch in channels[0]
        }
        for (gen, ch), value in channel_ends.items():
            runtime = self.book.runtime_gens if gen else self.book.runtime_ros
            if isinstance(value, RuntimeEnd):
                runtime[ch] = value
            else:
                runtime.pop(ch, None)
            self.book.native_timestamp(
                prog, value, gen_ch=ch if gen else None, ro_ch=None if gen else ch
            )
        self._merge_proofs(states)

    def _merge_proofs(self, states: list[PathState]) -> None:
        self.book.source_expressions = {
            name: value
            for name, value in states[0].source_expressions.items()
            if all(state.source_expressions.get(name) == value for state in states[1:])
        }
        unknown_sources = set().union(*(state.source_expressions for state in states))
        unknown_sources.difference_update(self.book.source_expressions)
        self.book.time_bounds.defined = (
            set.intersection(*(state.defined for state in states)) - unknown_sources
        )
        self.book.native_expressions = {
            key: choose_time(*(state.native_expressions[key] for state in states))
            for key in states[0].native_expressions
        }
        self.book.unknown_native_adcs = set().union(
            *(state.unknown_native_adcs for state in states)
        )
        self.book.time_bounds.merge_selections([state.selections for state in states])

    def translate(self, prog: TimelineProgramV2) -> None:
        super().translate(prog)
        self.book.output_time = None  # A taken edge need not have set the same s14.
        self.math.translate(prog)


class TimelineJumps:
    """Record and validate one body's graph; its macros own edge snapshots."""

    def __init__(self, book: TimelineBook):
        self.book = book
        self.regions: list[str] = []
        self.pending: dict[str, list[TimelineJump]] = {}
        # Registers are live on saved edges even while another path is prepared.
        self.live: dict[_Target, set[str]] = {}
        self.defined: set[str] = set()
        self.unreachable_after: str | None = None

    @contextmanager
    def region(self, description: str) -> Generator[None]:
        self.regions.append(description)
        try:
            yield
        finally:
            self.regions.pop()

    def require_reachable(self) -> None:
        if self.unreachable_after is not None:
            raise ValueError(
                f"unreachable timeline action after unconditional jump to {self.unreachable_after!r}; "
                "add a reachable label before this action"
            )

    def _check(self, action: str, name: object) -> None:
        if self.regions:
            raise ValueError(
                f"timeline {action} {name!r} is inside {self.regions[-1]}; "
                "label, jump and cond_jump are allowed only at the top level of _body"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(f"timeline {action} {name!r} requires a nonempty string")

    def label(self, name: str) -> TimelineLabel:
        self._check("label", name)
        if name in self.defined:
            raise ValueError(f"timeline label {name!r} is defined twice in _body")
        self.defined.add(name)
        incoming = self.pending.pop(name, [])
        source = TimelineLabel(
            self,
            incoming[0].target if incoming else _Target(name),
            incoming,
            fallthrough=self.unreachable_after is None,
        )
        if incoming:
            self.unreachable_after = None
        return source

    def jump(self, name: str, native: Jump | CondJump) -> TimelineJump:
        kind = "jump" if isinstance(native, Jump) else "cond_jump"
        self._check(f"{kind} to", name)
        self.require_reachable()
        if name in self.defined:
            raise ValueError(
                f"timeline jump to {name!r} is backward; general jumps must target a later label "
                "(use Repeat or a sweep for iteration)"
            )
        if isinstance(native, CondJump):
            if native.test in ("F", "NF"):
                raise ValueError(
                    f"timeline cond_jump to {name!r}: peripheral-flag test {native.test!r} is unsupported in _body"
                )
            if native.test not in ("Z", "NZ", "S", "NS"):
                raise ValueError(
                    f"timeline cond_jump to {name!r}: test {native.test!r} must be Z, NZ, S or NS in _body"
                )
        incoming = self.pending.setdefault(name, [])
        target = incoming[0].target if incoming else _Target(name)
        native.label = target
        source = TimelineJump(self, target, native)
        incoming.append(source)
        if kind == "jump":
            self.unreachable_after = name
        return source

    def finish(self) -> None:
        if self.pending:
            name = next(iter(self.pending))
            raise ValueError(f"timeline jump target {name!r} is never defined in _body")


def link_author_labels(prog: TimelineProgramV2) -> None:
    """Bind QICK's author identities to collision-free strings before IR linking."""
    # QICK's unlinked translation stores the logical identity as a key.
    unlinked: dict[object, str] = {key: addr for key, addr in prog.labels.items()}
    targets = [key for key in unlinked if isinstance(key, _Target)]
    if not targets:
        return
    labels = {key: addr for key, addr in unlinked.items() if isinstance(key, str)}
    allocated = set(labels)
    names = {
        target: make_label(f"_timeline_jump_{i}", allocated).name
        for i, target in enumerate(targets)
    }
    # Raw references keep native and pseudo-label precedence, but can still
    # name a body label by its original spelling when there is no native one.
    raw_targets = {
        target.name: name
        for target, name in names.items()
        if target.name not in labels and target.name not in PSEUDO_LABELS
    }
    prog.labels = labels | {name: unlinked[target] for target, name in names.items()}
    for inst in prog.prog_list:
        if isinstance(target := inst.get("LABEL"), _Target):
            inst["LABEL"] = names[target]
        elif isinstance(target, str) and target in raw_targets:
            inst["LABEL"] = raw_targets[target]
    for meta in prog.meta_infos:
        if meta["kind"] == "label" and isinstance(target := meta["name"], _Target):
            meta["name"] = names[target]
