"""Record timeline actions without acquiring data or touching hardware."""

from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, TypedDict, Unpack, override

from qick import QickConfig
from qick.asm_v2 import QickParam

from .modules.base import Module
from .sweep import SweepCfg
from .timeline import TimelineProgramV2
from .utils import sweep2param

_Action = Callable[[TimelineProgramV2], None]


class _TriggerOptions(TypedDict, total=False):
    """Optional trigger controls, with native defaults when omitted.

    ddr4/mr enable the corresponding capture buffers (default False).
    tag names this occurrence for time lookup (default None).
    timeline places at its end plus t; None uses scope or native reference timing.
    """

    ddr4: bool
    mr: bool
    tag: str | None
    timeline: str | None


class _RecordedProgram(TimelineProgramV2):
    """Replay initialization and body actions on every compilation."""

    def __init__(
        self,
        soccfg: QickConfig,
        initialize: Sequence[_Action],
        body: Sequence[_Action],
        **kwargs,
    ) -> None:
        self._initialize_actions = tuple(initialize)
        self._body_actions = tuple(body)
        super().__init__(soccfg, **kwargs)

    @override
    def _initialize(self, cfg: None) -> None:
        for action in self._initialize_actions:
            action(self)

    @override
    def _body(self, cfg: None) -> None:
        for action in self._body_actions:
            action(self)


class BuilderBranchCases:
    """Recording handle yielded by TimelineBuilder.branch, not a runtime selector.

    Use case(index) for selected zero-based sweep points, in any order. Record body
    operations on the originating builder inside each case. Handles are valid
    only inside their branch context; cases cannot overlap on the same handle.
    Timeline selection, timing and index validation belong to the program
    compiler at build, not this handle.
    """

    def __init__(
        self, record_case: Callable[[int], AbstractContextManager[str]]
    ) -> None:
        """Use record_case to open a case; normally yielded by builder.branch().

        The callback captures operations on its builder and enforces recording
        lifetime/overlap constraints. It yields the implicit case timeline name.
        """
        self._record_case = record_case

    def case(self, index: int) -> AbstractContextManager[str]:
        """Record a zero-based sweep point's body; omitted points do nothing.

        Yield <by>[<index>], the case's default timeline; only it and local
        forks are accessible inside. Record through the owning builder.
        Duplicate, non-integer and out-of-range
        indices raise ValueError at build. Calls after branch exit or build,
        and overlapping cases, raise ValueError immediately.
        Nested branches are supported. Repeat in a case fails at build because
        its entry/backedge advance reference; put Repeat outside the branch.
        """
        return self._record_case(index)


class TimelineBuilder:
    """Record a program, automatically declaring modules on their first emit.

    soccfg is a QICK compile profile, not a connected SoC. reps counts shots;
    initial_delay and relax_delay are us. Module objects are borrowed by identity:
    do not mutate them after recording. The same module may be recorded in
    multiple builders or declared in native programs. Each compiled program
    owns its declaration results, including waveform and pulse identifiers.
    Loops replay in call order, outside in; body actions replay in call order.
    build freezes this builder and compiles a TimelineProgramV2, with no
    acquisition or Schedule dependency. Later recording
    or build calls raise ValueError. Compiler/owner errors propagate.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        *,
        reps: int = 1,
        initial_delay: float = 1.0,
        relax_delay: float = 1.0,
    ) -> None:
        """Capture the compile profile and shot controls without compiling."""
        self._soccfg = soccfg
        self._reps = reps
        self._initial_delay = initial_delay
        self._relax_delay = relax_delay
        self._initialize_actions: list[_Action] = []
        self._body_actions: list[_Action] = []
        self._action_stack = [self._body_actions]
        self._modules: list[Module] = []
        self._built = False
        self._loop_names = {"reps"}
        self._timeline_name: str | None = None

    def _ensure_recording(self) -> None:
        if self._built:
            raise ValueError("TimelineBuilder is frozen after build")

    def _record(self, action: _Action) -> None:
        self._ensure_recording()
        self._action_stack[-1].append(action)

    @contextmanager
    def _capture(self) -> Generator[list[_Action]]:
        self._ensure_recording()
        actions: list[_Action] = []
        self._action_stack.append(actions)
        try:
            yield actions
        finally:
            self._action_stack.pop()

    def timeline(self, name: str) -> AbstractContextManager[None]:
        """Record a scoped default for omitted timeline arguments.

        Applies to emit, pulse, trigger, delay_timeline, sync and
        the repeat and branch parents. Branch cases shadow this default.
        name must exist at replay (default or forked); errors surface at build.
        Explicit names win, nested scopes override and exit restores the prior
        default. No fork/join or native delay/delay_auto change occurs. Close
        before build; recording after build raises ValueError.
        """
        self._ensure_recording()
        return self._timeline(name)

    @contextmanager
    def _timeline(self, name: str) -> Generator[None]:
        previous = self._timeline_name
        self._timeline_name = name
        try:
            with self._capture() as actions:
                yield
        finally:
            self._timeline_name = previous
        recorded = tuple(actions)

        def replay(prog: TimelineProgramV2) -> None:
            with prog.timeline(name):
                for action in recorded:
                    action(prog)

        self._record(replay)

    def repeat(
        self,
        name: str,
        n: int | str,
        *,
        advance_ref: bool,
        timeline: str | None = None,
        range_hint: tuple[int, int] | None = None,
    ) -> AbstractContextManager[None]:
        """Record a repeated body; replay program-owned timing rules at build.

        advance_ref is a required bool. False shadows the default with a
        shot-local accumulating timeline and preserves reference throughout the
        body. It permits branch cases and nested False, but rejects sync, native
        delay/delay_auto and inner True. True advances reference on entry and
        each iteration, leaving parent zero; it is forbidden in cases.
        timeline=None selects the scope, or default. Only parent/local forks are
        accessible; local names expire at exit. Time parameters remain in us.
        name identifies the counter. n is a nonnegative integer or existing
        register name. Register counts accept an optional guaranteed closed
        nonnegative integer range_hint; fixed counts forbid it. Zero discards
        the body; only True syncs entry. Invalid bool/count/range/timeline uses
        fail at build. QICK register exhaustion propagates. Recording after build
        raises ValueError. Close all recording contexts before build.
        """
        self._ensure_recording()
        return self._repeat(name, n, timeline, range_hint, advance_ref=advance_ref)

    @contextmanager
    def _repeat(
        self,
        name: str,
        n: int | str,
        timeline: str | None,
        range_hint: tuple[int, int] | None,
        *,
        advance_ref: bool,
    ) -> Generator[None]:
        with self._capture() as actions:
            yield
        recorded = tuple(actions)

        def replay(prog: TimelineProgramV2) -> None:
            with prog.repeat(
                name,
                n,
                timeline=timeline,
                range_hint=range_hint,
                advance_ref=advance_ref,
            ):
                for action in recorded:
                    action(prog)

        self._record(replay)

    def branch(
        self, *, by: str, align_timeline: bool, timeline: str | None = None
    ) -> AbstractContextManager[BuilderBranchCases]:
        """Record indexed cases without advancing the shared reference.

        by names an existing sweep loop. The yielded handle's case(index) records
        a zero-based point, in any order; omitted points execute no actions.
        timeline selects the parent; None uses the current scope or default.
        Each case implicitly forks it and yields <by>[<index>], shadowing the
        default. Only that case timeline and its local forks may be used;
        other names fail at build. Required align_timeline=True joins case ends
        into the parent, using static max or exact runtime pointwise max. False
        keeps the executed case end. Other timelines remain unchanged.
        Local fork names expire and unjoined local ends are discarded. True
        requires all ends to be reconstructible from branch-entry state.
        Case delay/delay_auto/sync fail at build; use delay_timeline. Repeat in
        a case fails at build; move it outside the branch. Branch inside Repeat
        is supported. Unconditional actions in this context fail at build; move
        them before or after the with block. Unknown selectors, invalid indices,
        timing and native register-capacity errors propagate from the program.
        Calls after build raise ValueError. Close before build; the handle
        expires on exit.
        """
        self._ensure_recording()
        return self._branch(by, align_timeline=align_timeline, timeline=timeline)

    @contextmanager
    def _branch(
        self, by: str, *, align_timeline: bool, timeline: str | None
    ) -> Generator[BuilderBranchCases]:
        self._ensure_recording()
        cases: list[tuple[int, tuple[_Action, ...]]] = []
        is_open = True
        case_active = False

        def ensure_case() -> None:
            self._ensure_recording()
            if not is_open:
                raise ValueError("branch.case requires an open builder branch")
            if case_active:
                raise ValueError("branch.case cannot overlap another case")

        @contextmanager
        def capture_case(index: int) -> Generator[str]:
            nonlocal case_active
            ensure_case()
            case_active = True
            previous = self._timeline_name
            self._timeline_name = f"{by}[{index}]"
            try:
                with self._capture() as case_actions:
                    yield self._timeline_name
                # Cases replay separately from the branch's surrounding actions.
                cases.append((index, tuple(case_actions)))
            finally:
                self._timeline_name = previous
                case_active = False

        def record_case(index: int) -> AbstractContextManager[str]:
            ensure_case()
            return capture_case(index)

        try:
            with self._capture() as actions:
                yield BuilderBranchCases(record_case)
        finally:
            is_open = False
        recorded = tuple(actions)
        recorded_cases = tuple(cases)

        def replay(prog: TimelineProgramV2) -> None:
            with prog.branch(
                by=by, align_timeline=align_timeline, timeline=timeline
            ) as branch:
                for action in recorded:
                    action(prog)
                for index, case in recorded_cases:
                    with branch.case(index):
                        for action in case:
                            action(prog)

        self._record(replay)

    def add_loop(self, name: str, count: int) -> None:
        """Record count points, outermost first; duplicate names raise ValueError.

        The shot loop reserves "reps". Loop count validation belongs to QICK at
        build; recording after build raises ValueError.
        """
        self._ensure_recording()
        if name in self._loop_names:
            raise ValueError(f"loop name already used: {name!r}")
        self._loop_names.add(name)
        self._initialize_actions.append(lambda prog: prog.add_loop(name, count))

    def add_sweep(self, name: str, spec: SweepCfg) -> QickParam:
        """Record spec.expts loop points and return its uniform QICK parameter.

        spec supplies inclusive start/stop values in the consumer's units.
        The returned parameter can be placed in module cfg before emit.
        Duplicate/reserved loop names or calls after build raise ValueError.
        """
        self.add_loop(name, spec.expts)
        return sweep2param(name, spec)

    def emit(
        self, module: Module, *, timeline: str | None = None, tag: str | None = None
    ) -> None:
        """Record an emission and declare this object once during initialization.

        None selects the scoped name, or default outside a scope.
        tag names this occurrence, not the module.
        Module declaration/lowering errors surface at build.
        """
        self._ensure_recording()
        self._declare(module)
        self._record(lambda prog: prog.emit(module, timeline=timeline, tag=tag))

    def _declare(self, module: Module) -> None:
        if not any(module is declared for declared in self._modules):
            self._modules.append(module)
            self._initialize_actions.append(lambda prog: prog.declare(module))

    def delay(self, t: float | QickParam, tag: str | None = None) -> None:
        """Record native reference advance by t us; tag exposes resolved time."""
        self._record(lambda prog: prog.delay(t, tag=tag))

    def delay_auto(
        self,
        t: float | QickParam = 0,
        *,
        gens: bool = True,
        ros: bool = True,
        tag: str | None = None,
    ) -> None:
        """Record native advance to selected generator/ADC ends plus t us.

        gens/ros select channel families; tag exposes the resolved advance.
        Compilation propagates native QICK timing failures.
        """
        self._record(lambda prog: prog.delay_auto(t, gens=gens, ros=ros, tag=tag))

    def delay_timeline(
        self,
        t: float | QickParam | str,
        *,
        timeline: str | None = None,
        tag: str | None = None,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        """Record end += t without moving reference; tag exposes static end.

        None timeline selects the scoped name, or default outside a scope.
        t is us, a QICK parameter, or an author-calculated nonnegative cycle
        register name. range_hint optionally guarantees inclusive integer-cycle
        bounds for a register. Unbounded values use runtime maxima; overlap and
        overflow are author-owned. Invalid hints and unknown/wait-only timelines
        fail during build, as in TimelineProgramV2.delay_timeline.
        """
        self._record(
            lambda prog: prog.delay_timeline(
                t, timeline=timeline, tag=tag, range_hint=range_hint
            )
        )

    def fork(self, source: str, *, into: str | list[str]) -> None:
        """Record new timeline names copying source's current end.

        source must exist; into is one fresh name or a list of distinct fresh
        names, captured now. Wait-only sources produce ordinary placement
        targets. No reference advance or output occurs. Unknown/duplicate names
        fail at build, and recording after build raises ValueError.
        """
        self._ensure_recording()
        recorded = into if isinstance(into, str) else list(into)
        self._record(lambda prog: prog.fork(source, into=recorded))

    def join(self, *sources: str, into: str) -> None:
        """Record max of existing sources and target into target's end.

        Sources remain unchanged; the target becomes ordinary, including when
        sources are wait-only. This is bookkeeping, not a context manager; it
        emits no output or reference advance. Unknown names fail at build with
        the program error. Recording after build raises ValueError.
        """
        self._record(lambda prog: prog.join(*sources, into=into))

    def sync(self, *, timeline: str | None = None) -> None:
        """Record reference advance to the selected end; unknown names fail at build.

        None selects the scoped name, or default outside a scope.
        """
        self._record(lambda prog: prog.sync(timeline=timeline))

    def pulse(
        self,
        ch: int,
        name: str,
        t: float | QickParam | Literal["auto"] = 0,
        tag: str | None = None,
        *,
        timeline: str | None = None,
    ) -> None:
        """Record playback of a declared pulse template on generator ch.

        name identifies a template declared by an emitted module. t is us or
        "auto"; tag identifies this occurrence. None uses the scoped name, or
        native reference placement outside a scope. Names place at end plus t
        and advance that end. Missing declarations/unsafe overlap fail at build.
        """
        self._record(lambda prog: prog.pulse(ch, name, t, tag, timeline=timeline))

    def trigger(
        self,
        ros: Sequence[int] | None = None,
        tts: Sequence[int] | None = None,
        pins: Sequence[int] | None = None,
        t: float | QickParam | None = 0,
        width: float | QickParam | None = None,
        **options: Unpack[_TriggerOptions],
    ) -> None:
        """Record ADC/time-tagger/pin triggers with native or timeline timing.

        ros/tts/pins are channel lists; None selects none. t is us (default 0),
        or native None for the current time register. width is us or a QICK
        parameter; None selects QICK's 10-cycle default. Keyword-only ddr4/mr
        enable capture buffers, default
        False. tag names the occurrence, default None. timeline defaults to
        None (scoped name, or native reference outside a scope); a name places
        at its end plus t and advances through ADC/trigger completion. Named t=None, unknown resources and
        invalid timing fail at build. Unknown keywords raise TypeError now.
        """
        self._ensure_recording()
        unknown = options.keys() - {"ddr4", "mr", "tag", "timeline"}
        if unknown:
            raise TypeError(f"unexpected trigger keyword(s): {sorted(unknown)}")
        recorded_ros = None if ros is None else tuple(ros)
        recorded_tts = None if tts is None else tuple(tts)
        recorded_pins = None if pins is None else tuple(pins)
        self._record(
            lambda prog: prog.trigger(
                recorded_ros, recorded_tts, recorded_pins, t, width, **options
            )
        )

    def build(self) -> TimelineProgramV2:
        """Freeze and compile; subsequent recording/build calls raise ValueError.

        The program retains the recorded actions without depending on builder
        state. Open recording contexts raise ValueError without freezing.
        Owner/compiler errors propagate, leaving the builder frozen even
        when compilation fails.
        """
        self._ensure_recording()
        if len(self._action_stack) != 1:
            raise ValueError("close all recording contexts before build")
        self._built = True
        return _RecordedProgram(
            self._soccfg,
            self._initialize_actions,
            self._body_actions,
            reps=self._reps,
            initial_delay=self._initial_delay,
            relax_delay=self._relax_delay,
        )
