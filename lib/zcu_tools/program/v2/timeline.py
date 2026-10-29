"""Native QICK programs with timeline ends relative to reference time."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager

from qick.asm_v2 import ConfigReadout, Macro, OpenLoop, Pulse, QickParam, Trigger

from ._timeline_bounds import CycleBounds, FixedLoopIndex, ProducerBounds
from ._timeline_macros import (
    BranchCases,
    FixedRepeatLoop,
    ModuleEmission,
    PreparedTimelineMacro,
    PreparedTimelineOpenLoop,
    TimelineAdvance,
    TimelineBoundary,
    TimelineBranch,
    TimelineFork,
    TimelineJoin,
    TimelineRepeatEnter,
    TimelineRepeatExit,
    TimelineRuntimeAdvance,
    TimelineSync,
    TimeProducer,
)
from ._timeline_outputs import (
    TimelineObservedPulse,
    TimelinePlacedConfig,
    TimelinePlacedPulse,
    TimelinePlacedTrigger,
)
from ._timeline_runtime import RuntimeEnd
from ._timeline_state import TimelineBook
from .base import MyProgramV2
from .ir import TimingLayout, TimingLoop
from .macro.delay import DelayRegAuto
from .modules.base import Module
from .resources import ProgramResourcesMixin


def _repeat_minimum(name: str, n: int | str, range_hint: tuple[int, int] | None) -> int:
    """Validate the public count contract and return its guaranteed minimum."""
    if isinstance(n, str):
        if (
            not isinstance(range_hint, tuple)
            or len(range_hint) != 2
            or any(type(value) is not int or value < 0 for value in range_hint)
            or range_hint[0] > range_hint[1]
        ):
            raise ValueError("register repeat requires a valid nonnegative range_hint")
        if n == name:
            raise ValueError("repeat n must differ from its counter register name")
        return range_hint[0]
    if type(n) is not int or n < 0:
        raise ValueError("repeat n must be a nonnegative integer or register name")
    if range_hint is not None:
        raise ValueError("integer repeat does not accept range_hint")
    return n


class TimelineProgramV2(MyProgramV2, ProgramResourcesMixin):
    """Author native QICK hooks with optional timeline ends (times in us).

    Constructor follows MyProgramV2: reps, initial_delay, relax_delay. Native
    operations retain QICK timing. Named outputs update their relative ends;
    sync and body completion move reference time. Each compile resets ends.
    All instructions share the existing IR and acquisition implementations.
    Simulation frontend migration is separate.
    """

    macro_list: list[Macro]

    def __init__(self, *args, **kwargs):
        self._timeline_book = TimelineBook()
        self._time_bounds = ProducerBounds()
        self._time_origin: Macro | None = None
        self._timeline_register_pool: list[str] = []
        self._timeline_math_active = False
        self._declared_modules: list[Module] = []
        self._declaring = False
        self._building_body = False
        self._branch_serial = 0
        self._emission_serial = 0
        self._emission_scope: tuple[int, int] | None = None
        super().__init__(*args, **kwargs)

    def _make_asm(self) -> None:
        self._timeline_register_pool = []
        self._time_bounds = ProducerBounds()
        for macro in self.macro_list:
            self._observe_time_producers(macro)
        super()._make_asm()
        self._time_bounds.audit(self)

    def _pop_native[T: Macro](self, expected: type[T]) -> T:
        source = self.macro_list.pop()
        if not isinstance(source, expected):
            raise RuntimeError(f"expected native {expected.__name__} macro")
        return source

    def _timeline_live_registers(self) -> set[str]:
        values = [
            *self._timeline_book.ends.values(),
            *self._timeline_book.runtime_gens.values(),
            *self._timeline_book.runtime_ros.values(),
        ]
        for frame in self._timeline_book.repeats:
            values.extend(frame.external.values())
        return self._timeline_book.pinned | {
            value.register for value in values if isinstance(value, RuntimeEnd)
        }

    def _observe_time_producers(self, macro: Macro) -> None:
        if isinstance(macro, PreparedTimelineMacro):
            self._observe_time_producers(macro.source)
            return
        self._time_bounds.observe(macro)
        if type(macro) is FixedRepeatLoop:
            self._time_bounds.register_fixed_loop(macro.index, macro, macro.closing)
        if isinstance(macro, TimeProducer):
            self._time_bounds.register(
                macro.register,
                macro.limits,
                tuple(macro.macros),
                macro.dependencies,
                macro.dependency_ranges,
            )
        if isinstance(macro, (ModuleEmission, TimeProducer)):
            for child in macro.macros:
                self._observe_time_producers(child)
        if isinstance(macro, TimelineBranch):
            for case in macro.cases:
                for child in case:
                    self._observe_time_producers(child)

    def _add_asm(self, inst, addr_inc=1):
        self._time_bounds.emission(
            inst, self.p_addr, self._time_origin, arithmetic=self._timeline_math_active
        )
        super()._add_asm(inst, addr_inc=addr_inc)

    def append_macro(self, macro: Macro) -> None:
        """Append a native macro, retaining its behavior and repeat pulse validation.

        Pulse subclasses are observed through their public preprocess timing
        inputs, including pulses supplied directly rather than through pulse().
        Other macros pass through unchanged.
        """
        if isinstance(macro, Pulse) and not isinstance(macro, TimelineObservedPulse):
            macro = TimelineObservedPulse(self._timeline_book, macro)
        super().append_macro(macro)

    def make_program(self) -> None:
        """Build the QICK shell with linked body markers and all-timeline closure.

        Body closure precedes body_end and final_wait/final_delay. Loop hooks,
        counters, cleanup and subroutines retain AveragerProgramV2's order.
        """
        # QICK 0.2.436 provides no public shell boundary hooks.
        self._reset_resources()
        self._emission_serial = 0
        self._emission_scope = None
        self._declared_modules.clear()
        self.append_macro(TimelineBoundary(self._timeline_book, "reset"))
        self.set_ext_counter(addr=self.COUNTER_ADDR)
        self._declaring = True
        try:
            self._initialize(None)
        finally:
            self._declaring = False
        self.append_macro(TimelineBoundary(self._timeline_book, "initialize_end"))
        self.delay_auto(self.initial_delay)
        for name, count, before, _after in self.loops:
            self.open_loop(count, name=name)
            if before is not None:
                self.extend_macros(before)
        self.append_macro(TimelineBoundary(self._timeline_book, "body_start"))
        self._building_body = True
        try:
            self._body(None)
        finally:
            self._building_body = False
        self.append_macro(TimelineSync(self._timeline_book, None))
        self.append_macro(TimelineBoundary(self._timeline_book, "body_end"))
        self.wait_auto(self.final_wait, no_warn=True)
        if self.final_delay is not None:
            self.delay_auto(self.final_delay)
        self.inc_ext_counter(addr=self.COUNTER_ADDR)
        for _name, _count, _before, after in self.loops[::-1]:
            if after is not None:
                self.extend_macros(after)
            self.close_loop()
        self._cleanup(None)
        self.end()
        for name, asm in self.subroutines.items():
            self.label(name)
            self.extend_macros(asm)
            self.ret()
        self.macro_list = [
            PreparedTimelineOpenLoop(self._timeline_book, macro)
            if isinstance(macro, OpenLoop)
            else PreparedTimelineMacro(self._timeline_book, macro)
            for macro in self.macro_list
        ]

    @property
    def timing_layout(self) -> TimingLayout:
        """Return linked body addresses and outer-to-inner axes after compile.

        Missing or duplicate compiler markers raise ValueError. Body end includes
        timeline closure but precedes shell timing; shell timing belongs to shot.
        """
        return TimingLayout.from_metadata(
            self.meta_infos,
            tuple(TimingLoop(name, count) for name, count, _, _ in self.loops),
        )

    @property
    def fixed_loop_indices(self) -> tuple[FixedLoopIndex, ...]:
        """Return Repeat-owned counter metadata in emission order after compile.

        Includes nonempty fixed repeats nested in module emissions, separate
        from QICK's top-level sweep loop_dict. No nested axis is added there.
        A recompile rebuilds these immutable records with their native writers.
        """
        return tuple(self._time_bounds.fixed_loops.values())

    def declare(self, module: Module) -> None:
        """Declare a borrowed module in _initialize; repeat declarations are harmless.

        Do not mutate it afterward. Outside _initialize raises ValueError;
        unsupported module types raise NotImplementedError.
        """
        if not self._declaring:
            raise ValueError("declare must be called in _initialize")
        if any(module is declared for declared in self._declared_modules):
            return
        module.declare(self)
        self._declared_modules.append(module)

    def emit(
        self, module: Module, *, timeline: str | None = None, tag: str | None = None
    ) -> None:
        """Emit a declared module once, with a fresh timing/tag occurrence.

        Omitted timeline selects ordinary default, not reference time. Internal
        events share its current end; rule-a duration advances that end only.
        DelayAuto is native reference advance and rejects explicit timelines.
        Undeclared objects raise ValueError; owner lowering failures propagate.
        """
        if not any(module is declared for declared in self._declared_modules):
            raise ValueError(f"{module.name}: module must be declared before emit")
        start = len(self.macro_list)
        previous_scope = self._emission_scope
        self._emission_scope = (self._emission_serial, 0)
        self._emission_serial += 1
        try:
            module.lower(self, timeline=timeline, tag=tag)
            macros = self.macro_list[start:]
        finally:
            self._emission_scope = previous_scope
            del self.macro_list[start:]
        self.append_macro(
            ModuleEmission(
                module,
                [PreparedTimelineMacro(self._timeline_book, macro) for macro in macros],
                tag,
            )
        )

    def emission_name(self, stem: str) -> str:
        """Allocate an internal register/loop name during a module's lower call.

        stem labels the owner's internal resource. Each call gets a distinct
        name in this emission, including nested emissions. Names reset on each
        compile; they do not reserve resources or rename explicit native loops.
        A collision with an explicit native name fails during compilation.
        Outside lower raises ValueError. Failed/nested lower calls restore the
        caller's naming scope, without changing declared module resources.
        """
        if self._emission_scope is None:
            raise ValueError("emission_name requires an active module lower call")
        emission, local = self._emission_scope
        self._emission_scope = (emission, local + 1)
        return f"__emission_{emission}_{local}_{stem}"

    @contextmanager
    def time_producer(
        self,
        register: str,
        bounds: CycleBounds,
        *,
        dependencies: tuple[str, ...] = (),
        dependency_ranges: Mapping[str, CycleBounds] | None = None,
    ) -> Generator[None]:
        """Certify framework-owned native operations that produce cycle values.

        register receives the inclusive bounds on every execution of this scope.
        dependencies names input registers whose producers must also be audited.
        dependency_ranges additionally constrains producer inputs, such as a
        table's valid indices. Its keys automatically join the audited inputs.
        Checks use conservative whole-program bounds, including later writes,
        so a subsequent index reset cannot hide an earlier invalid lookup.
        This seam is for known table/runtime-length owners, not arbitrary author
        macros. Only captured framework/native operations may write the result;
        later or alternate unregistered writes invalidate the proof at compile.
        Bounds are checked only if this result is consumed as time, so uint32
        words retain their bit-pattern contract for non-time consumers.
        """
        start = len(self.macro_list)
        yield
        macros = self.macro_list[start:]
        del self.macro_list[start:]
        if any(
            not type(macro).__module__.startswith(("qick.", "zcu_tools.program."))
            for macro in macros
        ):
            raise ValueError("custom macro cannot certify a time producer")
        self.append_macro(
            TimeProducer(
                self._timeline_book,
                register,
                bounds,
                macros,
                dependencies,
                dependency_ranges if dependency_ranges is not None else {},
            )
        )

    def advance_module(
        self,
        duration: float | QickParam,
        *,
        timeline: str | None,
        tag: str | None = None,
    ) -> None:
        """Add an owner's already tProc-aligned rule-a duration in us to its end.

        None selects default. Unlike arbitrary delay_timeline inputs, this value
        is not ceiled again. tag names the resolved end via get_time_param.
        """
        self.append_macro(
            TimelineAdvance(
                self._timeline_book,
                timeline if timeline is not None else "default",
                duration,
                tag=tag,
                aligned=True,
            )
        )

    def delay_timeline(
        self,
        t: float | QickParam | str,
        *,
        timeline: str = "default",
        tag: str | None = None,
    ) -> None:
        """Add time to a timeline without advancing reference or channel timestamps.

        Scalars and QickParam use us, ceiled to tProc cycles. A string names an
        existing nonnegative cycle register with compiler-known producer bounds.
        Wait-only targets, unknown bounds and possible signed overflow raise
        ValueError during compilation. tag exposes static ends as parameter 't';
        runtime ends have no affine get_time_param value.
        """
        self.append_macro(TimelineAdvance(self._timeline_book, timeline, t, tag=tag))

    def set_timestamp(
        self,
        val: float | QickParam,
        gen_ch: int | None = None,
        ro_ch: int | None = None,
    ) -> None:
        """Set a native channel timestamp in us using QICK's channel validation.

        During repeat preprocessing, record channels with newly scheduled events.
        Their last-iteration tails must not receive the external-duration debit.
        """
        super().set_timestamp(val, gen_ch=gen_ch, ro_ch=ro_ch)
        if self._timeline_book.repeats and not self._timeline_book.native_only:
            frame = self._timeline_book.repeats[-1]
            if gen_ch is not None:
                frame.written_gens.add(gen_ch)
            if ro_ch is not None:
                frame.written_ros.add(ro_ch)

    def decrement_timestamps(self, t: float | QickParam | None) -> None:
        """Rebase all relative ends when QICK moves reference by t us.

        None is native auto with no selected channels and changes no ends.
        Ends use a pointwise zero floor. A crossing floor promotes only that
        expression to runtime cycles; signed overflow fails during compile.
        """
        super().decrement_timestamps(t)
        if t is not None:
            self._timeline_book.decrement(t)

    def delay_reg_auto(self, time_reg: str, *, gens=True, ros=True) -> None:
        """Advance reference by selected native ends plus register cycles.

        time_reg contains nonnegative tProc cycles. Remaining ends debit the
        actual advance using certified producer bounds. No bounds are needed
        when the selected native ends already consume every timeline. This
        native register reference operation remains forbidden inside Repeat.
        """
        super().delay_reg_auto(time_reg, gens=gens, ros=ros)
        source = self._pop_native(DelayRegAuto)
        self.append_macro(TimelineRuntimeAdvance(self._timeline_book, source))

    def fork(self, source: str, *, into: str | list[str]) -> None:
        """Declare new timeline names in _body, copying source end at this point.

        source must already exist; into is one new name or a list of distinct
        new names. A wait-only source supplies its actual remaining end; new
        timelines permit placement while that source stays wait-only.
        Unknown source or duplicate targets raise ValueError during
        compile. Calling outside _body raises ValueError immediately. This emits
        no output and does not advance reference. Each compile resets names.
        """
        if not self._building_body:
            raise ValueError("fork must be called in _body")
        targets = (into,) if isinstance(into, str) else tuple(into)
        self.append_macro(TimelineFork(self._timeline_book, source, targets))

    @contextmanager
    def branch(self, *, by: str) -> Generator[BranchCases]:
        """Select one case per sweep point without padding paths to equal duration.

        by is an existing sweep loop name. Provide exactly one case() per point.
        Each case closes only timelines it uses. Different remaining ends become
        wait-only; join/fork derive ordinary targets, while sync consumes them.
        Forked names are case-local, even when every case uses the same spelling.
        Their ends join the case closing maximum and names expire at its exit.
        Measurement selectors raise ValueError. Call only while building _body.
        """
        if not self._building_body:
            raise ValueError("branch must be called in _body")
        cases = BranchCases(self)
        name = f"_timeline_branch_{self._branch_serial}"
        self._branch_serial += 1
        yield cases
        self.append_macro(TimelineBranch(self._timeline_book, by, cases.cases, name))

    @contextmanager
    def repeat(
        self,
        name: str,
        n: int | str,
        *,
        timeline: str = "default",
        range_hint: tuple[int, int] | None = None,
    ) -> Generator[None]:
        """Repeat a body n times on parent timeline (times remain in us).

        Entry syncs the parent. Each iteration closes parent and local ends;
        exit discards local names and leaves parent zero. Fixed counts exactly
        rebase outer ends. Unknown counts or case advances debit outer ends on
        the actual executed path. Nonzero outer ends remain wait-only; join/fork
        establish placement targets and full consumption restores placement. Only
        parent and locally forked names may be used inside. name is
        the native loop/counter name. n is a nonnegative integer or existing
        register name. Register counts require an author-guaranteed closed
        nonnegative integer range_hint; integer counts forbid range_hint. No
        hardware range check is added. Invalid counts/ranges raise ValueError;
        unknown/inaccessible names fail during compile. Zero counts still sync
        the parent but discard body operations. Register-valued timeline delays
        and differing Branch advances are supported. Native delay_reg_auto inside
        the body raises ValueError; it is not a timeline duration operation.
        """
        minimum = _repeat_minimum(name, n, range_hint)
        if n == 0:
            self.sync(timeline=timeline)
            start = len(self.macro_list)
            try:
                yield
            finally:
                del self.macro_list[start:]
            return
        variable = isinstance(n, str)
        entry = TimelineRepeatEnter(
            self._timeline_book, timeline, name if variable else None
        )
        self.append_macro(entry)
        fixed_loop = FixedRepeatLoop(name, n) if isinstance(n, int) else None
        if fixed_loop is None:
            self.open_inner_loop(name, name, n, range_hint=range_hint)
        else:
            self.append_macro(fixed_loop)
        try:
            yield
        finally:
            self.append_macro(
                TimelineRepeatExit(
                    self._timeline_book,
                    minimum,
                    range_hint[1] if range_hint is not None else minimum,
                    variable=range_hint is not None and minimum != range_hint[1],
                )
            )
            if isinstance(n, str):
                self.close_inner_loop(name, name, n)
            elif fixed_loop is not None:
                self.append_macro(fixed_loop.closing)

    def join(self, *sources: str, into: str) -> None:
        """Merge existing sources and target into target's latest end (us).

        Sources remain unchanged. Wait-only sources contribute actual ends;
        the target becomes ordinary and permits placement. Crossing maxima use
        runtime arithmetic without moving reference or emitting output. Unknown
        names raise ValueError; declare names with fork before joining them.
        """
        self.append_macro(TimelineJoin(self._timeline_book, sources, into))

    def sync(self, *, timeline: str = "default") -> None:
        """Move reference to the chosen end and floor every rebased end at zero.

        Only default exists initially; other names require fork in _body.
        Fully consumed wait-only bounds become ordinary zero ends again.
        Unknown names or unprovable swept floors raise ValueError during compile.
        """
        self.append_macro(TimelineSync(self._timeline_book, timeline))

    def send_readoutconfig(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Send an instantaneous config at timeline end+t us, or native t if omitted.

        Native args are t=0, tag=None. A timeline end becomes the scheduled config
        instant; config length and unrelated ADC windows do not contribute.
        Wait-only timelines reject placement even at t=0; first sync the bound
        or establish an ordinary scheduling point with join/fork.
        Declaration and timing errors propagate.
        """
        super().send_readoutconfig(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self._pop_native(ConfigReadout)
            self.append_macro(
                TimelinePlacedConfig(self._timeline_book, timeline, source)
            )

    def pulse_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Play runtime words at native t or timeline end+t, advancing a named end.

        kwargs are t (us), tag, freq_reg and gain_reg containing final hardware
        words. Native patching and template/register validation are unchanged.
        """
        super().pulse_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self._pop_native(Pulse)
            self.append_macro(
                TimelinePlacedPulse(self._timeline_book, timeline, source)
            )

    def send_readoutconfig_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Send runtime ADC words at native t or timeline end+t (us).

        kwargs are t, tag and freq_reg. A timeline end becomes the scheduled config
        instant, independent of config length. Wait-only timelines reject placement
        even at t=0; first sync the bound or establish an ordinary scheduling
        point with join/fork. Native patching and declaration/register validation
        are unchanged.
        """
        super().send_readoutconfig_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self._pop_native(ConfigReadout)
            self.append_macro(
                TimelinePlacedConfig(self._timeline_book, timeline, source)
            )

    def trigger(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Trigger ADC/tt/pins at native t or timeline end+t and advance named end.

        Native args select ros/tts/pins, t, width (us), ddr4/mr and tag. Named end
        includes ADC integration and trigger width, rounded up to tProc cycles.
        Named t=None raises ValueError at entry; native special values are unchanged.
        """
        super().trigger(*args, **kwargs)
        if timeline is not None:
            source = self._pop_native(Trigger)
            if source.t is None:
                raise ValueError("timeline trigger does not support t=None")
            self.append_macro(
                TimelinePlacedTrigger(self._timeline_book, timeline, source)
            )

    def pulse(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Play a pulse at native t or timeline end+t, advancing a named end.

        Native args are t=0 (us or 'auto'), tag=None. Named 'auto' takes the
        provably later timeline/channel end. Named end includes waveform length,
        rounded up to tProc cycles. Crossing swept maxima raise ValueError with
        a delay_timeline hint; native declaration errors propagate.
        """
        super().pulse(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self._pop_native(Pulse)
            self.append_macro(
                TimelinePlacedPulse(self._timeline_book, timeline, source)
            )
