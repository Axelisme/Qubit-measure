# tolerated(file-size): Native declaration ownership and timing share one compile lifecycle.
"""Native QICK programs with timeline ends relative to reference time."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TypedDict, override

from qick.asm_v2 import (
    AsmV2,
    ConfigReadout,
    Macro,
    OpenLoop,
    Pulse,
    QickParam,
    QickRawParam,
    Trigger,
)

from ._timeline_bounds import CycleBounds, FixedLoopIndex, ProducerBounds
from ._timeline_macros import (
    BranchCases,
    FixedRepeatLoop,
    ModuleEmission,
    PreparedTimelineMacro,
    PreparedTimelineOpenLoop,
    ReferenceGuard,
    RepeatTimelineBody,
    RepeatTimelineCapture,
    RepeatTimelineEnter,
    RepeatTimelineExit,
    TimelineAdvance,
    TimelineBoundary,
    TimelineBranch,
    TimelineFork,
    TimelineJoin,
    TimelineRuntimeAdvance,
    TimelineShellWait,
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
from ._timeline_selected import SelectedPlayback, TimelineSelectedPulse
from ._timeline_state import TimelineBook
from .base import MyProgramV2
from .ir import TimingLayout, TimingLoop
from .ir.instructions import BaseInst, RegWriteInst
from .macro.delay import DelayRegAuto
from .macro.table import TableSelection
from .modules.base import Module
from .resources import ProgramResourcesMixin
from .sweep import SweepCfg
from .utils import sweep2param


def _repeat_minimum(
    name: str | None, n: int | str, range_hint: tuple[int, int] | None
) -> int:
    """Validate the public count contract and return its guaranteed minimum."""
    if isinstance(n, str):
        if n == name:
            raise ValueError("repeat count register must differ from counter")
        if range_hint is None:
            return 0
        if (
            type(range_hint) is not tuple
            or len(range_hint) != 2
            or any(type(value) is not int or value < 0 for value in range_hint)
            or range_hint[0] > range_hint[1]
        ):
            raise ValueError("register repeat requires a valid nonnegative range_hint")
        return range_hint[0]
    if type(n) is not int or n < 0:
        raise ValueError("repeat n must be a nonnegative integer or register name")
    if range_hint is not None:
        raise ValueError("integer repeat does not accept range_hint")
    return n


class _RegisterOptions(TypedDict, total=False):
    """Non-null QICK allocation inputs, omitted to request automatic allocation."""

    name: str
    addr: int
    init: QickRawParam


class _TimelineScope(Macro):
    """Validate a lexical default without marking it used or changing any ends."""

    def __init__(self, book: TimelineBook, name: str):
        self.book = book
        self.name = name

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        self.book.require(self.name)

    def translate(self, prog) -> None:
        pass


class TimelineProgramV2(MyProgramV2, ProgramResourcesMixin):
    """Author native QICK hooks with optional timeline ends (times in us).

    Constructor follows MyProgramV2: reps, initial_delay, relax_delay. Native
    operations retain QICK timing. Named outputs update their relative ends;
    sync and body completion move reference time. Each compile resets ends.
    All instructions share the existing IR and acquisition implementations.
    get_time_param returns actual scalar/affine times. Runtime-only and
    non-affine timeline times raise RuntimeError, including as_array=True.
    Reading time parameters does not change scheduling or allocate time data.
    """

    macro_list: list[Macro]

    def __init__(self, *args, **kwargs):
        self._timeline_book = TimelineBook()
        self._time_bounds = ProducerBounds()
        self._time_origin: Macro | None = None
        self._timeline_register_pool: list[str] = []
        self._timeline_math_active = False
        self._declared_modules: list[Module] = []
        self._module_states: dict[int, tuple[Module, object]] = {}
        self._declaration_module: Module | None = None
        self._resource_owners: dict[tuple[str, str | int], Module] = {}
        self._declaring = False
        self._building_body = False
        self._timeline_default: str | None = None
        self._branch_serial = 0
        self._branch_cases: list[BranchCases] = []
        self._emission_serial = 0
        self._emission_scope: tuple[int, int] | None = None
        super().__init__(*args, **kwargs)

    def _make_asm(self) -> None:
        self._timeline_register_pool = []
        self._timeline_book.output_time = None
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
            if macro.selection is not None:
                self._time_bounds.register_selection(
                    macro.register, macro.selection, tuple(macro.macros)
                )
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
        instruction = BaseInst.from_dict(inst)
        if inst["CMD"] in ("JUMP", "CALL", "RET") or (
            "s14" in instruction.reg_write
            and not (
                isinstance(instruction, RegWriteInst) and instruction.is_pure_self_copy
            )
        ):
            self._timeline_book.output_time = None
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
        self._timeline_default = None
        self._emission_serial = 0
        self._emission_scope = None
        self._declared_modules.clear()
        self._module_states.clear()
        self._declaration_module = None
        self._resource_owners.clear()
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
        self.append_macro(TimelineShellWait(self._timeline_book, self.final_wait))
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
    def sweep_loops(self) -> tuple[TimingLoop, ...]:
        """Return declared top-level loop names/counts, including reps, in order.

        Available during _initialize after add_loop, before QICK's loop discovery.
        This immutable view excludes body-local Repeat counters and has no ASM
        register addresses. Recompilation rebuilds it from native declarations.
        """
        return tuple(TimingLoop(name, count) for name, count, _, _ in self.loops)

    def add_sweep(self, name: str, spec: SweepCfg) -> QickParam:
        """Declare a new uniform top-level loop and return its QICK parameter.

        Call in _initialize. spec.expts gives the number of points; inclusive
        start/stop use the consuming parameter's units. Loops are outer-to-inner
        in call order after reps. Duplicate loop names raise ValueError.
        """
        if any(loop.name == name for loop in self.sweep_loops):
            raise ValueError(f"loop name already used: {name!r}")
        self.add_loop(name, spec.expts)
        return sweep2param(name, spec)

    def append_loop_after(self, name: str, *macros: Macro) -> None:
        """Attach native operations after each iteration of a declared sweep loop.

        Call during _initialize, after add_loop. Operations execute after shot
        closure and before the loop test/backedge, in appended order. Unknown
        names or calls outside declaration raise ValueError.
        """
        if not self._declaring:
            raise ValueError("append_loop_after must be called in _initialize")
        for index, (loop_name, count, before, after) in enumerate(self.loops):
            if loop_name != name:
                continue
            if after is None:
                after = AsmV2()
                self.loops[index] = (loop_name, count, before, after)
            for macro in macros:
                after.append_macro(macro)
            return
        raise ValueError(f"sweep loop {name!r} is not declared")

    @property
    def fixed_loop_indices(self) -> tuple[FixedLoopIndex, ...]:
        """Return Repeat-owned counter metadata in emission order after compile.

        Includes nonempty fixed repeats nested in module emissions, separate
        from QICK's top-level sweep loop_dict. No nested axis is added there.
        A recompile rebuilds these immutable records with their native writers.
        """
        return tuple(self._time_bounds.fixed_loops.values())

    def set_module_state(self, module: Module, state: object) -> None:
        """Save this compile's declaration result for the exact module object.

        Module declare implementations call this after allocating their resources.
        The program retains both objects until recompilation. Replacing a result
        is allowed during declaration; calls outside _initialize raise ValueError.
        """
        if not self._declaring:
            raise ValueError("module state can only be saved during declaration")
        self._module_states[id(module)] = (module, state)

    def get_module_state[T](self, module: Module, state_type: type[T]) -> T:
        """Return this program's declaration result as state_type.

        module is the exact object passed to its declaration. Missing declarations
        or a mismatched result type raise ValueError. Results belong to this
        compile and must not be transferred to another program.
        """
        entry = self._module_states.get(id(module))
        if entry is None or entry[0] is not module:
            raise ValueError(
                f"{module.name}: module has no declaration on this program"
            )
        state = entry[1]
        if not isinstance(state, state_type):
            raise ValueError(
                f"{module.name}: expected {state_type.__name__} declaration"
            )
        return state

    def declare(self, module: Module) -> None:
        """Declare a borrowed module in _initialize; repeat declarations are harmless.

        Do not mutate it afterward. Outside _initialize raises ValueError;
        unsupported module types raise NotImplementedError. Generator and ADC
        declarations follow QICK: later declarations overwrite earlier ones,
        and authors own their consistency. Read gen_chs/ro_chs for settings.
        Explicit protect_readout_routes calls prevent later redeclaration.
        Register conflicts raise ValueError naming both objects; explicit
        register borrowing and Pulse registry sharing are retained.
        """
        if not self._declaring:
            raise ValueError("declare must be called in _initialize")
        if any(module is declared for declared in self._declared_modules):
            return
        previous = self._declaration_module
        self._declaration_module = module
        try:
            module.declare(self)
        finally:
            self._declaration_module = previous
        self._declared_modules.append(module)

    def _check_resource_owner(self, kind: str, key: str | int) -> None:
        owner = self._resource_owners.get((kind, key))
        module = self._declaration_module
        if owner is not None and module is not None and owner is not module:
            raise ValueError(
                f"{kind} {key!r}: {module.name} ({module!r}) conflicts with "
                f"{owner.name} ({owner!r})"
            )

    def _remember_resource_owner(self, kind: str, key: str | int) -> None:
        if self._declaration_module is not None:
            self._resource_owners.setdefault((kind, key), self._declaration_module)

    @override
    def add_reg(
        self,
        name: str | None = None,
        addr: int | None = None,
        init: QickRawParam | None = None,
        allow_reuse: bool = False,
    ) -> str:
        """Allocate a native register, tracking module name and address ownership.

        name/addr request an unused logical name/physical data-register address;
        None asks QICK to choose. init is QICK's optional swept initial value.
        Distinct module owners conflict with both names/objects in ValueError.
        Explicit allow_reuse retains QICK's scratch-register borrowing contract
        (addr/init must be None). Reading an existing register never claims it.
        Native calls outside declare(module) retain QICK validation.
        """
        if not allow_reuse:
            if name is not None:
                self._check_resource_owner("register", name)
            if addr is not None:
                self._check_resource_owner("register address", addr)
        # QICK's annotations omit None although omission requests automatic allocation.
        options: _RegisterOptions = {}
        if name is not None:
            options["name"] = name
        if addr is not None:
            options["addr"] = addr
        if init is not None:
            options["init"] = init
        result = super().add_reg(**options, allow_reuse=allow_reuse)
        self._remember_resource_owner("register", result)
        self._remember_resource_owner("register address", self.reg_dict[result].addr)
        return result

    @contextmanager
    def timeline(self, name: str) -> Generator[None]:
        """Select an existing name for omitted timeline arguments in this scope.

        Applies to emit, pulse, trigger, send_readoutconfig,
        send_readoutconfig_from_regs, pulse_from_regs, pulse_by_reg,
        pulse_from_length_reg, delay_timeline, sync and the repeat parent.
        The branch parent uses this default; cases shadow it. Nested
        scopes override it and restore it on exit, including exceptions.
        Explicit names win, including default. This neither forks
        nor joins and does not affect native delay/delay_auto. Call in _body;
        unknown names fail during compile, like other timeline operations.
        """
        if not self._building_body:
            raise ValueError("timeline must be called in _body")
        self.append_macro(_TimelineScope(self._timeline_book, name))
        previous = self._timeline_default
        self._timeline_default = name
        try:
            yield
        finally:
            self._timeline_default = previous
            self.append_macro(_TimelineScope(self._timeline_book, name))

    def emit(
        self, module: Module, *, timeline: str | None = None, tag: str | None = None
    ) -> None:
        """Emit a declared module once, with a fresh timing/tag occurrence.

        None selects the scoped name, or ordinary default outside a scope. Internal
        events share its current end; actual output ends plus owner post-delay
        determine the ceiled end without moving reference time. Snapshot source
        markers use module.name; nested emissions retain their own labels.
        Branch joins case ends without moving reference; ScanWith retains
        iteration reference synchronization.
        DelayAuto is native reference advance and rejects explicit timelines.
        Undeclared objects raise ValueError; owner lowering failures propagate.
        """
        if not any(module is declared for declared in self._declared_modules):
            raise ValueError(f"{module.name}: module must be declared before emit")
        timeline = timeline if timeline is not None else self._timeline_default
        start = len(self.macro_list)
        previous_scope = self._emission_scope
        self._emission_scope = (self._emission_serial, 0)
        self._emission_serial += 1
        try:
            with self.source_scope(module.name):
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

    @contextmanager
    def reference_guard(self, timeline: str) -> Generator[None]:
        """Require module lowering to preserve reference time during compilation.

        timeline names the parent placement target, validated even when empty.
        Use inside lower to protect forked children. Native timestamp advances
        and runtime timeline debits raise ValueError with the same restriction
        as branch cases. Raw instructions/custom macro translation bypass
        reference owners and remain the author's responsibility. This does not
        fork, join or reorder operations. Outside module lowering raises ValueError.
        """
        if self._emission_scope is None:
            raise ValueError("reference_guard requires an active module lower call")
        self.append_macro(ReferenceGuard(self._timeline_book, timeline, enter=True))
        try:
            yield
        finally:
            self.append_macro(
                ReferenceGuard(self._timeline_book, timeline, enter=False)
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
        selection: TableSelection | None = None,
    ) -> Generator[None]:
        """Certify framework-owned native operations that produce cycle values.

        register receives the inclusive bounds on every execution of this scope.
        dependencies names input registers whose producers must also be audited.
        dependency_ranges additionally constrains producer inputs, such as a
        table's valid indices. Its keys automatically join the audited inputs.
        selection optionally binds selected playback to a declared scalar table
        and captured index; only matching loader writers can write that relation.
        Ordinary time producers do not enable this additional relation audit.
        The same selection metadata also retains unpacked lookup values and
        the current index for exact legal-point timing of ordinary scalar loads.
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
                register,
                bounds,
                macros,
                dependencies,
                dependency_ranges if dependency_ranges is not None else {},
                selection,
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
        timeline: str | None = None,
        tag: str | None = None,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        """Add time to a timeline without advancing reference or channel timestamps.

        None selects the scoped name, or default outside a scope.
        Scalars and QickParam use us; the complete end plus t is ceiled to
        tProc cycles. In an emission, pending actual output ends are included
        before adding t as the owner post-delay. A string names an
        existing nonnegative integer-cycle register, including author-calculated
        values. range_hint optionally guarantees an inclusive (lo, hi) cycle
        range. With no known range, maxima execute at runtime; same-channel
        overlap and signed overflow are the author's responsibility. Certified
        producers and explicit hints retain finite-range checks. Hints on scalar
        times and invalid ranges raise ValueError. Wait-only targets still fail
        compilation. tag exposes static ends as 't'; runtime-only ends raise
        RuntimeError from get_time_param rather than returning a bound.
        """
        if range_hint is not None:
            if not isinstance(t, str):
                raise ValueError("range_hint requires a register time")
            _repeat_minimum(None, t, range_hint)
        selected = timeline if timeline is not None else self._timeline_default
        self.append_macro(
            TimelineAdvance(
                self._timeline_book,
                selected if selected is not None else "default",
                t,
                tag=tag,
                range_hint=range_hint,
            )
        )

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
        book = self._timeline_book
        if not book.native_only:
            if gen_ch is not None:
                book.record_native(self, gen_ch, val, generator=True)
            if ro_ch is not None:
                book.record_native(self, ro_ch, val, generator=False)
            for loop in book.loop_timelines:
                if gen_ch is not None:
                    loop.written_channels.add((True, gen_ch))
                if ro_ch is not None:
                    loop.written_channels.add((False, ro_ch))
            if book.repeats:
                frame = book.repeats[-1]
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
        self._timeline_book.require_reference_advance()
        super().decrement_timestamps(t)
        if t is not None:
            self._timeline_book.decrement(t)

    def delay_reg_auto(self, time_reg: str, *, gens=True, ros=True) -> None:
        """Advance reference by selected native ends plus register cycles.

        time_reg contains nonnegative tProc cycles. Remaining ends debit the
        actual advance using certified producer bounds. No bounds are needed
        when the selected native ends already consume every timeline. This
        operation is allowed in Repeat but forbidden in branch cases and
        reference-preserving module groups, like other reference operations.
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
    def branch(
        self, *, by: str, align_timeline: bool, timeline: str | None = None
    ) -> Generator[BranchCases]:
        """Select indexed sweep cases without advancing the shared reference.

        by names an existing sweep loop. case(index) selects its zero-based point,
        in any declaration order; omitted points execute no actions. Each case
        implicitly forks timeline, the parent (None selects the current scope,
        or default). case(index) yields its name, <by>[<index>], and shadows the
        default inside the case. Only that name and its local forks are usable;
        other timelines raise ValueError at compile. align_timeline is required:
        True joins all case ends into the parent (static or runtime pointwise
        max); False keeps the executed case end. Other timelines are unchanged.
        Both modes allow subsequent placement. Local fork names expire at case
        exit; unjoined local ends are discarded, not merged into their ancestor.
        True requires every end to be reconstructible from branch-entry state.
        Native delay/delay_auto/sync in cases raise ValueError; use delay_timeline.
        Repeat may accumulate inside cases; its body obeys the same reference
        prohibition. Branch inside Repeat is supported. Unconditional actions inside this context
        raise ValueError; put them before or after the with block.
        Duplicate, non-integer/out-of-range indices and measurement selectors
        raise ValueError. Register exhaustion propagates QICK errors unchanged.
        Call only while building _body.
        """
        if not self._building_body:
            raise ValueError("branch must be called in _body")
        selected = timeline if timeline is not None else self._timeline_default
        parent = selected if selected is not None else "default"
        cases = BranchCases(self, by)
        name = f"_timeline_branch_{self._branch_serial}"
        self._branch_serial += 1
        start = len(self.macro_list)
        self._branch_cases.append(cases)
        try:
            yield cases
        finally:
            self._branch_cases.pop()
        if any(
            not isinstance(macro, _TimelineScope) for macro in self.macro_list[start:]
        ):
            raise ValueError(
                "move unconditional branch actions before or after the with block"
            )
        self.append_macro(
            TimelineBranch(
                self._timeline_book,
                by,
                cases.cases,
                name,
                align_timeline=align_timeline,
                timeline=parent,
            )
        )

    @contextmanager
    def repeat(
        self,
        name: str,
        n: int | str,
        *,
        timeline: str | None = None,
        range_hint: tuple[int, int] | None = None,
    ) -> Generator[None]:
        """Repeat on a scoped, accumulating timeline without implicit sync.

        timeline=None selects the surrounding scope or default. Only this parent
        and local forks are accessible; local names expire at exit. Times use us.
        Explicit sync/delay/delay_auto may advance reference except in branch
        cases or reference-preserving module groups, which raise ValueError.
        A final top-level sync of the loop timeline and constant-zero entry
        select lowering without a tail register. Otherwise use a shot-local
        accumulator. A final sync with nonzero entry warns to sync before entry.
        n is a nonnegative integer or existing count register distinct from name.
        Optional range_hint guarantees a closed nonnegative integer count range,
        without runtime checks. Fixed counts forbid hints. Zero discards the body.
        Unhinted counts skip backedge bounds proofs. Invalid count/hint/name
        raises ValueError; QICK register exhaustion propagates unchanged.
        Runtime-only event getters raise RuntimeError, never return bounds.
        """
        selected = timeline if timeline is not None else self._timeline_default
        parent = selected if selected is not None else "default"
        minimum = _repeat_minimum(name, n, range_hint)
        start = len(self.macro_list)
        previous = self._timeline_default
        self._timeline_default = parent
        if n == 0:
            try:
                yield
            finally:
                self._timeline_default = previous
                del self.macro_list[start:]
            return
        entry = RepeatTimelineEnter(
            self._timeline_book, parent, name if isinstance(n, str) else None
        )
        self.append_macro(entry)
        fixed = FixedRepeatLoop(name, n) if isinstance(n, int) else None
        if fixed is None:
            self.open_inner_loop(name, name, n, range_hint=range_hint)
        else:
            self.append_macro(fixed)
        self.append_macro(RepeatTimelineBody(entry))
        body_start = len(self.macro_list)
        try:
            yield
        finally:
            self._timeline_default = previous
            body = self.macro_list[body_start:]
            entry.tail_sync = (
                bool(body)
                and type(body[-1]) is TimelineSync
                and body[-1].timeline == parent
            )
            entry.accumulator.simple = all(
                isinstance(macro, (TimelinePlacedPulse, TimelineAdvance))
                and macro.timeline == parent
                for macro in body
            )
            self.append_macro(
                RepeatTimelineExit(
                    entry,
                    n,
                    minimum,
                    range_hint[1]
                    if range_hint is not None
                    else (n if isinstance(n, int) else None),
                    name,
                )
            )
            if isinstance(n, str):
                self.close_inner_loop(name, name, n)
            elif fixed is not None:
                self.append_macro(fixed.closing)
            self.append_macro(RepeatTimelineCapture(entry))

    def join(self, *sources: str, into: str) -> None:
        """Merge existing sources and target into target's latest end (us).

        Sources remain unchanged. Wait-only sources contribute actual ends;
        the target becomes ordinary and permits placement. Crossing maxima use
        runtime arithmetic without moving reference or emitting output. Unknown
        names raise ValueError; declare names with fork before joining them.
        """
        self.append_macro(TimelineJoin(self._timeline_book, sources, into))

    def sync(self, *, timeline: str | None = None) -> None:
        """Move reference to the chosen end and floor every rebased end at zero.

        None selects the scoped name, or default outside a scope.
        Only default exists initially; other names require fork in _body.
        Fully consumed wait-only bounds become ordinary zero ends again.
        Unknown names or unprovable swept floors raise ValueError during compile.
        """
        selected = timeline if timeline is not None else self._timeline_default
        self.append_macro(
            TimelineSync(
                self._timeline_book, selected if selected is not None else "default"
            )
        )

    def send_readoutconfig(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Send an instantaneous config at native t or timeline end+t us.

        None uses the scoped name, or native reference timing outside a scope.
        Native args are t=0, tag=None. A timeline end becomes the scheduled config
        instant; config length and unrelated ADC windows do not contribute.
        Wait-only timelines reject placement even at t=0; first sync the bound
        or establish an ordinary scheduling point with join/fork.
        Declaration and timing errors propagate.
        """
        timeline = timeline if timeline is not None else self._timeline_default
        super().send_readoutconfig(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self._pop_native(ConfigReadout)
            self.append_macro(
                TimelinePlacedConfig(self._timeline_book, timeline, source)
            )

    def pulse_by_reg(
        self,
        ch: int,
        addr_reg: str,
        t: float | QickParam = 0.0,
        *,
        flat_top_pulse: bool = False,
        timeline: str | None = None,
        selection: SelectedPlayback | None = None,
    ) -> None:
        """Play native addresses, or place certified selected playback on timeline.

        None uses the scoped name, or native reference timing outside a scope.
        Native arguments retain their meanings. Timeline placement requires
        framework selection metadata pairing the address producer and candidates.
        Selector bounds, writes and overlap are checked during compilation.
        """
        timeline = timeline if timeline is not None else self._timeline_default
        if timeline is None:
            super().pulse_by_reg(ch, addr_reg, t, flat_top_pulse=flat_top_pulse)
            return
        if selection is None or selection.register != addr_reg:
            raise ValueError(
                "timeline address playback requires matching selection metadata"
            )
        from .macro.pluse_reg import PulseByReg

        self.append_macro(
            PulseByReg.from_base(self, ch, addr_reg, t, flat_top_pulse=flat_top_pulse)
        )
        source = self._pop_native(PulseByReg)
        self.append_macro(
            TimelineSelectedPulse(self._timeline_book, timeline, source, selection)
        )

    @override
    def pulse_from_length_reg(
        self,
        ch: int,
        name: str,
        *,
        length_reg: str,
        flat_top: bool,
        t: float | QickParam = 0.0,
        timeline: str | None = None,
        selection: SelectedPlayback | None = None,
    ) -> None:
        """Play a native length register or certified timeline-selected lengths.

        None uses the scoped name, or native reference timing outside a scope.
        Native arguments are unchanged. Timeline placement requires matching
        framework selection metadata; raw waveform length and post stay separate.
        Invalid producer bounds, writes and overlap fail during compilation.
        """
        timeline = timeline if timeline is not None else self._timeline_default
        super().pulse_from_length_reg(
            ch, name, length_reg=length_reg, flat_top=flat_top, t=t
        )
        if timeline is not None:
            if selection is None or selection.register != length_reg:
                raise ValueError(
                    "timeline length playback requires matching selection metadata"
                )
            from .macro.wmem import PulseFromLengthReg

            source = self._pop_native(PulseFromLengthReg)
            self.append_macro(
                TimelineSelectedPulse(self._timeline_book, timeline, source, selection)
            )

    def pulse_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Play runtime words at native t or timeline end+t, advancing a named end.

        None uses the scoped name, or native reference timing outside a scope.
        kwargs are t (us), tag, freq_reg and gain_reg containing final hardware
        words. Native patching and template/register validation are unchanged.
        """
        timeline = timeline if timeline is not None else self._timeline_default
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

        None uses the scoped name, or native reference timing outside a scope.
        kwargs are t, tag and freq_reg. A timeline end becomes the scheduled config
        instant, independent of config length. Wait-only timelines reject placement
        even at t=0; first sync the bound or establish an ordinary scheduling
        point with join/fork. Native patching and declaration/register validation
        are unchanged.
        """
        timeline = timeline if timeline is not None else self._timeline_default
        super().send_readoutconfig_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self._pop_native(ConfigReadout)
            self.append_macro(
                TimelinePlacedConfig(self._timeline_book, timeline, source)
            )

    def trigger(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Trigger ADC/tt/pins at native t or timeline end+t and advance named end.

        None uses the scoped name, or native reference timing outside a scope.

        Native args select ros/tts/pins, t, width (us), ddr4/mr and tag. Named end
        includes ADC integration and trigger width, rounded up to tProc cycles.
        Named t=None raises ValueError at entry; native special values are unchanged.
        """
        timeline = timeline if timeline is not None else self._timeline_default
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

        None uses the scoped name, or native reference timing outside a scope.

        Native args are t=0 (us or 'auto'), tag=None. Named 'auto' takes the
        provably later timeline/channel end. Named end includes waveform length,
        rounded up to tProc cycles. Crossing swept maxima raise ValueError with
        a delay_timeline hint; native declaration errors propagate.
        """
        timeline = timeline if timeline is not None else self._timeline_default
        super().pulse(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self._pop_native(Pulse)
            self.append_macro(
                TimelinePlacedPulse(self._timeline_book, timeline, source)
            )
