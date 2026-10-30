"""Builder recording and native-program compilation share a public contract.

The handwritten parity oracle remains until TimelineProgramV2 author hooks are
retired; changes to the common compiler must preserve parity, not old ASM bytes.
"""

from collections.abc import Callable

import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.program import v2
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _readout(name="read", *, gain: float | QickParam = 0.5):
    return PulseReadout(
        name,
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            ro_ch=0,
            freq=100,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.2,
            post_delay=0.05,
        ),
    )


def test_builder_sweep_reused_readout_and_timing_match_native(mock_soccfg: QickConfig):
    builder = v2.TimelineBuilder(
        mock_soccfg, reps=2, initial_delay=0.1, relax_delay=0.3
    )
    builder.add_loop("outer", 2)
    gain = builder.add_sweep(
        "gain", v2.SweepCfg(start=0.2, stop=0.4, step=0.1, expts=3)
    )
    readout = _readout(gain=gain)
    builder.emit(readout, tag="first")
    builder.delay_timeline(0.05)
    builder.sync()
    builder.delay(0.1, tag="gap")
    builder.emit(readout, tag="second")
    builder.delay_auto(0.1, gens=False, ros=True, tag="wait")
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("outer", 2)
            self.add_loop("gain", 3)
            self.readout = _readout(gain=QickSweep1D("gain", 0.2, 0.4))
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout, tag="first")
            self.delay_timeline(0.05)
            self.sync()
            self.delay(0.1, tag="gap")
            self.emit(self.readout, tag="second")
            self.delay_auto(0.1, gens=False, ros=True, tag="wait")

    expected = Native(mock_soccfg, reps=2, initial_delay=0.1, relax_delay=0.3)
    assert actual.prog_list == expected.prog_list
    assert actual.sweep_loops == expected.sweep_loops
    for tag in ("first", "second", "gap", "wait"):
        assert actual.get_time_param(tag, "t") == expected.get_time_param(tag, "t")
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_native_add_sweep_declares_axis_and_returns_uniform_parameter(mock_soccfg):
    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            gain = self.add_sweep(
                "gain", v2.SweepCfg(start=0.2, stop=0.4, step=0.1, expts=3)
            )
            self.readout = _readout(gain=gain)
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout)

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("gain", 3)
            self.readout = _readout(gain=QickSweep1D("gain", 0.2, 0.4))
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout)

    actual = Native(mock_soccfg)
    assert actual.prog_list == Explicit(mock_soccfg).prog_list
    assert [(axis.name, axis.count) for axis in actual.sweep_loops] == [
        ("reps", 1),
        ("gain", 3),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "build",
        "loop",
        "sweep",
        "emit",
        "parallel",
        "pulse",
        "trigger",
        "delay",
        "auto",
        "end",
        "sync",
    ],
)
def test_builder_freezes_after_build(mock_soccfg, operation):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.build()
    actions = {
        "build": builder.build,
        "loop": lambda: builder.add_loop("later", 2),
        "sweep": lambda: builder.add_sweep(
            "later", v2.SweepCfg(start=0, stop=1, step=1, expts=2)
        ),
        "emit": lambda: builder.emit(_readout()),
        "parallel": lambda: builder.emit_parallel([[_readout()]], timeline="default"),
        "pulse": lambda: builder.pulse(0, "later"),
        "trigger": builder.trigger,
        "delay": lambda: builder.delay(0.1),
        "auto": builder.delay_auto,
        "end": lambda: builder.delay_timeline(0.1),
        "sync": builder.sync,
    }
    with pytest.raises(ValueError, match="(frozen|built)"):
        actions[operation]()


@pytest.mark.parametrize("first", ["loop", "sweep"])
@pytest.mark.parametrize("second", ["loop", "sweep"])
def test_duplicate_builder_axes_raise_at_recording(mock_soccfg, first, second):
    builder = v2.TimelineBuilder(mock_soccfg)
    spec = v2.SweepCfg(start=0, stop=1, step=1, expts=2)
    actions = {
        "loop": lambda: builder.add_loop("scan", 2),
        "sweep": lambda: builder.add_sweep("scan", spec),
    }
    actions[first]()
    with pytest.raises(ValueError, match="scan"):
        actions[second]()
    # Rejection leaves only the accepted axis.
    assert [(axis.name, axis.count) for axis in builder.build().sweep_loops] == [
        ("reps", 1),
        ("scan", 2),
    ]


@pytest.mark.parametrize("resource", ["adc", "register", "register address"])
@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_conflicting_module_objects_name_both_owners(mock_soccfg, resource, frontend):
    if resource == "adc":
        first, second = _readout("first"), _readout("second")
    elif resource == "register address":
        first = DeclarationModule("first", lambda p: p.add_reg("a", addr=10))
        second = DeclarationModule("second", lambda p: p.add_reg("b", addr=10))
    else:
        first = v2.LoadValue("first", [], idx_reg="index", val_reg="shared")
        second = v2.LoadValue("second", [], idx_reg="index", val_reg="shared")

    def compile_program():
        if frontend == "builder":
            builder = v2.TimelineBuilder(mock_soccfg)
            builder.emit(first)
            builder.emit(second)
            return builder.build()

        class Native(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare(first)
                self.declare(second)

        return Native(mock_soccfg)

    with pytest.raises(ValueError, match="(readout|register)") as error:
        compile_program()
    for module in (first, second):
        assert module.name in str(error.value)
        assert repr(module) in str(error.value)


def _pulse(name, ch=1):
    return v2.Pulse(
        name,
        v2.PulseCfg(
            ch=ch,
            nqz=1,
            freq=100,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.1),
        ),
    )


def test_native_parallel_body_allocates_its_own_emission_scope(mock_soccfg):
    readout, pulse = _readout(), _pulse("drive")

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)
            self.declare(pulse)

        def _body(self, cfg):
            self.emit_parallel([[readout], [pulse]], timeline="default", tag="one")
            self.emit_parallel([[readout], [pulse]], timeline="default", tag="two")

    actual = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    result = ir.analyze_timing(
        ir.CompiledTimingInput(actual.prog_list, actual.labels, actual.timing_layout)
    )
    assert [
        event.cycle for event in result.shots[0].events if event.kind == "wave"
    ] == [
        # ceil(61 ADC clocks / 307.2 * 430.08 + .05 * 430.08) = 107.
        0,
        0,
        0,
        107,
        107,
        107,
    ]
    expected = actual.prog_list.copy()
    actual.compile()
    assert actual.prog_list == expected


def test_builder_parallel_and_native_outputs_match_native(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    readout = _readout()
    pulse = v2.Pulse(
        "drive",
        v2.PulseCfg(
            ch=1,
            nqz=1,
            freq=100,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.1),
        ),
        pulse_id="native_drive",
    )
    builder.emit_parallel([[readout], [pulse]], timeline="default", tag="parallel")
    builder.sync()
    builder.pulse(1, "native_drive", t=0.1, timeline="default", tag="pulse")
    builder.trigger(
        ros=[0],
        tts=[],
        pins=[],
        t=0.02,
        width=0.05,
        ddr4=False,
        mr=False,
        tag="trigger",
        timeline="default",
    )
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)
            self.declare(pulse)

        def _body(self, cfg):
            self.emit_parallel([[readout], [pulse]], timeline="default", tag="parallel")
            self.sync()
            self.pulse(1, "native_drive", t=0.1, timeline="default", tag="pulse")
            self.trigger(
                ros=[0],
                tts=[],
                pins=[],
                t=0.02,
                width=0.05,
                ddr4=False,
                mr=False,
                tag="trigger",
                timeline="default",
            )

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    for tag in ("pulse", "trigger"):
        assert actual.get_time_param(tag, "t") == expected.get_time_param(tag, "t")


def test_builder_preserves_borrowed_selector_and_shared_pulse_templates(mock_soccfg):
    cfg = v2.PulseCfg(
        ch=1,
        nqz=1,
        freq=100,
        gain=0.5,
        waveform=ConstWaveformCfg(length=0.1),
    )

    def modules():
        loader = v2.LoadValue("selector", [0, 1], idx_reg="scan", val_reg="selected")
        computed = v2.ComputedPulse(
            "chosen",
            val_reg="selected",
            pulses=[cfg, cfg.model_copy(update={"gain": 0.25})],
        )
        # Distinct objects intentionally share generator and template definitions.
        return [loader, computed, v2.Pulse("first", cfg), v2.Pulse("second", cfg)]

    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 2)
    for module in modules():
        builder.emit(module)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.modules = modules()
            for module in self.modules:
                self.declare(module)

        def _body(self, cfg):
            for module in self.modules:
                self.emit(module)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    np.testing.assert_array_equal(actual.compile_datamem(), expected.compile_datamem())


@pytest.mark.parametrize("method", ["delay_auto", "trigger"])
def test_builder_rejects_positional_boolean_controls(mock_soccfg, method):
    builder = v2.TimelineBuilder(mock_soccfg)
    # Dynamic caller intentionally exercises invalid call shapes.
    operation: Callable[..., None]
    if method == "delay_auto":
        operation = builder.delay_auto
        args = (0, False)
    else:
        operation = builder.trigger
        args = (None, None, None, 0, None, True)
    with pytest.raises(TypeError, match="positional"):
        operation(*args)


class DeclarationModule(v2.Module):
    """Synthetic owner declaring resources without emitting output."""

    def __init__(self, name, declare: Callable[[v2.TimelineProgramV2], object]):
        self.name = name
        self.declaration = declare

    def init(self, prog):
        raise NotImplementedError("native-only fixture")

    def run(self, prog, t=0.0):
        raise NotImplementedError("native-only fixture")

    def declare(self, prog):
        self.declaration(prog)

    def lower(self, prog, *, timeline, tag):
        pass


def test_protected_generator_conflict_names_protector_not_earlier_shared_user(
    mock_soccfg,
):
    shared = DeclarationModule("shared", lambda p: p.declare_gen(0, nqz=1))

    def protect(prog):
        prog.declare_gen(0, nqz=1)
        prog.declare_readout(0, length=0.1)
        prog.protect_readout_routes("protector", 0, [0])

    protector = DeclarationModule("protector", protect)
    conflicting = DeclarationModule("conflicting", lambda p: p.declare_gen(0, nqz=1))
    builder = v2.TimelineBuilder(mock_soccfg)
    for module in (shared, protector, conflicting):
        builder.emit(module)
    with pytest.raises(ValueError, match="generator") as error:
        builder.build()
    for module in (protector, conflicting):
        assert module.name in str(error.value)
        assert repr(module) in str(error.value)


def test_explicit_scratch_register_reuse_remains_valid(mock_soccfg):
    owner = DeclarationModule("owner", lambda p: p.add_reg("scratch"))
    borrower = DeclarationModule(
        "borrower", lambda p: p.add_reg("scratch", allow_reuse=True)
    )
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.emit(owner)
    builder.emit(borrower)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(owner)
            self.declare(borrower)

    assert actual.prog_list == Native(mock_soccfg).prog_list


def test_failed_build_also_freezes_builder(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.emit(_readout("first"))
    builder.emit(_readout("second"))
    with pytest.raises(ValueError, match="readout"):
        builder.build()
    with pytest.raises(ValueError, match="frozen"):
        builder.emit(_readout("later"))


def test_builder_rejects_unknown_trigger_keywords_at_recording(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg)
    operation: Callable[..., None] = builder.trigger
    with pytest.raises(TypeError, match="unexpected.*typo"):
        operation(typo=True)
