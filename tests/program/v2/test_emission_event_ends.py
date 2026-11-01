"""Actual waveform/ADC ends govern module emission at the public compiler seam."""

from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import lru_cache
from math import ceil

import numpy as np
import pytest
from qick.asm_v2 import Macro, QickParam, QickSpan, QickSweep1D
from zcu_tools.program.v2.modules import (
    ComputedPulse,
    LoadValue,
    Module,
    Pulse,
    PulseCfg,
)
from zcu_tools.program.v2.modules.delay import Delay, SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import GEN_MHZ, RO_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import (
    generator_duration,
    native_sweep,
    readout_duration,
    runtime_cycles,
    runtime_weight,
    tproc_ceil,
)
from tests.program.v2.timing_support import shots, waves

PROFILE = load_soccfg(3, 1)
RELAX = int(PROFILE.us2cycles(0.02))
PRE = tproc_ceil(PROFILE, 0.01)
TAIL = tproc_ceil(PROFILE, generator_duration(PROFILE, 0.101))


def _pulse_cfg(
    ch: int = 0,
    *,
    length: float | QickParam = 0.101,
    pre: float | QickParam = 0.01,
    post: float | QickParam = 0.0005,
) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=1000,
        gain=0.5,
        waveform=ConstWaveformCfg(length=length),
        pre_delay=pre,
        post_delay=post,
    )


def _program(
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
    *,
    fabric_clock: float = GEN_MHZ,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    profile = load_soccfg(3, 1)
    for generator in profile["gens"]:
        generator["f_fabric"] = fabric_clock
    return Program(profile, reps=2, initial_delay=0, relax_delay=0.02)


@pytest.mark.parametrize("post", [0.0005, 0.001])
@pytest.mark.parametrize("timeline", [None, "drive"])
def test_pulse_ceil_covers_actual_playback_plus_post_once(post, timeline):
    pulse = Pulse("drive", _pulse_cfg(post=post))

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(pulse, timeline=timeline, tag="first")
        prog.emit(pulse, timeline=timeline, tag="second")

    prog = _program([pulse], body)
    # Preserve the fractional waveform tail until the single post is added.
    spacing = PRE + tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 0.101) + Fraction(str(post))
    )
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[PRE, spacing + PRE]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing + RELAX
    ] * 2
    assert prog.get_time_param("second", "t") * TPROC_MHZ == pytest.approx(
        spacing + PRE
    )
    prog.compile()
    assert shots(prog) == result


@pytest.mark.parametrize("origin", [0, 40])
@pytest.mark.parametrize("swept_trigger", [False, True])
@pytest.mark.parametrize("offset, post", [(0.0013, 0.0005), (0.0011, 0.0012)])
def test_readout_trigger_quantization_does_not_extend_generator_end(
    origin, swept_trigger, offset, post
):
    trigger = (
        QickSweep1D("scan", offset, offset + 2 / TPROC_MHZ) if swept_trigger else offset
    )
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=10 / GEN_MHZ),
            pre_delay=0,
            post_delay=post,
            ro_ch=0,
            ro_length=13 / RO_MHZ,
            trig_offset=trigger,
        ),
    )

    def initialize(prog):
        prog.append_macro(_AllocateDuration())
        if swept_trigger:
            prog.add_loop("scan", 3)

    def body(prog):
        if origin:
            prog.write_reg("duration", origin)
            prog.timeline_inc("duration")
        prog.emit(readout)

    prog = _program([readout], body, initialize)
    result = shots(prog)
    # Scalar triggers ceil both .559104 and .473088 to 1. Author
    # sweeps start at native round(.559104/.473088), then step by one.
    # ADC length is18.2 cycles; generator is11.2 cycles plus post.
    steps = [0, 1, 2] if swept_trigger else [0]
    trigger_start = (
        int(prog.us2cycles(offset))
        if swept_trigger
        else tproc_ceil(prog.soccfg, offset)
    )
    generator_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 10 / GEN_MHZ) + Fraction(str(post))
    )
    adc_cycles = readout_duration(prog.soccfg, 13 / RO_MHZ) * Fraction(str(TPROC_MHZ))
    adc_shell = ceil(adc_cycles) if origin else round(adc_cycles)
    width = int(prog.us2cycles(0.02))
    assert [
        [
            (event.cycle - shot.start_cycle, event.value)
            for event in shot.events
            if event.kind == "data"
        ]
        for shot in result
    ] == [
        [(origin + trigger_start + step, 1), (origin + trigger_start + step + width, 0)]
        for step in steps
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        origin + generator_end for _ in steps
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        origin + trigger_start + adc_shell + step + RELAX for step in steps
    ] * 2
    # Native final reference can round a fractional ADC tail down. The physical
    # WAIT remains ceiled and blocks until integration completes at every point.
    (wait,) = [inst for inst in prog.prog_list if inst["CMD"] == "WAIT"]
    wait_cycles = int(str(wait["TIME"]).removeprefix("@"))
    assert all(
        generator_end + wait_cycles >= trigger_start + step + ceil(adc_cycles)
        for step in steps
    )


class _AllocateDuration(Macro):
    """Declare a native QICK register at preprocessing, before its writes."""

    def preprocess(self, prog):
        prog.add_reg("duration")

    def translate(self, prog):
        pass


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
def test_fractional_delay_from_register_origin_rounds_up_without_reference_advance(
    delay_type,
):
    delay = delay_type("gap", 0.0005)

    def body(prog):
        prog.write_reg("duration", 40)
        prog.timeline_inc("duration")
        prog.emit(delay)

    result = shots(
        _program([delay], body, lambda prog: prog.append_macro(_AllocateDuration()))
    )
    end = 40 + tproc_ceil(PROFILE, 0.0005)
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [end] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [end + RELAX] * 2


@pytest.mark.parametrize("swept_post", [False, True])
@pytest.mark.parametrize("kind", ["pulse", "readout"])
def test_runtime_origin_keeps_fractional_event_tail_until_post_is_added(
    kind, swept_post
):
    post = QickSweep1D("scan", 0.0005, 0.0005 + 4 / TPROC_MHZ) if swept_post else 0.0005
    if kind == "pulse":
        owner = Pulse("drive", _pulse_cfg(post=post))
    else:
        owner = PulseReadout(
            "read",
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                freq=1000,
                gain=0.5,
                waveform=ConstWaveformCfg(length=0.051),
                pre_delay=0.01,
                post_delay=post,
                ro_ch=0,
                ro_length=0.1,
                trig_offset=0.01,
            ),
        )

    def initialize(prog):
        prog.append_macro(_AllocateDuration())
        if swept_post:
            prog.add_loop("scan", 3)

    def body(prog):
        prog.write_reg("duration", 40)
        prog.timeline_inc("duration")
        prog.emit(owner)

    result = shots(_program([owner], body, initialize))
    # Generator tail is43.68 for pulse,22.4 for readout, never the ADC tail.
    # Native post sweep starts at zero and adds0/2/4 cycles.
    base = (
        40
        + PRE
        + tproc_ceil(
            PROFILE,
            generator_duration(PROFILE, 0.101 if kind == "pulse" else 0.051)
            + (Fraction(0) if swept_post else Fraction("0.0005")),
        )
    )
    ends = [base, base + 2, base + 4] if swept_post else [base]
    assert [waves(shot) for shot in result] == (
        [[40, 40 + PRE]] if kind == "readout" else [[40 + PRE]]
    ) * (2 * len(ends))
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == ends * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        (
            end
            if kind == "pulse"
            else max(
                end, 40 + PRE + tproc_ceil(PROFILE, readout_duration(PROFILE, 0.1))
            )
        )
        + RELAX
        for end in ends
    ] * 2
    if kind == "readout":
        assert [
            [
                (event.cycle - shot.start_cycle, event.value)
                for event in shot.events
                if event.kind == "data"
            ]
            for shot in result
        ] == [[(40 + PRE, 1), (40 + PRE + int(PROFILE.us2cycles(0.02)), 0)]] * (
            2 * len(ends)
        )


@pytest.mark.parametrize("swept_post", [False, True])
def test_affine_waveform_and_post_sweeps_ceil_complete_end(swept_post):
    post = QickSweep1D("scan", 0.0005, 0.0005 + 4 / TPROC_MHZ) if swept_post else 0.0005
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / GEN_MHZ, 89 / GEN_MHZ), post=post),
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program([pulse], body, lambda prog: prog.add_loop("scan", 3))
    # Registered lengths39,64,89 generator clocks =>43.68,71.68,99.68.
    # Start ceils to5. Scalar post is .21504; native swept post starts0.
    # Both produce ceil(5+43.68+post)=49 at the first point.
    posts = (
        [0, 2, 4] if swept_post else [Fraction("0.0005") * Fraction(str(TPROC_MHZ))] * 3
    )
    ends = [
        PRE + ceil(runtime_cycles(PROFILE, n) + p)
        for n, p in zip((39, 64, 89), posts, strict=True)
    ]
    result = shots(prog)
    assert prog.compile_datamem() is None
    assert [waves(shot) for shot in result] == [[PRE, end + PRE] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + RELAX for end in ends
    ] * 2


@pytest.mark.parametrize("post_ticks, extra", [(5e-11, 0), (2e-10, 1)])
def test_scalar_static_cross_grid_ceil_applies_noise_tolerance(post_ticks, extra):
    ratio = Fraction(str(TPROC_MHZ)) / Fraction(str(GEN_MHZ))
    length = ratio.denominator / GEN_MHZ
    end = ratio.numerator + extra
    pulse = Pulse(
        "precise", _pulse_cfg(length=length, pre=0, post=post_ticks / TPROC_MHZ)
    )
    prog = _program([pulse], lambda p: p.emit(pulse))
    # Exact 28-tick playback plus post: only the tail outside tolerance adds a tick.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [end] * 2


def test_runtime_cross_grid_length_sweep_ceil_without_implicit_time_table():
    pulse = Pulse(
        "cross_grid",
        _pulse_cfg(
            length=QickSweep1D("scan", 39 / GEN_MHZ, 115 / GEN_MHZ), pre=0, post=0
        ),
    )
    tail = Pulse("tail", _pulse_cfg(1, length=10 / GEN_MHZ, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda p: p.add_loop("scan", 3))
    assert [waves(shot) for shot in shots(prog)] == [
        [0, ceil(runtime_cycles(prog.soccfg, n))] for n in (39, 77, 115)
    ] * 2
    assert prog.compile_datamem() is None
    assert not any(i["CMD"] in {"ARITH", "DIV"} for i in prog.prog_list)


@lru_cache(maxsize=None)
def _signed_power_cost(value: int) -> int:
    if value <= 1:
        return value
    if value % 2 == 0:
        return _signed_power_cost(value // 2)
    return 1 + min(_signed_power_cost(value // 2), _signed_power_cost(value // 2 + 1))


@pytest.mark.parametrize("clock", [GEN_MHZ, RO_MHZ, 38.4, TPROC_MHZ])
def test_runtime_conversion_uses_minimum_csd_upper_weight_with_500us_budget(clock):
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 256 / clock, 258 / clock), pre=0, post=0),
    )
    tail = Pulse("tail", _pulse_cfg(1, length=3 / clock, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program(
        [pulse, tail], body, lambda p: p.add_loop("scan", 3), fabric_clock=clock
    )
    starts = [waves(shot)[1] for shot in shots(prog)]
    weight = runtime_weight(TPROC_MHZ, clock)
    assert starts == [(n * weight + 255) // 256 for n in (256, 257, 258)] * 2
    observed = starts[0]  # At n=256 the public end reveals the multiplier exactly.
    ratio = Fraction(str(TPROC_MHZ)) / Fraction(str(clock))
    counts = np.arange(int(500 * clock) + 1, dtype=np.int64)
    physical_numerator = counts * ratio.numerator
    scheduled = (counts * observed + 255) // 256
    error_numerator = scheduled * ratio.denominator - physical_numerator
    assert np.all(error_numerator >= 0)
    assert np.all(error_numerator <= TPROC_MHZ * ratio.denominator)
    admissible = []
    for candidate in range(
        ceil(256 * ratio),
        int(256 * (ratio + Fraction(str(TPROC_MHZ)) / int(500 * clock))) + 2,
    ):
        error = (
            (counts * candidate + 255) // 256
        ) * ratio.denominator - physical_numerator
        if np.all(error >= 0) and np.all(error <= TPROC_MHZ * ratio.denominator):
            admissible.append(candidate)
    assert observed == min(admissible, key=lambda w: (_signed_power_cost(w), w))
    shifts = [
        op
        for inst in prog.prog_list
        if isinstance(op := inst.get("OP"), str)
        and any(shift in op for shift in ("SL", "SR", "ASR"))
    ]
    assert all(int(op.split("#")[-1]) <= 15 for op in shifts)
    if clock == TPROC_MHZ:
        assert shifts == []


def test_runtime_length_sweep_keeps_correlated_nonnegative_product_bounds():
    pulse = Pulse(
        "wide",
        _pulse_cfg(
            length=QickSweep1D("scan", 3 / GEN_MHZ, 65535 / GEN_MHZ), pre=0, post=0
        ),
    )
    tail = Pulse("tail", _pulse_cfg(1, length=3 / GEN_MHZ, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda p: p.add_loop("scan", 3))
    assert [waves(shot)[1] for shot in shots(prog)] == [
        ceil(runtime_cycles(prog.soccfg, n)) for n in (3, 32769, 65535)
    ] * 2


@pytest.mark.parametrize("align", [False, True])
def test_branch_merge_preserves_runtime_cross_grid_length_producer(align):
    pulse = Pulse(
        "drive",
        _pulse_cfg(
            length=QickSweep1D("scan", 39 / GEN_MHZ, 77 / GEN_MHZ), pre=0, post=0
        ),
    )
    other = Pulse("other", _pulse_cfg(length=35 / GEN_MHZ, pre=0, post=0))
    tail = Pulse("tail", _pulse_cfg(1, length=3 / GEN_MHZ, pre=0, post=0))

    def body(prog):
        with prog.branch(by="scan", align_timeline=align) as cases:
            with cases.case(0):
                prog.emit(pulse)
            with cases.case(1):
                prog.emit(other)
        prog.emit(tail)

    prog = _program([pulse, other, tail], body, lambda p: p.add_loop("scan", 2))
    ends = (
        [ceil(runtime_cycles(prog.soccfg, n)) for n in (39, 77)]
        if align
        else [
            ceil(runtime_cycles(prog.soccfg, 39)),
            tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 35 / GEN_MHZ)),
        ]
    )
    assert [waves(shot) for shot in shots(prog)] == [[0, end] for end in ends] * 2
    assert prog.compile_datamem() is None


def test_runtime_cross_grid_product_overflow_fails_at_build():
    # Native waveform words fit 16 bits, and the final end fits signed32.
    # Only the required fixed-point product overflows, before ASR8.
    pulse = Pulse(
        "large", _pulse_cfg(length=QickSweep1D("scan", 25000, 25002), pre=0, post=0)
    )
    with pytest.raises(ValueError, match="overflow signed 32-bit"):
        _program(
            [pulse],
            lambda p: p.emit(pulse),
            lambda p: p.add_loop("scan", 3),
            fabric_clock=1,
        )


def test_selected_waveform_with_native_length_sweep_tracks_selected_end():
    selector = LoadValue("selector", [0, 1], "choice", "selected")
    playback = ComputedPulse(
        "chosen",
        val_reg="selected",
        pulses=[
            _pulse_cfg(
                length=QickSweep1D("scan", n / GEN_MHZ, (n + 76) / GEN_MHZ),
                pre=0,
                post=0,
            )
            for n in (39, 40)
        ],
    )
    tail = Pulse("tail", _pulse_cfg(1, length=10 / GEN_MHZ, pre=0, post=0))

    def initialize(prog):
        prog.add_loop("choice", 2)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.emit(selector)
        prog.emit(playback)
        prog.emit(tail)

    prog = _program([selector, playback, tail], body, initialize)
    # Candidate-specific fabric lengths use W=287, never longest-candidate padding.
    assert [waves(shot) for shot in shots(prog)] == [
        [0, ceil(runtime_cycles(prog.soccfg, n))] for n in (39, 77, 115, 40, 78, 116)
    ] * 2
    # Two selector words and two author-declared waveform addresses, no time table.
    data = prog.compile_datamem()
    assert data is not None and len(data) == 4


def test_nonaffine_sweep_ceil_uses_exact_registered_playback_at_each_point():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 0.101, 0.301), pre=0, post=0),
        pulse_id="drive",
    )
    tail = Pulse("tail", _pulse_cfg(1, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda prog: prog.add_loop("scan", 3))
    # QICK resolves 39,77,115 generator clocks. At clock ratio28/25,
    # W=287 gives ceil(n*287/256)=44,87,129, not affine44,87,130.
    assert prog.get_pulse_param("drive", "total_length", as_array=True) * GEN_MHZ == (
        pytest.approx(native_sweep(prog.soccfg, 0.101, 0.301, 3, gen_ch=0))
    )
    result = shots(prog)
    ends = [
        ceil(runtime_cycles(prog.soccfg, n))
        for n in native_sweep(prog.soccfg, 0.101, 0.301, 3, gen_ch=0)
    ]
    assert [waves(shot) for shot in result] == [[0, end] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL + RELAX for end in ends
    ] * 2
    assert prog.compile_datamem() is None
    prog.compile()
    assert shots(prog) == result


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("swept_post", [False, True])
def test_nonaffine_post_and_repeated_emissions_keep_exact_same_channel_spacing(
    descending, swept_post
):
    start, stop = (
        (115 / GEN_MHZ, 39 / GEN_MHZ) if descending else (39 / GEN_MHZ, 115 / GEN_MHZ)
    )
    post = (
        QickSweep1D("scan", 0.0002, 0.0002 + 0.1 / TPROC_MHZ) if swept_post else 0.0002
    )
    pulse = Pulse(
        "drive", _pulse_cfg(length=QickSweep1D("scan", start, stop), pre=0, post=post)
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program([pulse], body, lambda prog: prog.add_loop("scan", 3))
    # W=287 gives tails43.72265625/86.32421875/128.92578125.
    # Scalar post .086016 crosses 129; native swept post truncates to zero.
    posts = (
        native_sweep(prog.soccfg, 0.0002, 0.0002 + 0.1 / TPROC_MHZ, 3)
        if swept_post
        else [Fraction("0.0002") * Fraction(str(TPROC_MHZ))] * 3
    )
    ends = [
        ceil(runtime_cycles(prog.soccfg, n) + p)
        for n, p in zip(
            native_sweep(prog.soccfg, start, stop, 3, gen_ch=0), posts, strict=True
        )
    ]
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[0, end] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + RELAX for end in ends
    ] * 2


def test_multi_axis_nonaffine_sweep_reenters_nested_repeat_at_the_same_point():
    pulse = Pulse(
        "drive",
        _pulse_cfg(
            length=39 / GEN_MHZ
            + QickSpan("scan", 76 / GEN_MHZ)
            + QickSpan("fine", 2 / GEN_MHZ),
            pre=0,
            post=0,
        ),
    )

    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_loop("fine", 3)

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("outer", 2):
            prog.delay_by_timeline()
            with prog.repeat("inner", 2):
                prog.emit(pulse)
                prog.delay_by_timeline()
            prog.delay_by_timeline()

    prog = _program([pulse], body, initialize)
    # Generator grids39/40/41,77/78/79,115/116/117 at ratio28/25.
    # Each native sweep point executes four k=8 local bodies, not four
    # advances of a shared sequential dmem cursor.
    ends = [
        ceil(runtime_cycles(prog.soccfg, n + fine))
        for n in (39, 77, 115)
        for fine in (0, 1, 2)
    ]
    result = shots(prog)
    assert [waves(shot) for shot in result] == [
        [0, end, 2 * end, 3 * end] for end in ends
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        4 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        4 * end + RELAX for end in ends
    ] * 2


@pytest.mark.parametrize("swept_post", [False, True])
@pytest.mark.parametrize("length", [0.051, 0.201])
def test_readout_post_follows_generator_without_waiting_for_adc(length, swept_post):
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=length),
            pre_delay=0.01,
            post_delay=QickSweep1D("scan", 0.0005, 0.0005 + 4 / TPROC_MHZ)
            if swept_post
            else 0.0005,
            ro_ch=0,
            ro_length=0.1,
            trig_offset=0.01,
        ),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))

    def body(prog):
        prog.emit(readout)
        prog.emit(tail)

    prog = _program(
        [readout, tail],
        body,
        (lambda prog: prog.add_loop("scan", 3)) if swept_post else None,
    )
    result = shots(prog)
    posts = [0, 2, 4] if swept_post else [Fraction("0.0005") * Fraction(str(TPROC_MHZ))]
    ends = [
        PRE
        + ceil(
            generator_duration(prog.soccfg, length) * Fraction(str(TPROC_MHZ)) + post
        )
        for post in posts
    ]
    # Generator starts5 and lasts22.4/86.24 cycles. Add scalar post.21504
    # or native swept post0 once: ceil28/92, with swept steps2.
    assert [waves(shot) for shot in result] == [
        [0, PRE, point_end] for point_end in ends
    ] * 2
    assert [
        [
            (event.cycle - shot.start_cycle, event.value)
            for event in shot.events
            if event.kind == "data"
        ]
        for shot in result
    ] == [[(PRE, 1), (PRE + int(PROFILE.us2cycles(0.02)), 0)]] * (2 * len(ends))
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        point_end + TAIL for point_end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        point_end + TAIL + RELAX for point_end in ends
    ] * 2


def test_nonaffine_reference_debit_and_runtime_max_use_selected_point_not_padding():
    pulse = Pulse(
        "drive",
        _pulse_cfg(
            length=QickSweep1D("scan", 39 / GEN_MHZ, 115 / GEN_MHZ), pre=0, post=0
        ),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))
    loader = LoadValue("other", [60, 20, 100], "scan", "other_end")

    def body(prog):
        prog.fork("default", into=["drive", "other"])
        prog.emit(pulse, timeline="drive")
        prog.emit(loader)
        prog.timeline_inc("other_end", timeline="other")
        prog.delay(50 / TPROC_MHZ)
        prog.join("drive", "other", into="default")
        prog.emit(tail)

    prog = _program([pulse, tail, loader], body, lambda prog: prog.add_loop("scan", 3))
    result = shots(prog)
    # max(0,44/87/129-50) versus max(0,60/20/100-50) is10/37/79.
    ends = [
        max(50, other, ceil(runtime_cycles(prog.soccfg, n)))
        for n, other in zip((39, 77, 115), (60, 20, 100), strict=True)
    ]
    assert [waves(shot) for shot in result] == [[0, end] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL + RELAX for end in ends
    ] * 2


def test_nonaffine_boundary_does_not_certify_a_negative_same_channel_offset():
    pulse = Pulse(
        "drive",
        _pulse_cfg(
            length=QickSweep1D("scan", 25 / GEN_MHZ, 75 / GEN_MHZ), pre=0, post=0
        ),
        pulse_id="drive",
    )

    def body(prog):
        prog.emit(pulse)
        prog.pulse(0, "drive", t=-1 / TPROC_MHZ, timeline="default")

    # At the first point,28-1 starts before actual playback ends at28.
    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program([pulse], body, lambda prog: prog.add_loop("scan", 3))


def test_sweep_end_requires_a_discovered_native_axis_before_using_its_bounds():
    with pytest.raises(ValueError, match="missing.*native sweep axis"):
        _program(
            [], lambda prog: prog.timeline_inc(QickSweep1D("missing", 0.101, 0.301))
        )


def test_native_sweep_delay_preserves_author_declared_dmem_capacity():
    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_dmem([0] * 2045)

    prog = _program(
        [],
        lambda prog: prog.timeline_inc(
            QickSweep1D("scan", 43.68 / TPROC_MHZ, 128.8 / TPROC_MHZ)
        ),
        initialize,
    )
    result = shots(prog)
    memory = prog.compile_datamem()
    assert memory is not None
    assert len(memory) == 2045
    # Native start rounds43.68 to44; native step truncates42.56 to42.
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        44,
        86,
        128,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + RELAX for end in (44, 86, 128)
    ] * 2


def test_readout_length_sweep_adds_post_to_generator_without_adc_padding():
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(
                length=QickSweep1D("scan", 40 / GEN_MHZ, 116 / GEN_MHZ)
            ),
            pre_delay=0,
            post_delay=0.0002,
            ro_ch=0,
            ro_length=0.2,
            trig_offset=0,
        ),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))

    def body(prog):
        prog.emit(readout)
        prog.emit(tail)

    prog = _program([readout, tail], body, lambda prog: prog.add_loop("scan", 3))
    result = shots(prog)
    # Generator k=8 tails44.84375/87.4453125/130.046875 plus post.086016
    # ceil to45/88/131, even where ADC61 samples last85.4 cycles.
    ends = [
        ceil(
            runtime_cycles(prog.soccfg, n)
            + Fraction("0.0002") * Fraction(str(TPROC_MHZ))
        )
        for n in (40, 78, 116)
    ]
    assert [waves(shot) for shot in result] == [[0, 0, end] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + TAIL + RELAX for end in ends
    ] * 2


@pytest.mark.parametrize("kind", ["two", "bath"])
def test_nonaffine_composite_aggregate_and_dependent_placement(kind):
    cavity = _pulse_cfg(
        length=QickSweep1D("scan", 39 / GEN_MHZ, 115 / GEN_MHZ), pre=0, post=0.0002
    )
    qubit = _pulse_cfg(1, length=77 / GEN_MHZ, pre=0, post=0.0002)
    reset = (
        TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit)
        if kind == "two"
        else BathResetCfg(
            cavity_tone_cfg=cavity,
            qubit_tone_cfg=qubit,
            pi2_cfg=_pulse_cfg(0, length=20 / GEN_MHZ, pre=0, post=0.0002),
        )
    ).build("reset")

    prog = _program(
        [reset], lambda prog: prog.emit(reset), lambda prog: prog.add_loop("scan", 3)
    )
    result = shots(prog)
    # Cavity k=8 tails plus .086016 =>44/87/130. Static qubit stays87.
    # Pi2 starts at the cavity boundary and adds ceil(22.4+.086016)=23.
    post = Fraction("0.0002") * Fraction(str(TPROC_MHZ))
    cavity_ends = [ceil(runtime_cycles(prog.soccfg, n) + post) for n in (39, 77, 115)]
    qubit_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 77 / GEN_MHZ) + Fraction("0.0002")
    )
    pi2_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 20 / GEN_MHZ) + Fraction("0.0002")
    )
    cycles = [[0, 0]] * 3 if kind == "two" else [[0, 0, end] for end in cavity_ends]
    ends = [
        max(qubit_end, end + (pi2_end if kind != "two" else 0)) for end in cavity_ends
    ]
    assert [waves(shot) for shot in result] == cycles * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == ends * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + RELAX for end in ends
    ] * 2


@pytest.mark.parametrize("origin", [0, 40])
@pytest.mark.parametrize(
    "kind",
    ["pulse", "two", "bath_short", "bath_long"],
)
def test_reset_retains_child_posts_and_rounds_dependent_placement(kind, origin):
    cavity = _pulse_cfg()
    if kind == "pulse":
        reset = PulseResetCfg(pulse_cfg=cavity).build("reset")
    elif kind == "two":
        reset = TwoPulseResetCfg(
            pulse1_cfg=cavity,
            pulse2_cfg=_pulse_cfg(1, length=0.051, pre=0.02, post=0.05),
        ).build("reset")
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=cavity,
            qubit_tone_cfg=_pulse_cfg(
                1, length=0.401 if kind == "bath_long" else 0.051, pre=0.02, post=0.003
            ),
            pi2_cfg=_pulse_cfg(0, length=0.051),
        ).build("reset")

    def body(prog):
        if origin:
            prog.write_reg("duration", origin)
            prog.timeline_inc("duration")
        prog.emit(reset)

    result = shots(
        _program([reset], body, lambda prog: prog.append_macro(_AllocateDuration()))
    )
    cavity_end = PRE + tproc_ceil(
        PROFILE, generator_duration(PROFILE, 0.101) + Fraction("0.0005")
    )
    expected_waves = [PRE]
    end = cavity_end
    if kind != "pulse":
        qubit_start = tproc_ceil(PROFILE, 0.02)
        expected_waves.append(qubit_start)
        end = max(
            end,
            qubit_start
            + tproc_ceil(
                PROFILE,
                generator_duration(PROFILE, 0.401 if kind == "bath_long" else 0.051)
                + Fraction("0.05" if kind == "two" else "0.003"),
            ),
        )
    if kind.startswith("bath"):
        expected_waves.append(cavity_end + PRE)
        end = max(
            end,
            cavity_end
            + PRE
            + tproc_ceil(
                PROFILE, generator_duration(PROFILE, 0.051) + Fraction("0.0005")
            ),
        )
    assert [waves(shot) for shot in result] == [
        [origin + cycle for cycle in expected_waves]
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        origin + end
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        origin + end + RELAX
    ] * 2
