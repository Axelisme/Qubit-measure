"""Smoke tests for TimelineProgramV2 compilation without hardware.

These tests verify that the program class can be constructed and compiled
using a mock QickConfig built from a plain dict, exercising the real
_initialize / _body / compile pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import inf, nextafter
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from qick import QickConfig
from qick.asm_v2 import QickParam
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    PulseReadoutCfg,
    TableParam,
    TimelineBuilder,
    TimelineProgramV2,
)
from zcu_tools.program.v2.modules import Delay, SoftDelay
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg

from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prog(
    modules=None,
    sweep=None,
    n_gens: int = 2,
    n_readouts: int = 1,
    **cfg_kwargs,
) -> TimelineProgramV2:
    soccfg = load_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    return compile_modules(soccfg, modules=modules or [], sweep=sweep, **cfg_kwargs)


def _addr_inc(inst: Mapping[str, object]) -> int:
    return 2 if inst.get("CMD") == "WAIT" else 1


def _expected_p_addr(prog: TimelineProgramV2) -> int:
    if not prog.prog_list:
        return 0
    end_addrs = []
    for inst in prog.prog_list:
        p_addr = inst["P_ADDR"]
        assert isinstance(p_addr, int)
        end_addrs.append(p_addr + _addr_inc(inst))
    return max(end_addrs)


def _expected_line(prog: TimelineProgramV2) -> int:
    tracked_labels = sum(1 for info in prog.meta_infos if info.get("kind") == "label")
    return len(prog.prog_list) + tracked_labels


# ---------------------------------------------------------------------------
# Compile-phase tests
# ---------------------------------------------------------------------------


@pytest.fixture
def interpolating_soccfg() -> QickConfig:
    with (Path(__file__).parents[1] / "profiles" / "ordinary.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    profile["generators"][0].update(interpolation=4, has_mixer=True)
    return build_soccfg(profile)


def _frequency_builder(
    soccfg: QickConfig, start: float, stop: float, mixer_freq: float | None = 0.0
) -> TimelineBuilder:
    builder = TimelineBuilder(soccfg)
    frequency = builder.add_sweep(
        "frequency", SweepCfg(start=start, stop=stop, step=(stop - start) / 2, expts=3)
    )
    builder.emit(
        Pulse(
            "drive",
            PulseCfg(
                ch=0,
                nqz=1,
                freq=frequency,
                mixer_freq=mixer_freq,
                gain=0.5,
                waveform=ConstWaveformCfg(length=0.1),
            ),
        )
    )

    return builder


@pytest.mark.parametrize("direction", [1, -1])
def test_swept_frequency_range_error_reports_endpoints_and_dds_limit(
    interpolating_soccfg: QickConfig, direction: int
):
    half_range = interpolating_soccfg["gens"][0]["f_dds"] / 2
    start, stop = direction * (5 * half_range - 5), direction * (5 * half_range + 5)
    builder = _frequency_builder(interpolating_soccfg, start, stop)

    with pytest.raises(RuntimeError) as error:
        builder.build()

    message = str(error.value)
    assert "range" in message.lower()
    for value in (start, stop, -half_range, half_range):
        assert f"{value:g}" in message
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("has_mixer", [False, True])
def test_legal_swept_frequency_compiles_with_and_without_mixer(
    interpolating_soccfg: QickConfig, has_mixer: bool
):
    interpolating_soccfg["gens"][0]["has_mixer"] = has_mixer
    half_range = interpolating_soccfg["gens"][0]["f_dds"] / 2
    mixer = 4 * half_range if has_mixer else None
    center = (mixer or 0.0) + half_range / 4
    builder = _frequency_builder(interpolating_soccfg, center - 5, center + 5, mixer)

    program = builder.build()

    assert program.binprog["pmem"] is not None


@pytest.mark.parametrize(
    ("failure", "interpolation"),
    [
        (RuntimeError("conversion failed"), 4),
        (RuntimeError("conversion failed"), 1),
        (ValueError("conversion failed"), 4),
    ],
)
def test_frequency_conversion_preserves_other_dependency_failures(
    interpolating_soccfg: QickConfig, failure: Exception, interpolation: int
):
    class FailingConfig(QickConfig):
        def freq2reg(self, f, gen_ch=0, ro_ch=None):
            raise failure

    soccfg = FailingConfig(interpolating_soccfg.get_cfg())
    soccfg["gens"][0]["interpolation"] = interpolation
    builder = _frequency_builder(soccfg, 10, 20)

    with pytest.raises(type(failure)) as error:
        builder.build()

    assert error.value is failure


def _gain_builder(
    soccfg: QickConfig, gain: float | QickParam | SweepCfg, multiplier: float = 1.0
) -> TimelineBuilder:
    builder = TimelineBuilder(soccfg)
    if isinstance(gain, SweepCfg):
        gain = builder.add_sweep("gain", gain)
    builder.emit(
        Pulse(
            "drive",
            PulseCfg(
                ch=0,
                nqz=1,
                freq=100.0,
                gain=multiplier * gain,
                waveform=ConstWaveformCfg(length=0.1),
            ),
            pulse_id="gain_probe",
        )
    )
    return builder


@pytest.mark.parametrize("gain", [1.2, -1.2, nextafter(1.0, inf)])
def test_scalar_gain_outside_full_scale_fails_at_construction(gain: float) -> None:
    with pytest.raises(ValidationError) as error:
        PulseCfg(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
        )

    assert {item["loc"][0] for item in error.value.errors()} == {"gain"}


@pytest.mark.parametrize("gain", [1.2, -1.2])
def test_readout_scalar_gain_outside_full_scale_fails_at_construction(
    gain: float,
) -> None:
    with pytest.raises(ValidationError) as error:
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=100.0,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
            ro_ch=0,
            ro_length=0.1,
        )

    assert {item["loc"][0] for item in error.value.errors()} == {"gain"}


@pytest.mark.parametrize("gain", [1.2, -1.2])
def test_constant_qick_param_gain_outside_full_scale_fails_at_compile(
    mock_soccfg: QickConfig, gain: float
) -> None:
    builder = _gain_builder(mock_soccfg, QickParam(gain))

    with pytest.raises(ValueError, match="gain range") as error:
        builder.build()

    message = str(error.value)
    assert "gain_probe" in message
    assert "generator 0" in message
    assert f"[{gain}, {gain}]" in message
    assert "[-1, 1]" in message


@pytest.mark.parametrize("gain", [-1.0, 0.0, 1.0])
def test_scalar_gain_including_full_scale_compiles(
    mock_soccfg: QickConfig, gain: float
) -> None:
    assert _gain_builder(mock_soccfg, gain).build().binprog["pmem"] is not None


@pytest.mark.parametrize(
    ("start", "stop", "multiplier", "actual_range"),
    [
        (-0.5, 1.2, 1.0, "[-0.5, 1.2]"),
        (1.2, -0.5, 1.0, "[-0.5, 1.2]"),
        (-1.2, 0.5, 1.0, "[-1.2, 0.5]"),
        (0.5, -1.2, 1.0, "[-1.2, 0.5]"),
        (-0.6, 0.6, 2.0, "[-1.2, 1.2]"),
        (0.6, -0.6, -2.0, "[-1.2, 1.2]"),
    ],
)
def test_linear_gain_checks_actual_range_before_conversion(
    mock_soccfg: QickConfig,
    start: float,
    stop: float,
    multiplier: float,
    actual_range: str,
) -> None:
    sweep = SweepCfg(start=start, stop=stop, step=(stop - start) / 2, expts=3)
    builder = _gain_builder(mock_soccfg, sweep, multiplier)

    with pytest.raises(ValueError, match="gain range") as error:
        builder.build()

    message = str(error.value)
    assert "gain_probe" in message
    assert "generator 0" in message
    assert actual_range in message
    assert "[-1, 1]" in message


@pytest.mark.parametrize("start, stop", [(-1.0, 1.0), (1.0, -1.0)])
def test_linear_gain_full_scale_endpoints_compile(
    mock_soccfg: QickConfig, start: float, stop: float
) -> None:
    sweep = SweepCfg(start=start, stop=stop, step=stop, expts=3)
    assert _gain_builder(mock_soccfg, sweep).build().binprog["pmem"] is not None


@pytest.mark.parametrize(
    ("middle", "actual_range"), [(1.2, "[0.0, 1.2]"), (-1.2, "[-1.2, 0.0]")]
)
def test_table_gain_checks_interior_extrema(
    mock_soccfg: QickConfig, middle: float, actual_range: str
) -> None:
    builder = _gain_builder(mock_soccfg, TableParam("gain", [0.0, middle, 0.0]))
    builder.add_loop("gain", 3)

    with pytest.raises(ValueError, match="gain range") as error:
        builder.build()

    message = str(error.value)
    assert "gain_probe" in message
    assert "generator 0" in message
    assert actual_range in message
    assert "[-1, 1]" in message


def test_table_gain_full_scale_and_zero_compile(mock_soccfg: QickConfig) -> None:
    builder = _gain_builder(mock_soccfg, TableParam("gain", [-1.0, 0.0, 1.0]))
    builder.add_loop("gain", 3)
    assert builder.build().binprog["pmem"] is not None


@pytest.mark.parametrize("gain", [1.2, -1.2])
def test_native_pulse_gain_cannot_bypass_compile_check(
    mock_soccfg: QickConfig, gain: float
) -> None:
    class NativeProgram(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            del cfg
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0,
                "native_gain",
                style="const",
                freq=100,
                phase=0,
                gain=gain,
                length=0.1,
            )

        def _body(self, cfg: None) -> None:
            del cfg
            self.pulse(0, "native_gain")

    with pytest.raises(ValueError, match="native_gain.*generator 0.*gain range"):
        NativeProgram(mock_soccfg)


def test_empty_program_compiles():
    prog = _make_prog()
    assert prog.binprog is not None


def test_binprog_has_pmem():
    prog = _make_prog()
    assert "pmem" in prog.binprog


def test_delay_module_compiles():
    prog = _make_prog(modules=[Delay("wait", delay=1.0)])
    assert prog.binprog is not None


def test_softdelay_module_compiles():
    prog = _make_prog(modules=[SoftDelay("soft_wait", delay=0.5)])
    assert prog.binprog is not None


def test_multiple_delays_compile():
    modules = [
        SoftDelay("pre", delay=0.2),
        Delay("mid", delay=1.0),
        SoftDelay("post", delay=0.3),
    ]
    prog = _make_prog(modules=modules)
    assert prog.binprog is not None


def test_sweep_int_compiles():
    prog = _make_prog(sweep=[("my_loop", 10)])
    assert prog.binprog is not None


def test_sweep_cfg_compiles():
    sweep_cfg = SweepCfg(start=0.0, stop=9.0, expts=10, step=1.0)
    prog = _make_prog(sweep=[("freq_sweep", sweep_cfg)])
    assert prog.binprog is not None


def test_compile_keeps_labels_out_of_prog_list():
    prog = _make_prog(sweep=[("my_loop", 10)])

    assert prog.labels
    assert all(not ("LABEL" in inst and "CMD" not in inst) for inst in prog.prog_list)


def test_compile_refreshes_program_cursors():
    prog = _make_prog(sweep=[("my_loop", 10)])

    assert prog.p_addr == _expected_p_addr(prog)
    assert prog.line == _expected_line(prog)


def test_compile_refreshes_program_cursors_with_wait():
    prog = _make_prog(modules=[Delay("wait", delay=1.0)])

    assert any(inst.get("CMD") == "WAIT" for inst in prog.prog_list)
    assert prog.p_addr == _expected_p_addr(prog)
    assert prog.line == _expected_line(prog)


def test_reps_propagated():
    prog = _make_prog(reps=4)
    assert prog.reps == 4


def test_datamem_none_when_empty():
    prog = _make_prog()
    assert prog.compile_datamem() is None


# ---------------------------------------------------------------------------
# Parametrised channel counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gens,n_readouts", [(1, 1), (2, 1), (4, 2)])
def test_various_channel_counts(n_gens, n_readouts):
    prog = _make_prog(n_gens=n_gens, n_readouts=n_readouts)
    assert prog.binprog is not None
