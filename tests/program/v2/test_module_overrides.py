"""Constructor overrides preserve cfg routing at the public compile boundary."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import (
    MuxReadout,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
)
from zcu_tools.program.v2.modules.reset import (
    BathReset,
    BathResetCfg,
    PulseReset,
    PulseResetCfg,
    Reset,
    TwoPulseReset,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.program.v2.utils import is_qick_param

from tests.program.v2._mux_support import native_mux

ProgramKind = Literal["native", "builder", "legacy"]


def _compile(kind: ProgramKind, modules: Sequence[Module], swept: bool):
    soccfg = make_mock_soccfg(n_gens=3, n_readouts=1)
    if kind == "legacy":
        return ModularProgramV2(
            soccfg, modules=list(modules), sweep=[("point", 3)] if swept else None
        )
    if kind == "builder":
        builder = TimelineBuilder(soccfg)
        if swept:
            builder.add_loop("point", 3)
        for module in modules:
            builder.emit(module)
        return builder.build()

    class Native(TimelineProgramV2):
        def _initialize(self, cfg):
            if swept:
                self.add_loop("point", 3)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            for module in modules:
                self.emit(module)

    return Native(soccfg)


def _pulse_cfg(ch: int = 0) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=100,
        gain=0.2,
        waveform=ConstWaveformCfg(length=0.1),
    )


def _assert_same_compile(actual, expected):
    assert actual.prog_list == expected.prog_list
    assert actual.compiled_stream == expected.compiled_stream
    np.testing.assert_array_equal(actual.binprog["wmem"], expected.binprog["wmem"])
    np.testing.assert_array_equal(actual.compile_datamem(), expected.compile_datamem())


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize(
    "knob,start,stop",
    [
        ("freq", 200, 220),
        ("gain", 0.3, 0.5),
        ("phase", 30, 50),
        ("length", 0.2, 0.3),
    ],
)
def test_pulse_overrides_match_set_param_without_mutating_cfg(
    kind, swept, knob, start, stop
):
    cfg = _pulse_cfg()
    original = cfg.model_dump()
    value = QickSweep1D("point", start, stop) if swept else start
    equivalent = cfg.model_copy(deep=True)
    equivalent.set_param(knob, value)
    actual = _compile(kind, [Pulse("drive", cfg, **{knob: value})], swept)
    expected = _compile(kind, [Pulse("drive", equivalent)], swept)
    _assert_same_compile(actual, expected)
    assert cfg.model_dump() == original


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
def test_pulse_none_overrides_keep_cfg_and_registry_identity(kind):
    cfg = _pulse_cfg()
    actual = _compile(
        kind,
        [
            Pulse("first", cfg, freq=None, gain=None, phase=None, length=None),
            Pulse("second", cfg, gain=0.4),
        ],
        False,
    )
    changed = cfg.model_copy(deep=True)
    changed.set_param("gain", 0.4)
    expected = _compile(kind, [Pulse("first", cfg), Pulse("second", changed)], False)
    _assert_same_compile(actual, expected)


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("factory", [PulseReadout, Readout])
@pytest.mark.parametrize(
    "knob,start,stop",
    [
        ("freq", 200, 220),
        ("gain", 0.3, 0.5),
        ("length", 0.2, 0.3),
        ("ro_length", 0.4, 0.5),
    ],
)
def test_readout_overrides_match_set_param_without_mutating_cfg(
    kind, swept, factory, knob, start, stop
):
    cfg = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        freq=100,
        gain=0.2,
        waveform=ConstWaveformCfg(length=0.1),
        ro_ch=0,
        ro_length=0.2,
    )
    original = cfg.model_dump()
    value = QickSweep1D("point", start, stop) if swept else start
    equivalent = cfg.model_copy(deep=True)
    equivalent.set_param(knob, value)
    overrides = dict(freq=None, gain=None, length=None, ro_length=None)
    overrides[knob] = value
    readout = factory("read", cfg, **overrides)
    if knob == "ro_length" and swept:
        effective = readout.cfg.ro_length
        expected_value = equivalent.ro_length
        assert is_qick_param(effective) and is_qick_param(expected_value)
        np.testing.assert_array_equal(
            effective.to_array({"point": 3}), expected_value.to_array({"point": 3})
        )
        with pytest.raises(
            RuntimeError,
            match="QickRawParam can only be divided after steps have been defined",
        ) as actual_error:
            _compile(kind, [readout], swept)
        with pytest.raises(type(actual_error.value), match=str(actual_error.value)):
            _compile(kind, [factory("read", equivalent)], swept)
    else:
        actual = _compile(kind, [readout], swept)
        expected = _compile(kind, [factory("read", equivalent)], swept)
        _assert_same_compile(actual, expected)
    assert cfg.model_dump() == original


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("use_factory", [False, True])
@pytest.mark.parametrize(
    "variant,knob,start,stop",
    [
        ("pulse", "freq", 200, 220),
        ("pulse", "gain", 0.3, 0.5),
        ("pulse", "phase", 30, 50),
        ("pulse", "length", 0.2, 0.3),
        ("two", "gain1", 0.3, 0.5),
        ("two", "gain2", 0.3, 0.5),
        ("two", "freq1", 200, 220),
        ("two", "freq2", 200, 220),
        ("two", "length", 0.2, 0.3),
        ("bath", "qub_gain", 0.3, 0.5),
        ("bath", "qub_freq", 200, 220),
        ("bath", "qub_length", 0.2, 0.3),
        ("bath", "res_gain", 0.3, 0.5),
        ("bath", "res_freq", 200, 220),
        ("bath", "res_length", 0.2, 0.3),
        ("bath", "pi2_phase", 30, 50),
    ],
)
def test_reset_overrides_match_set_param_without_mutating_cfg(
    kind, swept, use_factory, variant, knob, start, stop
):
    if variant == "pulse":
        cfg = PulseResetCfg(pulse_cfg=_pulse_cfg())
        ctor = PulseReset
    elif variant == "two":
        cfg = TwoPulseResetCfg(pulse1_cfg=_pulse_cfg(), pulse2_cfg=_pulse_cfg(1))
        ctor = TwoPulseReset
    else:
        cfg = BathResetCfg(
            cavity_tone_cfg=_pulse_cfg(),
            qubit_tone_cfg=_pulse_cfg(1),
            pi2_cfg=_pulse_cfg(2),
        )
        ctor = BathReset
    if use_factory:
        ctor = Reset
    original = cfg.model_dump()
    value = QickSweep1D("point", start, stop) if swept else start
    equivalent = cfg.model_copy(deep=True)
    equivalent.set_param(knob, value)
    names = {
        "pulse": ("freq", "gain", "phase", "length"),
        "two": ("gain1", "gain2", "freq1", "freq2", "length"),
        "bath": (
            "qub_gain",
            "qub_freq",
            "qub_length",
            "res_gain",
            "res_freq",
            "res_length",
            "pi2_phase",
        ),
    }
    overrides = dict.fromkeys(names[variant])
    overrides[knob] = value
    actual = _compile(kind, [ctor("reset", cfg, **overrides)], swept)
    expected = _compile(kind, [ctor("reset", equivalent)], swept)
    _assert_same_compile(actual, expected)
    assert cfg.model_dump() == original


@pytest.mark.parametrize("knob", ["freq", "gain", "length", "ro_length"])
@pytest.mark.parametrize("swept", [False, True])
def test_mux_rejects_constructor_overrides(knob, swept):
    cfg = native_mux()
    original = cfg.model_dump()
    value = QickSweep1D("point", 0.2, 0.4) if swept else 0.2
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        MuxReadout("mux", cfg, **{knob: value})
    with pytest.raises(ValueError, match="does not support"):
        Readout("mux", cfg, **{knob: value})
    assert cfg.model_dump() == original


@pytest.mark.parametrize(
    "family", ["pulse", "readout", "readout_factory", "reset", "reset_factory"]
)
def test_unsupported_waveform_length_sweep_is_rejected_without_mutation(family):
    pulse_cfg = _pulse_cfg()
    pulse_cfg.waveform = GaussWaveformCfg(length=0.1, sigma=0.02)
    if family == "pulse":
        cfg, ctor = pulse_cfg, Pulse
    elif family.startswith("readout"):
        cfg = PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=100,
            gain=0.2,
            waveform=pulse_cfg.waveform,
            ro_ch=0,
            ro_length=0.2,
        )
        ctor = Readout if family.endswith("factory") else PulseReadout
    else:
        cfg = PulseResetCfg(pulse_cfg=pulse_cfg)
        ctor = Reset if family.endswith("factory") else PulseReset
    original = cfg.model_dump()
    with pytest.raises(
        ValueError, match="Gauss waveform length must not be a QickParam"
    ):
        ctor("module", cfg, length=QickSweep1D("point", 0.1, 0.3))
    assert cfg.model_dump() == original
