"""Reset phase cycling balances preparation without changing gate timing."""

import re

import numpy as np
import pytest
from qick.asm_v2 import QickRawParam
from zcu_tools.program.v2 import (
    DirectReadoutCfg,
    ModularProgramV2,
    PulseCfg,
    PulseReadoutCfg,
    TwoPulseResetCfg,
    make_mock_soccfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.zigzag.core import (
    ZigZagCfg,
    ZigZagModuleCfg,
    build_zigzag_modules,
)


def _base() -> ZigZagCfg:
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        freq=309.153205939,
        gain=0.9,
        waveform=ConstWaveformCfg(length=0.25),
    )
    tone = PulseCfg(
        ch=1,
        nqz=2,
        freq=5345.8,
        gain=0.11,
        waveform=ConstWaveformCfg(length=20.01),
        post_delay=0.35,
    )
    reset_pi = pulse.model_copy(deep=True)
    reset_pi.pre_delay = 20.36
    seed = pulse.model_copy(deep=True)
    seed.waveform.set_param("length", 0.125)
    readout = tone.model_copy(deep=True)
    readout.waveform.set_param("length", 1.0)
    readout.post_delay = 0.0
    return ZigZagCfg(
        n_times=4,
        relax_delay=20.0,
        modules=ZigZagModuleCfg(
            reset=TwoPulseResetCfg(pulse1_cfg=tone, pulse2_cfg=reset_pi),
            X90_pulse=seed,
            X180_pulse=pulse,
            readout=PulseReadoutCfg(
                pulse_cfg=readout,
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, gen_ch=1, ro_freq=5345.8, ro_length=1.0
                ),
            ),
        ),
    )


@pytest.mark.parametrize("reps", [2, 6])
@pytest.mark.parametrize("phase", [0.0, 30.0, 180.0])
def test_reset_phase_cycle_is_balanced_and_leaves_gate_words_unchanged(reps, phase):
    raw = _base().to_dict()
    raw.update(reps=reps, reset_phase_cycle=True)
    raw["modules"]["reset"]["pulse2_cfg"]["phase"] = phase
    cfg = ZigZagCfg.model_validate(raw)
    before = cfg.to_dict()
    soc = make_mock_soccfg()
    prog = ModularProgramV2(
        soc, cfg, build_zigzag_modules(cfg), sweep=[("times", cfg.n_times + 1)]
    )
    fixed_cfg = cfg.model_copy(update={"reset_phase_cycle": False}, deep=True)
    fixed_prog = ModularProgramV2(
        soc,
        fixed_cfg,
        build_zigzag_modules(fixed_cfg),
        sweep=[("times", cfg.n_times + 1)],
    )
    port_times = r"WPORT_WR (p\d+) wmem \[&\d+\] @(\d+)"
    assert re.findall(port_times, prog.asm()) == re.findall(
        port_times, fixed_prog.asm()
    )
    swept = [
        wave
        for wave in prog.waves
        if isinstance(wave.phase, QickRawParam) and wave.phase.spans
    ]
    assert len(swept) == 1
    phase_word = swept[0].phase
    assert isinstance(phase_word, QickRawParam)
    modulus = 1 << soc["gens"][0]["b_phase"]
    assert phase_word.steps is not None
    step = phase_word.steps["reps"]["step"]
    assert step == modulus // 2
    assert phase_word.steps["reps"]["span"] == step * (reps - 1)
    words = (phase_word.start + np.arange(reps, dtype=np.int64) * step) % modulus
    np.testing.assert_array_equal(words[::2], np.full(reps // 2, words[0]))
    np.testing.assert_array_equal(words[1::2], np.full(reps // 2, words[1]))
    assert words[0] != words[1]
    for pulse in [cfg.modules.X90_pulse, cfg.modules.X180_pulse]:
        assert pulse is not None
        assert isinstance(pulse.phase, float)
        pulse_name = prog.pulse_registry.calc_name(pulse)
        np.testing.assert_allclose(
            prog.get_pulse_param(pulse_name, "phase", as_array=True), pulse.phase
        )
    assert cfg.to_dict() == before


@pytest.mark.parametrize("reps", [0, 1, 3])
def test_unbalanced_reset_phase_cycle_fails_before_hardware(reps):
    raw = _base().to_dict()
    raw.update(reps=reps, reset_phase_cycle=True)
    with pytest.raises(ValueError, match="even reps"):
        ZigZagCfg.model_validate(raw)


def test_reset_phase_cycle_requires_a_supported_reset():
    raw = _base().to_dict()
    raw.update(reps=2, reset_phase_cycle=True)
    raw["modules"]["reset"] = None
    with pytest.raises(ValueError, match="requires a two-pulse or bath reset"):
        ZigZagCfg.model_validate(raw)
