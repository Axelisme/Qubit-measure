from __future__ import annotations

import math

import pytest
from zcu_tools.experiment.v2.singleshot.ac_stark import AcStarkCfg
from zcu_tools.experiment.v2_gui.adapters.singleshot.ac_stark import SsAcStarkAdapter
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    DirectValue,
    EvalValue,
    ReferenceSpec,
    SweepSpec,
    read_value_path,
    resolve_spec_path,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _ctx(*, with_pi_amp: bool = True) -> ExpContext:
    md = MetaDict()
    md.q_f = 4000.0
    md.r_f = 6200.0
    md.readout_f = 6210.0
    md.res_ch = 2
    md.qub_ch = 3
    md.rf_w = 4.0
    ml = ModuleLibrary()
    if with_pi_amp:
        ml.modules["pi_amp"] = PulseCfg(
            waveform=ConstWaveformCfg(length=0.4),
            ch=3,
            nqz=2,
            freq=4000.0,
            gain=0.3,
        )
    return ExpContext(md=md, ml=ml, soc=None, soccfg=None)


def test_prefill_matches_notebook_tone_roles_and_timing() -> None:
    ctx = _ctx()
    schema = SsAcStarkAdapter.cfg_definition().instantiate(ctx)
    assert read_value_path(schema.value, "modules.stark_pulse1.ch") == EvalValue(
        "res_ch"
    )
    assert read_value_path(schema.value, "modules.stark_pulse1.freq") == EvalValue(
        "readout_f"
    )
    assert read_value_path(schema.value, "modules.stark_pulse2.ch") == EvalValue(
        "qub_ch"
    )
    assert read_value_path(schema.value, "modules.stark_pulse2.freq") == EvalValue(
        "q_f"
    )
    assert read_value_path(
        schema.value, "modules.stark_pulse2.mixer_freq"
    ) == DirectValue(None)
    assert isinstance(
        read_value_path(schema.value, "modules.stark_pulse1.waveform.length"),
        EvalValue,
    )
    assert isinstance(
        read_value_path(schema.value, "modules.stark_pulse2.pre_delay"),
        EvalValue,
    )
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = ctx.ml.make_cfg(raw, AcStarkCfg)

    assert cfg.modules.reset is None
    assert cfg.modules.init_pulse is None
    cavity = cfg.modules.stark_pulse1
    qubit = cfg.modules.stark_pulse2
    assert cavity.ch == 2
    assert cavity.freq == 6210.0
    assert cavity.gain == 0.0
    assert cavity.waveform.length == pytest.approx(0.4 + 5.1 / (2 * math.pi * 4.0))
    assert qubit.ch == 3
    assert qubit.freq == 4000.0
    assert qubit.gain == 0.3
    assert qubit.waveform.length == 0.4
    assert qubit.pre_delay == pytest.approx(5.0 / (2 * math.pi * 4.0))
    assert qubit.post_delay == pytest.approx(3.1 / (2 * math.pi * 4.0))
    assert qubit.mixer_freq is None
    assert cfg.relax_delay == 5.5
    assert cfg.sweep.gain.expts == 301
    assert (cfg.sweep.freq.start, cfg.sweep.freq.stop, cfg.sweep.freq.expts) == (
        3300.0,
        4100.0,
        801,
    )

    cavity_spec = resolve_spec_path(schema.spec, "modules.stark_pulse1")
    qubit_spec = resolve_spec_path(schema.spec, "modules.stark_pulse2")
    gain_spec = resolve_spec_path(schema.spec, "sweep.gain")
    freq_spec = resolve_spec_path(schema.spec, "sweep.freq")
    assert isinstance(cavity_spec, ReferenceSpec)
    assert isinstance(qubit_spec, ReferenceSpec)
    assert isinstance(gain_spec, SweepSpec)
    assert isinstance(freq_spec, SweepSpec)
    assert "Cavity" in cavity_spec.label and "gain" in cavity_spec.label
    assert "Qubit" in qubit_spec.label and "freq" in qubit_spec.label
    assert "Cavity" in gain_spec.label
    assert "Qubit" in freq_spec.label


def test_prefill_keeps_metadata_links_after_context_values_change() -> None:
    ctx = _ctx()
    ctx.md.qub_ch = None
    ctx.md.qub_1_4_ch = 4
    schema = SsAcStarkAdapter.cfg_definition().instantiate(ctx)
    assert read_value_path(schema.value, "modules.stark_pulse2.ch") == EvalValue(
        "qub_1_4_ch"
    )

    ctx.md.q_f = 4050.0
    ctx.md.qub_1_4_ch = 5
    ctx.md.readout_f = 6220.0
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = ctx.ml.make_cfg(raw, AcStarkCfg)

    assert cfg.modules.stark_pulse1.freq == 6220.0
    assert cfg.modules.stark_pulse2.freq == 4050.0
    assert cfg.modules.stark_pulse2.ch == 5
    assert cfg.sweep.freq.start == 3350.0


def test_configured_qubit_mixer_frequency_stays_linked_to_q_f() -> None:
    ctx = _ctx()
    pulse = ctx.ml.modules["pi_amp"]
    assert isinstance(pulse, PulseCfg)
    pulse.mixer_freq = 4000.0
    schema = SsAcStarkAdapter.cfg_definition().instantiate(ctx)

    assert read_value_path(
        schema.value, "modules.stark_pulse2.mixer_freq"
    ) == EvalValue("q_f")
    ctx.md.q_f = 4050.0
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = ctx.ml.make_cfg(raw, AcStarkCfg)
    assert cfg.modules.stark_pulse2.mixer_freq == 4050.0


def test_prefill_uses_library_frequency_and_channel_without_metadata() -> None:
    ctx = _ctx()
    ctx.md.q_f = None
    ctx.md.qub_ch = None
    pulse = ctx.ml.modules["pi_amp"]
    assert isinstance(pulse, PulseCfg)
    pulse.freq = 4025.0
    pulse.ch = 6
    pulse.mixer_freq = 4025.0

    schema = SsAcStarkAdapter.cfg_definition().instantiate(ctx)
    assert read_value_path(schema.value, "modules.stark_pulse2.freq") == DirectValue(
        4025.0
    )
    assert read_value_path(schema.value, "modules.stark_pulse2.ch") == DirectValue(6)
    assert read_value_path(
        schema.value, "modules.stark_pulse2.mixer_freq"
    ) == DirectValue(4025.0)


def test_prefill_without_library_pi_amp_still_materializes() -> None:
    ctx = _ctx(with_pi_amp=False)
    raw = schema_to_raw_dict(
        SsAcStarkAdapter.cfg_definition().instantiate(ctx), ctx.md, ctx.ml
    )
    cfg = ctx.ml.make_cfg(raw, AcStarkCfg)

    assert cfg.modules.stark_pulse2.waveform.length == 0.3
    assert cfg.modules.stark_pulse1.waveform.length == pytest.approx(
        0.3 + 5.1 / (2 * math.pi * 4.0)
    )


def test_prefill_uses_pi_len_when_pi_amp_is_missing() -> None:
    ctx = _ctx(with_pi_amp=False)
    ctx.ml.modules["pi_len"] = PulseCfg(
        waveform=ConstWaveformCfg(length=0.6),
        ch=3,
        nqz=2,
        freq=4000.0,
        gain=0.2,
    )
    raw = schema_to_raw_dict(
        SsAcStarkAdapter.cfg_definition().instantiate(ctx), ctx.md, ctx.ml
    )
    cfg = ctx.ml.make_cfg(raw, AcStarkCfg)

    assert cfg.modules.stark_pulse2.waveform.length == 0.6
    assert cfg.modules.stark_pulse1.waveform.length == pytest.approx(
        0.6 + 5.1 / (2 * math.pi * 4.0)
    )


def test_invalid_linewidth_fails_before_run() -> None:
    ctx = _ctx()
    ctx.md.rf_w = 0.0
    with pytest.raises(ValueError, match="rf_w.*positive finite"):
        SsAcStarkAdapter.cfg_definition().instantiate(ctx)
