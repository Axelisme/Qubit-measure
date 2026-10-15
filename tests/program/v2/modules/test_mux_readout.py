"""Fixed mux compile, hardware validation and averaged-I/Q selection contracts."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import ModuleCfg, ModuleCfgFactory, readout
from zcu_tools.program.v2.modules.base import Module

from tests.program.v2._mux_support import (
    AcquisitionSoc,
    RecordingProgram,
    core_mux,
    mux_soccfg,
    native_mux,
)


def test_native_cfg_build_and_union_preserve_order_and_copy():
    core = core_mux()
    cfg = native_mux()
    for parsed in (
        ModuleCfgFactory.from_raw(core.to_dict()),
        TypeAdapter(ModuleCfg).validate_python(core.to_dict()),
        TypeAdapter(readout.ReadoutCfg).validate_python(core.to_dict()),
    ):
        module = readout.Readout("mux", parsed)
        assert isinstance(module, readout.MuxReadout)
        parsed.tones.reverse()
        assert [tone.ro_ch for tone in module.cfg.tones] == [4, 1]
    with pytest.raises(ValueError, match="rebuild"):
        cfg.set_param("freq", 1200.0)


@pytest.mark.parametrize("channels", [(4,), (4, 1, 7, 0, 6, 2, 5, 3)])
def test_real_compile_and_acquire_all_tones(channels):
    cfg = native_mux(channels)
    module = cfg.build("mux")
    soccfg = mux_soccfg()
    prog = RecordingProgram(soccfg, [module], sweep=[("offset", 3)])
    assert prog.gen_declarations == [
        dict(
            ch=0,
            nqz=1,
            mixer_freq=1000.0,
            ro_ch=channels[0],
            mux_freqs=[t.freq for t in cfg.tones],
            mux_gains=[t.gain for t in cfg.tones],
            mux_phases=[t.phase for t in cfg.tones],
        )
    ]
    assert prog.adc_declarations == [
        dict(ch=t.ro_ch, length=cfg.ro_length, freq=t.freq, phase=0.0, gen_ch=0)
        for t in cfg.tones
    ]
    pulse = prog.pulse_definitions[0]
    assert pulse == dict(
        ch=0,
        name="mux",
        style="const",
        mask=list(range(len(channels))),
        length=cfg.length,
    )
    assert prog.triggers == [(list(channels), dict(t=cfg.trig_offset))]
    assert prog.playbacks == [dict(ch=0, name="mux", t=cfg.pre_delay)]
    soc = AcquisitionSoc(soccfg.get_cfg())
    raw = prog.acquire(soc, progress=False)
    iq = module.collect_iq(prog, raw)
    np.testing.assert_allclose(
        iq, np.tile([[ch + 1, -ch - 1] for ch in channels], (3, 1, 1))
    )
    assert len(soc.gen_configs[0][1]) == len(channels)
    assert soc.pfb_configs[0][0] == "pfb0"
    assert soc.pfb_configs[0][2] is None
    assert [r["pfb_port"] for r in soc.pfb_configs[0][1]] == list(channels)


@pytest.mark.parametrize("updates", [dict(length=0.537), dict(ro_length=1.337)])
def test_blocking_covers_rounded_generator_and_adc_windows(updates):
    cfg = native_mux(**updates)
    module = cfg.build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    # Scheduled scalar starts round to tProc ticks, hardware windows to their clocks.
    tclk = prog.soccfg["tprocs"][0]["f_time"]
    ceil_t = lambda t: np.ceil(t * tclk) / tclk
    gen_duration = prog.cycles2us(prog.us2cycles(cfg.length, gen_ch=0), gen_ch=0)
    adc_duration = prog.cycles2us(prog.us2cycles(cfg.ro_length, ro_ch=4), ro_ch=4)
    expected = max(
        ceil_t(ceil_t(cfg.pre_delay) + gen_duration + ceil_t(cfg.post_delay)),
        ceil_t(ceil_t(cfg.trig_offset) + adc_duration),
    )
    assert module.total_length(prog) == pytest.approx(expected)
    assert module.run(prog, 2.0) == pytest.approx(2.0 + expected)
    assert module.allow_rerun()


@pytest.mark.parametrize(
    "fault",
    [
        "gen_type",
        "mixer",
        "gain",
        "phase",
        "tone_capacity",
        "gen_missing",
        "ro_type",
        "dynamic",
        "pfb_path",
        "pfb_capacity",
        "port_range",
        "duplicate_port",
        "ro_missing",
        "duplicate_channel",
        "short_gen",
        "short_adc",
    ],
)
def test_invalid_hardware_or_routes_fail_before_declaration(fault):
    cfg = native_mux()
    soccfg = mux_soccfg()
    gen, ros = soccfg["gens"][0], soccfg["readouts"]
    if fault == "gen_type":
        gen["type"] = "axis_signal_gen_v6"
    elif fault == "mixer":
        gen["has_mixer"] = False
    elif fault == "gain":
        gen["has_gain"] = False
    elif fault == "phase":
        gen["has_phase"] = False
    elif fault == "tone_capacity":
        gen["n_tones"] = 1
    elif fault == "gen_missing":
        cfg.gen_ch = 99
    elif fault == "ro_type":
        ros[4]["ro_type"] = "axis_pfb_readout_v3"
    elif fault == "dynamic":
        ros[4]["tproc_ctrl"] = 0
    elif fault == "pfb_path":
        ros[1]["ro_fullpath"] = "other"
    elif fault == "pfb_capacity":
        ros[1]["pfb_nout"] = 1
    elif fault == "port_range":
        ros[1]["pfb_port"] = 8
    elif fault == "duplicate_port":
        ros[1]["pfb_port"] = 4
    elif fault == "ro_missing":
        cfg.tones[0].ro_ch = 99
    elif fault == "duplicate_channel":
        cfg.tones[1].ro_ch = 4
    elif fault == "short_gen":
        cfg.length = 1e-8
    elif fault == "short_adc":
        cfg.ro_length = 1e-8
    with pytest.raises(ValueError, match="mux"):
        RecordingProgram(soccfg, [cfg.build("mux")])


class UnrelatedReadout(Module):
    name = "unrelated"

    def init(self, prog):
        prog.declare_readout(ch=8, length=0.2)
        prog.add_readoutconfig(ch=8, name="unrelated", freq=100.0)

    def run(self, prog, t=0.0):
        prog.send_readoutconfig(8, "unrelated", t=t)
        prog.trigger([8], t=t)
        return t + 0.3


@pytest.mark.parametrize("unrelated_first", [True, False])
def test_collect_iq_uses_current_declarations_and_read_index(unrelated_first):
    module = native_mux().build("mux")
    modules = (
        [UnrelatedReadout(), module]
        if unrelated_first
        else [module, UnrelatedReadout()]
    )
    prog = ModularProgramV2(mux_soccfg(), ProgramV2Cfg(), modules)
    selected = {
        4: np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2),
        1: 100 + np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2),
        8: np.zeros((1, 2), dtype=np.float64),
    }
    raw = [selected[ch] for ch in prog.ro_chs]
    selected[4][1, 0, 0, 0] = np.nan
    before = deepcopy(raw)
    iq = module.collect_iq(prog, raw, read_index=1)
    assert iq.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(iq[..., 0, :], selected[4][1])
    np.testing.assert_array_equal(iq[..., 1, :], selected[1][1])
    for actual, expected in zip(raw, before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "fault",
    [
        "uninitialized",
        "missing_channel",
        "count",
        "negative",
        "bool",
        "fraction",
        "bounds",
        "no_read",
        "iq_width",
        "dtype",
        "read_shape",
        "sweep_shape",
    ],
)
def test_collect_iq_rejects_invalid_averaged_input(fault):
    module = native_mux().build("mux")
    prog = RecordingProgram(mux_soccfg(), [module])
    raw = [np.zeros((2, 3, 2), dtype=np.float64) for _ in range(2)]
    index = 0
    if fault == "uninitialized":
        module = native_mux().build("mux")
    elif fault == "missing_channel":
        prog = RecordingProgram(mux_soccfg(), [])
    elif fault == "count":
        raw.pop()
    elif fault == "negative":
        index = -1
    elif fault == "bool":
        index = True
    elif fault == "fraction":
        index = 0.5
    elif fault == "bounds":
        index = 2
    elif fault == "no_read":
        raw[0] = np.zeros(2)
    elif fault == "iq_width":
        raw[0] = np.zeros((2, 3, 3))
    elif fault == "dtype":
        raw[0] = np.zeros((2, 3, 2), dtype=np.float32)
    elif fault == "read_shape":
        raw[0] = np.zeros((3, 3, 2))
    elif fault == "sweep_shape":
        raw[0] = np.zeros((2, 4, 2))
    with pytest.raises(ValueError, match="mux"):
        module.collect_iq(prog, raw, read_index=index)


@pytest.mark.parametrize("collision", ["gen", "readout"])
def test_routes_cannot_be_silently_redeclared(collision):
    first = native_mux((4,)).build("first")
    second_cfg = native_mux((1,) if collision == "gen" else (4,))
    if collision == "readout":
        soccfg = mux_soccfg()
        soccfg["gens"][1].update(soccfg["gens"][0])
        soccfg["gens"][1]["tproc_ch"] = 1
        second_cfg.gen_ch = 1
    else:
        soccfg = mux_soccfg()
    with pytest.raises(ValueError, match="second"):
        RecordingProgram(soccfg, [first, second_cfg.build("second")])
