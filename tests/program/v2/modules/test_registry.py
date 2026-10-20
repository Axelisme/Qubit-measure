from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.registry import PulseRegistry
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _make_cfg(freq: float | QickParam = 5000.0, phase=0.0, gain=0.5, ch=0, length=0.1):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=1,
        freq=freq,
        phase=phase,
        gain=gain,
    )


def test_register_new_pulse_returns_true():
    reg = PulseRegistry()
    assert reg.register("p0", _make_cfg()) is True
    assert reg.count == 1


def test_register_duplicate_returns_false():
    reg = PulseRegistry()
    cfg = _make_cfg()
    reg.register("p0", cfg)
    assert reg.register("p0_dup", _make_cfg()) is False
    assert reg.count == 1


def test_register_different_freq_is_new():
    reg = PulseRegistry()
    reg.register("p0", _make_cfg(freq=5000.0))
    assert reg.register("p1", _make_cfg(freq=5001.0)) is True
    assert reg.count == 2


def test_calc_name_stable_with_qickparam():
    reg = PulseRegistry()
    cfg1 = _make_cfg(freq=QickParam(start=5000.0, spans={"s": 1.0}))
    cfg2 = _make_cfg(freq=QickParam(start=5000.0, spans={"s": 1.0}))
    assert reg.calc_name(cfg1) == reg.calc_name(cfg2)


def test_runtime_adc_channel_is_part_of_registry_identity():
    reg = PulseRegistry()
    cfg = _make_cfg()
    names = [reg.calc_name(cfg, ro_ch=ch) for ch in (None, 0, 1)]
    assert len(set(names)) == 3
    for ch in (None, 0, 1):
        assert reg.register("first", cfg, ro_ch=ch) is True
        assert reg.register("again", cfg, ro_ch=ch) is False
    assert reg.count == 3


def test_runtime_adc_channel_reaches_generator_and_pulse(mock_prog):
    from zcu_tools.program.v2.modules import Pulse

    cfg = _make_cfg()
    cfg.mixer_freq = 500.0
    pulse = Pulse("mixed", cfg, ro_ch=1)
    pulse.init(mock_prog)
    assert mock_prog.only("declare_gen").kwargs["ro_ch"] == 1
    assert mock_prog.only("add_pulse").kwargs["ro_ch"] == 1


def test_runtime_adc_channel_without_mixer_only_matches_pulse(mock_prog):
    from zcu_tools.program.v2.modules import Pulse

    pulse = Pulse("unmixed", _make_cfg(), ro_ch=0)
    pulse.init(mock_prog)
    assert mock_prog.only("declare_gen").kwargs["ro_ch"] is None
    assert mock_prog.only("add_pulse").kwargs["ro_ch"] == 0
