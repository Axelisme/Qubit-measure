import pytest
from pydantic import BaseModel
from zcu_tools.program.v2.modules.readout import (
    AbsReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pulse_ro_cfg(ch=3, ro_ch=None):
    return PulseReadoutCfg.model_validate(
        {
            "gen_ch": ch,
            "nqz": 1,
            "freq": 700.0,
            "gain": 0.8,
            "waveform": ConstWaveformCfg(length=0.5),
            "ro_ch": ro_ch if ro_ch is not None else ch,
            "ro_length": 1.0,
        }
    )


def test_readout_cfg_namespace_dispatches_nested_pulse():
    class Modules(BaseModel):
        readout: Readout.Cfg
        pulse: PulseReadout.Cfg

    cfg = _make_pulse_ro_cfg()
    modules = Modules.model_validate({"readout": cfg.to_dict(), "pulse": cfg.to_dict()})
    assert modules.readout == cfg
    assert modules.pulse == cfg
    assert isinstance(Readout("readout", modules.readout), PulseReadout)


@pytest.mark.parametrize(
    "length,ro_length,expected_us",
    [(0.5, 2.0, 2.4), (3.0, 0.5, 3.5), (0.0, 0.0, 0.5)],
)
def test_nominal_duration_covers_adc_and_generator_without_clock_rounding(
    length: float, ro_length: float, expected_us: float
) -> None:
    cfg = _make_pulse_ro_cfg(ch=2, ro_ch=1)
    cfg.set_param("length", length)
    cfg.ro_length = ro_length
    cfg.pre_delay = 0.25
    cfg.post_delay = 0.25
    cfg.trig_offset = 0.4

    assert cfg.nominal_duration_us() == pytest.approx(expected_us)


class TestReadoutFactory:
    def test_dispatch_pulse(self):
        ro = Readout("ro", _make_pulse_ro_cfg())
        assert isinstance(ro, PulseReadout)

    def test_unknown_cfg_raises(self):
        class UnknownCfg(AbsReadoutCfg):
            type: str = "unknown"

            def build(self, name):
                raise NotImplementedError

            def set_param(self, name, value):
                raise NotImplementedError

        with pytest.raises(NotImplementedError):
            Readout("ro", UnknownCfg())

    def test_name_set(self):
        ro = Readout("myro", _make_pulse_ro_cfg())
        assert ro.name == "myro"

    def test_allow_rerun_pulse(self):
        assert Readout("ro", _make_pulse_ro_cfg()).allow_rerun() is True


class TestPulseReadoutRuntime:
    def test_init_wires_pulse_and_ro_window(self, mock_prog):
        ro = PulseReadout("ro", _make_pulse_ro_cfg(ch=3))
        ro.init(mock_prog)
        assert mock_prog.count("declare_readout") == 1
        assert mock_prog.count("declare_gen") == 1

    def test_run_calls_both_submodules(self, mock_prog):
        ro = PulseReadout("ro", _make_pulse_ro_cfg(ch=3))
        ro.init(mock_prog)
        ro.run(mock_prog, t=0.0)
        assert mock_prog.count("send_readout_config") == 1
        assert mock_prog.count("trigger") == 1
        assert mock_prog.count("pulse") == 1

    def test_runtime_regs_init_scratch_entries(self, mock_prog):
        ro = PulseReadout(
            "ro",
            _make_pulse_ro_cfg(ch=3),
            gain_val="gain_word",
            freq_val="freq_word",
            ro_freq_val="ro_freq_word",
        )

        ro.init(mock_prog)

        assert mock_prog.count("declare_readout") == 1
        assert mock_prog.count("declare_gen") == 2
        add_pulses = mock_prog.events_of("add_pulse")
        assert [event.kwargs["name"] for event in add_pulses] == [
            "pulse_0",
            "ro_runtime_pulse",
        ]
        add_readouts = mock_prog.events_of("add_readout_config")
        assert [event.kwargs["name"] for event in add_readouts] == [
            "ro_adc",
            "ro_adc_runtime",
        ]

    def test_runtime_regs_play_scratch_entries_from_wave_registers(self, mock_prog):
        ro = PulseReadout(
            "ro",
            _make_pulse_ro_cfg(ch=3),
            gain_val="gain_word",
            freq_val="freq_word",
            ro_freq_val="ro_freq_word",
        )
        ro.init(mock_prog)

        ro.run(mock_prog, t=0.25)

        assert mock_prog.only("send_readout_config_from_regs").kwargs == {
            "ch": 3,
            "name": "ro_adc_runtime",
            "t": 0.25,
            "tag": None,
            "freq_reg": "ro_freq_word",
        }
        assert mock_prog.only("pulse_from_regs").kwargs == {
            "ch": 3,
            "name": "ro_runtime_pulse",
            "t": 0.25,
            "tag": None,
            "freq_reg": "freq_word",
            "gain_reg": "gain_word",
        }
        assert mock_prog.count("send_readout_config") == 0
        assert mock_prog.count("pulse") == 0

    def test_runtime_ro_freq_only_leaves_pulse_on_normal_entry(self, mock_prog):
        ro = PulseReadout(
            "ro",
            _make_pulse_ro_cfg(ch=3),
            ro_freq_val="ro_freq_word",
        )
        ro.init(mock_prog)

        ro.run(mock_prog)

        assert mock_prog.only("send_readout_config_from_regs").kwargs == {
            "ch": 3,
            "name": "ro_adc_runtime",
            "t": 0.0,
            "tag": None,
            "freq_reg": "ro_freq_word",
        }
        assert mock_prog.only("pulse").kwargs["name"] == "pulse_0"

    def test_allow_rerun(self):
        assert PulseReadout("ro", _make_pulse_ro_cfg()).allow_rerun() is True


class TestTablePulseReadout:
    def test_rejects_empty_tables(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TablePulseReadout(
                "ro",
                _make_pulse_ro_cfg(),
                idx_reg="freq",
                freq_words=[],
                ro_freq_words=[],
            )

    def test_rejects_mismatched_table_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            TablePulseReadout(
                "ro",
                _make_pulse_ro_cfg(),
                idx_reg="freq",
                freq_words=[1, 2],
                ro_freq_words=[3],
            )


def test_flat_native_knobs_and_capture(mock_prog):
    cfg = _make_pulse_ro_cfg(ch=2, ro_ch=1)
    cfg.set_param("freq", 600.0)
    cfg.set_param("gain", 0.4)
    cfg.set_param("length", 0.75)
    cfg.set_param("ro_length", 2.0)
    with pytest.raises(ValueError, match="Unknown parameter: ro_freq"):
        cfg.set_param("ro_freq", 200.0)
    ro = Readout("flat", cfg)
    cfg.set_param("freq", 800.0)
    ro.init(mock_prog)
    assert mock_prog.only("declare_readout").kwargs == {"ch": 1, "length": 2.0}
    adc = mock_prog.only("add_readout_config").kwargs
    pulse = mock_prog.only("add_pulse").kwargs
    assert (adc["freq"], pulse["freq"]) == (600.0, 600.0)
    assert (adc["gen_ch"], pulse["ro_ch"], pulse["gain"]) == (2, 1, 0.4)


@pytest.mark.parametrize(
    "length,ro_length,expected_ticks", [(0.5, 2.0, 1034), (3.0, 0.5, 1508)]
)
def test_flat_readout_schedules_and_blocks_both_paths(
    mock_prog, length, ro_length, expected_ticks
):
    cfg = _make_pulse_ro_cfg(ch=2, ro_ch=1)
    cfg.set_param("length", length)
    cfg.ro_length = ro_length
    cfg.pre_delay = 0.25
    cfg.post_delay = 0.25
    cfg.trig_offset = 0.4
    ro = Readout("timed", cfg)
    ro.init(mock_prog)
    end = ro.run(mock_prog, t=1.0)
    assert mock_prog.only("send_readout_config").kwargs["t"] == 1.0
    assert mock_prog.only("trigger").kwargs == {"ros": [1], "t": 1.4}
    assert mock_prog.only("pulse").kwargs["t"] == 1.25
    assert end == pytest.approx(1.0 + expected_ticks / 430.08)


@pytest.mark.parametrize("flat_top", [False, True])
def test_native_readout_conversion_binds_arb_to_explicit_capture(tmp_path, flat_top):
    import numpy as np
    from zcu_tools import cfg_model as core
    from zcu_tools.program.v2.modules import ModuleCfgFactory
    from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg, FlatTopWaveformCfg
    from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

    first = tmp_path / "first"
    first.mkdir()
    np.savez(first / "shape.npz", time=[0.0, 0.5], idata=[0.0, 1.0], qdata=[0.0, 0.0])
    context = core.CfgValidationContext(assets=ArbWaveformDatabase.info_at(first))
    raw = {
        "type": "readout/pulse",
        "gen_ch": 1,
        "nqz": 1,
        "freq": 700.0,
        "gain": 0.4,
        "waveform": {
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "arb", "data": "shape"},
        }
        if flat_top
        else {"style": "arb", "data": "shape"},
        "ro_ch": 0,
        "ro_length": 3.0,
    }
    source = core.ModuleCfgFactory.from_raw(raw, context=context)
    native = ModuleCfgFactory.from_core(source, context=context)
    assert isinstance(native, PulseReadoutCfg)
    waveform = (
        native.waveform.raise_waveform
        if isinstance(native.waveform, FlatTopWaveformCfg)
        else native.waveform
    )
    assert isinstance(waveform, ArbWaveformCfg)
    assert waveform.path == (first / "shape.npz").resolve().as_posix()
    assert waveform.length == 0.5
    with pytest.raises(ValueError, match="assets"):
        ModuleCfgFactory.from_core(source, context=None)


def test_swept_readout_schedule_and_blocking_window(mock_prog):
    import numpy as np
    from qick.asm_v2 import QickParam

    cfg = _make_pulse_ro_cfg(ch=2, ro_ch=1)
    cfg.pre_delay = QickParam(0.25, {"scan": 0.1})
    cfg.trig_offset = QickParam(0.4, {"scan": 0.1})
    cfg.set_param("ro_length", QickParam(2.0, {"scan": 0.5}))
    ro = Readout("swept", cfg)
    ro.init(mock_prog)
    end = ro.run(mock_prog, t=QickParam(1.0, {"scan": 0.2}))
    sent = mock_prog.only("send_readout_config").kwargs["t"]
    trigger = mock_prog.only("trigger").kwargs["t"]
    pulse = mock_prog.only("pulse").kwargs["t"]
    loops = {"scan": 3}
    np.testing.assert_allclose(sent.to_array(loops), [1.0, 1.1, 1.2])
    np.testing.assert_allclose(trigger.to_array(loops), [1.4, 1.55, 1.7])
    np.testing.assert_allclose(pulse.to_array(loops), [1.25, 1.4, 1.55])
    assert isinstance(end, QickParam)
    end_times = np.asarray(end.to_array(loops))
    window_ends = np.array([3.4, 3.8, 4.2])
    assert np.all(end_times >= window_ends)
    assert np.all(end_times <= window_ends + 0.01)
