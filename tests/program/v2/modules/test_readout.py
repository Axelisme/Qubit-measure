import pytest
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    AbsReadoutCfg,
    DirectReadout,
    DirectReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pulse_cfg(ch=0, freq=5000.0):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=0.5),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=0.8,
    )


def _make_direct_cfg(ro_ch=0, ro_freq=5000.0):
    return DirectReadoutCfg(ro_ch=ro_ch, ro_length=1.0, ro_freq=ro_freq)


def _make_pulse_ro_cfg(ch=3, ro_ch=None):
    return PulseReadoutCfg(
        pulse_cfg=_make_pulse_cfg(ch=ch, freq=7000.0),
        ro_cfg=_make_direct_cfg(
            ro_ch=ro_ch if ro_ch is not None else ch, ro_freq=7000.0
        ),
    )


# ---------------------------------------------------------------------------
# Readout factory – dispatch
# ---------------------------------------------------------------------------


class TestReadoutFactory:
    def test_dispatch_direct(self):
        ro = Readout("ro", _make_direct_cfg())
        assert isinstance(ro, DirectReadout)

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
        ro = Readout("myro", _make_direct_cfg())
        assert ro.name == "myro"

    def test_allow_rerun_direct(self):
        assert Readout("ro", _make_direct_cfg()).allow_rerun() is True

    def test_allow_rerun_pulse(self):
        assert Readout("ro", _make_pulse_ro_cfg()).allow_rerun() is True


# ---------------------------------------------------------------------------
# DirectReadout / PulseReadout – init / run via mock prog
# ---------------------------------------------------------------------------


class TestDirectReadoutRuntime:
    def test_init_calls_declare_and_add(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg(ro_ch=2, ro_freq=6000.0))
        ro.init(mock_prog)
        assert mock_prog.only("declare_readout").kwargs == {"ch": 2, "length": 1.0}
        assert mock_prog.only("add_readout_config").kwargs == {
            "ch": 2,
            "name": "ro",
            "freq": 6000.0,
        }

    def test_init_passes_gen_ch_when_set(self, mock_prog):
        cfg = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=5000.0, gen_ch=1)
        ro = DirectReadout("ro", cfg)
        ro.init(mock_prog)
        assert mock_prog.only("add_readout_config").kwargs["gen_ch"] == 1

    def test_init_omits_gen_ch_when_none(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg())
        ro.init(mock_prog)
        assert "gen_ch" not in mock_prog.only("add_readout_config").kwargs

    def test_run_calls_send_readoutconfig_and_trigger(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg())
        ro.init(mock_prog)
        result = ro.run(mock_prog, t=0.5)
        assert mock_prog.only("send_readout_config").kwargs == {
            "ch": 0,
            "name": "ro",
            "t": 0.5,
        }
        assert mock_prog.only("trigger").kwargs == {"ros": [0], "t": 0.5}
        assert result == 0.5  # returns t unchanged

    def test_allow_rerun(self):
        assert DirectReadout("ro", _make_direct_cfg()).allow_rerun() is True


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
