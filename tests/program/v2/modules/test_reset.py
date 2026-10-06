from unittest.mock import MagicMock

import pytest
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.reset import (
    AbsResetCfg,
    BathReset,
    BathResetCfg,
    NoneReset,
    NoneResetCfg,
    PulseReset,
    PulseResetCfg,
    Reset,
    TwoPulseReset,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ml():
    return MagicMock()


def _make_pulse_cfg(ch=1, freq=4000.0, length=0.2, gain=0.5):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _make_pi2_cfg():
    return PulseCfg(
        waveform=GaussWaveformCfg(length=0.1, sigma=0.025),
        ch=2,
        nqz=1,
        freq=4000.0,
        gain=0.5,
    )


# ---------------------------------------------------------------------------
# PulseResetCfg
# ---------------------------------------------------------------------------


class TestPulseResetCfg:
    def test_pulse_cfg_as_string(self):
        ml = _ml()
        ml.get_module.return_value = _make_pulse_cfg()
        d = {"type": "reset/pulse", "pulse_cfg": "my_pulse"}
        cfg = PulseResetCfg.model_validate(d, context={"ml": ml})
        ml.get_module.assert_called_once_with("my_pulse")
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)


# ---------------------------------------------------------------------------
# TwoPulseResetCfg
# ---------------------------------------------------------------------------


class TestTwoPulseResetCfg:
    def test_pulse_cfgs_as_strings(self):
        ml = _ml()
        ml.get_module.return_value = _make_pulse_cfg()
        d = {
            "type": "reset/two_pulse",
            "pulse1_cfg": "p1",
            "pulse2_cfg": "p2",
        }
        cfg = TwoPulseResetCfg.model_validate(d, context={"ml": ml})
        assert ml.get_module.call_count == 2
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)


# ---------------------------------------------------------------------------
# BathResetCfg
# ---------------------------------------------------------------------------


class TestBathResetCfg:
    def test_cfgs_as_strings(self):
        ml = _ml()
        ml.get_module.return_value = _make_pulse_cfg()
        d = {
            "type": "reset/bath",
            "cavity_tone_cfg": "res_pulse",
            "qubit_tone_cfg": "qub_pulse",
            "pi2_cfg": "pi2_pulse",
        }
        cfg = BathResetCfg.model_validate(d, context={"ml": ml})
        assert ml.get_module.call_count == 3
        assert isinstance(cfg, BathResetCfg)
        assert isinstance(cfg.cavity_tone_cfg, PulseCfg)


# ---------------------------------------------------------------------------
# Reset factory – dispatch
# ---------------------------------------------------------------------------


class TestResetFactory:
    def test_dispatch_none(self):
        r = Reset("r", NoneResetCfg())
        assert isinstance(r, NoneReset)

    def test_dispatch_none_from_none_arg(self):
        r = Reset("r", None)
        assert isinstance(r, NoneReset)

    def test_dispatch_pulse(self):
        r = Reset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        assert isinstance(r, PulseReset)

    def test_dispatch_two_pulse(self):
        r = Reset(
            "r",
            TwoPulseResetCfg(
                pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
            ),
        )
        assert isinstance(r, TwoPulseReset)

    def test_dispatch_bath(self):
        r = Reset(
            "r",
            BathResetCfg(
                cavity_tone_cfg=_make_pulse_cfg(ch=1),
                qubit_tone_cfg=_make_pulse_cfg(ch=2),
                pi2_cfg=_make_pi2_cfg(),
            ),
        )
        assert isinstance(r, BathReset)

    def test_unknown_cfg_raises(self):
        class UnknownCfg(AbsResetCfg):
            type: str = "unknown_reset"

            def build(self, name):
                raise NotImplementedError

            def set_param(self, name, value):
                raise NotImplementedError

        with pytest.raises(NotImplementedError):
            Reset("r", UnknownCfg())

    def test_name_set(self):
        r = Reset("myname", NoneResetCfg())
        assert r.name == "myname"

    def test_allow_rerun_none(self):
        assert Reset("r", NoneResetCfg()).allow_rerun() is True

    def test_allow_rerun_pulse(self):
        assert (
            Reset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg())).allow_rerun() is True
        )

    def test_allow_rerun_two_pulse(self):
        assert (
            Reset(
                "r",
                TwoPulseResetCfg(
                    pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
                ),
            ).allow_rerun()
            is True
        )

    def test_allow_rerun_bath(self):
        assert (
            Reset(
                "r",
                BathResetCfg(
                    cavity_tone_cfg=_make_pulse_cfg(ch=1),
                    qubit_tone_cfg=_make_pulse_cfg(ch=2),
                    pi2_cfg=_make_pi2_cfg(),
                ),
            ).allow_rerun()
            is True
        )


# ---------------------------------------------------------------------------
# Cfg → Module construction + runtime via mock prog
# ---------------------------------------------------------------------------


class TestNoneResetRuntime:
    def test_init_is_noop(self, mock_prog):
        NoneReset("r", NoneResetCfg()).init(mock_prog)
        assert mock_prog.has_no_events()

    def test_run_returns_t(self, mock_prog):
        r = NoneReset("r", NoneResetCfg())
        assert r.run(mock_prog, t=1.5) == 1.5

    def test_total_length_zero(self, mock_prog):
        assert NoneReset("r", NoneResetCfg()).total_length(mock_prog) == 0.0


class TestPulseResetRuntime:
    def test_init_declares_gen(self, mock_prog):
        r = PulseReset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        r.init(mock_prog)
        assert mock_prog.count("declare_gen") == 1

    def test_run_emits_pulse(self, mock_prog):
        r = PulseReset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        assert mock_prog.count("pulse") == 1


class TestTwoPulseResetRuntime:
    def _make(self):
        return TwoPulseReset(
            "r",
            TwoPulseResetCfg(
                pulse1_cfg=_make_pulse_cfg(ch=1, length=0.2),
                pulse2_cfg=_make_pulse_cfg(ch=2, length=0.1),
            ),
        )

    def test_init_declares_both_gens(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        assert mock_prog.count("declare_gen") == 2

    def test_run_emits_both_pulses(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        assert mock_prog.count("pulse") == 2


class TestBathResetRuntime:
    def _make(self):
        return BathReset(
            "r",
            BathResetCfg(
                cavity_tone_cfg=_make_pulse_cfg(ch=1, freq=5000.0, length=5.0),
                qubit_tone_cfg=_make_pulse_cfg(ch=2, freq=4000.0, length=5.0),
                pi2_cfg=_make_pi2_cfg(),
            ),
        )

    def test_init_declares_three_gens(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        assert mock_prog.count("declare_gen") == 3

    def test_run_emits_three_pulses(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        assert mock_prog.count("pulse") == 3
