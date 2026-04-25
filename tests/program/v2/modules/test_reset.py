from unittest.mock import MagicMock

import pytest
from zcu_tools.program.v2.modules.base import ModuleCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.reset import (
    AbsReset,
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


def _pulse_dict(ch=1, freq=4000.0, length=0.2, gain=0.5):
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": length},
        "ch": ch,
        "nqz": 1,
        "freq": freq,
        "gain": gain,
    }


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
# NoneResetCfg
# ---------------------------------------------------------------------------


class TestNoneResetCfg:
    def test_from_dict_basic(self):
        cfg = NoneResetCfg.from_dict({"type": "reset/none"}, _ml())
        assert isinstance(cfg, NoneResetCfg)

    def test_from_dict_via_modulecfg_dispatch(self):
        cfg = ModuleCfg.from_dict({"type": "reset/none"}, _ml())
        assert isinstance(cfg, NoneResetCfg)

    def test_from_raw_instance_passthrough(self):
        existing = NoneResetCfg()
        assert NoneResetCfg.from_raw(existing, _ml()) is existing

    def test_set_param_always_raises(self):
        cfg = NoneResetCfg()
        with pytest.raises(ValueError):
            cfg.set_param("anything", 1.0)


# ---------------------------------------------------------------------------
# PulseResetCfg
# ---------------------------------------------------------------------------


class TestPulseResetCfg:
    def _dict(self):
        return {"type": "reset/pulse", "pulse_cfg": _pulse_dict()}

    def test_from_dict_basic(self):
        cfg = PulseResetCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert cfg.pulse_cfg.ch == 1

    def test_from_dict_pulse_cfg_as_dict(self):
        cfg = PulseResetCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, PulseResetCfg)
        assert cfg.pulse_cfg.freq == 4000.0

    def test_from_dict_pulse_cfg_as_string(self):
        ml = _ml()
        ml.get_module.return_value = _pulse_dict()
        d = {"type": "reset/pulse", "pulse_cfg": "my_pulse"}
        cfg = PulseResetCfg.from_dict(d, ml)
        ml.get_module.assert_called_once_with("my_pulse")
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)

    def test_from_dict_via_modulecfg_dispatch(self):
        cfg = ModuleCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)  # narrowed by isinstance above

    def test_from_dict_pulse_cfg_as_string_type_is_pulsecfg(self):
        ml = _ml()
        ml.get_module.return_value = _pulse_dict()
        cfg = PulseResetCfg.from_dict(
            {"type": "reset/pulse", "pulse_cfg": "my_pulse"}, ml
        )
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)

    def test_from_raw_instance_passthrough(self):
        existing = PulseResetCfg(pulse_cfg=_make_pulse_cfg())
        assert PulseResetCfg.from_raw(existing, _ml()) is existing

    def test_set_param_gain(self):
        cfg = PulseResetCfg(pulse_cfg=_make_pulse_cfg())
        cfg.set_param("gain", 0.9)
        assert cfg.pulse_cfg.gain == 0.9

    def test_set_param_freq(self):
        cfg = PulseResetCfg(pulse_cfg=_make_pulse_cfg())
        cfg.set_param("freq", 5000.0)
        assert cfg.pulse_cfg.freq == 5000.0

    def test_set_param_length(self):
        cfg = PulseResetCfg(pulse_cfg=_make_pulse_cfg(length=0.2))
        cfg.set_param("length", 0.5)
        assert cfg.pulse_cfg.waveform.length == 0.5

    def test_set_param_unknown_raises(self):
        cfg = PulseResetCfg(pulse_cfg=_make_pulse_cfg())
        with pytest.raises(ValueError):
            cfg.set_param("ro_ch", 0)


# ---------------------------------------------------------------------------
# TwoPulseResetCfg
# ---------------------------------------------------------------------------


class TestTwoPulseResetCfg:
    def _dict(self):
        return {
            "type": "reset/two_pulse",
            "pulse1_cfg": _pulse_dict(ch=1, freq=4000.0, length=0.2),
            "pulse2_cfg": _pulse_dict(ch=2, freq=3000.0, length=0.3),
        }

    def test_from_dict_basic(self):
        cfg = TwoPulseResetCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)
        assert isinstance(cfg.pulse2_cfg, PulseCfg)

    def test_from_dict_preserves_individual_channels(self):
        cfg = TwoPulseResetCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, TwoPulseResetCfg)
        assert cfg.pulse1_cfg.ch == 1
        assert cfg.pulse2_cfg.ch == 2

    def test_from_dict_pulse_cfgs_as_strings(self):
        ml = _ml()
        ml.get_module.return_value = _pulse_dict()
        d = {
            "type": "reset/two_pulse",
            "pulse1_cfg": "p1",
            "pulse2_cfg": "p2",
        }
        cfg = TwoPulseResetCfg.from_dict(d, ml)
        assert ml.get_module.call_count == 2
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)

    def test_from_dict_via_modulecfg_dispatch(self):
        cfg = ModuleCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)
        assert isinstance(cfg.pulse2_cfg, PulseCfg)

    def test_from_raw_instance_passthrough(self):
        existing = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        assert TwoPulseResetCfg.from_raw(existing, _ml()) is existing

    def test_set_param_gain1(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        cfg.set_param("gain1", 0.1)
        assert cfg.pulse1_cfg.gain == 0.1

    def test_set_param_gain2(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        cfg.set_param("gain2", 0.2)
        assert cfg.pulse2_cfg.gain == 0.2

    def test_set_param_freq1(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        cfg.set_param("freq1", 5000.0)
        assert cfg.pulse1_cfg.freq == 5000.0

    def test_set_param_freq2(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        cfg.set_param("freq2", 5000.0)
        assert cfg.pulse2_cfg.freq == 5000.0

    def test_set_param_length_sets_both(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(length=0.2),
            pulse2_cfg=_make_pulse_cfg(ch=2, length=0.3),
        )
        cfg.set_param("length", 0.5)
        assert cfg.pulse1_cfg.waveform.length == 0.5
        assert cfg.pulse2_cfg.waveform.length == 0.5

    def test_set_param_unknown_raises(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
        )
        with pytest.raises(ValueError):
            cfg.set_param("unknown", 0)


# ---------------------------------------------------------------------------
# BathResetCfg
# ---------------------------------------------------------------------------


class TestBathResetCfg:
    def _dict(self):
        return {
            "type": "reset/bath",
            "cavity_tone_cfg": _pulse_dict(ch=1, freq=5000.0, length=5.0),
            "qubit_tone_cfg": _pulse_dict(ch=2, freq=4000.0, length=5.0),
            "pi2_cfg": {
                "type": "pulse",
                "waveform": {"style": "gauss", "length": 0.1, "sigma": 0.025},
                "ch": 2,
                "nqz": 1,
                "freq": 4000.0,
                "gain": 0.5,
            },
        }

    def test_from_dict_basic(self):
        cfg = BathResetCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, BathResetCfg)
        assert isinstance(cfg.cavity_tone_cfg, PulseCfg)
        assert isinstance(cfg.qubit_tone_cfg, PulseCfg)
        assert isinstance(cfg.pi2_cfg, PulseCfg)

    def test_from_dict_via_modulecfg_dispatch(self):
        cfg = ModuleCfg.from_dict(self._dict(), _ml())
        assert isinstance(cfg, BathResetCfg)
        assert isinstance(cfg.cavity_tone_cfg, PulseCfg)
        assert isinstance(cfg.qubit_tone_cfg, PulseCfg)
        assert isinstance(cfg.pi2_cfg, PulseCfg)

    def test_from_dict_cfgs_as_strings(self):
        ml = _ml()
        ml.get_module.return_value = _pulse_dict()
        d = {
            "type": "reset/bath",
            "cavity_tone_cfg": "res_pulse",
            "qubit_tone_cfg": "qub_pulse",
            "pi2_cfg": "pi2_pulse",
        }
        cfg = BathResetCfg.from_dict(d, ml)
        assert ml.get_module.call_count == 3
        assert isinstance(cfg, BathResetCfg)
        assert isinstance(cfg.cavity_tone_cfg, PulseCfg)

    def test_from_raw_instance_passthrough(self):
        existing = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1, freq=5000.0),
            qubit_tone_cfg=_make_pulse_cfg(ch=2, freq=4000.0),
            pi2_cfg=_make_pi2_cfg(),
        )
        assert BathResetCfg.from_raw(existing, _ml()) is existing

    def test_set_param_qub_gain(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_pulse_cfg(ch=2),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("qub_gain", 0.1)
        assert cfg.qubit_tone_cfg.gain == 0.1

    def test_set_param_qub_freq(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_pulse_cfg(ch=2, freq=4000.0),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("qub_freq", 4500.0)
        assert cfg.qubit_tone_cfg.freq == 4500.0

    def test_set_param_res_gain(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_pulse_cfg(ch=2),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("res_gain", 0.9)
        assert cfg.cavity_tone_cfg.gain == 0.9

    def test_set_param_res_freq(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1, freq=5000.0),
            qubit_tone_cfg=_make_pulse_cfg(ch=2),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("res_freq", 5500.0)
        assert cfg.cavity_tone_cfg.freq == 5500.0

    def test_set_param_qub_length(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1, length=5.0),
            qubit_tone_cfg=_make_pulse_cfg(ch=2, length=5.0),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("qub_length", 3.0)
        assert cfg.qubit_tone_cfg.waveform.length == 3.0

    def test_set_param_res_length(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1, length=5.0),
            qubit_tone_cfg=_make_pulse_cfg(ch=2, length=5.0),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("res_length", 7.0)
        assert cfg.cavity_tone_cfg.waveform.length == 7.0

    def test_set_param_pi2_phase(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_pulse_cfg(ch=2),
            pi2_cfg=_make_pi2_cfg(),
        )
        cfg.set_param("pi2_phase", 90.0)
        assert cfg.pi2_cfg.phase == 90.0

    def test_set_param_unknown_raises(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_pulse_cfg(ch=2),
            pi2_cfg=_make_pi2_cfg(),
        )
        with pytest.raises(ValueError):
            cfg.set_param("unknown", 0)


# ---------------------------------------------------------------------------
# Reset factory – dispatch
# ---------------------------------------------------------------------------


class TestResetFactory:
    def test_dispatch_none(self):
        r = Reset("r", NoneResetCfg())
        assert isinstance(r.reset, NoneReset)

    def test_dispatch_none_from_none_arg(self):
        r = Reset("r", None)
        assert isinstance(r.reset, NoneReset)

    def test_dispatch_pulse(self):
        r = Reset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        assert isinstance(r.reset, PulseReset)

    def test_dispatch_two_pulse(self):
        r = Reset(
            "r",
            TwoPulseResetCfg(
                pulse1_cfg=_make_pulse_cfg(), pulse2_cfg=_make_pulse_cfg(ch=2)
            ),
        )
        assert isinstance(r.reset, TwoPulseReset)

    def test_dispatch_bath(self):
        r = Reset(
            "r",
            BathResetCfg(
                cavity_tone_cfg=_make_pulse_cfg(ch=1),
                qubit_tone_cfg=_make_pulse_cfg(ch=2),
                pi2_cfg=_make_pi2_cfg(),
            ),
        )
        assert isinstance(r.reset, BathReset)

    def test_unknown_cfg_raises(self):
        class UnknownCfg(AbsResetCfg):
            type: str = "unknown_reset"

        with pytest.raises(ValueError):
            Reset("r", UnknownCfg())  # type: ignore[arg-type]

    def test_name_delegated(self):
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

    def test_bind_reset_duplicate_raises(self):
        class NewCfg(AbsResetCfg):
            type: str = "new_unique_reset"

        @Reset.bind_reset(NewCfg)
        class _NewReset(AbsReset):
            def __init__(self, name: str, cfg: NewCfg) -> None:  # noqa: ARG002
                self.name = name

            def init(self, _prog) -> None: ...  # type: ignore[override]

            def total_length(self, _prog):
                return 0.0

            def run(self, _prog, t=0.0):
                return t

        with pytest.raises(ValueError):

            @Reset.bind_reset(NewCfg)
            class _AnotherReset(AbsReset):
                pass


# ---------------------------------------------------------------------------
# Cfg → Module construction + runtime via mock prog
# ---------------------------------------------------------------------------


class TestNoneResetRuntime:
    def test_init_is_noop(self, mock_prog):
        NoneReset("r", NoneResetCfg()).init(mock_prog)
        mock_prog.assert_not_called()

    def test_run_returns_t(self, mock_prog):
        r = NoneReset("r", NoneResetCfg())
        assert r.run(mock_prog, t=1.5) == 1.5

    def test_total_length_zero(self, mock_prog):
        assert NoneReset("r", NoneResetCfg()).total_length(mock_prog) == 0.0


class TestPulseResetRuntime:
    def test_init_declares_gen(self, mock_prog):
        r = PulseReset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        r.init(mock_prog)
        mock_prog.declare_gen.assert_called_once()

    def test_run_emits_pulse(self, mock_prog):
        r = PulseReset("r", PulseResetCfg(pulse_cfg=_make_pulse_cfg()))
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        mock_prog.pulse.assert_called_once()


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
        assert mock_prog.declare_gen.call_count == 2

    def test_run_emits_both_pulses(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        assert mock_prog.pulse.call_count == 2


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
        assert mock_prog.declare_gen.call_count == 3

    def test_run_emits_three_pulses(self, mock_prog):
        r = self._make()
        r.init(mock_prog)
        r.run(mock_prog, t=0.0)
        assert mock_prog.pulse.call_count == 3
