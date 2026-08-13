from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.experiment.v2.jpa.jpa_auto_optimize as jpa_mod
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.v2.jpa.jpa_auto_optimize import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
    JPAOptModuleCfg,
    JPAOptSweepCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules import ConstWaveformCfg


def _pulse(*, ch: int = 0, freq: float = 3000.0, gain: float = 0.3) -> PulseCfg:
    return PulseCfg(
        waveform=ConstWaveformCfg(length=1.0),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _cfg() -> JPAOptCfg:
    return JPAOptCfg(
        reps=1,
        rounds=2,
        dev={
            "jpa_flux_dev": FakeDeviceInfo(address="fake_flux", label="jpa_flux_dev"),
            "jpa_rf_dev": FakeDeviceInfo(address="fake_rf", label="jpa_rf_dev"),
        },
        modules=JPAOptModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=0, freq=3000.0, gain=0.3),
            readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0, gen_ch=0),
        ),
        sweep=JPAOptSweepCfg(
            jpa_flux=SweepCfg(start=0.1, stop=0.3, expts=3, step=0.1),
            jpa_freq=SweepCfg(start=7200.0, stop=7240.0, expts=5, step=10.0),
            jpa_power=SweepCfg(start=-20.0, stop=-10.0, expts=3, step=5.0),
        ),
    )


@pytest.mark.parametrize("num_points", [-1, 0, 1, 2, 3])
def test_auto_optimize_rejects_num_points_below_four_before_sampling_or_device_setup(
    monkeypatch: pytest.MonkeyPatch, num_points: int
) -> None:
    def boom(*args, **kwargs) -> None:
        raise AssertionError("auto-optimize reached sampling or device setup")

    monkeypatch.setattr(jpa_mod, "JPAOptimizer", boom)  # LHS sampling happens here
    monkeypatch.setattr(jpa_mod, "setup_devices", boom)  # device setup happens here
    monkeypatch.setattr(jpa_mod, "Schedule", boom)  # scan/acquire orchestration

    exp = AutoOptimizeExp()
    with pytest.raises(ValueError, match="num_points"):
        exp.run(None, None, _cfg(), num_points=num_points)
    assert exp.last_result is None


def test_auto_optimize_valid_budget_keeps_run_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls: list[int] = []
    monkeypatch.setattr(
        jpa_mod, "setup_devices", lambda _cfg, **kw: setup_calls.append(1)
    )
    monkeypatch.setattr(jpa_mod, "instant_plot", lambda _fig: None)
    monkeypatch.setattr(
        jpa_mod, "snr_as_signal", lambda _raw, ge_axis=0, **kw: np.array(5.0)
    )

    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)
    exp = AutoOptimizeExp()
    result = exp.run(soc, soccfg, _cfg(), num_points=4)

    assert isinstance(result, JPAOptimizeResult)
    assert result.params.shape == (4, 3)
    assert result.phases.shape == (4,)
    assert result.signals.shape == (4,)
    assert np.all(np.isfinite(result.signals))
    assert exp.last_result is result
    assert result.cfg_snapshot is not None
    assert len(setup_calls) >= 1
