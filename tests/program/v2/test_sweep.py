import pytest
from zcu_tools.program.v2.sweep import SweepCfg


def test_sweep_consistency_expts_zero():
    with pytest.raises(ValueError, match="expts must be greater than 0"):
        SweepCfg(start=1.0, stop=2.0, expts=0, step=0.1)


def test_sweep_consistency_expts_one_mismatch():
    with pytest.raises(
        ValueError, match="for expts == 1, start and stop must be the same value"
    ):
        SweepCfg(start=1.0, stop=2.0, expts=1, step=0.0)


def test_sweep_consistency_expts_one_step_nonzero():
    with pytest.raises(ValueError, match="for expts == 1, step must be 0"):
        SweepCfg(start=1.0, stop=1.0, expts=1, step=0.1)


def test_sweep_consistency_step_zero():
    with pytest.raises(ValueError, match="step must not be zero when expts > 1"):
        SweepCfg(start=1.0, stop=2.0, expts=5, step=0.0)


def test_sweep_consistency_stop_mismatch():
    with pytest.raises(ValueError, match="invalid sweep setting: stop must satisfy"):
        SweepCfg(start=1.0, stop=6.0, expts=5, step=1.0)  # Expected 5.0


def test_sweep_consistency_valid():
    cfg = SweepCfg(start=1.0, stop=5.0, expts=5, step=1.0)
    assert cfg.start == 1.0
    assert cfg.stop == 5.0
    assert cfg.expts == 5
    assert cfg.step == 1.0

    cfg_one = SweepCfg(start=2.5, stop=2.5, expts=1, step=0.0)
    assert cfg_one.start == 2.5
    assert cfg_one.stop == 2.5
    assert cfg_one.expts == 1
    assert cfg_one.step == 0.0
