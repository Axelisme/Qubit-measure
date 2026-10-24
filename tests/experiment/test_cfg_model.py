"""Program-backed experiment inputs keep both schema and update contracts."""

import pytest
from pydantic import ValidationError
from zcu_tools import experiment


def test_program_experiment_cfg_composes_defaults_validation_and_updates():
    class ExperimentCfg(experiment.ProgramExpCfg):
        frequency: float

    cfg = ExperimentCfg.model_validate({"frequency": 500.0, "reps": 10})
    assert cfg.to_dict() == {
        "frequency": 500.0,
        "reps": 10,
        "rounds": 1,
        "initial_delay": 1.0,
        "relax_delay": 1.0,
    }
    updated = cfg.with_updates(rounds=3, relax_delay=2.5)
    assert updated.rounds == 3
    assert updated.relax_delay == 2.5
    assert cfg.rounds == 1
    assert isinstance(cfg, experiment.ExpCfgModel)
    with pytest.raises(ValidationError):
        ExperimentCfg.model_validate({"frequency": 500.0, "reps": "invalid"})
