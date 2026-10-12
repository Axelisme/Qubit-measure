"""Program field descriptions do not change the configuration value contract."""

import pytest

from zcu_tools.program.v2.base import ProgramV2Cfg


@pytest.mark.parametrize("name", ["reps", "rounds", "initial_delay", "relax_delay"])
def test_described_program_fields_keep_load_and_dump_values(name: str) -> None:
    from zcu_tools.cfg_model import get_field_meta

    assert get_field_meta(ProgramV2Cfg.model_fields[name]) is not None

    defaults = {"reps": 1, "rounds": 1, "initial_delay": 1.0, "relax_delay": 1.0}
    assert ProgramV2Cfg().model_dump() == defaults
    assert ProgramV2Cfg.model_validate(defaults).model_dump() == defaults

    supplied = {"reps": 20, "rounds": 7, "initial_delay": 0.25, "relax_delay": 12.5}
    cfg = ProgramV2Cfg.model_validate(supplied)
    assert cfg.model_dump() == supplied
    assert ProgramV2Cfg.model_validate_json(cfg.model_dump_json()).model_dump() == supplied
