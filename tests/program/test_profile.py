"""Profile-to-QICK contract.

The legacy helper comparison is a migration pin. Remove it when the fixed
helpers are deleted; retain clock, validation and real-compile contracts.
"""

from pathlib import Path

import yaml
from zcu_tools import program
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


def test_yaml_profile_matches_ordinary_compile_config():
    with (Path(__file__).parent / "profiles" / "ordinary.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    actual = program.build_soccfg(profile)
    assert actual.get_cfg() == make_mock_soccfg(2, 1).get_cfg()
