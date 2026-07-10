"""The measure adapter surface re-exports the shared cfg identities."""

from __future__ import annotations

from zcu_tools.gui import cfg
from zcu_tools.gui.app.main import adapter


def test_measure_adapter_reexports_shared_cfg_identities() -> None:
    assert adapter.CfgSchema is cfg.CfgSchema
    assert adapter.ScalarSpec is cfg.ScalarSpec
    assert adapter.CfgSectionValue is cfg.CfgSectionValue
    assert adapter.make_default_value is cfg.make_default_value
