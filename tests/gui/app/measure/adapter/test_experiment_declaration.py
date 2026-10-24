"""Core declaration defaults at the public adapter boundary."""

from pathlib import Path

import numpy as np
from zcu_tools.datafile import load
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._load_support import write_native_file


def test_adapter_defaults_to_its_core_tag_and_preserves_an_override() -> None:
    class DefaultTag(BaseAdapter[SavingCfg, SavingResult, None, None]):
        exp_cls = SavingCore
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE, load_data=False)

    class CustomTag(DefaultTag):
        experiment_tag = "custom/tag"

    assert DefaultTag().experiment_tag == "test/saving"
    assert CustomTag().experiment_tag == "custom/tag"


def test_adapter_restores_from_the_declaration_without_constructing_core(
    tmp_path: Path,
) -> None:
    class NeedsArgs(SavingCore):
        def __init__(self, required: object) -> None:
            raise AssertionError("Restoring a declaration must not construct a core")

    class Adapter(BaseAdapter[SavingCfg, SavingResult, None, None]):
        exp_cls = NeedsArgs
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE, load_data=True)

    path = tmp_path / "data.h5"
    metadata = write_native_file(path, tag="test/saving", gain=2.0)
    restored = Adapter().restore(load(path))
    assert restored.cfg_issue is None
    assert restored.record.cfg == SavingCfg(gain=2.0)
    assert restored.record.metadata == metadata
    np.testing.assert_array_equal(
        restored.record.result.signal, [2 + 4j, 6 - 8j, 10 + 12j]
    )
