"""Core declaration defaults at the public adapter boundary."""

from pathlib import Path

import numpy as np
from zcu_tools.datafile import load
from zcu_tools.gui.app.measure.adapter import Adapter

from tests.experiment._saving_support import SavingCfg, SavingCore
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._native_save_support import SavingAdapter


def test_adapter_restores_from_the_declaration_without_constructing_core(
    tmp_path: Path,
) -> None:
    class NeedsArgs(SavingCore):
        def __init__(self, required: object) -> None:
            raise AssertionError("Restoring a declaration must not construct a core")

    definition = Adapter(
        spec=NeedsArgs.AXES_SPEC,
        make_form=SavingAdapter.make_form,
        run=SavingAdapter.run,
        guide=None,
    )

    path = tmp_path / "data.h5"
    metadata = write_native_file(path, tag="test/saving", gain=2.0)
    restored = definition.restore(load(path))
    assert restored.cfg_issue is None
    assert restored.record.cfg == SavingCfg(gain=2.0)
    assert restored.record.metadata == metadata
    np.testing.assert_array_equal(
        restored.record.result.signal, [2 + 4j, 6 - 8j, 10 + 12j]
    )
