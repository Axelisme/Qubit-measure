from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load
from zcu_tools.experiment.records import RunRecord

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    native_save_session,
)


def test_base_adapter_load_reads_native_record_and_full_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        path = session.native_data_path

        restored = AnalysisSavingAdapter.restore(load(path))
        record = restored.record

        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.frequency, [1000, 1001, 1002])
        assert restored.cfg_issue is None
