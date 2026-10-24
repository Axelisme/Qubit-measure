from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import Container, load
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._load_support import write_native_file
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
