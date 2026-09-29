"""GUI T1 adapter hands one typed run and analysis to the shared core."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Analysis, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.time_domain.t1 import (
    T1Adapter,
    T1AnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    LoadDataRequest,
    RunRequest,
    SaveDataRequest,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def _result() -> T1Result:
    times = np.linspace(0, 100, 101)
    signals = (0.2 + 0.8 * np.exp(-times / 20)).astype(np.complex128)
    return T1Result(times, signals)


def test_t1_gui_analysis_preserves_core_numbers_and_named_fit() -> None:
    result = _result()
    plots = Plots(NonPresentingHost())
    answer = T1Adapter().analyze(
        AnalyzeRequest(
            result, T1AnalyzeParams(skip=3), md=Mock(), ml=Mock(), predictor=None
        ),
        plots=plots,
    )
    plots.finish()
    assert isinstance(answer.analysis, T1Analysis)
    assert answer.t1 == pytest.approx(20, rel=0.01)
    assert answer.t1_err == answer.analysis.t1_err
    assert answer.to_summary_dict()["t1"] == answer.t1
    assert tuple(plots) == ("fit",)
    np.testing.assert_array_equal(
        plots["fit"].axes[0].lines[0].get_xdata(), result.times[3:]
    )
    plots.release()


def test_t1_gui_run_passes_captured_context_and_formal_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = T1Adapter()
    cfg = Mock()
    monkeypatch.setattr(adapter, "build_exp_cfg", lambda _raw, _req: cfg)
    result = _result()
    observed: list[tuple[object, QickContext]] = []

    def run(_exp: T1Exp, config: object, *, context: QickContext) -> T1Result:
        observed.append((config, context))
        return result

    monkeypatch.setattr(T1Exp, "run", run)
    plots = Plots(NonPresentingHost())
    req = RunRequest(soc=Mock(), soccfg=Mock(), device_snapshot={})
    assert adapter.run(req, {"uniform": False}, plots=plots) is result
    assert observed[0][0] is cfg
    assert observed[0][1].soc is req.soc
    assert observed[0][1].soccfg is req.soccfg
    assert observed[0][1].plots is plots
    plots.finish(present=False)
    plots.release()


def test_t1_gui_save_and_load_delegate_canonical_result_first_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _result()
    observed: list[tuple[T1Result, Path, str | None]] = []

    def save(
        _exp: T1Exp, data: T1Result, path: Path, *, comment: str | None = None
    ) -> None:
        observed.append((data, path, comment))

    def load(_exp: T1Exp, path: Path) -> T1Result:
        assert path == tmp_path / "data.hdf5"
        return result

    monkeypatch.setattr(T1Exp, "save", save)
    monkeypatch.setattr(T1Exp, "load", load)
    adapter = T1Adapter()
    path = str(tmp_path / "data.hdf5")
    adapter.save(
        SaveDataRequest(
            result,
            path,
            md=Mock(),
            ml=Mock(),
            chip_name="c",
            qub_name="q",
            res_name="r",
            active_label="a",
            comment="note",
        )
    )
    assert observed == [(result, Path(path), "note")]
    assert adapter.load(LoadDataRequest(path, md=Mock(), ml=Mock())) is result
