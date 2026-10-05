"""SA readout-frequency acquisition and native amplitude presentation."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.zcu_lab.v2.onetone._support import make_sa_cfg
from zcu_lab.v2.onetone.sa.core import SA_FreqCfg, SA_FreqExp, SA_FreqResult


def test_mock_soc_acquisition_preserves_readout_cfg_and_publishes_final_trace() -> None:
    cfg = make_sa_cfg()
    before = cfg.model_copy(deep=True)
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    result = SA_FreqExp().run(
        cfg,
        context=RunContext(soc, soccfg, plots, devices={}, cancel_signal=StopSignal()),
    )
    figures = plots.finish()
    assert cfg == before
    assert result.signals.shape == result.freqs.shape == (9,)
    assert np.all(np.isfinite(result.signals))
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), result.freqs)
    np.testing.assert_allclose(np.asarray(line.get_ydata()), np.abs(result.signals))
    plots.release()


def test_nullable_source_analysis_returns_none_and_keeps_native_figure(
    tmp_path: Path,
) -> None:
    result = SA_FreqResult(np.array([1.0, 2.0, 3.0]), np.array([3 + 4j, -5j, 12j]))
    source = RunRecord[SA_FreqCfg, SA_FreqResult](cfg=None, result=result)
    plots = Plots(NonPresentingHost())
    answer = SA_FreqExp().analyze(source, None, plots=plots)
    figures = plots.finish()
    assert answer is None
    line = figures["fit"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_ydata(), [5.0, 5.0, 12.0])
    np.testing.assert_array_equal(line.get_xdata(), result.freqs)
    plots.release()
    figures["fit"].savefig(tmp_path / "sa.png")
    assert (tmp_path / "sa.png").stat().st_size > 0
    with pytest.raises(ValueError, match="cfg"):
        SA_FreqExp().save(source, tmp_path / "missing-cfg.hdf5")


def test_canonical_roundtrip_preserves_hz_and_complex(tmp_path: Path) -> None:
    source = RunRecord(
        cfg=make_sa_cfg(),
        result=SA_FreqResult(np.array([5970.0, 6000.0]), np.array([1 + 2j, 3 - 4j])),
    )
    core = SA_FreqExp()
    path = tmp_path / "sa.hdf5"
    core.save(source, path)
    loaded = core.load(path)
    assert loaded.cfg == source.cfg
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    np.testing.assert_allclose(loaded.result.freqs, source.result.freqs)
    np.testing.assert_allclose(
        load_labber_data(str(path)).axes[0].values, source.result.freqs * 1e6
    )
