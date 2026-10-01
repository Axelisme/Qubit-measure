"""The Gaussian fake core uses explicit data, cfg and operation plots."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.datafile import load_labber_data, save_labber_data
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.fake import FakeCfg, FakeExp, FakeResult
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg


def make_cfg(noise_scale: float = 0.0) -> FakeCfg:
    return FakeCfg(
        sweep=SweepCfg(start=4.8, stop=5.2, expts=9, step=0.05),
        rounds=3,
        noise_scale=noise_scale,
        round_delay=0.0,
    )


@pytest.mark.parametrize("noise_scale", [0.0, 0.1])
def test_run_preserves_scalar_complex_noise_round_mean_and_final_trace(
    monkeypatch: pytest.MonkeyPatch, noise_scale: float
) -> None:
    cfg = make_cfg(noise_scale)
    before = cfg.model_copy(deep=True)
    rng = np.random.Generator(np.random.PCG64(23))
    monkeypatch.setattr(np.random, "randn", lambda: rng.standard_normal())
    stop = StopSignal()
    wait = Mock(return_value=False)
    monkeypatch.setattr(stop.event, "wait", wait)
    plots = Plots(NonPresentingHost())
    result = FakeExp().run(
        cfg, context=RunContext(None, None, plots, devices={}, cancel_signal=stop)
    )

    freqs = np.linspace(4.8, 5.2, 9)
    clean = np.exp(-((freqs - 5.0) ** 2) / (2 * 0.1**2))
    expected_rng = np.random.Generator(np.random.PCG64(23))
    rounds = [
        clean
        + noise_scale * expected_rng.standard_normal()
        + 1j * noise_scale * expected_rng.standard_normal()
        for _ in range(3)
    ]
    expected = np.mean(rounds, axis=0)
    assert cfg == before
    assert result.signals.dtype == np.complex128
    np.testing.assert_array_equal(result.freqs, freqs)
    np.testing.assert_array_equal(result.signals, expected)
    assert wait.call_count == 3
    wait.assert_called_with(0.0)
    figures = plots.finish()
    assert tuple(figures) == ("measurement",)
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), freqs)
    np.testing.assert_array_equal(line.get_ydata(), np.abs(expected))
    plots.release()


def test_run_uses_detached_cfg_through_all_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg()
    cfg.round_delay = 0.2
    delays: list[float] = []

    def wait(delay: float) -> bool:
        delays.append(delay)
        cfg.noise_scale = 10.0
        cfg.round_delay = 0.4
        return False

    stop = StopSignal()
    monkeypatch.setattr(stop.event, "wait", wait)
    plots = Plots(NonPresentingHost())
    result = FakeExp().run(
        cfg, context=RunContext(None, None, plots, devices={}, cancel_signal=stop)
    )
    assert delays == [0.2, 0.2, 0.2]
    expected = np.exp(-((result.freqs - 5.0) ** 2) / (2 * 0.1**2))
    np.testing.assert_allclose(result.signals, expected, atol=1e-15)
    plots.finish()
    plots.release()


@pytest.mark.parametrize(
    "control",
    [
        {"rounds": 0},
        {"noise_scale": -0.1},
        {"round_delay": -0.1},
    ],
)
def test_invalid_acquisition_controls_fail_before_run(
    control: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        FakeCfg.model_validate(control)


def test_notebook_b_does_not_replace_explicit_a_or_its_native_figures(
    tmp_path: Path,
) -> None:
    core = FakeExp()
    adapter = NotebookAdapter(
        core, devices={}, soc=object(), soccfg=object(), host=NonPresentingHost()
    )
    a = adapter.run(make_cfg())
    b_cfg = make_cfg()
    b_cfg.sweep = SweepCfg(start=5.4, stop=5.8, expts=9, step=0.05)
    b = adapter.run(b_cfg)
    answer = adapter.analyze(None, source=a)
    assert adapter.last_run is b
    assert answer.source is a
    assert answer.options is None and answer.result is None
    assert tuple(answer.figures) == ("fit",)
    line = answer.figures["fit"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), a.result.freqs)
    np.testing.assert_array_equal(line.get_ydata(), np.abs(a.result.signals))
    presentation = adapter.analysis_presentation
    assert presentation is not None
    presentation.release()
    answer.figures["fit"].axes[0].set_title("User annotation")
    answer.figures["fit"].savefig(tmp_path / "native.png")
    assert (tmp_path / "native.png").stat().st_size > 0

    saved = adapter.save(a, tmp_path / "a.hdf5")
    disk = load_labber_data(str(saved))
    assert disk.axes[0].unit == "Hz"
    np.testing.assert_array_equal(disk.axes[0].values, a.result.freqs * 1e6)
    assert np.iscomplexobj(disk.z)
    loaded = adapter.load(saved)
    assert loaded.cfg == a.cfg
    np.testing.assert_array_equal(loaded.result.signals, a.result.signals)
    adapter.load(adapter.save(b, tmp_path / "b.hdf5"))
    after_b = adapter.analyze(None, source=loaded)
    assert after_b.source is loaded
    np.testing.assert_array_equal(
        after_b.figures["fit"].axes[0].lines[0].get_ydata(), np.abs(a.result.signals)
    )
    with pytest.raises(FileExistsError):
        core.save(a, saved)
    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing.hdf5")
    assert adapter.last_run is not None
    np.testing.assert_array_equal(adapter.last_run.result.freqs, b.result.freqs)


def test_nullable_cfg_analysis_succeeds_but_canonical_save_rejects(
    tmp_path: Path,
) -> None:
    source = RunRecord[FakeCfg, FakeResult](
        cfg=None, result=FakeResult(np.array([5.0, 5.1]), np.array([3 + 4j, -5j]))
    )
    adapter = NotebookAdapter(FakeExp(), host=NonPresentingHost())
    answer = adapter.analyze(None, source=source)
    assert answer.source is source and answer.result is None
    np.testing.assert_array_equal(
        answer.figures["fit"].axes[0].lines[0].get_ydata(), [5, 5]
    )
    with pytest.raises(ValueError, match="RunRecord.cfg is None"):
        adapter.save(source, tmp_path / "unknown-cfg.hdf5")


def test_load_keeps_valid_data_when_acquisition_cfg_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "invalid-cfg.hdf5"
    signals = np.array([3 + 4j, -5j])
    save_labber_data(
        str(path),
        z=("Signal", "a.u.", signals),
        axes=[("Frequency", "Hz", np.array([5e6, 5.1e6]))],
        comment=json.dumps({"cfg": {"rounds": 0}}),
    )
    adapter = NotebookAdapter(FakeExp(), host=NonPresentingHost())
    with pytest.warns(UserWarning, match="Failed to validate loaded cfg"):
        source = adapter.load(path)
    assert source.cfg is None and adapter.last_run is source
    np.testing.assert_array_equal(source.result.signals, signals)
    answer = adapter.analyze(None, source=source)
    np.testing.assert_array_equal(
        answer.figures["fit"].axes[0].lines[0].get_ydata(), [5, 5]
    )
    with pytest.raises(ValueError, match="RunRecord.cfg is None"):
        adapter.save(source, tmp_path / "cannot-save.hdf5")


def test_failed_run_retains_previous_source_and_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = NotebookAdapter(
        FakeExp(), devices={}, soc=object(), soccfg=object(), host=NonPresentingHost()
    )
    source = adapter.run(make_cfg())
    answer = adapter.analyze(None, source=source)
    monkeypatch.setattr(
        np.random, "randn", Mock(side_effect=RuntimeError("noise source failed"))
    )
    with pytest.raises(RuntimeError, match="noise source failed"):
        adapter.run(make_cfg(0.1))
    assert adapter.last_run is source and adapter.analysis is answer


def test_bad_analysis_or_noncanonical_load_retains_success(tmp_path: Path) -> None:
    adapter = NotebookAdapter(
        FakeExp(), devices={}, soc=object(), soccfg=object(), host=NonPresentingHost()
    )
    source = adapter.run(make_cfg())
    answer = adapter.analyze(None, source=source)
    bad = RunRecord[FakeCfg, FakeResult](
        cfg=None, result=FakeResult(np.array([5.0, 5.1]), np.array([1j]))
    )
    with pytest.raises(ValueError, match="same first dimension"):
        adapter.analyze(None, source=bad)
    assert adapter.analysis is answer and adapter.last_run is source

    path = tmp_path / "wrong-units.hdf5"
    save_labber_data(
        str(path),
        z=("Signal", "a.u.", source.result.signals),
        axes=[("Frequency", "MHz", source.result.freqs)],
    )
    with pytest.raises(ValueError, match="unit"):
        adapter.load(path)
    assert adapter.analysis is answer and adapter.last_run is source
