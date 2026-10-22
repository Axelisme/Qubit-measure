"""Notebook native restore through synthetic cores, without hardware or widgets."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.datafile import load, save
from zcu_tools.device import DeviceManager
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    CfgUnavailableWarning,
    ExpCfgModel,
    GroupedAxesSpec,
    GroupedLoadData,
    PersistableExperiment,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._native_support import native_metadata


class NativeCfg(ExpCfgModel):
    gain: float = 1.0


@dataclass
class NativeResult:
    frequency: NDArray[np.float64]
    signal: NDArray[np.complex128]
    delay: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    decay: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


class NativeCore(PersistableExperiment[NativeResult, NativeCfg]):
    AXES_SPEC: ClassVar = AxesSpec(
        axes=(Axis("frequency", "Frequency", "Hz", scale=1e6),),
        z=ZSpec("signal", "Signal", "V", scale=1e-3),
        result_type=NativeResult,
        cfg_type=NativeCfg,
        tag="test/notebook-native",
    )

    def run(self, config: NativeCfg, *, context: RunContext) -> NativeResult:
        return NativeResult(
            np.array([4000.0, 5000.0]),
            np.array([2 + 3j, 4 - 5j], dtype=np.complex128) * config.gain,
        )

    def analyze(
        self, source: RunRecord[NativeCfg, NativeResult], options: None, *, plots: Plots
    ) -> float:
        return float(source.result.signal.real.sum())


def build_grouped(data: GroupedLoadData[NativeCfg]) -> NativeResult:
    signal, decay = data.variable("signal"), data.variable("decay")
    return NativeResult(signal.axes[0], signal.z, decay.axes[0], decay.z)


class GroupedCore(NativeCore):
    AXES_SPEC: ClassVar = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "signal",
                (VariableAxisSpec("Frequency", "Hz", "frequency", scale=1e6),),
                VariableZSpec("signal", "Signal", "V", scale=1e-3, dtype=np.complex128),
            ),
            VariableSpec(
                "decay",
                (VariableAxisSpec("Delay", "s", "delay", scale=1e-6),),
                VariableZSpec("decay", "Decay", "", dtype=np.float64),
            ),
        ),
        result_type=NativeResult,
        cfg_type=NativeCfg,
        tag="test/notebook-grouped-native",
        result_builder=build_grouped,
    )

    def run(self, config: NativeCfg, *, context: RunContext) -> NativeResult:
        result = super().run(config, context=context)
        return replace(
            result, delay=np.array([1.0, 2.0, 3.0]), decay=np.array([0.5, 0.3, 0.1])
        )


def assert_loaded_result(result: NativeResult, *, grouped: bool) -> None:
    """Check the synthetic experiment's known memory-unit data."""
    np.testing.assert_allclose(result.frequency, [4000.0, 5000.0])
    np.testing.assert_allclose(result.signal, [4 + 6j, 8 - 10j])
    if grouped:
        np.testing.assert_allclose(result.delay, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.decay, [0.5, 0.3, 0.1])


@pytest.mark.parametrize("grouped", [False, True], ids=["single", "grouped"])
@pytest.mark.parametrize("cfg_unavailable", [False, True], ids=["typed-cfg", "no-cfg"])
def test_native_load_publishes_shared_restore_and_retains_previous_on_failure(
    tmp_path: Path, grouped: bool, cfg_unavailable: bool
) -> None:
    core = GroupedCore() if grouped else NativeCore()
    adapter = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(core)
    previous = adapter.run(NativeCfg(gain=2.0))
    analysis = adapter.analyze(None)
    evidence = adapter.acquisition_for(previous)
    path = tmp_path / "data.h5"
    core.save_run(
        replace(previous, metadata=native_metadata(core.persistence_spec().tag)), path
    )
    stored = load(path)
    if cfg_unavailable:
        stored = replace(
            stored,
            docs={**stored.docs, "cfg": {**stored.docs["cfg"], "values": {"gain": []}}},
        )
        save(path, stored, replace=True)

    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing.h5")
    assert adapter.last_run is previous
    assert adapter.analysis is analysis
    assert adapter.acquisition_for(previous) == evidence

    if cfg_unavailable:
        with pytest.warns(CfgUnavailableWarning) as core_warning:
            expected = core.from_stored(stored).record
        with pytest.warns(CfgUnavailableWarning) as notebook_warning:
            try:
                loaded = adapter.load(path)
            except ValueError as error:
                pytest.fail(f"Notebook rejected valid native data: {error}")
        assert str(notebook_warning[0].message) == str(core_warning[0].message)
        assert loaded.cfg is None
    else:
        expected = core.from_stored(stored).record
        try:
            loaded = adapter.load(path)
        except ValueError as error:
            pytest.fail(f"Notebook rejected valid native data: {error}")
        assert loaded.cfg == NativeCfg(gain=2.0)

    assert adapter.last_run is loaded
    assert adapter.analysis is None
    assert adapter.run_presentation is None
    assert adapter.analysis_presentation is None
    with pytest.raises(ValueError, match="No acquisition evidence"):
        adapter.acquisition_for(loaded)
    for name in ("frequency", "signal", "delay", "decay"):
        np.testing.assert_allclose(
            getattr(loaded.result, name), getattr(expected.result, name)
        )
    assert_loaded_result(loaded.result, grouped=grouped)

    offline = NotebookAdapter(host=NonPresentingHost())(core)
    analyzed = offline.analyze(None, source=loaded)
    assert analyzed.source is loaded
    assert analyzed.result == 12.0
    assert offline.last_run is None

    with pytest.raises(ValueError, match="No acquisition evidence"):
        adapter.acquisition_for(previous)


def test_unknown_historical_evidence_loads_without_inventing_acquisition(
    tmp_path: Path,
) -> None:
    core = NativeCore()
    path = tmp_path / "unknown.h5"
    source = RunRecord(
        cfg=NativeCfg(gain=2.0),
        result=NativeResult(np.array([4000.0, 5000.0]), np.array([4 + 6j, 8 - 10j])),
    )
    metadata = native_metadata(core.persistence_spec().tag).model_copy(
        update={"started_at": None, "completion": "unknown"}
    )
    try:
        core.save_run(replace(source, metadata=metadata), path)
        adapter = NotebookAdapter(host=NonPresentingHost())(core)
        loaded = adapter.load(path)
    except ValueError as error:
        pytest.fail(
            f"Unknown historical evidence must not prevent offline load: {error}"
        )
    assert adapter.last_run is loaded
    assert loaded.cfg == NativeCfg(gain=2.0)
    assert_loaded_result(loaded.result, grouped=False)
    assert adapter.analyze(None).result == 12.0
    with pytest.raises(ValueError, match="No acquisition evidence"):
        adapter.acquisition_for(loaded)
    assert metadata_from_container(load(path)) == metadata


def test_labber_only_export_is_rejected_without_replacing_notebook_results(
    tmp_path: Path,
) -> None:
    adapter = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(NativeCore())
    previous = adapter.run(NativeCfg())
    analysis = adapter.analyze(None)
    evidence = adapter.acquisition_for(previous)
    exported = adapter.save(tmp_path / "export.hdf5")

    with pytest.raises(ValueError, match="format"):
        adapter.load(exported)

    assert adapter.last_run is previous
    assert adapter.analysis is analysis
    assert adapter.acquisition_for(previous) == evidence
