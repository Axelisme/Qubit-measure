"""Shared fixtures for dispersive-fit-gui tests."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.datafile import save_labber_data
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.simulate.fluxonium import FluxoniumParams

from ._entry_fakes import CoupledFluxonium, Fluxonium, Resonator


@pytest.fixture(autouse=True, name="_drain_qt_events")
def isolate_non_ui_qt_events() -> None:
    """Keep non-UI Dispersive tests independent of QApplication."""


@pytest.fixture
def onetone_hdf5(tmp_path):
    """A small 2D one-tone spectrum hdf5 (canonical x=flux, y=freq[Hz], z=signals.T).

    Returns (filepath, dev_values, freqs_GHz, signals[flux, freq]). Mirrors how
    FluxDepExp.save lays out the axes so ``load_data`` + ``format_rawdata`` recover
    GHz frequencies and the (flux, freq) signal grid.
    """
    dev_values = np.linspace(-5.0, 5.0, 16).astype(np.float64)
    freqs_ghz = np.linspace(5.0, 6.0, 40).astype(np.float64)
    rng = np.random.RandomState(0)
    signals = (
        rng.randn(len(dev_values), len(freqs_ghz))
        + 1j * rng.randn(len(dev_values), len(freqs_ghz))
    ).astype(np.complex128)

    filepath = str(tmp_path / "R1_flux_1")
    # Native labber_io: axes are inner-first [inner=x, outer=y]; z is (Ny, Nx).
    # The migrated loader sets dev=axes[0] (x), freq=axes[1] (y), signals=z.T,
    # so we store dev as the inner axis, freq as the outer axis, and z=signals.T
    # with shape (Nfreq, Ndev) — z.T then yields device-major (Ndev, Nfreq).
    save_labber_data(
        filepath,
        z=("Signal", "a.u.", signals.T),
        axes=[
            ("Flux device value", "a.u.", dev_values),  # inner axis (x)
            ("Frequency", "Hz", freqs_ghz * 1e9),  # outer axis (y)
        ],
    )
    return filepath + ".hdf5", dev_values, freqs_ghz, signals


@pytest.fixture
def analysis_entry(tmp_path):
    """An entry with synthetic fluxonium kinds and a nonzero predictor bias."""
    models = {
        "qubit/fluxonium/test_plain": Fluxonium,
        "qubit/fluxonium/test_coupled": CoupledFluxonium,
        "resonator/test_dispersive": Resonator,
    }
    before = {kind: component_registry.get(kind) for kind in component_registry.kinds()}
    try:
        for kind, model in models.items():
            component_registry.register(kind, model)
        entry = ResultEntry.create(
            "synthetic",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component(
            "q1", kind="qubit/fluxonium/test_plain", EJ=4.0, EC=1.0, EL=0.5
        )
        entry.setup.add_component(
            "q2", kind="qubit/fluxonium/test_coupled", EJ=6.0, EC=1.5, EL=0.7
        )
        entry.setup.add_component("r1", kind="resonator/test_dispersive", freq=5750.0)
        entry.save_aliases(EntryAliases(aliases={"qubit": "q1", "resonator": "r1"}))
        save_predictor_file(
            entry.predictors_path / "fit.json",
            PredictorFile(
                model="fluxonium",
                params=FluxoniumParams(
                    EJ=4.0,
                    EC=1.0,
                    EL=0.5,
                    flux_half=0.5,
                    flux_period=-2.0,
                    flux_bias=0.25,
                ),
                flux_unit="V",
                source=PredictorSource(run_id="source-run", created_at=None),
            ),
        )
        yield entry
    finally:
        for kind in models:
            component_registry.unregister(kind)
            if kind in before:
                component_registry.register(kind, before[kind])


@pytest.fixture
def analysis_target(analysis_entry):
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget

    return AnalysisTarget.open(
        analysis_entry.result_path.parent.parent, analysis_entry.result_path.name
    )
