"""Shared fixtures for dispersive-fit-gui tests."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.meta_tool import FluxDepFit, ParamsProject, QubitParams
from zcu_tools.utils.datasaver import save_labber_data


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
def params_json(tmp_path):
    """A params.json holding a fluxdep_fit section (dispersive's hard input).

    Returns (filepath, fluxdep_fit_dict). The fit gives (EJ, EC, EL), the flux
    alignment, and a plot_transitions.r_f (so the bare_rf priority chain has a
    middle tier to exercise).
    """
    fluxdep_fit = {
        "params": {"EJ": 4.0, "EC": 1.0, "EL": 0.5},
        "flux_half": 0.5,
        "flux_int": 1.0,
        "flux_period": 2.0,
        "plot_transitions": {"r_f": 5.3, "transitions": [[0, 1]]},
    }
    path = str(tmp_path / "params.json")
    params = QubitParams(path)
    params.ensure_project(ParamsProject("Q1", "Q1"))
    params.set_fluxdep_fit(
        FluxDepFit(
            EJ=4.0,
            EC=1.0,
            EL=0.5,
            flux_half=0.5,
            flux_int=1.0,
            flux_period=2.0,
            plot_transitions={"r_f": 5.3, "transitions": [[0, 1]]},
        )
    )
    return path, fluxdep_fit
