"""Tests for dispersive LoadService — onetone hdf5 → OnetoneEntry (GHz)."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.dispersive.services.load import LoadService
from zcu_tools.gui.app.dispersive.state import DispersiveState, FluxoniumInputs
from zcu_tools.resources.predictor_file import PredictorFlux


def _inputs() -> FluxoniumInputs:
    return FluxoniumInputs(
        params=(4.0, 1.0, 0.5),
        flux=PredictorFlux(0.25, {"bias": 2.0}, {"bias": "native"}),
        bare_rf_seed=5.3,
    )


def test_load_onetone_requires_fit_inputs(onetone_hdf5):
    path, *_ = onetone_hdf5
    st = DispersiveState()
    with pytest.raises(RuntimeError, match="fit inputs"):
        LoadService(st).load_onetone(path)


def test_load_onetone_writes_entry_in_ghz(onetone_hdf5):
    path, dev_values, freqs_ghz, signals = onetone_hdf5
    st = DispersiveState()
    st.set_fit_inputs(_inputs())

    name = LoadService(st).load_onetone(path)

    assert name == "R1_flux_1.hdf5"
    assert st.onetone is not None
    raw = st.onetone.raw
    np.testing.assert_allclose(raw["freqs"], freqs_ghz)  # GHz, not Hz
    np.testing.assert_allclose(raw["dev_values"], dev_values)
    assert raw["signals"].shape == signals.shape


def test_load_onetone_derives_flux_from_recorded_predictor(onetone_hdf5):
    path, _dev_values, _freqs, _signals = onetone_hdf5
    st = DispersiveState()
    st.set_fit_inputs(_inputs())

    LoadService(st).load_onetone(path)

    assert st.onetone is not None
    raw = st.onetone.raw
    np.testing.assert_allclose(raw["fluxs"][[0, -1]], [-2.25, 2.75])
    assert st.onetone.device == "unnamed_device"


def test_external_without_snapshot_rejects_multiple_predictor_devices(onetone_hdf5):
    path, *_ = onetone_hdf5
    st = DispersiveState()
    st.set_fit_inputs(
        FluxoniumInputs(
            params=(4.0, 1.0, 0.5),
            flux=PredictorFlux(
                0.25, {"bias": 2.0, "coil": -4.0}, {"bias": "V", "coil": "A"}
            ),
            bare_rf_seed=5.3,
        )
    )
    with pytest.raises(ValueError, match="snapshot.*bias.*coil"):
        LoadService(st).load_onetone(path)
    assert st.onetone is None
