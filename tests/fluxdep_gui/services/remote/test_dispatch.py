"""Tests for the fluxdep remote dispatch handlers + method-spec validation.

Qt-free: the handlers only touch ``adapter.ctrl`` (a real ``Controller``), so a
tiny stub adapter exercises the whole RPC handler surface without a QApplication
or the socket server.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.services.remote.dispatch import (
    _HANDLERS,
    METHOD_REGISTRY,
)
from zcu_tools.fluxdep_gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.fluxdep_gui.services.remote.method_specs import METHOD_SPECS
from zcu_tools.fluxdep_gui.services.remote.param_spec import validate_params
from zcu_tools.fluxdep_gui.state import FluxDepState


class _StubAdapter:
    """Minimal stand-in: handlers reach the façade via ``adapter.ctrl`` only."""

    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl


def _adapter() -> _StubAdapter:
    return _StubAdapter(Controller(FluxDepState()))


def _call(adapter: _StubAdapter, method: str, raw_params: dict) -> dict:
    """Validate params against the spec (as the service does) then dispatch."""
    spec = METHOD_REGISTRY[method]
    params = validate_params(spec.params, raw_params) if spec.params else raw_params
    return dict(spec.handler(adapter, params))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry coherence
# ---------------------------------------------------------------------------


def test_registry_specs_and_handlers_match():
    assert set(_HANDLERS) == set(METHOD_SPECS)
    assert set(METHOD_REGISTRY) == set(METHOD_SPECS)


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def test_project_setup_and_info():
    adapter = _adapter()
    res = _call(
        adapter,
        "project.setup",
        {"chip_name": "Q5_2D", "qub_name": "Q1", "result_dir": "/tmp/out"},
    )
    assert res == {"ok": True}
    info = _call(adapter, "project.info", {})
    assert info["chip_name"] == "Q5_2D"
    assert info["qub_name"] == "Q1"
    # explicit result_dir is kept; omitted database_path is derived from chip/qubit
    assert info["result_dir"] == "/tmp/out"
    assert info["database_path"] == os.path.join("result", "Q5_2D", "Q1")


def test_state_check_reflects_project_and_spectra(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    check = _call(adapter, "state.check", {})
    assert check == {"has_project": False, "spectrum_count": 0, "has_active": False}

    _call(adapter, "project.setup", {"chip_name": "Q5_2D", "qub_name": "Q1"})
    _call(adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"})
    check = _call(adapter, "state.check", {})
    assert check["has_project"] is True
    assert check["spectrum_count"] == 1


# ---------------------------------------------------------------------------
# Spectrum collection
# ---------------------------------------------------------------------------


def test_spectrum_load_list_remove(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    loaded = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )
    name = loaded["name"]
    listing = _call(adapter, "spectrum.list", {})
    assert [s["name"] for s in listing["spectrums"]] == [name]
    assert listing["spectrums"][0]["spec_type"] == "OneTone"
    assert listing["spectrums"][0]["aligned"] is False

    _call(adapter, "spectrum.set_active", {"name": name})
    assert adapter.ctrl.state.active_spectrum == name

    removed = _call(adapter, "spectrum.remove", {"name": name})
    assert removed == {"ok": True}
    assert _call(adapter, "spectrum.list", {})["spectrums"] == []


def test_spectrum_load_rejects_bad_spec_type(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "spectrum.load", {"filepath": filepath, "spec_type": "Bogus"})
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_spectrum_set_active_unknown_name():
    adapter = _adapter()
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "spectrum.set_active", {"name": "nope"})
    assert exc.value.code is ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# Alignment / points / selection
# ---------------------------------------------------------------------------


def test_alignment_and_points_set(spectrum_hdf5):
    filepath, dev_values, _freqs_ghz, _ = spectrum_hdf5
    # set_points takes paired (dev_value, freq) selected points: equal-length.
    point_freqs = [5.0 + 0.1 * i for i in range(len(dev_values))]
    adapter = _adapter()
    name = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )["name"]

    _call(adapter, "alignment.set", {"name": name, "flux_half": 0.0, "flux_int": 1.0})
    assert adapter.ctrl.state.spectrums[name].aligned is True

    _call(
        adapter,
        "points.set",
        {
            "name": name,
            "dev_values": dev_values.tolist(),
            "freqs": point_freqs,
        },
    )
    assert adapter.ctrl.state.spectrums[name].points_selected is True

    cloud = _call(adapter, "selection.pointcloud", {})
    assert len(cloud["fluxs"]) == len(cloud["freqs"]) == len(dev_values)

    mask = [True] * len(cloud["fluxs"])
    res = _call(adapter, "selection.set", {"selected": mask})
    assert res == {"ok": True}


def test_selection_set_rejects_wrong_length(spectrum_hdf5):
    filepath, dev_values, _freqs_ghz, _ = spectrum_hdf5
    point_freqs = [5.0 + 0.1 * i for i in range(len(dev_values))]
    adapter = _adapter()
    name = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )["name"]
    _call(
        adapter,
        "points.set",
        {"name": name, "dev_values": dev_values.tolist(), "freqs": point_freqs},
    )
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "selection.set", {"selected": [True, False]})
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_points_set_rejects_non_array(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    name = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )["name"]
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "points.set", {"name": name, "dev_values": "x", "freqs": [1.0]})
    assert exc.value.code is ErrorCode.INVALID_PARAMS


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_empty_collection_precondition():
    adapter = _adapter()
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "export.spectrums", {})
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_export_writes_file(spectrum_hdf5, tmp_path):
    filepath, dev_values, _freqs_ghz, _ = spectrum_hdf5
    point_freqs = [5.0 + 0.1 * i for i in range(len(dev_values))]
    adapter = _adapter()
    name = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )["name"]
    _call(adapter, "alignment.set", {"name": name, "flux_half": 0.0, "flux_int": 1.0})
    _call(
        adapter,
        "points.set",
        {"name": name, "dev_values": dev_values.tolist(), "freqs": point_freqs},
    )
    out = str(tmp_path / "spectrums.hdf5")
    res = _call(adapter, "export.spectrums", {"filepath": out})
    assert res["path"] == out


# ---------------------------------------------------------------------------
# resources.versions
# ---------------------------------------------------------------------------


def test_resources_versions_snapshot(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    _call(adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"})
    versions = _call(adapter, "resources.versions", {})["versions"]
    assert isinstance(versions, dict)
    assert any(k.startswith("spectrum:") for k in versions)


def test_spectrum_load_transpose_axes(transposed_spectrum_hdf5):
    import numpy as np

    filepath, flux, freqs_ghz, _signals = transposed_spectrum_hdf5
    adapter = _adapter()
    loaded = _call(
        adapter,
        "spectrum.load",
        {"filepath": filepath, "spec_type": "OneTone", "transpose_axes": True},
    )
    entry = adapter.ctrl.state.spectrums[loaded["name"]]
    np.testing.assert_allclose(entry.raw["dev_values"], flux)
    np.testing.assert_allclose(entry.raw["freqs"], freqs_ghz)


def test_spectrum_load_transpose_defaults_false(spectrum_hdf5):
    # omitting transpose_axes must behave as before (validate_params fills False)
    filepath, dev_values, *_ = spectrum_hdf5
    adapter = _adapter()
    import numpy as np

    loaded = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"}
    )
    entry = adapter.ctrl.state.spectrums[loaded["name"]]
    np.testing.assert_allclose(entry.raw["dev_values"], dev_values)


def test_spectrum_load_processed_roundtrip(spectrum_hdf5, tmp_path):
    import numpy as np

    # build + export via one adapter, restore via spectrum.load_processed in another
    filepath, *_ = spectrum_hdf5
    a1 = _adapter()
    loaded = _call(a1, "spectrum.load", {"filepath": filepath, "spec_type": "OneTone"})
    name = loaded["name"]
    _call(a1, "alignment.set", {"name": name, "flux_half": 0.0, "flux_int": 1.0})
    _call(
        a1,
        "points.set",
        {"name": name, "dev_values": [0.0, 2.0], "freqs": [5.0, 5.5]},
    )
    out = str(tmp_path / "spectrums.hdf5")
    _call(a1, "export.spectrums", {"filepath": out})

    a2 = _adapter()
    res = _call(a2, "spectrum.load_processed", {"filepath": out})
    assert res["names"] == [name]
    entry = a2.ctrl.state.spectrums[name]
    assert entry.aligned and entry.points_selected
    np.testing.assert_allclose(entry.points["freqs"], [5.0, 5.5])


# ---------------------------------------------------------------------------
# Database-search fit (v2)
# ---------------------------------------------------------------------------


@pytest.fixture
def _fit_db(tmp_path):
    """A tiny synthetic fluxonium database for fit.search dispatch tests."""
    import h5py

    M, L = 21, 4
    fluxs = np.linspace(0.0, 0.5, M).astype(np.float64)
    params = np.array(
        [[3.0, 1.0, 0.5], [5.0, 1.2, 0.4], [6.0, 0.9, 0.6]], dtype=np.float64
    )
    energies = np.zeros((len(params), M, L), dtype=np.float64)
    for n, (EJ, EC, EL) in enumerate(params):
        for lvl in range(L):
            energies[n, :, lvl] = lvl * (EC + EL) + EJ * np.cos(2 * np.pi * fluxs) * 0.1
    path = tmp_path / "fit_db.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fluxs", data=fluxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)
    return str(path)


def _seed_points(adapter, spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    loaded = _call(
        adapter, "spectrum.load", {"filepath": filepath, "spec_type": "TwoTone"}
    )
    name = loaded["name"]
    _call(adapter, "alignment.set", {"name": name, "flux_half": 0.0, "flux_int": 0.5})
    _call(
        adapter,
        "points.set",
        {
            "name": name,
            "dev_values": [0.0, 1.0, 2.0, 3.0],
            "freqs": [5.0, 5.1, 5.2, 5.3],
        },
    )
    return name


def test_fit_set_params_and_result(_fit_db, spectrum_hdf5):
    adapter = _adapter()
    _seed_points(adapter, spectrum_hdf5)
    _call(
        adapter,
        "fit.set_params",
        {
            "database_path": _fit_db,
            "EJb": [0.1, 50.0],
            "ECb": [0.01, 10.0],
            "ELb": [0.01, 10.0],
            "transitions": {"transitions": [[0, 1], [0, 2]]},
            "r_f": 5.5,
        },
    )
    res = _call(adapter, "fit.result", {})
    assert res["has_result"] is False
    assert res["params"] is None
    assert res["database_path"] == _fit_db
    assert res["transitions"]["transitions"] == [[0, 1], [0, 2]]
    # r_f injected into transitions and surfaced on the result
    assert res["r_f"] == 5.5


def test_fit_search_returns_params_and_records(_fit_db, spectrum_hdf5):
    adapter = _adapter()
    _seed_points(adapter, spectrum_hdf5)
    _call(
        adapter,
        "fit.set_params",
        {
            "database_path": _fit_db,
            "EJb": [0.1, 50.0],
            "ECb": [0.01, 10.0],
            "ELb": [0.01, 10.0],
            "transitions": {"transitions": [[0, 1], [0, 2]]},
        },
    )
    res = _call(adapter, "fit.search", {})
    assert set(res) == {"EJ", "EC", "EL"}
    # result now recorded on State
    assert adapter.ctrl.state.fit.has_result
    after = _call(adapter, "fit.result", {})
    assert after["has_result"] is True
    assert after["params"]["EJ"] == res["EJ"]


def test_fit_search_fast_fails_without_db(spectrum_hdf5):
    adapter = _adapter()
    _seed_points(adapter, spectrum_hdf5)
    with pytest.raises(RemoteError) as exc:
        _call(adapter, "fit.search", {})
    assert exc.value.code == ErrorCode.PRECONDITION_FAILED


def test_fit_set_params_rejects_bad_transitions(_fit_db):
    adapter = _adapter()
    with pytest.raises(RemoteError) as exc:
        _call(
            adapter,
            "fit.set_params",
            {
                "database_path": _fit_db,
                "EJb": [0.1, 50.0],
                "ECb": [0.01, 10.0],
                "ELb": [0.01, 10.0],
                "transitions": {"transitions": [[0, 1, 2]]},  # bad pair
            },
        )
    assert exc.value.code == ErrorCode.INVALID_PARAMS


def test_fit_export_params_roundtrip(tmp_path, _fit_db, spectrum_hdf5):
    adapter = _adapter()
    _call(adapter, "project.setup", {"chip_name": "Q9", "qub_name": "Q1"})
    _seed_points(adapter, spectrum_hdf5)
    _call(
        adapter,
        "fit.set_params",
        {
            "database_path": _fit_db,
            "EJb": [0.1, 50.0],
            "ECb": [0.01, 10.0],
            "ELb": [0.01, 10.0],
            "transitions": {"transitions": [[0, 1]]},
        },
    )
    _call(adapter, "fit.search", {})
    out = str(tmp_path / "params.json")
    res = _call(adapter, "fit.export_params", {"savepath": out})
    assert res["path"] == out

    from zcu_tools.notebook.persistance import load_result

    loaded = load_result(out)
    assert loaded.get("fluxdep_fit") is not None


def test_fit_set_params_accepts_null_freqs(_fit_db, spectrum_hdf5):
    adapter = _adapter()
    _seed_points(adapter, spectrum_hdf5)
    # omit r_f/sample_f entirely → stored as None (not 0.0)
    _call(
        adapter,
        "fit.set_params",
        {
            "database_path": _fit_db,
            "EJb": [0.1, 50.0],
            "ECb": [0.01, 10.0],
            "ELb": [0.01, 10.0],
            "transitions": {"transitions": [[0, 1], [0, 2]]},
        },
    )
    res = _call(adapter, "fit.result", {})
    assert res["r_f"] is None
    assert res["sample_f"] is None
    assert adapter.ctrl.state.fit.r_f is None


def test_fit_set_params_explicit_null(_fit_db, spectrum_hdf5):
    adapter = _adapter()
    _seed_points(adapter, spectrum_hdf5)
    _call(
        adapter,
        "fit.set_params",
        {
            "database_path": _fit_db,
            "EJb": [0.1, 50.0],
            "ECb": [0.01, 10.0],
            "ELb": [0.01, 10.0],
            "transitions": {"transitions": [[0, 1]]},
            "r_f": 5.5,
            "sample_f": None,
        },
    )
    assert adapter.ctrl.state.fit.r_f == 5.5
    assert adapter.ctrl.state.fit.sample_f is None
