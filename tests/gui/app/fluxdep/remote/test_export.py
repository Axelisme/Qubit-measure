"""Native export guards, output policy and round trips through the RPC route."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import load_spectrums
from zcu_tools.gui.app.fluxdep.services.export import default_export_path
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.resources.qubit_params import (
    DispersiveFit,
    FluxDepFit,
    ParamsProject,
    QubitParams,
)

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _entry(name: str, *, empty: bool = False, aligned: bool = True) -> SpectrumEntry:
    points = np.array([] if empty else [0.4, 0.8], dtype=np.float64)
    return SpectrumEntry(
        name=name,
        spec_type="OneTone" if empty else "TwoTone",
        raw={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.2, 0.7]),
            "freqs": np.array([4.0, 5.0]),
            "signals": np.array([[1 + 2j, 3j], [4j, 5 - 6j]], dtype=np.complex128),
        },
        points={"dev_values": points, "fluxs": points + 0.2, "freqs": points + 4},
        flux_half=1.4 if empty else 2.4,
        flux_int=2.4 if empty else 3.4,
        flux_period=2.0,
        aligned=aligned,
        points_completed=True,
    )


def _assert_error(reply: RouteReply, code: str, reason: str = "") -> None:
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == code
    if reason:
        assert reply["error"].get("reason") == reason


@pytest.mark.parametrize("overwrite", [False, True])
def test_spectrum_export_roundtrips_native_data_and_leaves_observations_unchanged(
    route_harness: RouteHarness,
    tmp_path: Path,
    overwrite: bool,
) -> None:
    state = route_harness.ctrl.state
    for entry in (_entry("空:star*", empty=True), _entry("nonempty")):
        state.put_spectrum(entry)
    client = route_harness.client()
    assert route_harness.request(client, "project.info")["ok"]
    assert route_harness.request(client, "spectrum.list")["ok"]
    for name in state.spectrums:
        assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    before = state.version.snapshot()
    path = tmp_path / "exported" / "spectrums.h5"
    reply = route_harness.request(
        client, "export.spectrums", filepath=str(path), overwrite=overwrite
    )
    assert reply["ok"]
    assert "result" in reply
    assert reply["result"] == {"filepath": str(path)}
    loaded = load_spectrums(str(path))
    assert set(loaded) == set(state.spectrums)
    for name, original in state.spectrums.items():
        restored = loaded[name]
        assert restored.get("type") == original.spec_type
        assert restored["flux_half"] == original.flux_half
        assert restored["flux_int"] == original.flux_int
        assert restored["flux_period"] == original.flux_period
        for key in ("dev_values", "fluxs", "freqs", "signals"):
            np.testing.assert_array_equal(restored["spectrum"][key], original.raw[key])
        for key in ("dev_values", "fluxs", "freqs"):
            np.testing.assert_array_equal(restored["points"][key], original.points[key])
    assert state.version.snapshot() == before
    # Export adds no new observations, but must not consume existing ones.
    assert route_harness.request(client, "spectrum.reset_points", name="nonempty")["ok"]


def test_export_defaults_create_only_and_explicit_overwrite_replaces_output(
    route_harness: RouteHarness,
    tmp_path: Path,
) -> None:
    state = route_harness.ctrl.state
    result_dir = tmp_path / "native-results"
    route_harness.ctrl.setup_project(
        ProjectInfo(chip_name="chip", qub_name="Q1", result_dir=str(result_dir))
    )
    state.put_spectrum(_entry("source"))
    client = route_harness.client()
    for method in ("project.info", "spectrum.list"):
        assert route_harness.request(client, method)["ok"]
    assert route_harness.request(client, "spectrum.snapshot", name="source")["ok"]
    path = Path(default_export_path(str(result_dir)))
    reply = route_harness.request(client, "export.spectrums")
    assert reply["ok"]
    assert "result" in reply
    assert reply["result"] == {"filepath": str(path)}
    before_bytes = path.read_bytes()

    state.put_spectrum(_entry("source", empty=True))
    assert route_harness.request(client, "spectrum.snapshot", name="source")["ok"]
    reply = route_harness.request(
        client, "export.spectrums", filepath=None, overwrite=None
    )
    _assert_error(reply, "controller_error")
    assert path.read_bytes() == before_bytes
    # Failed I/O neither rolls back nor invalidates a complete observation.
    assert route_harness.request(client, "export.spectrums", overwrite=True)["ok"]
    restored = load_spectrums(str(path))["source"]
    assert restored["points"]["freqs"].size == 0
    assert restored.get("type") == "OneTone"


@pytest.mark.parametrize(
    ("method", "missing"),
    [
        ("export.spectrums", "project"),
        ("export.spectrums", "collection"),
        ("export.spectrums", "empty-leaf"),
        ("fit.export_params", "project"),
        ("fit.export_params", "collection"),
        ("fit.export_params", "empty-leaf"),
        ("fit.export_params", "fit"),
    ],
)
def test_export_requires_complete_owner_inputs_on_each_connection(
    route_harness: RouteHarness,
    tmp_path: Path,
    method: str,
    missing: str,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_entry("empty", empty=True))
    state.set_fit_result((5, 1, 0.5))
    first, second = route_harness.client(), route_harness.client()
    for client in (first, second):
        if client is first or missing != "project":
            assert route_harness.request(client, "project.info")["ok"]
        if client is first or missing != "collection":
            assert route_harness.request(client, "spectrum.list")["ok"]
        if client is first or missing != "empty-leaf":
            assert route_harness.request(client, "spectrum.snapshot", name="empty")[
                "ok"
            ]
        if client is first or missing != "fit":
            assert route_harness.request(client, "fit.result")["ok"]
    output = tmp_path / "blocked" / "output"
    params = {"filepath" if method == "export.spectrums" else "savepath": str(output)}
    reply = route_harness.request(second, method, **params)
    _assert_error(reply, "precondition_failed", "stale_version")
    assert not output.parent.exists()
    assert route_harness.request(first, method, **params)["ok"]


@pytest.mark.parametrize("method", ["export.spectrums", "fit.export_params"])
def test_export_stale_leaf_and_gui_fit_change_require_explicit_reread(
    route_harness: RouteHarness,
    tmp_path: Path,
    method: str,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_entry("empty", empty=True))
    state.put_spectrum(_entry("empty-neighbor"))
    state.set_fit_result((5, 1, 0.5))
    client = route_harness.client()
    for read in ("project.info", "spectrum.list", "fit.result"):
        assert route_harness.request(client, read)["ok"]
    for name in state.spectrums:
        assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    state.put_spectrum(_entry("empty", empty=True))
    output = tmp_path / "output"
    params = {"filepath" if method == "export.spectrums" else "savepath": str(output)}
    assert route_harness.request(client, "resources.versions")["ok"]
    _assert_error(
        route_harness.request(client, method, **params),
        "precondition_failed",
        "stale_version",
    )
    assert not output.exists()
    assert route_harness.request(client, "spectrum.snapshot", name="empty")["ok"]
    if method == "fit.export_params":
        state.set_fit_result((6, 1, 0.5))
        _assert_error(
            route_harness.request(client, method, **params),
            "precondition_failed",
            "stale_version",
        )
        assert not output.exists()
        assert route_harness.request(client, "fit.result")["ok"]
    assert route_harness.request(client, method, **params)["ok"]


@pytest.mark.parametrize(
    ("method", "reason", "with_fit"),
    [
        ("export.spectrums", "no_spectrums", False),
        ("fit.export_params", "no_fit_result", False),
        ("fit.export_params", "no_aligned_spectrum", True),
    ],
)
def test_export_readiness_failure_precedes_output_creation(
    route_harness: RouteHarness,
    tmp_path: Path,
    method: str,
    reason: str,
    with_fit: bool,
) -> None:
    client = route_harness.client()
    if with_fit:
        route_harness.ctrl.state.put_spectrum(_entry("unaligned", aligned=False))
        route_harness.ctrl.state.set_fit_result((5, 1, 0.5))
        assert route_harness.request(client, "spectrum.snapshot", name="unaligned")[
            "ok"
        ]
    for read in ("project.info", "spectrum.list", "fit.result"):
        assert route_harness.request(client, read)["ok"]
    path = tmp_path / "not-created" / "output"
    param = "filepath" if method == "export.spectrums" else "savepath"
    _assert_error(
        route_harness.request(client, method, **{param: str(path)}),
        "precondition_failed",
        reason,
    )
    assert not path.parent.exists()


def test_params_export_merges_sections_and_uses_first_aligned_even_if_empty(
    route_harness: RouteHarness,
    tmp_path: Path,
) -> None:
    state = route_harness.ctrl.state
    result_dir = tmp_path / "results"
    route_harness.ctrl.setup_project(
        ProjectInfo(chip_name="chip", qub_name="Q1", result_dir=str(result_dir))
    )
    state.put_spectrum(_entry("ignored-unaligned", aligned=False))
    state.put_spectrum(_entry("first-aligned", empty=True))
    state.put_spectrum(_entry("later-aligned"))
    state.set_fit_params(
        "db", (1, 2), (2, 3), (3, 4), {"transitions1": [(0, 1)], "r_f": 6.2}, None, None
    )
    state.set_fit_result((5, 1, 0.5))
    path = result_dir / "params.json"
    existing = QubitParams(str(path))
    existing.ensure_project(ParamsProject(chip_name="existing", qub_name="Q0"))
    existing.set_fluxdep_fit(
        FluxDepFit(EJ=3, EC=2, EL=1, flux_half=0, flux_int=1, flux_period=2)
    )
    existing.set_dispersive_fit(DispersiveFit(g=0.2, bare_rf=6.4))
    independent = existing.get_dispersive_fit()

    client = route_harness.client()
    for read in ("project.info", "spectrum.list", "fit.result"):
        assert route_harness.request(client, read)["ok"]
    for name in state.spectrums:
        assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    versions = state.version.snapshot()
    for params in ({}, {"savepath": None}, {"savepath": str(path)}):
        reply = route_harness.request(client, "fit.export_params", **params)
        assert reply["ok"]
        assert "result" in reply
        assert reply["result"] == {"savepath": str(path)}
        restored = QubitParams(str(path), readonly=True)
        assert restored.get_dispersive_fit() == independent
        assert restored.require_project().chip_name == "chip"
        fit = restored.require_fluxdep_fit()
        assert fit.params == (5, 1, 0.5)
        assert fit.flux_half == 1.4
        assert fit.flux_int == 2.4
        assert fit.flux_period == 2
        assert fit.plot_transitions == {"transitions1": [[0, 1]], "r_f": 6.2}
        assert fit.timestamp is not None
    assert state.version.snapshot() == versions
    # Export must not repair a different connection's absent observation.
    other = route_harness.client()
    _assert_error(
        route_harness.request(
            other,
            "fit.set_params",
            database_path="db",
            EJb=[1, 2],
            ECb=[2, 3],
            ELb=[3, 4],
            transitions={},
        ),
        "precondition_failed",
        "stale_version",
    )


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("export.spectrums", {"filepath": 123}),
        ("export.spectrums", {"overwrite": 1}),
        ("fit.export_params", {"savepath": []}),
    ],
)
def test_malformed_export_is_rejected_without_writing(
    route_harness: RouteHarness,
    method: str,
    params: dict[str, object],
) -> None:
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

    client = route_harness.client()
    with pytest.raises(RemoteError) as caught:
        route_harness.request(client, method, **params)
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.version.snapshot() == {}


def test_native_hdf5_failure_keeps_the_successful_prefix_without_cleanup(
    route_harness: RouteHarness,
    tmp_path: Path,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_entry("first"))
    # Opaque names may collide with native HDF5 datasets. The second group
    # cannot replace the first spectrum's flux_half dataset.
    state.put_spectrum(_entry("first/flux_half"))
    client = route_harness.client()
    for read in ("project.info", "spectrum.list"):
        assert route_harness.request(client, read)["ok"]
    for name in state.spectrums:
        assert route_harness.request(client, "spectrum.snapshot", name=name)["ok"]
    before = state.version.snapshot()
    path = tmp_path / "partial.h5"
    _assert_error(
        route_harness.request(client, "export.spectrums", filepath=str(path)),
        "controller_error",
    )
    prefix = load_spectrums(str(path))
    assert list(prefix) == ["first"]
    np.testing.assert_array_equal(
        prefix["first"]["points"]["freqs"], state.spectrums["first"].points["freqs"]
    )
    assert state.version.snapshot() == before
