from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.services import arb_waveform
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from ._helpers import Fixture, dispatch_handler


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001
    yield


@pytest.fixture(scope="module", autouse=True)
def repository_state_guard() -> Iterator[None]:
    before = vars(ArbWaveformDatabase)["_database_path"]
    yield
    assert vars(ArbWaveformDatabase)["_database_path"] == before


@pytest.fixture()
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setattr(ArbWaveformDatabase, "_database_path", None)
    monkeypatch.setattr(arb_waveform, "gettempdir", lambda: str(tmp_path))
    fx = Fixture()
    db_path = tmp_path / "Database" / "chip_a" / "q1" / "2026" / "06" / "Data_0626"
    fx.state.session_env = replace(
        fx.state.session_env,
        chip_name="chip_a",
        qub_name="q1",
        database_path=str(db_path),
    )
    return fx


def _recipe() -> dict[str, object]:
    return {
        "segments": [
            {"duration": 0.001, "formula": "sin(2*pi*t)"},
            {"duration": 0.001, "formula": "I*cos(2*pi*t)"},
        ],
        "normalize": "peak",
    }


def test_set_list_and_preview_round_trip(fixture: Fixture, tmp_path: Path) -> None:
    result = dispatch_handler(
        fixture.ctrl,
        "arb_waveform.set",
        {"name": "arb_wav1", "recipe": _recipe(), "overwrite": False},
    )

    assert result == {"success": True, "status": "created"}
    assert fixture.state.version.get("arb_waveforms") == 1

    asset_path = (
        tmp_path / "Database" / "chip_a" / "q1" / "arb_waveforms" / "arb_wav1.npz"
    )
    assert asset_path.exists()

    listed = dispatch_handler(fixture.ctrl, "arb_waveform.list", {})
    assert listed == {"waveforms": ["arb_wav1"]}

    preview = dispatch_handler(
        fixture.ctrl,
        "arb_waveform.preview",
        {"name": "arb_wav1"},
    )
    assert preview["recipe"] == _recipe()
    assert Path(str(preview["preview_figure"])).exists()
    assert fixture.state.version.get("arb_waveforms") == 1


def test_list_without_project_database_path_returns_empty(fixture: Fixture) -> None:
    fixture.state.session_env = replace(fixture.state.session_env, database_path="")

    assert fixture.ctrl.arb_waveforms.list_data_keys() == []
    assert fixture.ctrl.arb_waveforms.list_infos() == []
    assert dispatch_handler(fixture.ctrl, "arb_waveform.list", {}) == {"waveforms": []}


def test_set_existing_without_overwrite_reports_reason(fixture: Fixture) -> None:
    params = {"name": "arb_wav1", "recipe": _recipe(), "overwrite": False}
    dispatch_handler(fixture.ctrl, "arb_waveform.set", params)

    with pytest.raises(RemoteError) as exc:
        dispatch_handler(fixture.ctrl, "arb_waveform.set", params)

    assert exc.value.code is ErrorCode.PRECONDITION_FAILED
    assert exc.value.reason == "data_key_exists"
    assert exc.value.data == {"data_key": "arb_wav1"}
    assert fixture.state.version.get("arb_waveforms") == 1


def test_preview_missing_name_reports_available(fixture: Fixture) -> None:
    with pytest.raises(RemoteError) as exc:
        dispatch_handler(
            fixture.ctrl,
            "arb_waveform.preview",
            {"name": "missing"},
        )

    assert exc.value.code is ErrorCode.INVALID_PARAMS
    assert exc.value.reason == "data_key_not_found"
    data = exc.value.data
    assert data is not None
    assert data["available"] == []


@pytest.mark.parametrize(
    "recipe",
    [
        {"segments": [{"duration": 0.001, "formula": "unknown(t)"}]},
        {"normalize": True, "segments": [{}]},
        {"normalize": "peak", "segments": [{}]},
    ],
)
def test_set_invalid_recipe_reports_reason(
    fixture: Fixture, recipe: dict[str, object]
) -> None:
    with pytest.raises(RemoteError) as exc:
        dispatch_handler(
            fixture.ctrl,
            "arb_waveform.set",
            {
                "name": "arb_wav1",
                "recipe": recipe,
                "overwrite": False,
            },
        )

    assert exc.value.code is ErrorCode.INVALID_PARAMS
    assert exc.value.reason == "invalid_recipe"
    assert fixture.state.version.get("arb_waveforms") == 0


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("failed_query", ["preview", "inspect"])
def test_save_succeeds_when_a_derived_query_is_unavailable(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    failed_query: str,
) -> None:
    assets = fixture.ctrl.arb_waveforms
    if overwrite:
        assets.set_formula("pulse", _recipe(), overwrite=False)
    before = fixture.state.version.get("arb_waveforms")
    recipe = {"segments": [{"duration": 0.002, "formula": "0.5"}], "normalize": "none"}

    def fail_export(*_args: object, **_kwargs: object) -> str:
        raise OSError("PNG export unavailable")

    monkeypatch.setattr(arb_waveform, "render_preview_png", fail_export)
    if failed_query == "inspect":
        monkeypatch.setattr(ArbWaveformDatabase, "inspect", fail_export)
    result = dispatch_handler(
        fixture.ctrl,
        "arb_waveform.set",
        {"name": "pulse", "recipe": recipe, "overwrite": overwrite},
    )
    assert result == {
        "success": True,
        "status": "overwritten" if overwrite else "created",
    }
    saved = assets.load_data("pulse")
    assert saved.recipe is not None
    assert saved.recipe.to_dict() == recipe
    assert fixture.state.version.get("arb_waveforms") == before + 1

    with pytest.raises(OSError, match="PNG export unavailable"):
        dispatch_handler(fixture.ctrl, "arb_waveform.preview", {"name": "pulse"})
    retained = assets.load_data("pulse")
    assert retained.recipe == saved.recipe
    assert retained.idata.tolist() == saved.idata.tolist()
    assert fixture.state.version.get("arb_waveforms") == before + 1


def test_asset_port_rename_and_delete_update_the_same_revision(
    fixture: Fixture,
) -> None:
    assets = fixture.ctrl.arb_waveforms
    assert assets.set_formula("pulse", _recipe(), overwrite=False) == "created"
    assets.rename("pulse", "renamed")
    assert assets.list_data_keys() == ["renamed"]
    assert [info.data_key for info in assets.list_infos()] == ["renamed"]
    assert fixture.state.version.get("arb_waveforms") == 2
    assets.delete("renamed")
    assert assets.list_data_keys() == []
    assert fixture.state.version.get("arb_waveforms") == 3
