"""Entry-scoped predictor selection through the shared public control seam."""

import errno
import os
from pathlib import Path
from typing import NoReturn

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.transmon import TransmonParams


@pytest.fixture
def files(tmp_path: Path):
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    control = PredictorControlFacet(
        PredictorService(ContextControlFacet(context=context))
    )
    entry = ResultEntry.create(
        "files",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    return context, control, entry


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "predictor scan denied"),
        OSError(errno.EIO, "predictor scan failed"),
    ],
    ids=["permission-denied", "io-error"],
)
def test_list_files_propagates_scan_failures(
    files, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    context, control, entry = files
    context.install_entry(entry)
    entry.predictors_path.mkdir()

    def fail_scan(path: object) -> NoReturn:
        raise error

    with monkeypatch.context() as filesystem:
        filesystem.setattr(os, "scandir", fail_scan)
        with pytest.raises(type(error), match="predictor scan"):
            control.list_files()


def test_list_files_rejects_predictor_directory_is_file(files) -> None:
    context, control, entry = files
    context.install_entry(entry)
    entry.predictors_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        control.list_files()


def load_outcome(
    control: PredictorControlFacet, filename: str
) -> PredictorFile | Exception:
    """Observe either a returned file or the public failure without masking its type."""
    try:
        return control.load_file(filename)
    except (
        NotImplementedError,
        FailedPreconditionError,
        InvalidInputError,
        OSError,
        ValueError,
    ) as error:
        return error


def test_files_follow_open_entry_without_point_selection(files, tmp_path: Path):
    context, control, entry = files
    assert control.list_files() == ()
    context.install_entry(entry)
    assert control.list_files() == ()
    assert not entry.predictors_path.exists()
    entry.predictors_path.mkdir()
    (entry.predictors_path / "z.json").write_text("{}", encoding="utf-8")
    (entry.predictors_path / "a.json").write_text("{}", encoding="utf-8")
    (entry.predictors_path / "notes.txt").write_text("notes", encoding="utf-8")
    (entry.predictors_path / "folder.json").mkdir()
    assert control.list_files() == ("a.json", "z.json")
    second = ResultEntry.create(
        "second",
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    context.install_entry(second)
    assert control.list_files() == ()
    assert not second.predictors_path.exists()


@pytest.mark.parametrize(
    "params", [FluxoniumParams(EJ=4.5), TransmonParams(EJmax=17.0, d=0.1)]
)
def test_load_file_preserves_partial_model_and_metadata_without_writes(files, params):
    context, control, entry = files
    context.install_entry(entry)
    entry.predictors_path.mkdir()
    expected = PredictorFile(
        params.model,
        params,
        PredictorFlux(0.2, {"coil": -2.0, "bias": 4.0}, {"coil": "mA", "bias": "V"}),
        PredictorSource("run-1", None),
    )
    path = entry.predictors_path / "trial.json"
    save_predictor_file(path, expected)
    before = path.read_bytes()
    assert load_outcome(control, "trial.json") == expected
    assert path.read_bytes() == before
    assert context.get_session_env().parameters is None


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.json",
        "/tmp/escape.json",
        "nested/a.json",
        "nested\\a.json",
        ".",
        "",
        "trial.txt",
    ],
)
def test_load_file_rejects_non_entry_filenames(files, filename):
    context, control, entry = files
    context.install_entry(entry)
    assert isinstance(load_outcome(control, filename), InvalidInputError)


def test_load_file_requires_open_entry(files):
    _, control, _ = files
    assert isinstance(load_outcome(control, "trial.json"), FailedPreconditionError)


@pytest.mark.parametrize(
    "filename, content, error",
    [
        ("missing.json", None, FileNotFoundError),
        ("bad.json", "not json", ValueError),
        ("invalid.json", "{}", ValueError),
    ],
)
def test_load_file_propagates_owner_failures(files, filename, content, error):
    context, control, entry = files
    context.install_entry(entry)
    entry.predictors_path.mkdir()
    if content is not None:
        (entry.predictors_path / filename).write_text(content, encoding="utf-8")
    assert isinstance(load_outcome(control, filename), error)
