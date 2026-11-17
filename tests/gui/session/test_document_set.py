"""Cached document-set reads and detached capture construction."""

import pytest


@pytest.mark.parametrize("point", [None, "first"])
def test_document_set_queries_keep_installed_content_and_three_revisions(
    selected,
    point: str | None,
) -> None:
    from zcu_tools.gui.session import parameters as parameter_api
    from zcu_tools.resources.entry import document_revision

    _, _, entry = selected
    view = entry.parameters(point)
    library = entry.library(point)
    aliases = entry.aliases(point)
    documents = parameter_api.DocumentSet(view, library, aliases)
    expected = tuple(document_revision(handle) for handle in (view, library, aliases))
    for revision in expected:
        revision.path.unlink(missing_ok=True)
    assert documents.content().value("Q1.rate") == 4300.0
    assert documents.library_content().waveform_keys() == ()
    assert documents.aliases.rows == ()
    assert tuple((item.path, item.revision) for item in documents.revisions()) == tuple(
        (item.path, item.revision) for item in expected
    )
    assert tuple(item.document for item in documents.revisions()) == (
        "parameters",
        "library",
        "aliases",
    )


@pytest.mark.parametrize("point", [None, "first"])
def test_capture_from_set_detaches_content_and_keeps_explicit_storage(
    selected,
    point: str | None,
) -> None:
    from zcu_tools.gui.session import parameters as parameter_api
    from zcu_tools.gui.session.context_types import DocumentSetRef
    from zcu_tools.resources.entry.storage import capture_storage

    _, _, entry = selected
    view = entry.parameters(point)
    documents = parameter_api.DocumentSet(
        view, entry.library(point), entry.aliases(point)
    )
    target = (
        DocumentSetRef(entry_name="entry")
        if point is None
        else DocumentSetRef(entry_name="entry", point=point)
    )
    storage = capture_storage(view)
    captured = parameter_api.DocumentSetCapture.from_set(
        documents,
        target=target,
        roles=None,
        storage=storage,
    )
    view.Q1.rate = 5100.0
    assert captured.value("Q1.rate") == 4300.0
    assert documents.content().value("Q1.rate") == 5100.0
    assert captured.target == target
    assert captured.capture_storage().point == point
    assert captured.capture_storage().entry_name == "entry"
    assert captured.library.waveform_keys() == ()
