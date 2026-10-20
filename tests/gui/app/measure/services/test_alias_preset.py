"""Caller-owned alias presets cross the public measure composition seams."""

from __future__ import annotations

import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.catalog_loader import (
    SourceExperimentCatalogLoader,
    SourcePackage,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.aliases import EntryAliases

from tests.gui._progress_fakes import DirectProgressTransport
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@contextmanager
def composed_context(
    seam: str,
    rows: tuple[AliasRow, ...],
    root: Path,
    qapp: QApplication,
) -> Generator[ContextControlPort]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    # Own the background adapter even when construction rejects the preset.
    background = BackgroundRunner()
    from unittest.mock import patch

    try:
        with patch(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            return_value=background,
        ):
            if seam == "controller":
                controller = Controller(
                    state=state,
                    bus=bus,
                    registry=registry,
                    view=None,
                    project_root=str(root),
                    progress_transport=DirectProgressTransport(),
                    owner_scheduler=QtOwnerScheduler(),
                    alias_presets={"default": rows},
                )
                context = controller.context_control
            else:
                services = build_app_services(
                    state=state,
                    bus=bus,
                    registry=registry,
                    progress_transport=DirectProgressTransport(),
                    resource_versions=state.version.snapshot,
                    render_host=lambda: None,
                    owner_scheduler=QtOwnerScheduler(),
                    project_root=str(root),
                    alias_presets={"default": rows},
                )
                context = services.context_control
        yield context
    finally:
        background.quiesce()
        background.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("seam", ("services", "controller"))
@pytest.mark.parametrize(
    "rows",
    (
        (),
        (
            AliasRow("drive", "Q7"),
            AliasRow("sense", "$drive.resonator"),
            AliasRow("bias", ""),
        ),
    ),
    ids=("empty", "custom"),
)
def test_caller_preset_preserves_order_links_and_unbound_rows(
    seam: str, rows: tuple[AliasRow, ...], tmp_path: Path, qapp: QApplication
) -> None:
    with composed_context(seam, rows, tmp_path, qapp) as context:
        assert context.get_alias_rows() == rows
        context.set_alias_rows((AliasRow("replacement", "R2"),))
        assert context.get_alias_rows() == (AliasRow("replacement", "R2"),)
    # A second composition does not borrow the first session's edited table.
    with composed_context(seam, rows, tmp_path, qapp) as context:
        assert context.get_alias_rows() == rows


@pytest.mark.parametrize("seam", ("services", "controller"))
def test_invalid_caller_preset_fails_at_composition(
    seam: str, tmp_path: Path, qapp: QApplication
) -> None:
    with (
        pytest.raises(InvalidInputError, match="duplicate"),
        composed_context(
            seam, (AliasRow("drive", "Q1"), AliasRow("drive", "Q2")), tmp_path, qapp
        ),
    ):
        pytest.fail("Duplicate caller aliases must not construct a session")


NAMED_PRESETS = {
    "default": (AliasRow("driver", ""), AliasRow("sense", "$driver.sense")),
    "probe": (AliasRow("probe", "Q2"), AliasRow("spare", "")),
}


@pytest.mark.parametrize("saved", [None, (), (AliasRow("saved", "R1"),)])
def test_point_table_replaces_named_default_without_automatic_initialization(
    saved: tuple[AliasRow, ...] | None, tmp_path: Path, qapp: QApplication
) -> None:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "sample",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind="fake/drive/a", sense="R1")
        entry.setup.add_component("R1", kind="fake/sensor")
        entry.new_point("first")
        if saved is not None:
            entry.save_aliases(
                EntryAliases(aliases={row.alias: row.value for row in saved}), point="first"
            )
        try:
            controller = Controller(
                state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
                registry=Registry(),
                view=None,
                bus=BaseEventBus(),
                project_root=str(tmp_path),
                alias_presets=NAMED_PRESETS,
            )
        except TypeError as error:
            pytest.fail(f"Public composition must accept named alias presets: {error}")
        try:
            context = controller.context_control
            assert context.get_alias_rows() == NAMED_PRESETS["default"]
            controller.setup_control.open_entry(EntryRequest("sample"))
            assert context.get_alias_rows() == ()
            context.use_point(PointUseRequest("first"))
            assert context.get_alias_rows() == (() if saved is None else saved)
        finally:
            controller.close()
            qapp.processEvents()
        assert registry_state() == installed
    finally:
        restore_registry(before)


def registry_factory() -> tuple[Registry, SourceExperimentCatalogLoader]:
    registry = Registry()
    loader = SourceExperimentCatalogLoader(
        sources=(SourcePackage("tests.gui.app.measure", Path(__file__).parents[1]),),
        reload_modules=("tests.gui.app.measure._synthetic",),
        preserved_modules=(),
        catalog_module="tests.gui.app.measure._synthetic",
        current_registry=registry,
    )
    return registry, loader


@pytest.mark.parametrize("presets", [NAMED_PRESETS, {"default": ()}, None])
def test_measure_behavior_retains_complete_named_presets(
    presets: dict[str, tuple[AliasRow, ...]] | None,
) -> None:
    original_sys, original_thread = sys.excepthook, threading.excepthook
    try:
        try:
            behavior = MeasureGuiBehavior(registry_factory, alias_presets=presets)
        except TypeError as error:
            pytest.fail(f"Public composition must accept named alias presets: {error}")
        assert behavior.alias_presets == presets
    finally:
        sys.excepthook, threading.excepthook = original_sys, original_thread


@pytest.mark.parametrize("presets", [{}, {"probe": (AliasRow("probe", "Q2"),)}])
def test_measure_behavior_rejects_catalog_without_default(
    presets: dict[str, tuple[AliasRow, ...]],
) -> None:
    original_sys, original_thread = sys.excepthook, threading.excepthook
    try:
        with pytest.raises(ValueError, match="default"):
            MeasureGuiBehavior(registry_factory, alias_presets=presets)
    except TypeError as error:
        pytest.fail(f"Named alias preset admission is missing: {error}")
    finally:
        sys.excepthook, threading.excepthook = original_sys, original_thread
