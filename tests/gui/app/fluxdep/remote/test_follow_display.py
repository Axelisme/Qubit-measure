"""Remote command outcomes projected into the visible Fluxdep window."""

from pathlib import Path

import pytest
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.resources.entry import ResultEntry

from tests.gui._user_input import ComponentDriver
from tests.gui.app.fluxdep._components import synthetic_flux_context
from tests.gui.app.fluxdep._search_result import perfect_fit_result
from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _result(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] and "result" in reply, reply
    return reply["result"]


def _context_id(reply: RouteReply) -> int:
    context = _result(reply)["context"]
    assert isinstance(context, dict)
    identifier = context["context_id"]
    assert isinstance(identifier, int)
    return identifier


def _active(driver: ComponentDriver) -> list[str]:
    return [row.text.splitlines()[0] for row in driver.list_rows() if row.selected]


def _spectra(driver: ComponentDriver, name: str, *, stage: str | None = None) -> None:
    assert driver.tabs("Spectra").current == "Spectra"
    assert _active(driver) == [name]
    for item in ("align", "sample"):
        assert driver.observe(object_name=f"fluxdep.stage.{item}").checked == (
            item == stage
        )
    if stage is not None:
        finish = driver.observe(object_name="fluxdep.picker.finish")
        assert finish.visible and not finish.enabled
    else:
        assert [axis.xlabel for axis in driver.plots().axes] == ["Device value"]


def _fit(driver: ComponentDriver, stage: str) -> None:
    assert driver.tabs("Fit").current == "Fit"
    assert driver.observe(object_name=f"fluxdep.fit.stage.{stage}").checked


def _select_spectrum(driver: ComponentDriver, name: str) -> None:
    caption = next(
        row.text for row in driver.list_rows() if row.text.startswith(f"{name}\n")
    )
    driver.select_list_row(caption)


def _picker(h: RouteHarness, link: ClientLink, name: str, kind: str = "line") -> int:
    return _context_id(
        h.request(link, "spectrum.interactive.open", name=name, kind=kind)
    )


def test_remote_open_follows_a_named_picker_from_fit(
    joint_route: RouteHarness, shown_window, tmp_path: Path
):
    h = joint_route
    with shown_window(h.ctrl) as driver:
        link = h.client()
        a = _picker(h, link, "a")
        _result(
            h.request(
                link,
                "spectrum.interactive.command",
                name="a",
                context_id=a,
                command="move_line",
                params={"role": "half", "position": 0.2},
            )
        )
        driver.screenshot(tmp_path / "01-align-a.png")
        _spectra(driver, "a", stage="align")
        draft_text = driver.observe(object_name="fluxdep.picker.status").text
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "02-on-fit.png")
        b = _picker(h, link, "b")
        driver.screenshot(tmp_path / "03-remote-align-b.png")
        _spectra(driver, "b", stage="align")
        assert driver.observe(object_name="fluxdep.picker.status").text != draft_text
        assert _picker(h, link, "a") == a  # opening an existing draft still follows
        driver.screenshot(tmp_path / "04-back-to-draft-a.png")
        _spectra(driver, "a", stage="align")
        assert driver.observe(object_name="fluxdep.picker.status").text == draft_text
        _result(
            h.request(
                link,
                "spectrum.interactive.command",
                name="b",
                context_id=b,
                command="swap_lines",
            )
        )
        driver.screenshot(tmp_path / "05-edit-follows-b.png")
        _spectra(driver, "b", stage="align")


@pytest.mark.parametrize(
    ("kind", "terminal"), [("line", "finish"), ("sample", "cancel")]
)
def test_remote_unedited_terminal_follows_published_preview(
    joint_route: RouteHarness, shown_window, tmp_path: Path, kind: str, terminal: str
):
    h = joint_route
    with shown_window(h.ctrl) as driver:
        link = h.client()
        b = _picker(h, link, "b", kind)
        driver.screenshot(tmp_path / "01-draft-b.png")
        _spectra(driver, "b", stage="align" if kind == "line" else "sample")
        _select_spectrum(driver, "a")
        driver.screenshot(tmp_path / "02-selected-a.png")
        _spectra(driver, "a")
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "03-away-on-fit.png")
        _result(
            h.request(
                link,
                "spectrum.interactive.command",
                name="b",
                context_id=b,
                command=terminal,
            )
        )
        driver.screenshot(tmp_path / "04-published-b.png")
        _spectra(driver, "b")


def test_remote_filter_follows_open_edit_finish_and_cancel(
    joint_route: RouteHarness, shown_window, tmp_path: Path
):
    h = joint_route
    with shown_window(h.ctrl) as driver:
        link = h.client()
        driver.screenshot(tmp_path / "01-on-spectra.png")
        current = _context_id(h.request(link, "selection.interactive.open"))
        driver.screenshot(tmp_path / "02-filter-draft.png")
        _fit(driver, "filter")
        assert not driver.observe("Apply").enabled
        assert not driver.observe(object_name="fluxdep.picker.finish").enabled
        for step, command in enumerate(("clear", "cancel", "finish"), start=3):
            if command == "finish":
                current = _context_id(h.request(link, "selection.interactive.open"))
                driver.screenshot(tmp_path / "05-reopened-filter.png")
                _fit(driver, "filter")
            driver.select_tab("Spectra")
            driver.screenshot(tmp_path / f"{step}-before-{command}.png")
            _result(
                h.request(
                    link,
                    "selection.interactive.command",
                    context_id=current,
                    command=command,
                )
            )
            driver.screenshot(tmp_path / f"{step}-after-{command}.png")
            _fit(driver, "filter")
            if command == "clear":
                assert not driver.observe("Apply").enabled
            else:
                assert driver.observe(object_name="fluxdep.filter.published").visible
                assert not driver.observe(object_name="fluxdep.filter.edit").enabled
        driver.click(object_name="fluxdep.fit.stage.search")
        driver.screenshot(tmp_path / "06-fit-on-search.png")
        _result(h.request(link, "selection.interactive.open"))
        driver.screenshot(tmp_path / "07-search-to-filter.png")
        _fit(driver, "filter")


def test_remote_load_remove_export_and_restore_follow_collection(
    joint_route: RouteHarness, shown_window, tmp_path: Path, spectrum_hdf5
):
    h = joint_route
    path, *_ = spectrum_hdf5
    with shown_window(h.ctrl) as driver:
        link = h.client()
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "01-on-fit.png")
        _result(h.request(link, "spectrum.load", filepath=path, spec_type="OneTone"))
        driver.screenshot(tmp_path / "02-loaded.png")
        _spectra(driver, Path(path).name)
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "03-before-remove.png")
        _result(h.request(link, "spectrum.remove", name=Path(path).name))
        driver.screenshot(tmp_path / "04-removed-active.png")
        assert driver.tabs("Spectra").current == "Spectra"
        assert not _active(driver)
        assert not driver.plots().axes
        _select_spectrum(driver, "b")
        driver.screenshot(tmp_path / "05-selected-b.png")
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "06-before-export.png")
        checkpoint = str(tmp_path / "checkpoint.h5")
        _result(h.request(link, "export.spectrums", filepath=checkpoint))
        driver.screenshot(tmp_path / "07-export-keeps-b.png")
        _spectra(driver, "b")
        for index, name in enumerate(("a", "b", "empty"), start=8):
            _result(h.request(link, "spectrum.remove", name=name))
            driver.screenshot(tmp_path / f"{index:02}-removed-{name}.png")
            assert _active(driver) == (["b"] if name == "a" else [])
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "11-before-restore.png")
        _result(h.request(link, "spectrum.load_processed", filepath=checkpoint))
        driver.screenshot(tmp_path / "12-restored-first.png")
        _spectra(driver, "a")


def test_remote_fit_params_and_export_follow_search_and_show(
    joint_route: RouteHarness,
    shown_window,
    tmp_path: Path,
    search_database_hdf5: str,
    fluxonium_entry: ResultEntry,
):
    h = joint_route
    with shown_window(h.ctrl) as driver:
        link = h.client()
        h.ctrl.configure_flux_context(synthetic_flux_context)
        _setup(h, link, fluxonium_entry)
        driver.screenshot(tmp_path / "01-on-spectra.png")
        _result(
            h.request(
                link,
                "fit.set_params",
                search_db=search_database_hdf5,
                EJb=[2, 15],
                ECb=[0.2, 2],
                ELb=[0.1, 2],
                transitions={"transitions": [[0, 1]]},
            )
        )
        driver.screenshot(tmp_path / "02-fit-params-search.png")
        _fit(driver, "search")
        # Publish a fresh synthetic fit without issuing another follow command.
        inputs = h.ctrl.capture_search()
        h.ctrl.record_search_result(
            perfect_fit_result(inputs.fluxs, inputs.freqs, inputs.bounds), inputs
        )
        driver.select_tab("Spectra")
        driver.screenshot(tmp_path / "03-before-fit-export.png")
        _result(h.request(link, "fit.export_params"))
        driver.screenshot(tmp_path / "04-fit-export-show.png")
        _fit(driver, "show")
        assert driver.observe(object_name="fluxdep.show.canvas").visible
        assert any(axis.xlabel == "Flux" for axis in driver.plots().axes)


def _setup(h: RouteHarness, link: ClientLink, entry: ResultEntry) -> None:
    _result(
        h.request(
            link,
            "project.setup",
            project_root=str(entry.result_path.parent.parent),
            entry="synthetic",
        )
    )


def test_remote_reads_setup_and_rejected_edit_do_not_follow(
    joint_route: RouteHarness,
    shown_window,
    tmp_path: Path,
    search_database_hdf5: str,
    fluxonium_entry: ResultEntry,
):
    h = joint_route
    h.ctrl.set_fit_params(
        search_database_hdf5,
        ParamBounds(EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2)),
        {"transitions": [(0, 1)]},
        None,
        None,
    )
    h.ctrl.state.set_fit_result((5.0, 1.0, 0.5))
    with shown_window(h.ctrl) as driver:
        link = h.client()
        b = _picker(h, link, "b")
        driver.screenshot(tmp_path / "01-align-b.png")
        _picker(h, link, "a", "sample")
        driver.screenshot(tmp_path / "02-sample-a.png")
        _spectra(driver, "a", stage="sample")
        for index, (method, params) in enumerate(
            (
                ("spectrum.list", {}),
                ("spectrum.snapshot", {"name": "b"}),
                ("spectrum.render", {"name": "b"}),
                ("interactive.read", {"target": "spectrum", "name": "b"}),
                ("interactive.render", {"target": "spectrum", "name": "b"}),
                ("selection.snapshot", {}),
                ("selection.render", {}),
                ("fit.result", {}),
                ("fit.render", {}),
                ("fit.database_info", {"path": search_database_hdf5}),
            ),
            start=3,
        ):
            _result(h.request(link, method, **params))
            driver.screenshot(tmp_path / f"{index:02}-{method}.png")
            _spectra(driver, "a", stage="sample")
        rejected = h.request(
            link,
            "spectrum.interactive.command",
            name="b",
            context_id=b,
            command="not_a_command",
        )
        assert not rejected["ok"], rejected
        driver.screenshot(tmp_path / "13-rejected-b-stays-on-a.png")
        _spectra(driver, "a", stage="sample")
        driver.select_tab("Fit")
        driver.screenshot(tmp_path / "14-fit-before-setup.png")
        _setup(h, link, fluxonium_entry)
        driver.screenshot(tmp_path / "15-project-header-only.png")
        _fit(driver, "filter")
        assert (
            driver.observe(object_name="fluxdep.session.project").text
            == "Entry: synthetic / Q2"
        )
        driver.select_tab("Spectra")
        driver.screenshot(tmp_path / "16-sample-still-on-a.png")
        _spectra(driver, "a", stage="sample")
