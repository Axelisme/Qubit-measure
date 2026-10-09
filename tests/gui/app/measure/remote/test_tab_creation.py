"""Remote tab creation uses current sources without a birth target."""

from zcu_tools.gui.app.measure.remote.handlers.tab import h_tab_new
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryRequest

from tests.gui.app.measure.remote._helpers import Fixture


def test_remote_creation_allows_empty_source_and_keeps_global_aliases(
    qapp, tmp_path, monkeypatch
) -> None:
    background = BackgroundRunner()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )
    application = Fixture(project_root=str(tmp_path), headless=True, empty_project=True)
    ctrl = application.ctrl
    rows = (AliasRow("qubit", "Q2"), AliasRow("resonator", "$qubit.resonator"))
    ctrl.context_control.set_alias_rows(rows)
    try:
        created = h_tab_new(application.service, {"adapter_name": "fake"})
        tab_id = created["tab_id"]
        assert isinstance(tab_id, str)
        assert ctrl.list_tab_ids() == [tab_id]
        interaction = ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None and not interaction.has_context
        ctrl.setup_control.new_entry(EntryRequest("entry"))
        interaction = ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None and interaction.has_context
        assert ctrl.context_control.get_alias_rows() == rows
        assert ctrl.list_tab_ids() == [tab_id]
    finally:
        assert background.quiesce()
        for tab_id in tuple(ctrl.list_tab_ids()):
            ctrl.close_tab(tab_id)
        application.stop()
        qapp.processEvents()
