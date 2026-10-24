"""History Load survives a missing form without admitting a fake empty cfg."""

from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import PulseCfg, PulseSpec, Template
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.form_contracts import CfgPreconditionError
from zcu_tools.gui.app.measure.services.load import LoadDataError
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._history_form_support import (
    SourceDependentAdapter,
    history_controller,
)
from tests.gui.app.measure._load_support import write_native_file


def test_open_history_retains_native_data_when_factory_has_no_source(
    qapp: QApplication, tmp_path: Path
) -> None:
    path = tmp_path / "history.h5"
    metadata = write_native_file(path, tag="test/saving", gain=0.9)
    with history_controller(tmp_path) as ctrl:
        outcome = ctrl.tab_control.open_tab_from_file("history", str(path))
        tab_id = ctrl.tab_control.get_active_tab_id()
        assert tab_id is not None
        snapshot = ctrl.tab_control.get_tab_snapshot(tab_id)
        assert snapshot.run is not None
        record = snapshot.run.result
        assert isinstance(record, RunRecord)
        assert record.metadata == metadata
        assert isinstance(record.result, SavingResult)
        np.testing.assert_allclose(
            record.result.signal, [0.9 + 1.8j, 2.7 - 3.6j, 4.5 + 5.4j]
        )
        assert outcome.analysis_source_issue is not None
        resource = ctrl.cfg_resources.lookup(tab_id)
        assert resource.unavailable_reason == "factory: history_source is missing"
        with pytest.raises(
            CfgPreconditionError, match="factory: history_source is missing"
        ):
            resource.accept()
        with pytest.raises(
            CfgPreconditionError, match="factory: history_source is missing"
        ):
            _ = resource.input
        assert resource.unavailable_reason is not None


def test_unavailable_reset_updates_nonfactory_error_and_reraises_without_a_form(
    qapp: QApplication, tmp_path: Path
) -> None:
    path = tmp_path / "history.h5"
    write_native_file(path, tag="test/saving")
    with history_controller(tmp_path) as ctrl:
        ctrl.tab_control.open_tab_from_file("history", str(path))
        tab_id = ctrl.tab_control.get_active_tab_id()
        assert tab_id is not None
        resource = ctrl.cfg_resources.lookup(tab_id)
        ctrl.setup_control.new_entry(EntryRequest("current"))
        ctrl.setup_control.new_point(PointCreateRequest("p", "", CommandOrigin("gui")))
        ctrl.context_control.add_component("Q1", "history/source", {})
        ctrl.context_control.set_alias_rows((AliasRow("history", "Q1"),))
        ctrl.context_control.set_module(
            ctrl.context_control.read_context().target,
            "Q1.pulses.template",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.flat"),
        )
        with pytest.raises(
            ValueError, match="Adapter 'history', pulse: Unknown template 'missing'"
        ) as failure:
            resource.reset()
        assert type(failure.value) is ValueError
        assert resource.unavailable_reason == str(failure.value)
        with pytest.raises(CfgPreconditionError, match="Unknown template 'missing'"):
            resource.accept()


@pytest.mark.parametrize("failure", ["format", "tag"])
def test_rejected_history_load_removes_partial_tab_and_restores_focus(
    qapp: QApplication, tmp_path: Path, failure: str
) -> None:
    valid = tmp_path / "valid.h5"
    bad = tmp_path / "bad.h5"
    write_native_file(valid, tag="test/saving")
    if failure == "format":
        bad.write_text("not native", encoding="utf-8")
    else:
        write_native_file(bad, tag="wrong/tag" if failure == "tag" else "test/saving")
    with history_controller(tmp_path) as ctrl:
        ctrl.tab_control.open_tab_from_file("history", str(valid))
        previous = ctrl.tab_control.get_active_tab_id()
        before = ctrl.tab_control.list_tab_ids()
        with pytest.raises(LoadDataError):
            ctrl.tab_control.open_tab_from_file("history", str(bad))
        assert ctrl.tab_control.list_tab_ids() == before
        assert ctrl.tab_control.get_active_tab_id() == previous


def test_regular_new_tab_still_propagates_factory_error(
    qapp: QApplication, tmp_path: Path
) -> None:
    with history_controller(tmp_path) as ctrl:
        with pytest.raises(ValueError, match="factory: history_source is missing"):
            ctrl.tab_control.new_tab("history")
        assert ctrl.tab_control.list_tab_ids() == []


class TemplateHistoryCfg(SavingCfg):
    pulse: PulseCfg | None = None


def broken_template_form(ctx: FormContext) -> Form[TemplateHistoryCfg]:
    del ctx
    return Form(TemplateHistoryCfg, {"pulse": Template("missing")})


BrokenTemplateAdapter = Adapter(
    spec=SourceDependentAdapter.spec,
    make_form=broken_template_form,
    run=SourceDependentAdapter.run,
    guide=None,
)


def test_history_creation_does_not_isolate_template_binding_errors(
    qapp: QApplication, tmp_path: Path
) -> None:
    registry = Registry()
    registry.register("broken-template", BrokenTemplateAdapter)
    with history_controller(tmp_path, registry=registry) as ctrl:
        with pytest.raises(ValueError, match="broken-template.*pulse"):
            ctrl.tab_control.open_tab_from_file("broken-template", "unused.h5")
        assert ctrl.tab_control.list_tab_ids() == []
