"""Acquisition publishes a RunRecord carrying its historical evidence."""

from dataclasses import replace

from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore
from tests.gui.app.measure._record_support import require_record

from ._run_support import make_inputs, make_plots, make_run_service, make_state


def test_run_publishes_acquisition_metadata_inside_the_record() -> None:
    state, tab_id, adapter = make_state()
    cfg = SavingCfg()
    result = SavingCore().run(cfg, context=None)
    adapter.run.return_value = RunRecord(cfg, result)
    history = native_metadata()
    admitted = make_inputs(state, tab_id, adapter)
    inputs = replace(
        admitted,
        request=replace(admitted.request, soc=None, soccfg=None),
        snapshot=history.snapshot,
        experiment_tag=history.experiment,
    )
    service, _gate, background, handles = make_run_service(state)
    token = service.start_run(inputs, plots=make_plots())
    background.run_work()
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished", outcome
    record = require_record(state.get_tab(tab_id).run)
    assert record is not None and record.metadata is not None
    assert record.result is result
    assert record.metadata.experiment == "test/native"
    assert record.metadata.snapshot == history.snapshot
    assert record.metadata.completion == "complete"
    assert record.metadata.finished_at is not None
    assert record.metadata.device_snapshot == {}
