"""Notebook run roots remain independent across synchronous acquisitions."""

from zcu_tools.device import DeviceManager
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost

from tests.notebook._adapter_fakes import Cfg, RecordingCore


def test_notebook_allocates_fresh_run_scope_and_retains_partial_evidence() -> None:
    core = RecordingCore()
    adapter = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(core)
    first = adapter.run(Cfg())
    context = core.contexts[-1]
    assert hasattr(context, "scope"), "Notebook RunContext must carry an explicit scope"
    assert context.scope.active_path == "run"
    assert context.scope.cancel("run").cancelled
    second = adapter.run(Cfg())
    assert core.contexts[-1].scope is not context.scope
    assert not core.contexts[-1].scope.cancelled
    assert adapter.acquisition_for(second).completion == "complete"
    assert first is not second
