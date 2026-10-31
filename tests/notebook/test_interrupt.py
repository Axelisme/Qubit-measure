"""Notebook SIGINT requests cooperate with a synchronous experiment core."""

import signal
from collections.abc import Callable

import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.context import RunContext
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment
from zcu_tools.plotting.plots import NonPresentingHost

from tests.notebook._adapter_fakes import Cfg, RecordingCore


class InterruptCore(RecordingCore):
    """Run an authored action before producing the ordinary recording result."""

    def __init__(self, action: Callable[[RunContext], None]) -> None:
        super().__init__()
        self.action = action

    def run(self, config: Cfg, *, context: RunContext) -> float:
        self.action(context)
        return super().run(config, context=context)


def notebook(core: InterruptCore) -> NotebookExperiment[InterruptCore]:
    return NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=NonPresentingHost()
    )(core)


def test_first_sigint_requests_stop_and_publishes_returned_record() -> None:
    def author(context: RunContext) -> None:
        signal.raise_signal(signal.SIGINT)
        assert context.scope.stop.is_set()
        assert context.scope.stop.reason

    adapter = notebook(InterruptCore(author))
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        record = adapter.run(Cfg(scale=7.0))
    except KeyboardInterrupt:
        pytest.fail("The first SIGINT must request a cooperative stop")
    assert record.result == 7.0
    assert adapter.last_run is record
    assert adapter.acquisition_for(record).completion == "stopped"
    assert signal.getsignal(signal.SIGINT) is previous_handler
