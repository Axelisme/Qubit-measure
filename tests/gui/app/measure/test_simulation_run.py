"""A registered synthetic experiment acquires, stops, fails and saves via RPC."""

import sys
from threading import Event

import numpy as np
import pytest
from zcu_tools.datafile import load
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.experiment._saving_support import SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import SavingAdapter
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure.remote._helpers import call, open_client


def NoiseAdapter_run(cfg, *, context):
    prog = ModularProgramV2(
        context.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout")
        ],
        sweep=[("point", 3)],
        reps=1,
    )
    acquired = prog.acquire(
        context.soc, progress=False, cancel_flag=context.scope.stop, rounds=3
    )
    assert acquired is not None
    iq = acquired[0][0]
    return SavingResult(np.array([1.0, 2.0, 3.0]), iq[:, 0] + 1j * iq[:, 1])


NoiseAdapter = Adapter(
    spec=SavingAdapter.spec,
    make_form=SavingAdapter.make_form,
    run=NoiseAdapter_run,
    guide=None,
    analysis=SavingAdapter.analysis,
    post_analysis=SavingAdapter.post_analysis,
)


@pytest.fixture
def noise_run(qapp, tmp_path, outcome):
    env = white_noise(std_adc=0, seed=17)
    source = env.soc.binding.execute
    entered, release = Event(), Event()
    fail = outcome == "failure_then_retry"

    def execute(request):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("synthetic source failed")
        if outcome == "stopped" and request.index == 1:
            entered.set()
            # The caller releases after cancel; fixture teardown also releases on failure.
            if not release.wait(30):
                raise RuntimeError("test did not release held round")
        return source(request)

    env.soc.binding.execute = execute
    registry = Registry()
    registry.register("noise", NoiseAdapter)
    old_hook = sys.excepthook
    behavior = MeasureGuiBehavior(
        lambda: (registry, Loader()),
        clean=True,
        project_root=str(tmp_path),
    )
    assembly = behavior.assemble(ControlOptions(port=0))
    ctrl, window, remote = (
        assembly.controller,
        assembly.window,
        assembly.control_adapter,
    )
    if (
        not isinstance(ctrl, Controller)
        or not isinstance(window, MainWindow)
        or not isinstance(remote, RemoteControlAdapter)
    ):
        raise AssertionError(
            "measure composition must provide its real Window/RPC owners"
        )
    sock = None
    try:
        ctrl.install_soc(env.soc, env.soccfg)
        ctrl.setup_control.new_entry(EntryRequest("offline"))
        ctrl.setup_control.new_point(
            PointCreateRequest("point", "", CommandOrigin("gui"))
        )
        tab = ctrl.new_tab("noise")
        sock = open_client(remote.start())
        yield env, ctrl, sock, tab, entered, release
    finally:
        release.set()
        if sock is not None:
            sock.close()
        try:
            behavior.close(assembly)
        finally:
            # The board and its source resources remain caller-owned.
            env.devices.close_all_devices()
        sys.excepthook = old_hook
        qapp.processEvents()


@pytest.mark.parametrize("outcome", ["complete", "stopped", "failure_then_retry"])
def test_window_rpc_acquisition_saves_iq_and_public_environment_identity(
    noise_run, outcome
):
    env, ctrl, sock, tab, entered, release = noise_run
    started = call(sock, "tab.run_start", {"tab_id": tab})["result"]["operation_id"]
    if outcome == "stopped":
        # The accepted run reaches the held second round through real owner delivery.
        qt_until(entered.is_set, label="held simulation round")
        assert call(sock, "tab.run_cancel")["result"]["cancelled"]
        release.set()
    # The started run completes (or fails) after release; this deadline only guards hangs.
    terminal = call(sock, "operation.await", {"operation_id": started, "timeout": 30})[
        "result"
    ]
    if outcome == "failure_then_retry":
        assert terminal["status"] == "failed"
        state = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
        assert not state["result_state"]["available"]
        started = call(sock, "tab.run_start", {"tab_id": tab})["result"]["operation_id"]
        # The one-shot failure is consumed; the retry now delivers its terminal outcome.
        terminal = call(
            sock, "operation.await", {"operation_id": started, "timeout": 30}
        )["result"]
    assert terminal["status"] == ("cancelled" if outcome == "stopped" else "finished")
    saved = call(sock, "tab.save_data", {"tab_id": tab})
    assert saved["ok"], saved
    runs = ctrl.save_control.list_saved_runs()
    assert len(runs) == 1 and runs[0].data_path is not None
    stored = load(runs[0].data_path)
    variable = next(iter(stored.vars.values()))
    np.testing.assert_array_equal(variable.values, 0)
    assert metadata_from_container(stored).completion == (
        "stopped" if outcome == "stopped" else "complete"
    )
    assert (
        getattr(metadata_from_container(stored).provenance, "simulation_id", None)
        == env.soc.binding.provenance.identifier
    )
