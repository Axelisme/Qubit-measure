"""Registered synthetic FTF experiment uses real Window/RPC Run and Save."""

import sys
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
import yaml
from zcu_tools.datafile import load
from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
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
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.experiment._saving_support import SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import SavingAdapter
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure.remote._helpers import call, open_client
from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


class FTFAcquisitionAdapter(SavingAdapter):
    capabilities: ClassVar[AdapterCapabilities] = replace(
        SavingAdapter.capabilities, requires_soc=True
    )

    def run(self, req, cfg, *, context):
        modules = [
            PulseCfg(
                ch=2,
                nqz=1,
                freq=300.0,
                phase=90.0,
                gain=-0.1,
                waveform=ConstWaveformCfg(length=0.008),
            ).build("microwave"),
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=6000.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.02),
                ro_length=0.03,
            ).build("ro"),
        ]
        program = ModularProgramV2(
            context.soccfg, ProgramV2Cfg(reps=2), modules=modules, sweep=[("point", 3)]
        )
        acquired = program.acquire(
            context.soc, progress=False, cancel_flag=context.scope.stop
        )
        assert acquired is not None
        iq = acquired[0][0]
        return RunRecord(
            cfg, SavingResult(np.array([1.0, 2.0, 3.0]), iq[:, 0] + 1j * iq[:, 1])
        )


@pytest.fixture
def ftf_window(qapp, tmp_path):
    """Own the FTF runtime until accepted operations and Qt deliveries drain."""
    (tmp_path / "ftf.yaml").write_text(yaml.safe_dump(ftf_document()), encoding="utf-8")
    registry = Registry()
    registry.register("ftf", FTFAcquisitionAdapter)
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
        raise AssertionError("registered FTF run needs actual Window/RPC owners")
    sock = None
    env = ftf(tmp_path / "ftf.yaml", workers=1, warmup=False)
    try:
        ctrl.install_soc(env.soc, env.soccfg)
        sock = open_client(remote.start())
        yield ctrl, sock
    finally:
        try:
            # Assertion failures still own the Run. Request cancellation, then
            # observe settlement before the normal 5000ms worker/delivery drain.
            for operation in ctrl.operation_control.active_operations():
                ctrl.operation_control.cancel_operation(operation.op)
            qt_until(
                lambda: ctrl.active_operation_count() == 0,
                label="FTF teardown terminal delivery",
                watchdog_ms=60_000,
            )
            behavior.close(assembly)
            # Only a successful drain permits releasing caller resources or Qt
            # owners. A timeout must leave them alive, not dispose a live QThread.
            try:
                env.devices.close_all_devices()
            finally:
                qapp.processEvents()
        finally:
            sys.excepthook = old_hook
            if sock is not None:
                sock.close()


def test_ftf_yaml_environment_runs_and_saves_only_public_iq(ftf_window):
    ctrl, sock = ftf_window
    ctrl.setup_control.new_entry(EntryRequest("offline"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    tab = ctrl.new_tab("ftf")
    started = call(sock, "tab.run_start", {"tab_id": tab})["result"]["operation_id"]
    # The accepted Run must settle. Cold Numba compilation can outlast the
    # RPC wait; 60s only guards deadlock, not acquisition performance.
    qt_until(
        lambda: ctrl.active_operation_count() == 0,
        label="FTF Run terminal delivery",
        watchdog_ms=60_000,
    )
    terminal = call(
        sock,
        "operation.await",
        {"operation_id": started, "timeout": 15},
        timeout_s=20,
    )["result"]
    assert terminal["reason"] == "completed", terminal
    assert terminal["status"] == "finished", terminal
    saved = call(sock, "tab.save_data", {"tab_id": tab})
    assert saved["ok"], saved
    runs = ctrl.save_control.list_saved_runs()
    assert len(runs) == 1 and runs[0].data_path is not None
    stored = load(runs[0].data_path)
    values = next(iter(stored.vars.values())).values
    assert np.isfinite(values).all()
    assert np.any(values != 0)
    assert metadata_from_container(stored).completion == "complete"
    assert metadata_from_container(stored).provenance is not None
    assert metadata_from_container(stored).provenance.simulation_id is not None
