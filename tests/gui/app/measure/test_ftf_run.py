"""Registered synthetic FTF experiment uses real Window/RPC Run and Save."""

import sys
from dataclasses import replace
from typing import ClassVar

import numpy as np
import yaml
from qtpy.QtWidgets import QDialog
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.context_types import CommandOrigin, EntryRequest, PointCreateRequest
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.experiment._saving_support import SavingResult
from tests.gui.app.measure._native_save_support import SavingAdapter
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure.remote._helpers import call, open_client
from tests.resources._ftf_data import ftf_document


class FTFAcquisitionAdapter(SavingAdapter):
    capabilities: ClassVar[AdapterCapabilities] = replace(SavingAdapter.capabilities, requires_soc=True)

    def run(self, req, cfg, *, context):
        modules = [
            PulseCfg(ch=2, nqz=1, freq=300., phase=90., gain=-.1, waveform=ConstWaveformCfg(length=.008)).build("microwave"),
            PulseCfg(ch=1, nqz=1, freq=0., gain=.2, waveform=ConstWaveformCfg(length=.008), post_delay=.008).build("flux"),
            PulseReadoutCfg(gen_ch=0, ro_ch=0, nqz=1, freq=6000., gain=.1, waveform=ConstWaveformCfg(length=.02), ro_length=.03).build("ro"),
        ]
        program = ModularProgramV2(context.soccfg, ProgramV2Cfg(reps=2), modules=modules, sweep=[("point",3)])
        iq = program.acquire(context.soc, progress=False, cancel_flag=context.cancel_signal)[0][0]
        return RunRecord(cfg, SavingResult(np.array([1.,2.,3.]), iq[:,0]+1j*iq[:,1]))


def test_ftf_yaml_environment_runs_and_saves_only_public_iq(qapp, tmp_path):
    path = tmp_path/"environment.py"
    path.write_text("from pathlib import Path\nfrom zcu_tools.experiment.ftf import ftf\ndef build_environment():\n    return ftf(Path(__file__).with_name('ftf.yaml'))\n", encoding="utf-8")
    (tmp_path/"ftf.yaml").write_text(yaml.safe_dump(ftf_document()), encoding="utf-8")
    registry = Registry()
    registry.register("ftf", FTFAcquisitionAdapter)
    old_hook = sys.excepthook
    behavior = MeasureGuiBehavior(lambda: (registry, Loader()), simulate=path, clean=True, project_root=str(tmp_path))
    assembly = behavior.assemble(ControlOptions(port=0))
    ctrl, window, remote = assembly.controller, assembly.window, assembly.control_adapter
    if not isinstance(ctrl, Controller) or not isinstance(window, MainWindow) or not isinstance(remote, RemoteControlAdapter):
        raise AssertionError("registered FTF run needs actual Window/RPC owners")
    sock = None
    try:
        ctrl.setup_control.new_entry(EntryRequest("offline"))
        ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
        tab = ctrl.new_tab("ftf")
        sock = open_client(remote.start())
        started = call(sock, "tab.run_start", {"tab_id":tab})["result"]["operation_id"]
        terminal = call(sock, "operation.await", {"operation_id":started, "timeout":15}, timeout_s=20)["result"]
        assert terminal["status"] == "finished", terminal
        saved = call(sock, "tab.save_data", {"tab_id":tab})
        assert saved["ok"], saved
        runs = ctrl.save_control.list_saved_runs()
        assert len(runs) == 1 and runs[0].data_path is not None
        stored = load_run_data(runs[0].data_path)
        values = next(iter(stored.payload.variables.values())).data.values
        assert np.isfinite(values).all()
        assert np.any(values != 0)
        assert stored.metadata.completion == "complete"
        assert stored.metadata.provenance is not None
        assert stored.metadata.provenance.simulation_id is not None
    finally:
        if sock is not None:
            sock.close()
        behavior.close(assembly)
        for dialog in window.findChildren(QDialog):
            dialog.reject()
        window.dispose_control_watches()
        window.deleteLater()
        sys.excepthook = old_hook
        qapp.processEvents()
