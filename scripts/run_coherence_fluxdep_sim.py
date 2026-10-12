"""Run the complete native Coherence fluxdep workflow without hardware or a GUI.

--plan, --tunables, --seed and --sim read current CoherencePlan, CoherenceTunables,
Calibrated and SimParams JSON. --result-path and --database-path are distinct
isolated entry directories, not existing measurement data. The entry owner
calculates workflow roots; Engine writes journal/manifest, acquisitions and
plots. Existing run directories are refused, never overwritten.

One fast FakeDevice uses plan.flux_dev. Its current value drives the simulated
board at each acquisition. Other device names in native cfg fail normally.
There is no hardware connection, resume, retry or old-format fallback.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.device import BaseDeviceInfo, DeviceManager, FakeDevice
from zcu_tools.experiment.acquisition import (
    capture_device_snapshot,
    capture_software_provenance,
)
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import new_run_id, workflow_roots

from zcu_lab.workflows.coherence_fluxdep.workflow import (
    CoherencePlan,
    CoherenceTunables,
    coherence_fluxdep,
)
from zcu_lab.workflows.common.calibration import Calibrated


class _Devices:
    """Translate the Engine device port to one host-owned native manager."""

    def __init__(self, manager: DeviceManager) -> None:
        self.manager = manager

    def capture_snapshot(self) -> Mapping[str, JsonObject]:
        """Capture connected drivers before each callback in the Engine worker."""
        return capture_device_snapshot(self.manager.get_all_devices())

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Read detached live settings, identifying missing names at use."""
        devices = self.manager.get_all_devices()
        if name not in devices:
            raise ValueError(f"Device not connected: {name!r}")
        return devices[name].get_info().model_copy(deep=True)

    def setup_device(
        self, name: str, cfg: BaseDeviceInfo, cancel_signal: StopSignal
    ) -> None:
        """Revalidate full selected-model cfg and pass the original stop event."""
        devices = self.manager.get_all_devices()
        if name not in devices:
            raise ValueError(f"Device not connected: {name!r}")
        driver = devices[name]
        validated = driver.info_model.model_validate(cfg.model_dump(mode="python"))
        driver.setup(validated, progress=False, stop_event=cancel_signal.event)


class _Arguments(argparse.Namespace):
    plan: Path
    tunables: Path
    seed: Path
    sim: Path
    result_path: Path
    database_path: Path
    run_id: str | None


def _arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tunables", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--run-id", help="New opaque run ID; generated if omitted")
    return parser.parse_args(argv, namespace=_Arguments())


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shipped Coherence declaration with explicit offline inputs and paths.

    None selects process arguments. JSON inputs use current public models,
    including host-resolved qub_pulse/readout module cfgs and a historical
    RunSnapshot in the plan, with a scalar Calibrated seed. Native experiment
    templates are not accepted. Calibration derives relax timing from the
    workflow's T1 estimate; initial timing/device fields use native defaults.
    The result/database paths are exact isolated entry directories. A missing
    run_id is generated once, without collision retries.

    Print run_id, lifecycle, commit count, reason and both workflow roots as
    JSON. Return only after execution and FakeDevice cleanup. Non-done runs
    exit with status 1; input, startup, I/O and cleanup errors propagate.
    Never connect hardware or modify real data. Native software capture may run
    read-only Git commands; it never requests live instrument information.
    """
    args = _arguments(argv)
    plan = CoherencePlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    tunables = CoherenceTunables.model_validate_json(
        args.tunables.read_text(encoding="utf-8")
    )
    seed = TypeAdapter(Calibrated).validate_json(args.seed.read_text(encoding="utf-8"))
    sim = SimParams.model_validate_json(args.sim.read_text(encoding="utf-8"))
    run_id = args.run_id if args.run_id is not None else new_run_id()
    roots = workflow_roots(
        result_path=args.result_path, database_path=args.database_path, run_id=run_id
    )
    soc, soccfg = make_mock_soc(sim=sim)
    flux = FakeDevice(fast_mode=True)
    flux.set_output("on")
    manager = DeviceManager()
    manager.register_device(plan.flux_dev, flux)
    plots = Plots(NonPresentingHost())
    try:
        soc.set_flux_source(flux.get_value)
        engine = Engine(
            EnginePorts(
                plots,
                TQDMProgressBar,
                UtcClock(),
                context=seed,
                soc=soc,
                soccfg=soccfg,
                devices=_Devices(manager),
            )
        )
        engine.start(
            coherence_fluxdep,
            plan=plan,
            tunables=tunables,
            paths=RunPaths(roots.result, roots.database),
            identity=RunIdentity(
                run_id,
                capture_software_provenance(),
                entry_id=str(plan.snapshot.entry_id),
                entry_label=plan.snapshot.entry_name,
            ),
        )
        status = engine.execute(run_id)
    finally:
        soc.set_flux_source(None)
        manager.close_all_devices()
        plots.finish(present=False)
        plots.release()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "lifecycle": status.lifecycle,
                "committed_seq": status.committed_seq,
                "reason": status.reason,
                "metadata_root": str(roots.result),
                "data_root": str(roots.database),
            }
        )
    )
    if status.lifecycle != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
