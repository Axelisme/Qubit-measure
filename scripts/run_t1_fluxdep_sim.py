"""Run the complete native T1 fluxdep workflow without hardware or a GUI.

--plan, --tunables, --seed and --sim read current T1Plan, T1Tunables,
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
from datetime import UTC, datetime
from pathlib import Path
from socket import gethostname

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import TypeAdapter
from zcu_tools.device import DeviceInfo, DeviceManager, FakeDevice
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    DeviceSnapshot,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
)
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar
from zcu_tools.resources.entry import new_run_id, workflow_roots

from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.t1 import T1Plan, T1Tunables, t1_fluxdep


class _Clock:
    """Adapt UTC wall time and cancellable waits to the Engine clock port."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC instant."""
        return datetime.now(UTC)

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        """Wait for an aware target or cancellation; past targets return now."""
        seconds = max(0.0, (target - self.now()).total_seconds())
        cancel_signal.event.wait(seconds)


class _Plots:
    """Render Engine-owned figures with Agg, without pyplot or GUI state."""

    def axes(self, name: str) -> Axes:
        """Create one named figure's axes, with a headless rendering canvas."""
        figure = Figure()
        figure.set_label(name)
        FigureCanvasAgg(figure)
        return figure.subplots()

    def refresh(self, figure: Figure) -> None:
        """Render the supplied Engine-owned figure; rendering errors propagate."""
        figure.canvas.draw()


class _Devices:
    """Translate the Engine device port to one host-owned native manager."""

    def __init__(self, manager: DeviceManager) -> None:
        self.manager = manager

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        """Set absolute native flux through the driver, observing cancellation.

        The native owner validates names, model, address and settings.
        Unsupported flux setters and driver failures propagate.
        """
        info = self.manager.get_info(name)
        info.set_flux(value)
        self.setup({name: info}, cancel_signal)

    def setup(
        self, settings: Mapping[str, DeviceInfo], cancel_signal: StopSignal
    ) -> None:
        """Apply native device settings in order through the same manager."""
        self.manager.setup_devices(
            settings, progress=False, cancel_signal=cancel_signal.event
        )


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
    """Run the shipped T1 declaration with explicit offline inputs and paths.

    None selects process arguments. JSON inputs use current public models,
    including a historical RunSnapshot in the plan and scalar Calibrated seed.
    The result/database paths are exact isolated entry directories. A missing
    run_id is generated once, without collision retries.

    Print run_id, lifecycle, commit count, reason and both workflow roots as
    JSON. Return only after execution and FakeDevice cleanup. Non-done runs
    exit with status 1; input, startup, I/O and cleanup errors propagate.
    Never connect hardware, modify real data or execute another process.
    """
    args = _arguments(argv)
    plan = T1Plan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    tunables = T1Tunables.model_validate_json(args.tunables.read_text(encoding="utf-8"))
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
    try:
        soc.set_flux_source(flux.get_value)
        engine = Engine(
            EnginePorts(
                _Plots(),
                TQDMProgressBar,
                _Clock(),
                context=seed,
                soc=soc,
                soccfg=soccfg,
                devices=_Devices(manager),
            )
        )
        engine.start(
            t1_fluxdep,
            plan=plan,
            tunables=tunables,
            paths=RunPaths(roots.result, roots.database),
            identity=RunIdentity(
                run_id,
                DeviceSnapshot(manager.get_all_info()),
                gethostname(),
                entry_id=str(plan.snapshot.entry_id),
                entry_label=plan.snapshot.entry_name,
                soc_fingerprint="MockQickSoc",
            ),
        )
        status = engine.execute(run_id)
    finally:
        soc.set_flux_source(None)
        manager.close_all_devices()
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
