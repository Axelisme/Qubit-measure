"""Independent FTF loop losses, frozen at acquisition admission."""

from pathlib import Path
from uuid import uuid4

import numpy as np

from zcu_tools.experiment.single_fluxonium.drift import LogOU, LossRow, write_truth
from zcu_tools.resources.ftf import FTFSettings


class FTFCoherenceDrift:
    """Own three independent loop namespaces and one environment truth session.

    settings must have validated drift enabled. seed is the acquisition root seed;
    None selects fresh entropy once. truth_log selects a JSONL file, or None uses
    ~/.local/state/zcu-tools/simulation/ftf/<session>/truth.jsonl. No file is created
    before sample. This owner hides stream derivation and frozen chip assembly;
    call synchronously from the source. Invalid selected losses raise ValueError.
    """

    def __init__(
        self, settings: FTFSettings, *, seed: int | None, truth_log: Path | None
    ) -> None:
        drift = settings.drift
        if drift is None:
            raise ValueError("FTF coherence drift requires enabled settings")
        self._settings = settings
        self._session = uuid4().hex
        self._path = (
            truth_log
            if truth_log is not None
            else (
                Path.home()
                / ".local/state/zcu-tools/simulation/ftf"
                / self._session
                / "truth.jsonl"
            )
        )
        root = np.random.SeedSequence(seed).entropy
        self._processes: dict[str, dict[str, LogOU]] = {}
        for loop, circuit, targets in (
            ("f1", settings.chip.f1, drift.f1),
            ("coupler", settings.chip.coupler, drift.coupler),
            ("f2", settings.chip.f2, drift.f2),
        ):
            processes: dict[str, LogOU] = {}
            if targets is not None:
                parameters = [
                    ("q_cap", circuit.q_cap, targets.q_cap),
                    ("aphi", circuit.aphi, targets.aphi),
                ]
                if loop == "f1" and drift.f1 is not None:
                    parameters.append(("xqp", settings.chip.f1.xqp, drift.f1.xqp))
                elif loop == "f2" and drift.f2 is not None:
                    parameters.append(("xqp", settings.chip.f2.xqp, drift.f2.xqp))
                for name, value, parameter in parameters:
                    if parameter is not None:
                        if value is None or value <= 0:
                            raise ValueError(
                                f"drift.{loop}.{name} requires positive loss"
                            )
                        processes[name] = LogOU(
                            value,
                            sigma=parameter.sigma,
                            tau_s=parameter.tau_s,
                            rng=np.random.default_rng(
                                np.random.SeedSequence(
                                    root,
                                    spawn_key=(
                                        0x44524946,
                                        *loop.encode(),
                                        *name.encode(),
                                    ),
                                )
                            ),
                        )
            self._processes[loop] = processes

    def sample(self, sim_now: float) -> FTFSettings:
        """Freeze all losses at nondecreasing finite simulated seconds and log them.

        The first call returns YAML losses. Later calls use exact elapsed-time OU
        transitions. All three loops get one truth row, including constant loops;
        coupler q_ind/xqp are null. Returned settings disable drift for replay.
        Backward/nonfinite time raises ValueError; truth-log I/O errors propagate.
        """
        updates = {
            loop: {name: process.sample(sim_now) for name, process in processes.items()}
            for loop, processes in self._processes.items()
        }
        chip = self._settings.chip.model_copy(
            update={
                "f1": self._settings.chip.f1.model_copy(update=updates["f1"]),
                "coupler": self._settings.chip.coupler.model_copy(
                    update=updates["coupler"]
                ),
                "f2": self._settings.chip.f2.model_copy(update=updates["f2"]),
            }
        )
        for loop, losses in (
            ("f1", LossRow(chip.f1.q_cap, chip.f1.q_ind, chip.f1.xqp, chip.f1.aphi)),
            ("coupler", LossRow(chip.coupler.q_cap, None, None, chip.coupler.aphi)),
            ("f2", LossRow(chip.f2.q_cap, chip.f2.q_ind, chip.f2.xqp, chip.f2.aphi)),
        ):
            write_truth(
                self._path, losses, session=self._session, sim_now=sim_now, qubit=loop
            )
        return self._settings.model_copy(update={"chip": chip, "drift": None})
