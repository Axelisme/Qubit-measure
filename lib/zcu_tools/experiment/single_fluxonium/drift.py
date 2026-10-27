"""Compose independent loss-drift streams and acquisition truth I/O."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from zcu_tools.resources.single_fluxonium import SingleFluxoniumSettings
from zcu_tools.simulate import drift as numeric_drift


@dataclass(frozen=True)
class LossRow:
    """One loop's effective losses for a private acquisition truth row.

    q_cap/q_ind are positive dimensionless quality factors, or None for absent
    channels. xqp is a nonnegative dimensionless quasiparticle fraction, or None
    for a circuit without that channel. aphi is nonnegative Phi0/sqrt(Hz).
    Values must be finite. Construction does not perform validation or I/O.
    """

    q_cap: float | None
    q_ind: float | None
    xqp: float | None
    aphi: float


def write_truth(
    path: Path,
    losses: LossRow,
    *,
    session: str,
    sim_now: float,
    qubit: str | None = None,
) -> None:
    """Append one JSONL acquisition row, creating parent directories as needed.

    path selects a caller-owned private log. losses contains effective channel
    values; session is the environment UUID, sim_now is finite simulated seconds,
    and qubit is an optional stable loop key (None omits it for standalone single).
    Rows add UTC ISO wall_time. Nonfinite numbers raise ValueError and filesystem
    errors propagate; neither failure is hidden or retried.
    """
    record = {
        "session": session,
        "sim_now": sim_now,
        "wall_time": datetime.now(timezone.utc).isoformat(),
        "q_cap": losses.q_cap,
        "q_ind": losses.q_ind,
        "xqp": losses.xqp,
        "aphi": losses.aphi,
    }
    if qubit is not None:
        record["qubit"] = qubit
    line = json.dumps(record, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)


class CoherenceDrift:
    """Own independent log-loss streams and one session's private truth log.

    settings must have validated drift enabled. seed is the acquisition root
    seed (None chooses fresh entropy for a standalone source). truth_log selects
    a JSONL file, or None for ~/.local/state/zcu-tools/simulation/
    single-fluxonium/<session>/truth.jsonl. No file exists until sample succeeds.
    qubit optionally names a stable circuit identity, included in each truth row
    and the drift stream namespace without consuming sampling RNG. None keeps
    the standalone single stream unchanged. session optionally supplies an
    environment-owned UUID for a shared log; None creates a new UUID.
    I/O errors propagate. Call methods synchronously from the source owner.
    """

    def __init__(
        self,
        settings: SingleFluxoniumSettings,
        *,
        seed: int | None,
        truth_log: Path | None,
        qubit: str | None = None,
        session: str | None = None,
    ) -> None:
        self.settings = settings
        self.session = uuid4().hex if session is None else session
        self._qubit = qubit
        namespace = () if qubit is None else tuple(qubit.encode("utf-8"))
        self.path = (
            truth_log
            if truth_log is not None
            else (
                Path.home()
                / ".local/state/zcu-tools/simulation/single-fluxonium"
                / self.session
                / "truth.jsonl"
            )
        )
        drift = settings.drift
        if drift is None:
            raise ValueError("coherence drift requires enabled settings")
        root = np.random.SeedSequence(seed).entropy
        circuit = settings.chip.qubit
        self._processes: dict[str, numeric_drift.LogOU] = {}
        for index, (name, value, parameter) in enumerate(
            (
                ("q_cap", circuit.q_cap, drift.q_cap),
                ("xqp", circuit.xqp, drift.xqp),
                ("aphi", circuit.aphi, drift.aphi),
            )
        ):
            if parameter is not None:
                if value is None or value <= 0:
                    raise ValueError(f"drift.{name} requires positive loss")
                self._processes[name] = numeric_drift.LogOU(
                    value,
                    sigma=parameter.sigma,
                    tau_s=parameter.tau_s,
                    rng=np.random.default_rng(
                        np.random.SeedSequence(
                            root, spawn_key=(0x44524946, *namespace, index)
                        )
                    ),
                )

    def sample(self, sim_now: float) -> SingleFluxoniumSettings:
        """Freeze effective settings at sim_now and append exactly one truth row.

        The first call returns the configured losses. Later calls advance each
        stream by elapsed simulated seconds. Returned settings disable drift so
        private replay uses these values without clocks or further sampling.
        """
        values = {
            name: process.sample(sim_now) for name, process in self._processes.items()
        }
        qubit = self.settings.chip.qubit.model_copy(update=values)
        effective = self.settings.model_copy(
            update={
                "chip": self.settings.chip.model_copy(update={"qubit": qubit}),
                "drift": None,
            }
        )
        write_truth(
            self.path,
            LossRow(qubit.q_cap, qubit.q_ind, qubit.xqp, qubit.aphi),
            session=self.session,
            sim_now=sim_now,
            qubit=self._qubit,
        )
        return effective
