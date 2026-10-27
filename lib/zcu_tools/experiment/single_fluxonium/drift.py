"""Exact, independently sampled log-loss Ornstein-Uhlenbeck drift."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from zcu_tools.resources.single_fluxonium import SingleFluxoniumSettings


class LogOU:
    """A positive parameter whose logarithm follows an exact OU transition.

    mean is the positive YAML value: the stationary log mean is log(mean),
    not the linear arithmetic mean. sigma is stationary log standard deviation
    (>=0); tau_s is positive correlation time in seconds. rng is exclusively
    owned by this process. The first sample fixes its time origin and returns
    mean, so startup does not draw a stationary random initial condition.
    """

    def __init__(
        self, mean: float, *, sigma: float, tau_s: float, rng: np.random.Generator
    ) -> None:
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("mean must be finite and positive")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if not math.isfinite(tau_s) or tau_s <= 0:
            raise ValueError("tau_s must be finite and positive")
        self._mean = mean
        self._sigma = sigma
        self._tau_s = tau_s
        self._rng = rng
        self._deviation = 0.0
        self._time: float | None = None

    def sample(self, time_s: float) -> float:
        """Return the loss value at finite nondecreasing simulated seconds.

        Equal times return unchanged values without consuming RNG. Backward or
        nonfinite time raises ValueError. Arbitrarily large gaps use the exact
        stationary limit; tiny gaps retain precision through expm1.
        """
        if not math.isfinite(time_s) or (
            self._time is not None and time_s < self._time
        ):
            raise ValueError("time must be finite and nondecreasing")
        if self._time is not None and time_s > self._time and self._sigma > 0:
            gap = (time_s - self._time) / self._tau_s
            self._deviation = (
                math.exp(-gap) * self._deviation
                + self._sigma
                * math.sqrt(-math.expm1(-2 * gap))
                * self._rng.standard_normal()
            )
        self._time = time_s
        return self._mean * math.exp(self._deviation)


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
        self._processes: dict[str, LogOU] = {}
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
                self._processes[name] = LogOU(
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
        record = {
            "session": self.session,
            "sim_now": sim_now,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "q_cap": qubit.q_cap,
            "q_ind": qubit.q_ind,
            "xqp": qubit.xqp,
            "aphi": qubit.aphi,
        }
        if self._qubit is not None:
            record["qubit"] = self._qubit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return effective
