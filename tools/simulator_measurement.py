"""Shared serial session lifecycle and clocks for simulator timing tools.

Callers supply prepared sessions, inspection and resource-owning contexts.
No scene, reference I/O or Engine construction happens here.
Public evolution/history contracts follow ADR-0072 and ADR-0078.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from time import perf_counter, process_time
from typing import Literal

from pydantic import BaseModel, Field


class Sample[O](BaseModel):
    """One timed invocation and its untimed observation.

    Durations are finite, non-negative elapsed seconds. Zero is a valid reading
    of a clock that did not advance within its resolution. Negative or nonfinite
    durations raise pydantic.ValidationError, a ValueError subclass. Process CPU
    may exceed wall time. Readings below clock resolution do not establish speed.
    """

    wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    """Finite, non-negative elapsed wall seconds, including a literal zero."""
    process_seconds: float = Field(ge=0, allow_inf_nan=False)
    """Finite, non-negative whole-process CPU seconds, not calling-thread CPU."""
    observation: O
    """Caller-defined inspection result, computed after both clocks stop."""


@dataclass(frozen=True)
class Session[T, O]:
    """Prepared callbacks borrowed from a caller-owned context.

    invoke(telemetry) performs exactly one Engine.evolve, without setup/cleanup.
    inspect(result, telemetry) computes untimed diagnostics with the same flag.
    clear_history invalidates routing history without resetting RNG or models.
    Callback failures propagate. The enclosing context must release resources
    even on failure; neither this adapter nor measure_sample closes them.
    """

    invoke: Callable[[bool], T]
    """Perform one evolve with the requested telemetry flag; return its result."""
    inspect: Callable[[T, bool], O]
    """Inspect the raw result untimed and return caller-defined diagnostics."""
    clear_history: Callable[[], None]
    """Clear routing history untimed, leaving the session and model alive."""
    on_sample: Callable[[str, Sample[O] | None], None] | None = None
    """Untimed runner/mode observer; None sample marks an unmeasured mode.

    Mode names are first_evolve, warm_clear_history and warm_retain_history.
    measure_sample alone does not notify this callback. Observer errors propagate.
    """


def measure_sample[T, O](
    session: Session[T, O],
    *,
    telemetry: bool,
    wall_clock: Callable[[], float],
    process_clock: Callable[[], float],
) -> Sample[O]:
    """Time one prepared invocation, then inspect after both clocks stop.

    session is already prepared and history-adjusted. telemetry is forwarded
    unchanged. Clocks return monotonic floating-point seconds. Negative or
    nonfinite elapsed durations raise ValidationError, a ValueError subclass.
    Clock resolution can produce a valid zero, which is recorded unchanged.
    No epsilon, sample dropping, retry or cross-clock comparison is applied.
    Operation/inspection failures propagate. The caller owns session cleanup.
    This function does not open, close, warm or clear a session.
    """
    wall, process = wall_clock(), process_clock()
    result = session.invoke(telemetry)
    process_seconds, wall_seconds = process_clock() - process, wall_clock() - wall
    return Sample(
        wall_seconds=wall_seconds,
        process_seconds=process_seconds,
        observation=session.inspect(result, telemetry),
    )


def _observed_sample[T, O](
    session: Session[T, O],
    mode: str,
    *,
    telemetry: bool,
    wall_clock: Callable[[], float],
    process_clock: Callable[[], float],
) -> Sample[O]:
    measured = measure_sample(
        session,
        telemetry=telemetry,
        wall_clock=wall_clock,
        process_clock=process_clock,
    )
    if session.on_sample is not None:
        session.on_sample(mode, measured)
    return measured


class SampleRunner[T, O]:
    """Own first/warm sessions between serial sample requests.

    factory returns a context holding a prepared Session. first opens a fresh
    session per sample, without warmups. warm_clear_history and
    warm_retain_history retain a session and warm it once on opening.
    Warmups use the first request's telemetry flag; later flags do not rewarm.
    Samples return immediately and are not stored. Use as a context manager or
    close in finally, including after failures. No process/disk-cache reset occurs.
    """

    def __init__(
        self,
        factory: Callable[[], AbstractContextManager[Session[T, O]]],
        *,
        mode: Literal["first", "warm_clear_history", "warm_retain_history"],
        warmups: int = 2,
        wall_clock: Callable[[], float] = perf_counter,
        process_clock: Callable[[], float] = process_time,
    ) -> None:
        """Configure session ownership, mode, untimed warmups and second clocks.

        No session opens until sample(). Unknown modes or negative warmups
        raise ValueError before opening resources. warmups is ignored for first.
        Default clocks are perf_counter and whole-process process_time.
        """
        if mode not in ("first", "warm_clear_history", "warm_retain_history"):
            raise ValueError(f"unknown measurement mode: {mode}")
        if warmups < 0:
            raise ValueError("warmups must be nonnegative")
        self._factory = factory
        self._mode = mode
        self._warmups = warmups
        self._wall_clock = wall_clock
        self._process_clock = process_clock
        self._stack = ExitStack()
        self._session: Session[T, O] | None = None

    def sample(self, *, telemetry: bool, empty_history: bool = False) -> Sample[O]:
        """Return one sample, optionally clearing history for a diagnostic.

        Setup, warmup, clearing and inspection are untimed. empty_history clears
        immediately before this invocation, without changing the mode or warming
        again. Callbacks and invalid durations propagate; enclosing context exit
        releases the retained session on failures as well as success.
        """
        if self._mode == "first":
            with self._factory() as session:
                if empty_history:
                    session.clear_history()
                return _observed_sample(
                    session,
                    "first_evolve",
                    telemetry=telemetry,
                    wall_clock=self._wall_clock,
                    process_clock=self._process_clock,
                )
        if self._session is None:
            self._session = self._stack.enter_context(self._factory())
            for _ in range(self._warmups):
                self._session.invoke(telemetry)
        if self._mode == "warm_clear_history" or empty_history:
            self._session.clear_history()
        return _observed_sample(
            self._session,
            self._mode,
            telemetry=telemetry,
            wall_clock=self._wall_clock,
            process_clock=self._process_clock,
        )

    def close(self) -> None:
        """Release retained resources; a later sample opens and warms anew.

        Repeated close without a new sample does nothing. Cleanup errors
        propagate. This is a reset, not terminal shutdown.
        """
        self._stack.close()
        self._session = None

    def __enter__(self) -> "SampleRunner[T, O]":
        """Return this runner; context exit closes its session."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release resources after successful or failed context execution."""
        self.close()


@dataclass(frozen=True)
class ModeMeasurements[O]:
    """Retained samples for one mode, without consumer-specific summaries."""

    mode: str
    """first_evolve, warm_clear_history or warm_retain_history."""
    samples: list[Sample[O]]
    """Ordered retained invocations; empty only for an unmeasured mode."""
    unmeasured_reason: str | None = None
    """None for measured modes; otherwise the reason samples were skipped."""


def measure_modes[T, O](
    factory: Callable[[], AbstractContextManager[Session[T, O]]],
    *,
    telemetry: bool = True,
    wall_clock: Callable[[], float] = perf_counter,
    process_clock: Callable[[], float] = process_time,
) -> list[ModeMeasurements[O]]:
    """Collect the family-table protocol from caller-prepared session contexts.

    Retain five first_evolve samples, each with a fresh session. Each warm mode
    opens one session, invokes two untimed warmups, then retains five samples.
    warm_clear_history clears before every sample; warm_retain_history does not.
    If the first wall duration exceeds 60 seconds, keep it and one immediate
    warm_retain_history sample in that same session, without warmup or clearing.
    warm_clear_history then has no samples and an explicit reason.
    All invocations use telemetry unchanged. Observers receive retained samples
    or None for a skipped mode, untimed. No process/disk-cache state is changed.
    Setup, evolve, inspection, observer and cleanup failures propagate; every
    opened context exits on failure. No partial result list is returned, but
    observers can preserve completed samples. Factories own resource release.
    """
    with factory() as session:
        first = _observed_sample(
            session,
            "first_evolve",
            telemetry=telemetry,
            wall_clock=wall_clock,
            process_clock=process_clock,
        )
        if first.wall_seconds > 60:
            if session.on_sample is not None:
                session.on_sample("warm_clear_history", None)
            retained = _observed_sample(
                session,
                "warm_retain_history",
                telemetry=telemetry,
                wall_clock=wall_clock,
                process_clock=process_clock,
            )
            return [
                ModeMeasurements("first_evolve", [first]),
                ModeMeasurements("warm_clear_history", [], "slow family not measured"),
                ModeMeasurements("warm_retain_history", [retained]),
            ]
    with SampleRunner(
        factory, mode="first", wall_clock=wall_clock, process_clock=process_clock
    ) as runner:
        samples = [first] + [runner.sample(telemetry=telemetry) for _ in range(4)]
    modes = [ModeMeasurements("first_evolve", samples)]
    for name in ("warm_clear_history", "warm_retain_history"):
        with SampleRunner(
            factory, mode=name, wall_clock=wall_clock, process_clock=process_clock
        ) as runner:
            samples = [runner.sample(telemetry=telemetry) for _ in range(5)]
        modes.append(ModeMeasurements(name, samples))
    return modes
