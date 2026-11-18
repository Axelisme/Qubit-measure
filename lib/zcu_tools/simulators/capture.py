"""Simulator-owned metadata for FTF and single-fluxonium debug captures."""

from typing import TypedDict

from zcu_tools.program.v2.simulation import AcquireOutcome
from zcu_tools.resources.simulation_schema import Values
from zcu_tools.simulate.boundary import EnvironmentSnapshot
from zcu_tools.simulate.evolution import CaptureBuffer, write_capture


class CaptureSampling(TypedDict):
    """Binding RNG inputs: actual root seed and zero-based acquire sequence."""

    root_seed: int
    sequence: int


class CaptureRound(TypedDict):
    """A round's zero-based index, native DC values by name, and settings identity."""

    index: int
    dc_values: dict[str, float]
    model_identity: str


class AcquireCaptureMetadata(TypedDict):
    """Normal-return flag, captured attempts, and the settings they refer to.

    Sampling is null without a round snapshot in the outcome. effective_settings
    holds simulator-specific JSON by observed round model identity, including
    sampled losses. admission_settings supplies the settings at begin only when
    the outcome has no round snapshots.
    """

    complete: bool
    sampling: CaptureSampling | None
    rounds: list[CaptureRound]
    effective_settings: dict[str, dict[str, object]]
    admission_settings: dict[str, object] | None


class AcquireCaptureSettings:
    """Retain immutable settings for one captured acquire, beginning at admission.

    Settings are supplied by the owning simulator after any drift sampling.
    The generic capture writer knows only opaque JSON (ADR-0078).
    """

    def __init__(self, admission_settings: Values) -> None:
        self._admission = admission_settings
        self._by_identity: dict[str, Values] = {}

    def record(self, snapshot: EnvironmentSnapshot, settings: Values) -> None:
        """Keep the immutable settings represented by this successful snapshot."""
        self._by_identity.setdefault(snapshot.model_identity, settings)

    def write(self, buffer: CaptureBuffer, outcome: AcquireOutcome) -> str | None:
        """Publish after evolution and return the basename or None on writer failure.

        Keep every round in outcome.rounds, including failed attempts. Include
        effective settings only for round identities observed by this capture.
        Detached rounds can name identities absent from effective_settings.
        """
        rounds: list[CaptureRound] = [
            {
                "index": item.index,
                "dc_values": dict(item.snapshot.values),
                "model_identity": item.snapshot.model_identity,
            }
            for item in outcome.rounds
        ]
        request = outcome.rounds[0].acquire if rounds else None
        metadata: AcquireCaptureMetadata = {
            "complete": outcome.complete,
            "sampling": (
                None
                if request is None
                else {"root_seed": request.root_seed, "sequence": request.sequence}
            ),
            "rounds": rounds,
            "effective_settings": {
                identity: self._by_identity[identity].model_dump(mode="json")
                for identity in dict.fromkeys(item["model_identity"] for item in rounds)
                if identity in self._by_identity
            },
            "admission_settings": None
            if rounds
            else self._admission.model_dump(mode="json"),
        }
        path = write_capture(buffer, metadata=dict(metadata))
        return None if path is None else path.name
