"""Deterministic request identities and author-visible evaluation receipts."""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
import scipy

from .controls import ControlSequence
from .model import NumericModel, SolverSettings
from .state import DensityState

NUMERIC_VERSION = "density-direct-v2"


@dataclass(frozen=True)
class EvaluationReceipt:
    """Author-only deterministic computation provenance, never quantum/noise memory.

    origin: direct solver, persisted exact direct-record, or at least one U table
    interval (other intervals can still be direct). content_identities: immutable
    file digests actually used/written, empty without an injected library.
    elapsed_seconds: complete evolve wall time, including miss solve/write/load.
    """

    origin: Literal["direct", "direct-record", "table"]
    content_identities: tuple[str, ...]
    elapsed_seconds: float


def exact_key(
    model: NumericModel,
    settings: SolverSettings,
    initial: DensityState,
    controls: ControlSequence,
    endpoints: tuple[float, float],
) -> str:
    """Hash all deterministic density dependencies, excluding shots/seeds/noise.

    Static arrays are hashed by content. Callbacks/envelopes require reproducible
    caller descriptors. Validate this contract before any cache hit. start/end
    use the same quantized event origin as Evolution; no callable repr/YAML text.
    """
    if (
        model.hamiltonian_at is not None or model.collapse_at is not None
    ) and not model.dynamics_identity:
        raise ValueError("persistent callbacks require dynamics_identity")
    if any(not pulse.envelope.identity for pulse in controls.pulses):
        raise ValueError("persistent envelope requires reproducible identity")
    pulse_keys = [
        (
            p.source,
            p.start_ns,
            p.end_ns,
            p.frequency_ghz,
            p.phase_rad,
            p.phase_mode,
            p.breakpoints_ns,
            p.envelope.identity,
            None
            if p.envelope.constant is None
            else (complex(p.envelope.constant).real, complex(p.envelope.constant).imag),
        )
        for p in controls.pulses
    ]
    event_keys = [(e.source, e.time_ns, e.delta_rad) for e in controls.phase_events]
    basis = model.basis
    description = (
        NUMERIC_VERSION,
        np.__version__,
        scipy.__version__,
        model.identity,
        model.dynamics_identity,
        model.hamiltonian_at is not None,
        model.collapse_at is not None,
        basis.identity,
        basis.labels,
        pulse_keys,
        event_keys,
        model.event_times_ns,
        endpoints,
        settings.rtol,
        settings.atol,
        float(settings.max_step_ns).hex(),
        tuple(s.name for s in model.sources),
    )
    digest = sha256(json.dumps(description, sort_keys=True, allow_nan=False).encode())
    matrices = [
        model.hamiltonian_ghz,
        basis.vectors,
        initial.density,
        *model.collapse_operators,
    ]
    for source in model.sources:
        matrices.append(source.in_phase_ghz)
        if source.quadrature_ghz is not None:
            matrices.append(source.quadrature_ghz)
    for matrix in matrices:
        digest.update(str(matrix.shape).encode())
        digest.update(np.asarray(matrix, dtype="<c16").tobytes())
    return digest.hexdigest()
