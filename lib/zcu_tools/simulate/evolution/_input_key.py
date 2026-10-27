"""Full input identity for acquire-wide routing usage, not numerical results."""

from collections.abc import Sequence
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from ._validation import shape
from .engine import TimelineBatch
from .secular import ModelSource


class _InputDigest:
    """Length-framed bulk content hashing; never constructs per-row objects."""

    def __init__(self) -> None:
        self.value = sha256()

    def text(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.value.update(len(encoded).to_bytes(8, "little"))
        self.value.update(encoded)

    def array(self, value: NDArray[np.generic]) -> None:
        self.text(value.dtype.str)
        self.text(str(value.shape))
        self.value.update(value.tobytes(order="C"))

    def continuation(self, batch: TimelineBatch, state_shape: tuple[int, ...]) -> None:
        """Validate source dimensions and hash the paired absolute/DC inputs."""
        if batch.start_ns is None or batch.initial_densities is None:
            self.text("source initial density")
        else:
            shape(
                batch.initial_densities,
                (len(batch.points), *state_shape),
                "initial densities",
            )
            self.text("point-aligned continuation")
            self.array(batch.start_ns)
            self.array(batch.initial_densities)


def input_key(
    source: ModelSource, batches: Sequence[TimelineBatch], budget: float
) -> bytes:
    """Digest full input records before geometry/planning, with source identity.

    The DC identity describes the entire source builder, including nonzero flux
    levels. Initial density and optional continuous Hamiltonian expansion are
    separate source inputs and are hashed by content. Every batch/track field,
    including unused envelope candidates and beyond-readout values, participates.
    The source's midpoint subdivision policy also participates. Other fixed
    kernel policies belong to the owning Engine's in-process lifetime, not to
    a persistent or cross-engine key namespace.
    """
    digest = _InputDigest()
    loops = batches[0].flux.level.shape[1]
    digest.text(source.model((0,) * loops).identity)
    digest.array(source.initial_density)
    digest.text(float(budget).hex())
    digest.text(str(source.slow_envelope_segments_per_period))
    try:
        expansion = source.flux_expansion()
    except ValueError:
        # The source API uses ValueError to report an absent optional expansion.
        digest.text("no expansion")
    else:
        digest.text("expansion")
        digest.array(expansion.static)
        digest.text(str(len(expansion.terms)))
        for term in expansion.terms:
            digest.text(str(term.loop))
            digest.text(term.kind)
            digest.array(term.matrix)
    digest.text(str(len(batches)))
    for batch in batches:
        digest.array(batch.points)
        digest.array(batch.end_ns)
        digest.continuation(batch, source.initial_density.shape)
        for times in batch.segment_boundaries:
            digest.array(np.asarray(times, dtype=np.float64))
        digest.text("segment shifts")
        for shifts in batch.segment_shifts:
            digest.text(str(len(shifts)))
            for shift in shifts:
                digest.text(float(shift.start_ns).hex())
                digest.text(float(shift.end_ns).hex())
                digest.text(shift.content.decode())
        digest.array(batch.flux.offsets)
        digest.array(batch.flux.start_ns)
        digest.array(batch.flux.level)
        if batch.flux_waveform is None:
            digest.text("no waveform")
        else:
            digest.text("waveform")
            digest.array(batch.flux_waveform.offsets)
            digest.array(batch.flux_waveform.time_ns)
            digest.array(batch.flux_waveform.flux)
        digest.text(str(len(batch.tracks)))
        for track in batch.tracks:
            digest.text(track.port)
            for array in (
                track.start_ns,
                track.duration_ns,
                track.frequency_ghz,
                track.phase_rad,
                track.amplitude,
                track.envelope_index,
                track.active,
            ):
                digest.array(array)
            digest.text(str(len(track.envelopes)))
            for envelope in track.envelopes:
                # The immutable record already hashes kind, complex samples and
                # their time axis. Reuse it rather than serializing every pulse.
                digest.text(envelope.identity)
    return digest.value.digest()
