"""Shot mask registrations and their shared physical-to-logical read mapping."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

type ShotMaskSource = Callable[[Mapping[int, NDArray[np.float64]]], NDArray[np.bool_]]


def _validated_mask(
    keep: object, shape: tuple[int, int], source_position: int
) -> NDArray[np.bool_]:
    if isinstance(keep, np.ndarray) and keep.shape == shape and keep.dtype == np.bool_:
        return keep
    raise ValueError(
        f"shot mask source {source_position} must return a bool array of shape {shape}; "
        f"got shape {getattr(keep, 'shape', None)}, dtype {getattr(keep, 'dtype', None)}"
    )


class ShotMasks:
    """Store sources in callback order and hide the union of their reads."""

    def __init__(self) -> None:
        self._sources: list[tuple[dict[int, tuple[int, ...]], ShotMaskSource]] = []
        self._hidden: dict[int, set[int]] = {}

    def register(
        self, consumed_reads: Mapping[int, Sequence[int]], source: ShotMaskSource
    ) -> None:
        """Snapshot physical ADC/read selections and append source in call order."""
        reads = {ch: tuple(indices) for ch, indices in consumed_reads.items()}
        for ch, indices in reads.items():
            for index in indices:
                if type(index) is not int or index < 0:
                    raise ValueError(
                        f"shot mask source ADC {ch}: physical read index {index!r} "
                        "must be a nonnegative integer"
                    )
        self._sources.append((reads, source))
        for ch, indices in reads.items():
            self._hidden.setdefault(ch, set()).update(indices)

    def visible_reads(self, channel: int, physical_count: int) -> tuple[int, ...]:
        """Return remaining physical indices in logical read order."""
        hidden = self._hidden.get(channel, ())
        return tuple(index for index in range(physical_count) if index not in hidden)

    def logical_index(self, channel: int, physical_index: int) -> int | None:
        """Map a physical index to logical order, or None if a source consumes it."""
        hidden = self._hidden.get(channel, ())
        if physical_index in hidden:
            return None
        return physical_index - sum(index < physical_index for index in hidden)

    def validate_read_counts(self, counts: Mapping[int, int]) -> None:
        """Reject undeclared ADCs or read indices outside compiled trigger counts."""
        for ch, indices in self._hidden.items():
            if ch not in counts:
                raise ValueError(f"shot mask source consumes undeclared ADC {ch}")
            for index in indices:
                if index >= counts[ch]:
                    raise ValueError(
                        f"shot mask source ADC {ch}: physical read index {index!r} "
                        f"is outside [0, {counts[ch]})"
                    )

    def evaluate(
        self,
        iq: Mapping[int, NDArray[np.float64]],
        shape: tuple[int, int],
    ) -> NDArray[np.bool_]:
        """Pass selected length-normalized reads and AND the resulting shot masks.

        iq maps physical ADCs to (points, reps, reads, 2). shape is (points, reps).
        Each callback gets its own arrays. A non-bool mask or a shape mismatch
        raises ValueError with the one-based registration position. Callback
        exceptions propagate.
        """
        mask = np.ones(shape, dtype=np.bool_)
        for position, (consumed, source) in enumerate(self._sources, start=1):
            keep = source(
                {ch: iq[ch][:, :, reads, :] for ch, reads in consumed.items()}
            )
            mask &= _validated_mask(keep, shape, position)
        return mask
