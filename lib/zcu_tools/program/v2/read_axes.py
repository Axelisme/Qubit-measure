"""Compile read labels into joint shot-vector outer-product sums."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.cfg_model import ReadoutClassifier
from zcu_tools.program.acquisition import ShotSumProcessor

from .macro.readout import ReadoutEmission
from .modules.readout import MuxReadout, PulseReadout, classification_processor
from .readout_observation import observe_readouts
from .timeline import TimelineProgramV2


@dataclass(frozen=True)
class IQ:
    """An unclassified read axis with vector (1, I, Q).

    read names one compiled readout emission. tone is None for PulseReadout,
    or the zero-based cfg tone position for MuxReadout. Values use normalized
    ADC coordinates without IQ-offset subtraction.
    """

    read: str
    tone: int | None = None


@dataclass(frozen=True)
class Classify:
    """A one-hot classified read axis, in g/e/other, g/e, or g/e/f order.

    read names one compiled emission. tone is None for PulseReadout or a
    zero-based MuxReadout cfg tone position. None also selects a mux's only tone;
    multi-tone mux requires an explicit position and lists available tones on error.
    classifier overrides that read's calibration; None selects its cfg calibration.
    Missing calibration and circle radius raise ValueError at shot_sums construction.
    Calibration is detached.
    """

    read: str
    tone: int | None = None
    classifier: ReadoutClassifier | None = None


def shot_sums(
    prog: TimelineProgramV2, axes: Sequence[Classify | IQ]
) -> ShotSumProcessor:
    """Resolve compiled read labels and return a per-point shot-sum processor.

    prog is a compiled TimelineProgramV2. axes lists vectors
    in output-axis order; repeat a read for second moments. Empty axes raise
    ValueError before read-label resolution or acquisition. Timeline labels are
    emit tags, or module names when untagged.
    Each label must identify one emission; missing/ambiguous labels raise
    ValueError listing available labels. QICK rejects duplicate explicit tags
    during compilation with RuntimeError, before this helper runs. Repeat/Branch
    triggers on a selected ADC and declared/actual read-count mismatches raise
    ValueError. Classify snapshots its override or cfg classifier; missing
    calibration, missing circle radius and invalid tone selection raise ValueError.
    The returned processor accepts ADC-keyed float64 arrays shaped
    (points, reps, reads, 2) and returns (points, *axes), summing outer products
    over reps. It borrows inputs without mutation and rejects read-count changes.
    """
    if not axes:
        raise ValueError("axes must contain at least one read")
    labels, counts, unsupported = _collect_reads(prog)
    resolved = [_resolve_axis(prog, axis, labels, counts, unsupported) for axis in axes]

    def process(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        for ch, count in counts.items():
            actual = iq[ch].shape
            if len(actual) != 4 or actual[2] != count or actual[-1] != 2:
                raise ValueError(
                    f"ADC {ch}: expected (points, reps, {count}, 2); actual {actual}"
                )
        product = None
        for ch, index, classify in resolved:
            values = iq[ch][:, :, index]
            vector = (
                np.concatenate([np.ones((*values.shape[:-1], 1)), values], axis=-1)
                if classify is None
                else classify(values)
            )
            product = (
                vector
                if product is None
                else product[..., None]
                * vector.reshape(
                    *vector.shape[:2], *((1,) * (product.ndim - 2)), vector.shape[-1]
                )
            )
        if product is None:
            raise ValueError("axes must contain at least one read")
        return product.sum(axis=1, dtype=np.float64)

    return process


type _LabelReads = dict[str, list[tuple[PulseReadout | MuxReadout, dict[int, int]]]]


def _collect_reads(
    prog: TimelineProgramV2,
) -> tuple[_LabelReads, dict[int, int], dict[int, str]]:
    """Resolve labels and trigger indices from static source readout observations."""
    counts = dict.fromkeys(prog.ro_chs, 0)
    labels: _LabelReads = {}
    unsupported: dict[int, str] = {}
    for observation in observe_readouts(prog):
        macro = observation.source
        if isinstance(macro, ReadoutEmission) and isinstance(
            macro.owner, (PulseReadout, MuxReadout)
        ):
            owner = macro.owner
            channels = (
                [owner.cfg.ro_ch]
                if isinstance(owner, PulseReadout)
                else [tone.ro_ch for tone in owner.cfg.tones]
            )
            label = owner.name if macro.label is None else macro.label
            labels.setdefault(label, []).append(
                (owner, {ch: counts[ch] for ch in channels})
            )
        if isinstance(macro, Trigger):
            control = observation.control_scope
            for ch in macro.ros:
                if control is not None:
                    unsupported[ch] = control
                counts[ch] += 1
    return labels, counts, unsupported


def _resolve_axis(
    prog: TimelineProgramV2,
    axis: Classify | IQ,
    labels: _LabelReads,
    counts: dict[int, int],
    unsupported: dict[int, str],
) -> tuple[int, int, Callable[[NDArray[np.float64]], NDArray[np.float64]] | None]:
    """Validate a label/tone and snapshot its optional classifier before acquisition."""
    matches = labels.get(axis.read, [])
    if len(matches) != 1:
        raise ValueError(
            f"label {axis.read!r} expected one emission, actual {len(matches)}; available labels: {list(labels)}"
        )
    owner, indices = matches[0]
    if isinstance(owner, PulseReadout):
        if axis.tone is not None:
            raise ValueError(
                f"label {axis.read!r}: pulse readout expects tone=None; actual {axis.tone!r}"
            )
        ch, classifier = owner.cfg.ro_ch, owner.cfg.classifier
    else:
        tone_index = axis.tone
        if (
            isinstance(axis, Classify)
            and tone_index is None
            and len(owner.cfg.tones) == 1
        ):
            tone_index = 0
        if type(tone_index) is not int or not 0 <= tone_index < len(owner.cfg.tones):
            raise ValueError(
                f"label {axis.read!r}: mux expects tone in [0, {len(owner.cfg.tones)}); "
                f"actual {axis.tone!r}; available tones: {list(range(len(owner.cfg.tones)))}"
            )
        tone = owner.cfg.tones[tone_index]
        ch, classifier = tone.ro_ch, tone.classifier
    if ch in unsupported:
        raise ValueError(
            f"ADC {ch}: trigger inside {unsupported[ch]} is unsupported for read labels"
        )
    declared = prog.ro_chs[ch]["trigs"]
    if counts[ch] != declared:
        raise ValueError(
            f"ADC {ch}: declares {declared} reads per shot but emitted {counts[ch]} triggers"
        )
    classify = None
    if isinstance(axis, Classify):
        classifier = axis.classifier if axis.classifier is not None else classifier
        if classifier is None:
            raise ValueError(f"label {axis.read!r}: classifier missing")
        classify = classification_processor(classifier)
    return ch, indices[ch], classify


def split_sums(
    sums: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Split a final (count, sum I, sum Q) axis into counts and complex means.

    sums contains float64 shot sums. Its last axis must have length three or
    ValueError reports the actual shape. Return counts with that axis removed
    (a borrowed view) and detached complex128 means of the same shape. Empty
    cells and unmeasured NaN counts give NaN real and imaginary components,
    without divide warnings. NaN counts remain NaN.
    No input values are changed; counts are not normalized.
    """
    if sums.ndim < 1 or sums.shape[-1] != 3:
        raise ValueError(
            f"split_sums expected final axis length 3; actual shape {sums.shape}"
        )
    counts = sums[..., 0]
    means = np.full(counts.shape, complex(np.nan, np.nan), dtype=np.complex128)
    np.divide(
        sums[..., 1] + 1j * sums[..., 2],
        counts,
        out=means,
        where=(counts != 0) & ~np.isnan(counts),
    )
    return counts, means
