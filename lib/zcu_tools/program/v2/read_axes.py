"""Compile read labels into joint shot-vector outer-product sums."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam, Trigger

from zcu_tools.cfg_model import LinearClassifier, ReadoutClassifier
from zcu_tools.program.acquisition import ShotReducer
from zcu_tools.program.shot_processing import AdditiveShotReducer, ShotIQ

from .macro.readout import ReadoutEmission
from .modules.readout import MuxReadout, PulseReadout, classification_processor
from .readout_labels import emission_label
from .readout_observation import ReadoutPath, observe_readout_paths
from .timeline import TimelineProgramV2

_QUADRATIC_IQ_WIDTH = 6
_CUBIC_IQ_WIDTH = 10


@dataclass(frozen=True)
class IQ:
    """Unclassified read axis (1, I, Q, I², IQ, Q²), with optional cubics.

    order=2 (default) uses the six components above. order=3 appends
    (I³, I²Q, IQ², Q³). Other orders raise ValueError.

    read names one logical compiled readout, present once on each Branch path.
    tone is None for PulseReadout or a zero-based MuxReadout cfg tone position.
    None also selects a mux's only tone; multi-tone mux requires an explicit
    position and lists available tones on error. By default values use normalized
    ADC coordinates without IQ-offset subtraction. offset_corrected=True
    subtracts the native QICK offset from I and Q before forming moments.
    shot_sums accepts that opt-in only for one quadratic axis on a single ADC
    with one integrated read per shot and no registered shot masks.
    """

    read: str
    tone: int | None = None
    order: Literal[2, 3] = 2
    offset_corrected: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.order not in (2, 3):
            raise ValueError(f"IQ order must be 2 or 3, got {self.order!r}")


@dataclass(frozen=True)
class Classify:
    """A one-hot classified read axis, in g/e/other, g/e, or g/e/f order.

    read names one logical compiled readout, present once on each Branch path.
    tone is None for PulseReadout or a zero-based MuxReadout cfg tone position.
    None also selects a mux's only tone; multi-tone mux requires an explicit
    position and lists available tones on error.
    classifier overrides that read's calibration; None selects its cfg calibration.
    Missing calibration and circle radius raise ValueError at shot_sums construction.
    Calibration is detached.
    """

    read: str
    tone: int | None = None
    classifier: ReadoutClassifier | None = None


def shot_sums(
    prog: TimelineProgramV2, axes: Sequence[Classify | IQ]
) -> ShotReducer[NDArray[np.float64]]:
    """Resolve compiled read labels and return a per-point shot-sum reducer.

    prog is a compiled TimelineProgramV2. axes lists vectors
    in output-axis order. IQ includes all moments through its requested order;
    repeated reads form outer products of those features. Empty axes raise
    ValueError before read-label resolution or acquisition. Timeline labels are
    emit tags, or module names when untagged. Labels and ADC indices refer to
    logical reads after removing reads consumed by registered mask sources.
    Each label must identify one visible emission on every feasible Branch path.
    Mutually exclusive untagged emissions may share a module name when their
    selected ADC, read index, frequency (MHz), window (us), phase (degrees) and
    cfg classifier agree. Generator gain and waveform may differ. The compiler
    guarantees equal read counts across Branch cases.
    Missing/ambiguous labels, unequal settings/indices and triggers inside Repeat
    or a forward jump range on selected ADCs raise ValueError. Read labels
    cannot be indexed across opaque macros whose triggers are not visible in
    the source path; the error reports the compiled and observed counts.
    QICK rejects duplicate explicit tags at compile time with RuntimeError.
    Classify snapshots its override or cfg classifier; missing
    calibration, missing circle radius and invalid tone selection raise ValueError.
    The reducer borrows ADC-keyed float64 (points, reps, logical_reads, 2)
    arrays and accumulates (points, *axes) outer-product sums of kept shots.
    IQ NaNs in kept shots propagate. Corrected IQ uses the selected compiled
    logical read's native offset (including the static PFB policy). Unsupported
    ADC/read counts, edge counting, masks or additional axes raise ValueError.
    The reducer rejects read-count changes and does not modify its input or
    the program's compiled read counts.
    """
    if not axes:
        raise ValueError("axes must contain at least one read")
    offset_corrected = _validate_iq_correction(prog, axes)
    paths = iter(observe_readout_paths(prog))
    reference = next(paths)
    labels, resolved = _resolve_path(prog, axes, reference)
    for path in paths:
        other_labels, _ = _resolve_path(prog, axes, path)
        for axis in axes:
            _check_path_read(axis, labels, other_labels, reference, path)
    counts = {
        axis.channel: prog.logical_read_count(
            axis.channel, prog.ro_chs[axis.channel]["trigs"]
        )
        for axis in resolved
    }

    def round_sums(iq: ShotIQ, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        if offset_corrected and (prog.has_shot_mask_sources or not mask.all()):
            raise ValueError("offset_corrected IQ does not support shot masks")
        for ch, count in counts.items():
            actual = iq[ch].shape
            if len(actual) != 4 or actual[2] != count or actual[-1] != 2:
                raise ValueError(
                    f"ADC {ch}: expected (points, reps, {count}, 2); actual {actual}"
                )
        product = None
        for axis in resolved:
            vector = axis.features(iq[axis.channel][:, :, axis.index])
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
        if offset_corrected or mask.all():
            return product.sum(axis=1, dtype=np.float64)
        kept = mask.reshape(*mask.shape, *((1,) * (product.ndim - 2)))
        return product.sum(axis=1, dtype=np.float64, where=kept)

    return AdditiveShotReducer(round_sums, tuple(axis.size for axis in resolved))


def _validate_iq_correction(
    prog: TimelineProgramV2, axes: Sequence[Classify | IQ]
) -> bool:
    selected = next(
        (axis for axis in axes if isinstance(axis, IQ) and axis.offset_corrected),
        None,
    )
    if selected is None:
        return False
    if len(axes) != 1 or selected.order != 2:
        raise ValueError("offset_corrected IQ requires one quadratic IQ axis")
    if prog.has_shot_mask_sources:
        raise ValueError("offset_corrected IQ does not support shot masks")
    if len(prog.ro_chs) != 1:
        raise ValueError("offset_corrected IQ requires a single ADC")
    readout = next(iter(prog.ro_chs.values()))
    if readout["edge_counting"] or readout["trigs"] != 1:
        raise ValueError(
            "offset_corrected IQ requires one integrated read per shot, not edge counting"
        )
    return True


type _LabelReads = dict[str, list[tuple[PulseReadout | MuxReadout, dict[int, int]]]]


@dataclass(frozen=True)
class _ResolvedAxis:
    channel: int
    index: int
    features: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    size: int


def _iq_features(
    values: NDArray[np.float64], *, order: Literal[2, 3]
) -> NDArray[np.float64]:
    i, q = values[..., 0], values[..., 1]
    ii, qq = i**2, q**2
    components = [np.ones(values.shape[:-1]), i, q, ii, i * q, qq]
    if order == 3:
        components.extend((ii * i, ii * q, i * qq, qq * q))
    return np.stack(components, axis=-1)


def _corrected_iq_features(
    values: NDArray[np.float64], *, prog: TimelineProgramV2, channel: int
) -> NDArray[np.float64]:
    # QICK sets the static PFB frequency registers during acquire config, not build.
    return _iq_features(values - prog.native_readout_iq_offset(channel), order=2)


def _path_description(path: ReadoutPath) -> str:
    return "Branch cases " + ", ".join(f"{name}[{index}]" for name, index in path.cases)


def _resolve_path(
    prog: TimelineProgramV2,
    axes: Sequence[Classify | IQ],
    path: ReadoutPath,
) -> tuple[_LabelReads, list[_ResolvedAxis]]:
    labels, observed, unsupported = _collect_reads(prog, path)
    try:
        resolved = []
        for axis in axes:
            matches = labels.get(axis.read, [])
            if len(matches) != 1:
                raise ValueError(
                    f"label {axis.read!r}: expected one emission, actual {len(matches)}; "
                    f"available labels: {list(labels)}"
                )
            result = _resolve_axis(prog, axis, matches[0], unsupported)
            ch = result.channel
            compiled = prog.ro_chs[ch]["trigs"]
            if observed[ch] != compiled:
                raise ValueError(
                    f"ADC {ch}: compiled {compiled} reads, observed {observed[ch]} "
                    "source triggers; cannot resolve read labels across opaque triggers"
                )
            resolved.append(result)
        return labels, resolved
    except ValueError as error:
        if not path.cases:
            raise
        raise ValueError(f"{_path_description(path)}: {error}") from error


def _parameter_key(value: float | QickParam) -> tuple[float, dict[str, float]]:
    return (value.start, value.spans) if isinstance(value, QickParam) else (value, {})


def _adc_settings(
    read: tuple[PulseReadout | MuxReadout, dict[int, int]], axis: Classify | IQ
) -> dict[str, object]:
    owner, indices = read
    if isinstance(owner, PulseReadout):
        return {
            "ro_ch": owner.cfg.ro_ch,
            "read index": indices[owner.cfg.ro_ch],
            "freq": _parameter_key(owner.cfg.freq),
            "ro_length/window": _parameter_key(owner.cfg.ro_length),
            "phase": _parameter_key(owner.cfg.phase),
            "classifier": owner.cfg.classifier,
        }
    tone = owner.cfg.tones[0 if axis.tone is None else axis.tone]
    return {
        "ro_ch": tone.ro_ch,
        "read index": indices[tone.ro_ch],
        "freq": tone.freq,
        "ro_length/window": owner.cfg.ro_length,
        "phase": tone.phase,
        "classifier": tone.classifier,
    }


def _check_path_read(
    axis: Classify | IQ,
    labels: _LabelReads,
    other_labels: _LabelReads,
    reference: ReadoutPath,
    path: ReadoutPath,
) -> None:
    expected = _adc_settings(labels[axis.read][0], axis)
    actual = _adc_settings(other_labels[axis.read][0], axis)
    mismatches = [name for name, value in expected.items() if value != actual.get(name)]
    if mismatches:
        raise ValueError(
            f"{_path_description(reference)} vs {_path_description(path)}: "
            f"label {axis.read!r} differs in {', '.join(mismatches)}"
        )


def _collect_reads(
    prog: TimelineProgramV2,
    path: ReadoutPath,
) -> tuple[_LabelReads, dict[int, int], dict[int, str]]:
    """Collect logical label indices and physical trigger counts on one source path."""
    counts = dict.fromkeys(prog.ro_chs, 0)
    labels: _LabelReads = {}
    unsupported: dict[int, str] = {}
    for observation in path.observations:
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
            indices = {
                ch: index
                for ch in channels
                if (index := prog.logical_read_index(ch, counts[ch])) is not None
            }
            if indices:
                labels.setdefault(emission_label(macro), []).append((owner, indices))
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
    read: tuple[PulseReadout | MuxReadout, dict[int, int]],
    unsupported: dict[int, str],
) -> _ResolvedAxis:
    """Validate a label/tone and bind its feature conversion before acquisition."""
    owner, indices = read
    if isinstance(owner, PulseReadout):
        if axis.tone is not None:
            raise ValueError(
                f"label {axis.read!r}: pulse readout expects tone=None; actual {axis.tone!r}"
            )
        ch, classifier = owner.cfg.ro_ch, owner.cfg.classifier
    else:
        tone_index = axis.tone
        if tone_index is None and len(owner.cfg.tones) == 1:
            tone_index = 0
        if type(tone_index) is not int or not 0 <= tone_index < len(owner.cfg.tones):
            raise ValueError(
                f"label {axis.read!r}: mux expects tone in [0, {len(owner.cfg.tones)}); "
                f"actual {axis.tone!r}; available tones: {list(range(len(owner.cfg.tones)))}"
            )
        tone = owner.cfg.tones[tone_index]
        ch, classifier = tone.ro_ch, tone.classifier
    if ch not in indices:
        raise ValueError(
            f"label {axis.read!r}: ADC {ch} read is consumed by a shot mask source"
        )
    if ch in unsupported:
        raise ValueError(
            f"ADC {ch}: trigger inside {unsupported[ch]} is unsupported for read labels"
        )
    if isinstance(axis, IQ):
        features = (
            partial(_corrected_iq_features, prog=prog, channel=ch)
            if axis.offset_corrected
            else partial(_iq_features, order=axis.order)
        )
        return _ResolvedAxis(
            ch,
            indices[ch],
            features,
            _QUADRATIC_IQ_WIDTH if axis.order == 2 else _CUBIC_IQ_WIDTH,
        )
    classifier = axis.classifier if axis.classifier is not None else classifier
    if classifier is None:
        raise ValueError(f"label {axis.read!r}: classifier missing")
    return _ResolvedAxis(
        ch,
        indices[ch],
        classification_processor(classifier),
        2 if isinstance(classifier, LinearClassifier) else 3,
    )


def split_sums(
    sums: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Split final (count, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²) into counts and complex means.

    Only the first three components contribute to the result.
    sums contains float64 shot sums. Its last axis must have length six or
    ValueError reports the actual shape. Return counts with that axis removed
    (a borrowed view) and detached complex128 means of the same shape. Empty
    cells and unmeasured NaN counts give NaN real and imaginary components,
    without divide warnings. NaN counts remain NaN.
    No input values are changed; counts are not normalized.
    """
    if sums.ndim < 1 or sums.shape[-1] != _QUADRATIC_IQ_WIDTH:
        raise ValueError(
            f"split_sums expected final axis length 6; actual shape {sums.shape}"
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


@dataclass(frozen=True)
class CubicIQMoments:
    """Count and uncentered, per-shot moments from one cubic IQ axis.

    counts borrows the shot count per cell. first, second and third are
    detached arrays with one, two or three trailing length-two I/Q axes.
    They are divided by counts and are NaN for nonpositive or NaN counts.
    """

    counts: NDArray[np.float64]
    first: NDArray[np.float64]
    second: NDArray[np.float64]
    third: NDArray[np.float64]


def split_cubic_sums(sums: NDArray[np.float64]) -> CubicIQMoments:
    """Decode float64 IQ(order=3) sums into counts and normalized raw moments.

    Preserve all leading axes. Raise ValueError unless the final axis has
    the ten cubic IQ components. Do not mutate sums or center the moments.
    """
    if sums.ndim < 1 or sums.shape[-1] != _CUBIC_IQ_WIDTH:
        raise ValueError(
            f"Expected cubic IQ sums with final axis length 10, got {sums.shape}"
        )
    counts = sums[..., 0]
    raw = sums / np.where(counts > 0, counts, np.nan)[..., None]
    return CubicIQMoments(
        counts=counts,
        first=raw[..., 1:3],
        second=raw[..., [[3, 4], [4, 5]]],
        third=raw[..., [[[6, 7], [7, 8]], [[7, 8], [8, 9]]]],
    )
