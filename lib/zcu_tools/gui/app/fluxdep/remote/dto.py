"""Named JSON projections; domain values and policy remain with native owners."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from typing_extensions import TypedDict as ExtTypedDict

from zcu_tools.gui.app.fluxdep.state import SpecType
from zcu_tools.gui.interactive.flux_pick import FluxPickInfo
from zcu_tools.gui.project import ProjectInfoPayload
from zcu_tools.gui.remote.param_spec import InputSchema


class PngImage(TypedDict):
    """Native PNG: png_b64 is ASCII base64; bytes is its decoded byte count."""

    png_b64: str
    bytes: int


class CommandDeclaration(TypedDict):
    """Available command: name is plugin/reserved ID; schema is its JSON input."""

    name: str
    schema: InputSchema


class EmptyPluginInfo(TypedDict):
    """No plugin-specific operation status is currently exposed."""


class LineStateReply(TypedDict):
    """Device-axis reference lines and presentation flags from one snapshot."""

    flux_half: float
    flux_int: float
    conjugate: bool
    magnitude_only: bool


class OneToneStateReply(TypedDict):
    """Dimensionless prominence [0,5] and full ordered device-axis peak indices."""

    threshold: float
    peak_indices: list[int]


class TwoToneStateReply(TypedDict):
    """Detector/tool settings and counts from one exact native projection.

    threshold is [1,20]; sigma is [0,5] (gaussian requires >=0.001).
    smooth_method is wavelet/gaussian. width is normalized radius [0,0.1],
    mode is select/erase. mask_shape is [device_count, frequency_count].
    masked_count counts true cells; point_count counts detected device/GHz points.
    """

    threshold: float
    sigma: float
    smooth_method: Literal["wavelet", "gaussian"]
    width: float
    mode: Literal["select", "erase"]
    mask_shape: list[int]
    masked_count: int
    point_count: int


class SelectionStateReply(TypedDict):
    """Joint-cloud input state, not the published selection.

    min_distance and width are normalized [0,0.1]; width is radius.
    mode is select/erase. selected is the full input mask in captured insertion
    order; selected_count counts kept points after native downsampling.
    """

    min_distance: float
    width: float
    mode: Literal["select", "erase"]
    selected: list[bool]
    selected_count: int


class ContextReplyBase(TypedDict):
    """One owner-lifetime identity's committed state and native image.

    context_id is positive, not an edit revision. closed means this identity
    retired, even if a successor exists. plugin is its native plugin ID.
    commands lists current declarations (empty when closed). can_undo follows
    Session history and is false when closed. figure depicts this exact capture.
    """

    context_id: int
    closed: bool
    plugin: str
    commands: list[CommandDeclaration]
    can_undo: bool
    figure: PngImage


class LineContextReply(ContextReplyBase):
    """Line context: literal spectrum_name, line state and alignment status."""

    kind: Literal["line"]
    spectrum_name: str
    state: LineStateReply
    info: FluxPickInfo


class OneToneContextReply(ContextReplyBase):
    """OneTone context: literal spectrum_name, peak state and empty info."""

    kind: Literal["onetone"]
    spectrum_name: str
    state: OneToneStateReply
    info: EmptyPluginInfo


class TwoToneContextReply(ContextReplyBase):
    """TwoTone context: literal spectrum_name, detector/mask state, empty info."""

    kind: Literal["twotone"]
    spectrum_name: str
    state: TwoToneStateReply
    info: EmptyPluginInfo


class SelectionContextReply(ContextReplyBase):
    """Joint context: no spectrum target, full-cloud state and empty info."""

    kind: Literal["selection"]
    spectrum_name: None
    state: SelectionStateReply
    info: EmptyPluginInfo


InteractiveContextReply = (
    LineContextReply | OneToneContextReply | TwoToneContextReply | SelectionContextReply
)


class LineChanges(TypedDict):
    """Before/after native device positions and conjugation flags."""

    flux_half_before: float
    flux_half_after: float
    flux_int_before: float
    flux_int_after: float
    conjugate_before: bool
    conjugate_after: bool


class OneToneChanges(TypedDict):
    """Prominence before/after and nonnegative added/removed device peak counts."""

    threshold_before: float
    threshold_after: float
    peaks_added: int
    peaks_removed: int


class TwoToneChanges(TypedDict):
    """Nonnegative mask-cell and detected device/GHz positional differences."""

    mask_added: int
    mask_removed: int
    points_added: int
    points_removed: int


class SelectionChanges(TypedDict):
    """Nonnegative kept-index added/removed counts; duplicates stay distinct."""

    points_added: int
    points_removed: int


class InteractiveEffect(TypedDict):
    """Successful command receipt for the returned identity.

    command is the invoked ID; closed equals context.closed. changes compares
    the snapshots immediately before/after this command, including inverse Undo.
    """

    command: str
    closed: bool
    changes: LineChanges | OneToneChanges | TwoToneChanges | SelectionChanges


class InteractiveReply(TypedDict):
    """Read/open/command receipt; inactive read has both fields null.

    context belongs to the requested identity, never a synchronous successor.
    effect is null for reads/opens and populated only after a successful command.
    """

    context: InteractiveContextReply | None
    effect: InteractiveEffect | None


class SpectrumListItem(TypedDict):
    """One spectrum's workflow and availability projection.

    name is the opaque loaded identifier. spec_type is OneTone or TwoTone.
    aligned marks committed alignment. points_completed marks completion even
    with zero points. point_count is a nonnegative annotation count, not a gate.
    """

    name: str
    spec_type: SpecType
    aligned: bool
    points_completed: bool
    point_count: int


class SpectrumListReply(TypedDict):
    """spectrums lists loaded entries in collection insertion order."""

    spectrums: list[SpectrumListItem]


class PointcloudReply(TypedDict):
    """fluxs (Phi_0) and freqs (GHz) are paired, complete published points."""

    fluxs: list[float]
    freqs: list[float]


class FitParametersReply(TypedDict):
    """EJ, EC and EL are the accepted fit's energy parameters in GHz."""

    EJ: float
    EC: float
    EL: float


class TransitionWire(ExtTypedDict, extra_items=list[list[int]]):
    """Native transition choices encoded as JSON arrays.

    Extra keys name transition groups; each value lists integer [from, to] pairs.
    Optional r_f/sample_f are the native frequency entries in GHz. Omission means
    no such transition frequency is set; precedence remains kernel-owned.
    """

    r_f: NotRequired[float]
    sample_f: NotRequired[float]


class FitResultReply(TypedDict):
    """Complete editable fit snapshot and current result availability.

    has_result marks a stored successful result. params is its EJ/EC/EL or None.
    database_path is the search database file, not the project's raw root.
    EJb/ECb/ELb are [lower, upper] GHz bounds. transitions contains native choices.
    r_f/sample_f are separate optional GHz inputs, None when unset.
    """

    has_result: bool
    params: FitParametersReply | None
    database_path: str
    EJb: list[float]
    ECb: list[float]
    ELb: list[float]
    transitions: TransitionWire
    r_f: float | None
    sample_f: float | None


class StateCheckReply(TypedDict):
    """has_project excludes placeholders; spectrum_count is loaded count.

    has_active indicates a selected display spectrum, not a live input context.
    """

    has_project: bool
    spectrum_count: int
    has_active: bool


class ResourceVersionsReply(TypedDict):
    """versions is the raw live resource counter snapshot; missing keys stay absent."""

    versions: dict[str, int]


class AxisSnapshot(TypedDict):
    """Axis extent without full arrays.

    count is nonnegative; minimum/maximum are None only for an empty axis.
    unit distinguishes native device values, normalized flux Phi_0, and GHz.
    """

    count: int
    minimum: float | None
    maximum: float | None
    unit: Literal["native", "Phi_0", "GHz"]


class RawAxesReply(TypedDict):
    """dev_values/fluxs/freqs are native/Phi_0/GHz extents.

    signals_shape is [device_count, frequency_count]; no complex matrix is sent.
    """

    dev_values: AxisSnapshot
    fluxs: AxisSnapshot
    freqs: AxisSnapshot
    signals_shape: list[int]


class PublishedPointsReply(TypedDict):
    """Paired dev_values (native), fluxs (Phi_0), freqs (GHz) in published order."""

    dev_values: list[float]
    fluxs: list[float]
    freqs: list[float]


class SpectrumAbsentReply(TypedDict):
    """name is a legal opaque identifier; exists=false means no live entry."""

    name: str
    exists: Literal[False]


class SpectrumPresentReply(TypedDict):
    """Complete editable live spectrum snapshot.

    name is opaque; exists is true. spec_type determines the native picking tool.
    aligned and points_completed mark committed stages, not point availability.
    alignment_seeded marks inherited calibration. flux_half/flux_int/flux_period
    are native device coordinates and period. raw_axes contains extents/shape;
    points contains all committed annotations, including empty completed arrays.
    """

    name: str
    exists: Literal[True]
    spec_type: SpecType
    aligned: bool
    points_completed: bool
    alignment_seeded: bool
    flux_half: float
    flux_int: float
    flux_period: float
    raw_axes: RawAxesReply
    points: PublishedPointsReply


SpectrumSnapshotReply = SpectrumAbsentReply | SpectrumPresentReply


class SelectionSnapshotReply(TypedDict):
    """selected is the published joint-cloud mask, or None for all points.

    min_distance is the native normalized selection distance, not GHz or flux.
    """

    selected: list[bool] | None
    min_distance: float


class ProjectSetupReply(TypedDict):
    """project is the applied native identity/path receipt; it grants no observation."""

    project: ProjectInfoPayload


class NameReply(TypedDict):
    """name identifies the spectrum loaded or reset by this command."""

    name: str


class NamesReply(TypedDict):
    """names are processed-loader identifiers in publication order."""

    names: list[str]


class SpectrumRemovedReply(TypedDict):
    """name is the removed identifier; removed=true confirms successful removal."""

    name: str
    removed: Literal[True]


class ActiveSpectrumReply(TypedDict):
    """active_spectrum is the display selection, or None after clearing it."""

    active_spectrum: str | None


class FitUpdatedReply(TypedDict):
    """fit is the complete replacement receipt; it does not start a search."""

    fit: FitResultReply


class SpectrumsExportedReply(TypedDict):
    """filepath is the native owner's actual processed-spectrum output path."""

    filepath: str


class ParamsExportedReply(TypedDict):
    """savepath is the native owner's actual merged params JSON output path."""

    savepath: str


class OperationActivityReply(TypedDict):
    """token is a retained positive app handle, not an MCP session ID.

    status is pending or terminal. error is a failure message or None.
    """

    token: int
    status: Literal["pending", "finished", "failed", "cancelled"]
    error: str | None


class OperationStatusReply(TypedDict):
    """activity is the requested/current activity, or None if none has started."""

    activity: OperationActivityReply | None


class OperationOutcomeReply(TypedDict):
    """status is terminal; error is the native failure message or None."""

    status: Literal["finished", "failed", "cancelled"]
    error: str | None


class OperationStartedReply(TypedDict):
    """token is the positive SearchOwner handle for the admitted search."""

    token: int


class OperationCancelledReply(TypedDict):
    """token identifies the handle; cancel_requested=true is not a terminal outcome."""

    token: int
    cancel_requested: Literal[True]


class OperationAwaitReply(TypedDict):
    """token identifies the retained handle; reason reports how waiting ended.

    outcome is terminal on completed, otherwise possibly None. feedback is the
    native user-feedback string or None. Timeout does not cancel or reveal state.
    """

    token: int
    reason: Literal["completed", "timeout", "user_feedback"]
    outcome: OperationOutcomeReply | None
    feedback: str | None
