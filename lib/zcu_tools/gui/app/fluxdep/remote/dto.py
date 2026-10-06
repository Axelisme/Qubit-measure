"""Named JSON projections; domain values and policy remain with native owners."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from typing_extensions import TypedDict as ExtTypedDict

from zcu_tools.gui.app.fluxdep.state import SpecType
from zcu_tools.gui.project import ProjectInfoPayload


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
