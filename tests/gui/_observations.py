"""Qt-free visible-state contracts returned by the component input driver.

Snapshots contain only immutable values, not Qt widgets, artists or result data.
Text preserves displayed units and formatting; empty text follows each field's
visibility contract rather than providing hidden values.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WidgetObservation:
    """Displayed text, effective availability, identity and hover help of a widget."""

    kind: str  # Qt widget class name, e.g. QPushButton or QLabel.
    text: str  # Rendered widget value/caption; empty when hidden or without text.
    enabled: bool  # Effective Qt enabled flag, including disabled ancestors.
    visible: bool  # On-screen Qt region exists, including ancestor/viewport clipping.
    fully_visible: bool  # Entire widget rectangle survives ancestor/viewport clipping.
    object_name: str  # Stable objectName if provided by the UI, otherwise empty.
    tooltip: str  # Hover help; empty when hidden or without a tooltip.
    bounds: tuple[int, int, int, int] | None  # Visible screen x/y/width/height.
    caption_fits: bool  # Visible button text fits its Qt contents rectangle.
    checked: (
        bool | None
    )  # Visible checkable button state; None if hidden or not checkable.


@dataclass(frozen=True)
class TextInputObservation:
    """Visible line-edit text, focus, editing policy and accessible description."""

    text: str  # Rendered input, empty when hidden.
    focused: bool  # Visible input currently owns keyboard focus.
    read_only: bool  # Qt read-only policy; copying remains possible.
    description: str  # Accessible description of the visible input and hints.


@dataclass(frozen=True)
class TreeRowObservation:
    """One on-screen tree row; fully clipped/hidden cells have empty captions.

    Paths retain their depth and columns retain their original indices even when
    column-zero captions are not on screen. Empty captions are not wildcards.
    """

    tree_name: str  # Containing tree's objectName; empty if the UI provides none.
    path: tuple[str, ...]  # Root-to-row column-zero captions; unseen levels empty.
    columns: tuple[
        str, ...
    ]  # Original column indices; unseen cells empty, units unchanged.
    expanded: bool  # Whether the row currently exposes its children.


@dataclass(frozen=True)
class ListRowObservation:
    """Rendered caption, hover help and selection of one visible list row."""

    text: str  # Displayed row caption, preserving its rendered units.
    tooltip: str  # Row hover help; empty when no tooltip is provided.
    selected: bool  # Whether the visible row is selected.


@dataclass(frozen=True)
class TableRowObservation:
    """Visible table row captions, hover help and selection in column order."""

    columns: tuple[str, ...]  # Fully clipped cells have empty captions.
    tooltips: tuple[str, ...]  # Hover help for the corresponding visible cells.
    selected: bool  # Whether this row contains a selected item.


@dataclass(frozen=True)
class VisibleObservation:
    """Visible widgets only, each described without retaining a Qt object."""

    widgets: tuple[WidgetObservation, ...]  # On-screen widgets in widget-tree order.
    tree_rows: tuple[
        TreeRowObservation, ...
    ]  # Rows intersecting visible tree viewports.


@dataclass(frozen=True)
class StatusObservation:
    """Text currently visible in the window's bottom status bar."""

    text: str  # Current message or newline-joined direct labels; may be empty.


@dataclass(frozen=True)
class DialogObservation:
    """A visible dialog's captions and placement on its current screen."""

    title: str  # Displayed dialog window title; empty if no title is provided.
    texts: tuple[str, ...]  # Visible widget texts, in widget-tree order.
    frame_within_screen: bool  # Entire frame lies within available screen bounds.
    height_fraction: float  # Client height divided by available screen height.


@dataclass(frozen=True)
class ScrollObservation:
    """Visible scroll affordances of one scroll area."""

    vertical_scrollable: bool  # Visible vertical scrollbar has a nonempty range.


@dataclass(frozen=True)
class TabObservation:
    """Selected tab caption and all captions in the matched visible tab bar."""

    current: str  # Selected tab caption; empty if there is no current tab.
    captions: tuple[str, ...]  # All tab captions in the matched bar, in tab order.


@dataclass(frozen=True)
class LegendObservation:
    """Visible legend captions from visible matplotlib canvases, in draw order."""

    texts: tuple[str, ...]  # Visible legend entries; empty if none are visible.


@dataclass(frozen=True)
class PlotAxesObservation:
    """Visible axis labels and rendered line-curve count for one matplotlib axes."""

    xlabel: str  # Horizontal label text; empty if the label or axis is hidden.
    ylabel: str  # Vertical label text; empty if the label or axis is hidden.
    visible_curves: int  # Nontransparent stroked line paths intersecting the axes.


@dataclass(frozen=True)
class PlotLabelBoundsObservation:
    """Rendered horizontal ylabel bounds in one visible canvas's pixel coordinates."""

    canvas_name: str  # Public objectName of the canvas containing this axes.
    canvas_left: float  # Left canvas boundary in renderer coordinates.
    axes_left: float  # Left edge of the data axes, excluding labels and ticks.
    ylabel_left: float  # Left edge of the rendered vertical label.
    ylabel_right: float  # Right edge of the rendered vertical label.
    canvas_right: float  # Right canvas boundary in renderer coordinates.


@dataclass(frozen=True)
class PlotObservation:
    """Visible axes in canvas/axes order, without retaining artists or result data."""

    axes: tuple[PlotAxesObservation, ...]  # Empty when no plot axes are visible.
