"""CfgFormWidget — renders a CfgSchema as an interactive Qt form.

Spec/Value split design:
- populate(schema) reads spec to build widget structure, value to fill initial values.
  The schema is never mutated; the form owns its own internal state.
- read_values() rebuilds a CfgSectionValue purely from current widget state.
- Callers assemble a new CfgSchema with CfgSchema(schema.spec, form.read_values()).

Each CfgNodeSpec type maps to a specific widget strategy:
- ScalarSpec      → scalar input widget (spin / combo / checkbox / line edit)
- SweepSpec       → _SweepRow: three inline inputs (start / stop / expts)
- MultiSweepSpec  → one _SweepRow per axis
- CfgSectionSpec  → collapsible sub-form
- ModuleRefSpec   → QComboBox for module name + collapsible sub-form from chosen spec
- WaveformRefSpec → same as ModuleRefSpec
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from zcu_tools.gui.event_bus import GuiEvent

logger = logging.getLogger(__name__)

from qtpy.QtCore import (  # type: ignore[attr-defined]
    Qt,
    QTimer,
    Signal,  # type: ignore[attr-defined]
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .widgets import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import (
        CfgNodeSpec,
        CfgNodeValue,
        CfgSchema,
        CfgSectionSpec,
        CfgSectionValue,
        ChannelValue,
        ModuleRefSpec,
        ModuleRefValue,
        ScalarSpec,
        WaveformRefSpec,
        WaveformRefValue,
    )
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


# ---------------------------------------------------------------------------
# Scalar widget helpers (shared with analyze params in main_window)
# ---------------------------------------------------------------------------


def make_value_widget(
    type_: type,
    default: Any,
    choices: Optional[list],
    editable: bool = True,
    decimals: Optional[int] = None,
) -> QWidget:
    """Build an input widget from raw field attributes."""
    if choices:
        w = QComboBox()
        w.addItems([str(c) for c in choices])
        idx = w.findText(str(default))
        if idx >= 0:
            w.setCurrentIndex(idx)
        w.setEnabled(editable)
        return w
    if type_ is bool:
        w = QCheckBox()
        w.setChecked(bool(default))
        w.setEnabled(editable)
        return w
    if type_ is int:
        w = QSpinBox()
        w.setRange(-(2**31), 2**31 - 1)
        w.setValue(int(default))
        w.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        w.setEnabled(editable)
        return w
    if type_ is float:
        w = TrimDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(decimals if decimals is not None else 6)
        w.setValue(float(default))
        w.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        w.setEnabled(editable)
        return w
    w = QLineEdit(str(default))
    w.setEnabled(editable)
    return w


def read_value_widget(w: QWidget, type_: type, fallback: Any = None) -> Any:
    """Read the current value from a widget created by make_value_widget."""
    if isinstance(w, QComboBox):
        txt = w.currentText()
        return type_(txt) if type_ is not str else txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, TrimDoubleSpinBox):
        return w.value()
    if isinstance(w, QLineEdit):
        return type_(w.text())
    return fallback


def make_scalar_widget(spec: "ScalarSpec", value: Any) -> QWidget:
    """Build an input widget from a ScalarSpec and initial value."""
    return make_value_widget(
        spec.type, value, spec.choices, spec.editable, spec.decimals
    )


def read_scalar_widget(w: QWidget, spec: "ScalarSpec") -> Any:
    """Read the current value from a widget created by make_scalar_widget."""
    return read_value_widget(w, spec.type, fallback=None)


# ---------------------------------------------------------------------------
# _SweepRow — inline three-cell widget for SweepSpec
# ---------------------------------------------------------------------------


class _SweepRow(QWidget):
    def __init__(
        self,
        start: float,
        stop: float,
        expts: int,
        step: Optional[float],
        editable: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._original_step = step

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._start = TrimDoubleSpinBox()
        self._start.setRange(-1e12, 1e12)
        self._start.setDecimals(6)
        self._start.setValue(start)
        self._start.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._start.setMinimumWidth(30)
        self._start.setEnabled(editable)

        self._stop = TrimDoubleSpinBox()
        self._stop.setRange(-1e12, 1e12)
        self._stop.setDecimals(6)
        self._stop.setValue(stop)
        self._stop.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._stop.setMinimumWidth(30)
        self._stop.setEnabled(editable)

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setValue(expts)
        self._expts.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._expts.setMinimumWidth(30)
        self._expts.setEnabled(editable)

        layout.addWidget(QLabel("start"))
        layout.addWidget(self._start, stretch=1)
        layout.addWidget(QLabel("stop"))
        layout.addWidget(self._stop, stretch=1)
        layout.addWidget(QLabel("pts"))
        layout.addWidget(self._expts)

    def read_back(self) -> tuple[float, float, int, Optional[float]]:
        return (
            self._start.value(),
            self._stop.value(),
            self._expts.value(),
            self._original_step,
        )


# ---------------------------------------------------------------------------
# _ChannelRow — single QLineEdit accepting int or md-key string
# ---------------------------------------------------------------------------


def _resolve_channel(text: str, md: "Optional[MetaDict]") -> Optional[int]:
    """Resolve a channel text: int string → int directly; md-key → lookup."""
    try:
        v = int(text)
        if v >= 0:
            return v
        return None
    except ValueError:
        pass
    if md is None:
        return None
    try:
        raw = getattr(md, text, None)
        if isinstance(raw, int) and raw >= 0:
            return raw
    except Exception:
        pass
    return None


class _ChannelRow(QWidget):
    validity_changed: Signal = Signal(bool)

    def __init__(
        self,
        chosen: Union[int, str],
        md: "Optional[MetaDict]",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._md = md
        self._valid = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._edit = QLineEdit(str(chosen))
        self._edit.setMinimumWidth(60)
        self._edit.textChanged.connect(self._refresh_ghost)
        layout.addWidget(self._edit, stretch=1)

        self._ghost = QLabel()
        self._ghost.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._ghost)

        self._refresh_ghost()

    def _refresh_ghost(self) -> None:
        text = self._edit.text().strip()
        try:
            v = int(text)
            if v >= 0:
                self._ghost.setText("")
                self._ghost.setStyleSheet("color: gray; font-style: italic;")
                self._set_valid(True)
                return
        except ValueError:
            pass
        # md-key path
        resolved = _resolve_channel(text, self._md)
        if resolved is not None:
            self._ghost.setText(f"= {resolved}")
            self._ghost.setStyleSheet("color: gray; font-style: italic;")
            self._set_valid(True)
        else:
            self._ghost.setText("= ?")
            self._ghost.setStyleSheet("color: red; font-style: italic;")
            self._set_valid(False)

    def _set_valid(self, valid: bool) -> None:
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)

    def is_valid(self) -> bool:
        return self._valid

    def refresh_md(self, md: "Optional[MetaDict]") -> None:
        self._md = md
        self._refresh_ghost()

    def read_back(self) -> "ChannelValue":
        from zcu_tools.gui.adapter import ChannelValue

        text = self._edit.text().strip()
        try:
            v = int(text)
            if v >= 0:
                return ChannelValue(chosen=v, resolved=None)
        except ValueError:
            pass
        resolved = _resolve_channel(text, self._md)
        return ChannelValue(chosen=text, resolved=resolved)


# ---------------------------------------------------------------------------
# _CollapsibleSection
# ---------------------------------------------------------------------------


class _CollapsibleSection(QWidget):
    def __init__(
        self,
        label: str,
        collapsible: bool = True,
        collapsed: bool = False,
        no_header: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = None  # type: ignore[assignment]

        if not no_header:
            if collapsible:
                header = QWidget()
                header_row = QHBoxLayout(header)
                header_row.setContentsMargins(0, 0, 0, 0)
                header_row.setSpacing(2)

                self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
                self._toggle_btn.setFixedWidth(16)
                self._toggle_btn.setFlat(True)
                self._toggle_btn.setCheckable(True)
                self._toggle_btn.setChecked(not collapsed)
                self._toggle_btn.clicked.connect(self._on_toggle)
                header_row.addWidget(self._toggle_btn)
                header_row.addWidget(QLabel(f"<b>{label}</b>"), stretch=1)
                outer.addWidget(header)
            else:
                if label:
                    outer.addWidget(QLabel(f"<b>{label}</b>"))

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 2, 0, 2)
        body_layout.setSpacing(2)
        outer.addWidget(self._body)

        self._form = QFormLayout()
        self._form.setContentsMargins(0, 0, 0, 0)
        body_layout.addLayout(self._form)

        if collapsible and not no_header:
            self._body.setVisible(not collapsed)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        if self._toggle_btn is not None:
            self._toggle_btn.setText("▼" if checked else "▶")

    @property
    def form(self) -> QFormLayout:
        return self._form


# ---------------------------------------------------------------------------
# CfgFormWidget
# ---------------------------------------------------------------------------


class CfgFormWidget(QWidget):
    """Renders a CfgSchema as an interactive Qt form.

    populate(schema) builds the widget tree from spec+value; the schema is
    never mutated. read_values() returns a fresh CfgSectionValue from widget state.
    Callers compose a new CfgSchema with CfgSchema(schema.spec, form.read_values()).
    """

    validity_changed: Signal = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._spec: Optional["CfgSectionSpec"] = None
        self._root_widget: Optional[QWidget] = None
        self._ml: Optional["ModuleLibrary"] = None
        self._md: Optional["MetaDict"] = None
        self._bus: Optional["EventBus"] = None
        self._channel_rows: list[_ChannelRow] = []
        self._bus_cb: Optional[Callable[[], None]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        outer.addWidget(scroll)

        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(4, 4, 4, 4)
        self._inner_layout.setSpacing(4)
        self._inner_layout.addStretch()
        scroll.setWidget(self._inner)

    # ------------------------------------------------------------------

    def populate(
        self,
        schema: "CfgSchema",
        ml: Optional["ModuleLibrary"] = None,
        md: Optional["MetaDict"] = None,
        bus: Optional["EventBus"] = None,
    ) -> None:
        """Build widget tree from schema. Schema is never mutated."""
        self._spec = schema.spec
        self._ml = ml
        self._md = md
        self._channel_rows = []
        self._clear_inner()
        self._teardown_md_updates()

        section_widget = self._build_section(schema.spec, schema.value, top_level=True)
        self._root_widget = section_widget
        self._inner_layout.insertWidget(self._inner_layout.count() - 1, section_widget)
        logger.debug("CfgFormWidget.populate: built form root section widget")

        if self._channel_rows:
            self._setup_md_updates(bus)
            for row in self._channel_rows:
                row.validity_changed.connect(self._on_channel_validity_changed)
        # Emit initial validity state
        self.validity_changed.emit(self._is_valid())

    def read_values(self) -> "CfgSectionValue":
        """Return a new CfgSectionValue from current widget state."""
        if self._spec is None or self._root_widget is None:
            raise RuntimeError("populate() must be called before read_values()")
        return self._read_section(self._spec, self._root_widget)

    def _is_valid(self) -> bool:
        return all(row.is_valid() for row in self._channel_rows)

    def _on_channel_validity_changed(self) -> None:
        self.validity_changed.emit(self._is_valid())

    def read_schema(self) -> "CfgSchema":
        """Return a new CfgSchema combining the stored spec with current widget values."""
        from zcu_tools.gui.adapter import CfgSchema

        if self._spec is None:
            raise RuntimeError("populate() must be called before read_schema()")
        return CfgSchema(spec=self._spec, value=self.read_values())

    def _teardown_md_updates(self) -> None:
        if self._bus is not None and self._bus_cb is not None:
            self._bus.unsubscribe(GuiEvent.MD_CHANGED, self._bus_cb)
            self._bus.unsubscribe(GuiEvent.CONTEXT_CHANGED, self._bus_cb)
            self._bus_cb = None
        self._bus = None

    def _setup_md_updates(self, bus: Optional["EventBus"]) -> None:
        if bus is None:
            from zcu_tools.gui.event_bus import EventBus

            bus = EventBus()
        self._bus = bus
        self._bus_cb = self._refresh_channel_ghosts
        self._bus.subscribe(GuiEvent.MD_CHANGED, self._bus_cb)
        self._bus.subscribe(GuiEvent.CONTEXT_CHANGED, self._bus_cb)

    def _refresh_channel_ghosts(self) -> None:
        for row in self._channel_rows:
            row.refresh_md(self._md)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _clear_inner(self) -> None:
        while self._inner_layout.count() > 1:
            item = self._inner_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()

    def _build_section(
        self,
        spec: "CfgSectionSpec",
        value: "CfgSectionValue",
        top_level: bool = False,
        no_header: bool = False,
    ) -> QWidget:
        label = spec.label or ("Config" if top_level else "")
        collapsible = spec.collapsible and not top_level and not no_header

        container = _CollapsibleSection(
            label, collapsible=collapsible, collapsed=False, no_header=no_header
        )
        container._child_widgets = {}  # type: ignore[attr-defined]
        container._hidden_fields: dict[str, Any] = {}  # type: ignore[attr-defined]

        from zcu_tools.gui.adapter import LiteralSpec as _LiteralSpec
        from zcu_tools.gui.adapter import ScalarSpec as _ScalarSpec
        from zcu_tools.gui.adapter import ScalarValue as _ScalarValue

        for key, node_spec in spec.fields.items():
            node_val = value.fields.get(key)
            if isinstance(node_spec, _LiteralSpec):
                # LiteralSpec: no widget, value is fixed by spec
                container._hidden_fields[key] = _ScalarValue(node_spec.value)  # type: ignore[attr-defined]
                continue
            if isinstance(node_spec, _ScalarSpec) and node_spec.hidden:
                raw = node_val.value if isinstance(node_val, _ScalarValue) else None
                container._hidden_fields[key] = _ScalarValue(raw)  # type: ignore[attr-defined]
                continue
            w, row_widget = self._build_node(node_spec, node_val)
            container.form.addRow(_label_for(node_spec, key) + ":", row_widget)
            container._child_widgets[key] = w  # type: ignore[attr-defined]

        return container

    def _build_node(
        self,
        node_spec: "CfgNodeSpec",
        node_val: "Optional[CfgNodeValue]",
    ) -> tuple[QWidget, QWidget]:
        from zcu_tools.gui.adapter import (
            CfgSectionSpec,
            CfgSectionValue,
            ChannelSpec,
            ChannelValue,
            ModuleRefSpec,
            ModuleRefValue,
            MultiSweepSpec,
            MultiSweepValue,
            ScalarSpec,
            ScalarValue,
            SweepSpec,
            SweepValue,
            WaveformRefSpec,
            WaveformRefValue,
            make_default_value,
        )

        if isinstance(node_spec, ChannelSpec):
            chosen = node_val.chosen if isinstance(node_val, ChannelValue) else 0
            w = _ChannelRow(chosen, self._md)
            self._channel_rows.append(w)
            return w, w

        if isinstance(node_spec, ScalarSpec):
            val = (
                node_val.value
                if isinstance(node_val, ScalarValue)
                else (
                    node_spec.choices[0]
                    if node_spec.choices
                    else {int: 0, float: 0.0, bool: False, str: ""}[node_spec.type]
                )
            )
            w = make_scalar_widget(node_spec, val)
            return w, w

        if isinstance(node_spec, SweepSpec):
            sv = (
                node_val
                if isinstance(node_val, SweepValue)
                else SweepValue(0.0, 1.0, 11)
            )
            w = _SweepRow(sv.start, sv.stop, sv.expts, sv.step, node_spec.editable)
            return w, w

        if isinstance(node_spec, MultiSweepSpec):
            mv = node_val if isinstance(node_val, MultiSweepValue) else None
            container = QWidget()
            layout = QFormLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            rows: list[_SweepRow] = []
            for axis, axis_spec in node_spec.axes.items():
                if mv is not None:
                    sv = mv.axes.get(axis, SweepValue(0.0, 1.0, 11))
                else:
                    sv = SweepValue(0.0, 1.0, 11)
                row = _SweepRow(
                    sv.start, sv.stop, sv.expts, sv.step, axis_spec.editable
                )
                rows.append(row)
                layout.addRow(f"  {axis}:", row)
            container._sweep_rows = rows  # type: ignore[attr-defined]
            container._axis_names = list(node_spec.axes.keys())  # type: ignore[attr-defined]
            return container, container

        if isinstance(node_spec, CfgSectionSpec):
            cv = (
                node_val if isinstance(node_val, CfgSectionValue) else CfgSectionValue()
            )
            sub = self._build_section(node_spec, cv, top_level=False)
            return sub, sub

        if isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            ref_val: Optional[Union[ModuleRefValue, WaveformRefValue]] = None
            if isinstance(node_spec, ModuleRefSpec) and isinstance(
                node_val, ModuleRefValue
            ):
                ref_val = node_val
            elif isinstance(node_spec, WaveformRefSpec) and isinstance(
                node_val, WaveformRefValue
            ):
                ref_val = node_val
            if ref_val is None:
                first = node_spec.allowed[0] if node_spec.allowed else CfgSectionSpec()
                label = first.label or "Custom"
                default_val = make_default_value(first)
                ref_val = (
                    ModuleRefValue(f"<Custom:{label}>", default_val)
                    if isinstance(node_spec, ModuleRefSpec)
                    else WaveformRefValue(f"<Custom:{label}>", default_val)
                )
            return self._build_ref_field(node_spec, ref_val)

        raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")  # type: ignore[unreachable]

    def _build_ref_field(
        self,
        node_spec: "Union[ModuleRefSpec, WaveformRefSpec]",
        node_val: "Union[ModuleRefValue, WaveformRefValue]",
    ) -> tuple[QWidget, QWidget]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row: [ toggle_btn ] [ combo ]
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        toggle_btn = QToolButton()
        toggle_btn.setArrowType(Qt.RightArrow)  # type: ignore[attr-defined]
        toggle_btn.setAutoRaise(True)
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(False)
        toggle_btn.setStyleSheet("background: transparent; border: none;")

        combo = QComboBox()
        items = self._ref_items(node_spec)
        combo.addItems(items)

        # Set current selection
        chosen = node_val.chosen_key
        if chosen in items:
            combo.setCurrentText(chosen)
        else:
            custom_label = f"<Custom:{_spec_for_chosen(node_spec, chosen).label}>"
            if custom_label in items:
                combo.setCurrentText(custom_label)
            else:
                combo.setCurrentIndex(0)

        header_layout.addWidget(toggle_btn)
        header_layout.addWidget(combo, stretch=1)
        layout.addWidget(header)

        container._child_widgets = {}  # type: ignore[attr-defined]
        container._spec = node_spec  # type: ignore[attr-defined]

        # Build initial sub-section (no header: outer toggle replaces it)
        # Named module: derive spec from ml (same as _on_ref_changed Named path)
        # Custom: fall back to _spec_for_chosen by label
        chosen_key = node_val.chosen_key
        chosen_spec: "CfgSectionSpec" = _spec_for_chosen(node_spec, chosen_key)
        if not chosen_key.startswith("<Custom:") and self._ml is not None:
            try:
                from zcu_tools.gui.adapter import ModuleRefSpec
                from zcu_tools.gui.cfg_schemas import (
                    module_cfg_to_value,
                    waveform_cfg_to_value,
                )

                if isinstance(node_spec, ModuleRefSpec):
                    cfg = self._ml.get_module(chosen_key)
                    chosen_spec, _ = module_cfg_to_value(cfg)
                else:
                    cfg = self._ml.get_waveform(chosen_key)
                    chosen_spec, _ = waveform_cfg_to_value(cfg)
            except Exception:
                pass
        sub = self._build_section(chosen_spec, node_val.value, no_header=True)
        sub.setVisible(False)
        layout.addWidget(sub)
        container._sub_section_widget = sub  # type: ignore[attr-defined]
        container._sub_spec = chosen_spec  # type: ignore[attr-defined]
        container._child_widgets["_sub"] = sub  # type: ignore[attr-defined]

        def _on_toggle(checked: bool) -> None:
            sub_w = getattr(container, "_sub_section_widget", None)
            if sub_w is not None:
                sub_w.setVisible(checked)
            toggle_btn.setArrowType(
                Qt.DownArrow if checked else Qt.RightArrow  # type: ignore[attr-defined]
            )

        toggle_btn.toggled.connect(_on_toggle)
        container._toggle_btn = toggle_btn  # type: ignore[attr-defined]

        combo._container = container  # type: ignore[attr-defined]
        combo.currentIndexChanged.connect(
            lambda: self._on_ref_changed(node_spec, combo, container)
        )

        return combo, container

    def _ref_items(
        self, node_spec: "Union[ModuleRefSpec, WaveformRefSpec]"
    ) -> list[str]:
        """Build ComboBox item list: ml named modules + Custom:<label> per allowed spec."""
        from zcu_tools.gui.adapter import ModuleRefSpec

        items: list[str] = []

        # Named modules from ml filtered by allowed spec labels
        if self._ml is not None:
            allowed_labels = {s.label for s in node_spec.allowed}
            if isinstance(node_spec, ModuleRefSpec):
                for name, mod in self._ml.modules.items():
                    try:
                        from zcu_tools.gui.cfg_schemas import module_cfg_to_value

                        s, _ = module_cfg_to_value(mod)
                        if s.label in allowed_labels:
                            items.append(name)
                    except Exception:
                        pass
            else:
                for name, wav in self._ml.waveforms.items():
                    try:
                        from zcu_tools.gui.cfg_schemas import waveform_cfg_to_value

                        s, _ = waveform_cfg_to_value(wav)
                        if s.label in allowed_labels:
                            items.append(name)
                    except Exception:
                        pass

        # Custom options per allowed spec
        for s in node_spec.allowed:
            items.append(f"<Custom:{s.label}>")

        return items

    def _on_ref_changed(
        self,
        node_spec: "Union[ModuleRefSpec, WaveformRefSpec]",
        combo: QComboBox,
        container: QWidget,
    ) -> None:
        from zcu_tools.gui.adapter import (
            CfgSectionValue,
            ModuleRefSpec,
            inherit_from,
            make_default_value,
        )

        chosen = combo.currentText()
        layout = container.layout()
        assert layout is not None

        # Read current values before destroying old sub-section
        old_spec = getattr(container, "_sub_spec", None)
        old_sub = getattr(container, "_sub_section_widget", None)
        if old_sub is not None and old_spec is not None:
            try:
                old_val: CfgSectionValue = self._read_section(old_spec, old_sub)
            except Exception:
                old_val = CfgSectionValue()
        else:
            old_val = CfgSectionValue()

        # Remove old sub-section widget
        if old_sub is not None:
            layout.removeWidget(old_sub)
            old_sub.deleteLater()
            container._sub_section_widget = None  # type: ignore[attr-defined]

        # Determine new spec and value
        if chosen.startswith("<Custom:"):
            # Custom: start from defaults then inherit matching fields from old sub-section
            chosen_spec = _spec_for_chosen(node_spec, chosen)
            new_val = make_default_value(chosen_spec)
            if old_spec is not None:
                new_val = inherit_from(old_val, old_spec, chosen_spec)
        else:
            # Named module: use ml's cfg as-is (do not inherit)
            chosen_spec = None
            new_val = CfgSectionValue()
            if self._ml is not None:
                try:
                    if isinstance(node_spec, ModuleRefSpec):
                        cfg = self._ml.get_module(chosen)
                    else:
                        cfg = self._ml.get_waveform(chosen)
                    from zcu_tools.gui.cfg_schemas import (
                        module_cfg_to_value,
                        waveform_cfg_to_value,
                    )

                    if isinstance(node_spec, ModuleRefSpec):
                        chosen_spec, new_val = module_cfg_to_value(cfg)
                    else:
                        chosen_spec, new_val = waveform_cfg_to_value(cfg)
                except Exception as e:
                    logger.error("Error loading %r from library: %s", chosen, e)

            if chosen_spec is None:
                chosen_spec = (
                    node_spec.allowed[0] if node_spec.allowed else CfgSectionSpec()
                )  # type: ignore[attr-defined]
                new_val = make_default_value(chosen_spec)

        container._sub_spec = chosen_spec  # type: ignore[attr-defined]

        # Build and attach new sub-section; visibility follows toggle state
        sub = self._build_section(chosen_spec, new_val, no_header=True)
        toggle_btn = getattr(container, "_toggle_btn", None)
        sub.setVisible(toggle_btn.isChecked() if toggle_btn is not None else False)
        layout.addWidget(sub)
        container._sub_section_widget = sub  # type: ignore[attr-defined]
        container._child_widgets["_sub"] = sub  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Read-back helpers
    # ------------------------------------------------------------------

    def _read_section(
        self,
        spec: "CfgSectionSpec",
        container: QWidget,
    ) -> "CfgSectionValue":
        from zcu_tools.gui.adapter import (
            CfgSectionSpec,
            CfgSectionValue,
            ChannelSpec,
            LiteralSpec,
            ModuleRefSpec,
            ModuleRefValue,
            MultiSweepSpec,
            MultiSweepValue,
            ScalarSpec,
            ScalarValue,
            SweepSpec,
            SweepValue,
            WaveformRefSpec,
            WaveformRefValue,
        )

        child_widgets = getattr(container, "_child_widgets", {})
        hidden_fields: dict[str, Any] = getattr(container, "_hidden_fields", {})
        fields: dict[str, Any] = {}

        for key, node_spec in spec.fields.items():
            if isinstance(node_spec, LiteralSpec):
                fields[key] = ScalarValue(node_spec.value)
                continue
            if isinstance(node_spec, ScalarSpec) and node_spec.hidden:
                if key in hidden_fields:
                    fields[key] = hidden_fields[key]
                continue

            w = child_widgets.get(key)
            if w is None:
                continue

            if isinstance(node_spec, ScalarSpec):
                fields[key] = ScalarValue(read_scalar_widget(w, node_spec))

            elif isinstance(node_spec, SweepSpec):
                assert isinstance(w, _SweepRow)
                start, stop, expts, step = w.read_back()
                fields[key] = SweepValue(start, stop, expts, step)

            elif isinstance(node_spec, MultiSweepSpec):
                rows: list[_SweepRow] = getattr(w, "_sweep_rows", [])
                axis_names: list[str] = getattr(
                    w, "_axis_names", list(node_spec.axes.keys())
                )
                axes: dict[str, SweepValue] = {}
                for axis, row in zip(axis_names, rows):
                    start, stop, expts, step = row.read_back()
                    axes[axis] = SweepValue(start, stop, expts, step)
                fields[key] = MultiSweepValue(axes=axes)

            elif isinstance(node_spec, ChannelSpec):
                assert isinstance(w, _ChannelRow)
                fields[key] = w.read_back()

            elif isinstance(node_spec, CfgSectionSpec):
                fields[key] = self._read_section(node_spec, w)

            elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
                # w is the combo; container is combo._container
                combo = w
                cont_widget = getattr(combo, "_container", None)
                sub_spec: "CfgSectionSpec" = getattr(
                    cont_widget, "_sub_spec", CfgSectionSpec()
                )
                sub_widget = getattr(cont_widget, "_sub_section_widget", None)
                if sub_widget is not None:
                    sub_val = self._read_section(sub_spec, sub_widget)
                else:
                    sub_val = CfgSectionValue()

                # Always use "<Custom:label>" so _find_allowed_spec can match by label.
                # Named vs Custom distinction is only needed for UI display, not for
                # schema_to_dict which now always expands the value tree directly.
                canonical_key = f"<Custom:{sub_spec.label}>"

                if isinstance(node_spec, ModuleRefSpec):
                    fields[key] = ModuleRefValue(
                        chosen_key=canonical_key, value=sub_val
                    )
                else:
                    fields[key] = WaveformRefValue(
                        chosen_key=canonical_key, value=sub_val
                    )

        return CfgSectionValue(fields=fields)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _label_for(node_spec: "CfgNodeSpec", key: str) -> str:
    label = getattr(node_spec, "label", "")
    return label if label else key


def _spec_for_chosen(
    ref_spec: "Union[ModuleRefSpec, WaveformRefSpec]",
    chosen_key: str,
) -> "CfgSectionSpec":
    """Find the CfgSectionSpec from allowed[] matching the chosen key."""
    from zcu_tools.gui.adapter import CfgSectionSpec

    # Strip "<Custom:label>" prefix
    if chosen_key.startswith("<Custom:"):
        label = chosen_key[len("<Custom:") : -1]
    else:
        label = chosen_key

    for s in ref_spec.allowed:
        if s.label == label:
            return s
    return ref_spec.allowed[0] if ref_spec.allowed else CfgSectionSpec()
