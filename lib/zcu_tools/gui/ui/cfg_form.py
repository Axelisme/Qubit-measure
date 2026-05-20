"""CfgFormWidget — renders a CfgSchema as an interactive Qt form.

Each CfgNode type maps to a specific widget strategy:
- ScalarField      → scalar input widget (spin / combo / checkbox / line edit)
- SweepField       → _SweepRow: three inline inputs (start / stop / expts)
                     step field is stored but hidden; future phase adds mode toggle
- MultiSweepField  → one _SweepRow per axis with axis label
- CfgSection       → collapsible QGroupBox containing a sub-form
- ModuleRefField   → QComboBox for module_name + collapsible expanded_content sub-form

read_schema() returns a deep copy of the original schema with all values updated
from current widget state. The original schema passed to populate() is never mutated.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.gui.adapter import (
        CfgNode,
        CfgSchema,
        CfgSection,
        ModuleRefField,
        WaveformRefField,
        ScalarField,
        SweepField,
    )


# ---------------------------------------------------------------------------
# Scalar widget helpers (shared with analyze params in main_window)
# ---------------------------------------------------------------------------


def make_scalar_widget(field: "ScalarField") -> QWidget:
    """Build an input widget from a ScalarField."""
    if field.choices:
        w = QComboBox()
        w.addItems([str(c) for c in field.choices])
        idx = w.findText(str(field.value))
        if idx >= 0:
            w.setCurrentIndex(idx)
        w.setEnabled(field.editable)
        return w
    if field.type is bool:
        w = QCheckBox()
        w.setChecked(bool(field.value))
        w.setEnabled(field.editable)
        return w
    if field.type is int:
        w = QSpinBox()
        w.setRange(-(2**31), 2**31 - 1)
        w.setValue(int(field.value))
        w.setEnabled(field.editable)
        return w
    if field.type is float:
        w = QDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(6)
        w.setValue(float(field.value))
        w.setEnabled(field.editable)
        return w
    w = QLineEdit(str(field.value))
    w.setEnabled(field.editable)
    return w


def read_scalar_widget(w: QWidget, field: "ScalarField") -> Any:
    """Read the current value from a widget created by make_scalar_widget."""
    if isinstance(w, QComboBox):
        txt = w.currentText()
        return field.type(txt) if field.type is not str else txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, QDoubleSpinBox):
        return w.value()
    if isinstance(w, QLineEdit):
        return field.type(w.text())
    return field.value


# ---------------------------------------------------------------------------
# _SweepRow — inline three-cell widget for SweepField
# ---------------------------------------------------------------------------


class _SweepRow(QWidget):
    """Inline widget for a SweepField: [start] [stop] [expts].

    step is stored from the original field but not shown; a future phase
    can add a mode toggle button to switch between expts and step inputs.
    """

    def __init__(self, field: "SweepField", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._original_step = field.step  # preserved for read_back round-trip

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._start = QDoubleSpinBox()
        self._start.setRange(-1e12, 1e12)
        self._start.setDecimals(6)
        self._start.setValue(field.start)
        self._start.setEnabled(field.editable)

        self._stop = QDoubleSpinBox()
        self._stop.setRange(-1e12, 1e12)
        self._stop.setDecimals(6)
        self._stop.setValue(field.stop)
        self._stop.setEnabled(field.editable)

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setValue(field.expts)
        self._expts.setEnabled(field.editable)

        layout.addWidget(QLabel("start"))
        layout.addWidget(self._start, stretch=1)
        layout.addWidget(QLabel("stop"))
        layout.addWidget(self._stop, stretch=1)
        layout.addWidget(QLabel("pts"))
        layout.addWidget(self._expts)

    def read_back(self) -> tuple[float, float, int, Optional[float]]:
        """Return (start, stop, expts, step) — step preserved from original."""
        return (
            self._start.value(),
            self._stop.value(),
            self._expts.value(),
            self._original_step,
        )


# ---------------------------------------------------------------------------
# _CollapsibleSection — QGroupBox with toggle button
# ---------------------------------------------------------------------------


class _CollapsibleSection(QWidget):
    """A labelled section that can be collapsed/expanded via a toggle button."""

    def __init__(
        self,
        label: str,
        collapsible: bool = True,
        collapsed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if collapsible:
            self._toggle_btn = QPushButton(label)
            self._toggle_btn.setCheckable(True)
            self._toggle_btn.setChecked(not collapsed)
            self._toggle_btn.setStyleSheet("text-align: left; padding: 2px 4px;")
            self._toggle_btn.clicked.connect(self._on_toggle)
            outer.addWidget(self._toggle_btn)
        else:
            outer.addWidget(QLabel(f"<b>{label}</b>"))
            self._toggle_btn = None  # type: ignore[assignment]

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 2, 0, 2)
        body_layout.setSpacing(2)
        outer.addWidget(self._body)

        self._form = QFormLayout()
        self._form.setContentsMargins(0, 0, 0, 0)
        body_layout.addLayout(self._form)

        if collapsible:
            self._body.setVisible(not collapsed)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)

    @property
    def form(self) -> QFormLayout:
        return self._form


# ---------------------------------------------------------------------------
# _NodeWidget — wraps the widget and the original CfgNode for read-back
# ---------------------------------------------------------------------------


class _NodeWidget:
    """Associates a rendered widget with the CfgNode it represents."""

    __slots__ = ("node", "widget")

    def __init__(self, node: "CfgNode", widget: QWidget) -> None:
        self.node = node
        self.widget = widget


# ---------------------------------------------------------------------------
# CfgFormWidget — top-level form
# ---------------------------------------------------------------------------


class CfgFormWidget(QWidget):
    """Renders a CfgSchema as an interactive Qt form.

    Call populate(schema) to build the form; read_schema() to retrieve
    a new CfgSchema with current widget values (original is not mutated).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._schema: Optional["CfgSchema"] = None
        self._root_widget: Optional[QWidget] = None
        self._ml: Optional["ModuleLibrary"] = None
        self._node_widgets: list[
            tuple[str, _NodeWidget]
        ] = []  # Kept for compatibility but unused

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
        self, schema: "CfgSchema", ml: Optional["ModuleLibrary"] = None
    ) -> None:
        """Build widget tree from schema. Clears any previous form."""
        self._schema = schema
        self._ml = ml
        self._clear_inner()

        section_widget = self._build_section(schema.root, top_level=True)
        self._root_widget = section_widget
        # insert before the trailing stretch
        self._inner_layout.insertWidget(self._inner_layout.count() - 1, section_widget)
        logger.debug("CfgFormWidget.populate: built form root section widget")

    def read_schema(self) -> "CfgSchema":
        """Return a deep-copy of the original schema with widget values applied."""
        if self._schema is None or self._root_widget is None:
            raise RuntimeError("populate() must be called before read_schema()")
        new_schema = copy.deepcopy(self._schema)
        self._apply_to_section(new_schema.root, self._root_widget)
        return new_schema

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    def _clear_inner(self) -> None:
        while self._inner_layout.count() > 1:  # keep trailing stretch
            item = self._inner_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()

    def _build_section(self, section: "CfgSection", top_level: bool = False) -> QWidget:
        label = section.label or ("Config" if top_level else "")
        collapsible = section.collapsible and not top_level

        container = _CollapsibleSection(label, collapsible=collapsible, collapsed=False)
        container._child_widgets = {}

        for key, node in section.fields.items():
            w, row_widget = self._build_node(node)
            container.form.addRow(_label_for(node, key) + ":", row_widget)
            container._child_widgets[key] = w

        return container

    def _build_node(self, node: "CfgNode") -> tuple[QWidget, QWidget]:
        """Return (value_widget, row_widget).

        For most nodes value_widget == row_widget; for complex nodes
        (section, module) the row_widget is a container while value_widget
        is the primary interactive control.
        """
        from zcu_tools.gui.adapter import (
            CfgSection,
            ModuleRefField,
            WaveformRefField,
            MultiSweepField,
            ScalarField,
            SweepField,
        )

        if isinstance(node, ScalarField):
            w = make_scalar_widget(node)
            return w, w

        if isinstance(node, SweepField):
            w = _SweepRow(node)
            return w, w

        if isinstance(node, MultiSweepField):
            container = QWidget()
            layout = QFormLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            rows: list[_SweepRow] = []
            for axis, sf in node.sweeps.items():
                row = _SweepRow(sf)
                rows.append(row)
                layout.addRow(f"  {axis}:", row)
            # store rows list on container for read-back
            container._sweep_rows = rows  # type: ignore[attr-defined]
            return container, container

        if isinstance(node, CfgSection):
            sub = self._build_section(node, top_level=False)
            return sub, sub

        if isinstance(node, ModuleRefField):
            return self._build_module_ref(node)

        if isinstance(node, WaveformRefField):
            return self._build_waveform_ref(node)

        raise TypeError(f"Unknown CfgNode type: {type(node)}")

    def _build_module_ref(self, node: "ModuleRefField") -> tuple[QWidget, QWidget]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        combo = QComboBox()
        if node.type_filter is not None and self._ml is not None:
            items = [
                name
                for name, mod in self._ml.modules.items()
                if isinstance(mod, node.type_filter)
            ]
        else:
            items = list(node.available_modules) if node.available_modules else []

        if "<Custom>" not in items:
            items.append("<Custom>")
        combo.addItems(items)

        if node.module_name and node.module_name in items:
            combo.setCurrentText(node.module_name)
        else:
            combo.setCurrentText("<Custom>")

        layout.addWidget(combo)

        container._child_widgets = {}

        # expanded_content sub-form (collapsed by default)
        if node.expanded_content is not None:
            sub = self._build_section(node.expanded_content, top_level=False)
            # force collapsed on ModuleRefField sub-section
            if hasattr(sub, "_toggle_btn") and sub._toggle_btn is not None:
                sub._toggle_btn.setChecked(False)
                sub._body.setVisible(False)
            layout.addWidget(sub)
            container._sub_section_widget = sub
            container._child_widgets["expanded_content"] = sub
        else:
            container._sub_section_widget = None

        combo._container = container
        combo.currentIndexChanged.connect(
            lambda: self._on_module_changed(node, combo, container)
        )

        return combo, container

    def _build_waveform_ref(self, node: "WaveformRefField") -> tuple[QWidget, QWidget]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        combo = QComboBox()
        if node.type_filter is not None and self._ml is not None:
            items = [
                name
                for name, wav in self._ml.waveforms.items()
                if isinstance(wav, node.type_filter)
            ]
        else:
            items = list(node.available_waveforms) if node.available_waveforms else []

        if "<Custom>" not in items:
            items.append("<Custom>")
        combo.addItems(items)

        if node.waveform_name and node.waveform_name in items:
            combo.setCurrentText(node.waveform_name)
        else:
            combo.setCurrentText("<Custom>")

        layout.addWidget(combo)

        container._child_widgets = {}

        # expanded_content sub-form (collapsed by default)
        if node.expanded_content is not None:
            sub = self._build_section(node.expanded_content, top_level=False)
            if hasattr(sub, "_toggle_btn") and sub._toggle_btn is not None:
                sub._toggle_btn.setChecked(False)
                sub._body.setVisible(False)
            layout.addWidget(sub)
            container._sub_section_widget = sub
            container._child_widgets["expanded_content"] = sub
        else:
            container._sub_section_widget = None

        combo._container = container
        combo.currentIndexChanged.connect(
            lambda: self._on_waveform_changed(node, combo, container)
        )

        return combo, container

    def _on_module_changed(
        self, node: "ModuleRefField", combo: QComboBox, container: QWidget
    ) -> None:
        name = combo.currentText()
        layout = container.layout()
        assert layout is not None

        # 1. Clear old sub-section widget
        if (
            hasattr(container, "_sub_section_widget")
            and container._sub_section_widget is not None
        ):
            layout.removeWidget(container._sub_section_widget)
            container._sub_section_widget.deleteLater()
            container._sub_section_widget = None
            if (
                hasattr(container, "_child_widgets")
                and "expanded_content" in container._child_widgets
            ):
                del container._child_widgets["expanded_content"]

        # 2. Reset or load expanded content
        if name == "<Custom>":
            node.module_name = None
            if node.custom_template is not None:
                node.expanded_content = copy.deepcopy(node.custom_template)
            else:
                node.expanded_content = None
        else:
            node.module_name = name
            if self._ml is not None:
                try:
                    mod_cfg = self._ml.get_module(name)
                    node.expanded_content = self._make_override_section_for_module(
                        mod_cfg
                    )
                except Exception as e:
                    logger.error(f"Error loading module {name} from ModuleLibrary: {e}")
                    node.expanded_content = None
            else:
                node.expanded_content = None

        # 3. Build and add new sub-section widget (collapsed by default, or open if user selects it)
        if node.expanded_content is not None:
            sub = self._build_section(node.expanded_content, top_level=False)
            layout.addWidget(sub)
            container._sub_section_widget = sub
            if hasattr(container, "_child_widgets"):
                container._child_widgets["expanded_content"] = sub

    def _on_waveform_changed(
        self, node: "WaveformRefField", combo: QComboBox, container: QWidget
    ) -> None:
        name = combo.currentText()
        layout = container.layout()
        assert layout is not None

        # 1. Clear old sub-section widget
        if (
            hasattr(container, "_sub_section_widget")
            and container._sub_section_widget is not None
        ):
            layout.removeWidget(container._sub_section_widget)
            container._sub_section_widget.deleteLater()
            container._sub_section_widget = None
            if (
                hasattr(container, "_child_widgets")
                and "expanded_content" in container._child_widgets
            ):
                del container._child_widgets["expanded_content"]

        # 2. Reset or load expanded content
        if name == "<Custom>":
            node.waveform_name = None
            if node.custom_template is not None:
                node.expanded_content = copy.deepcopy(node.custom_template)
            else:
                node.expanded_content = None
        else:
            node.waveform_name = name
            if self._ml is not None:
                try:
                    wav_cfg = self._ml.get_waveform(name)
                    node.expanded_content = self._make_override_section_for_module(
                        wav_cfg
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading waveform {name} from ModuleLibrary: {e}"
                    )
                    node.expanded_content = None
            else:
                node.expanded_content = None

        # 3. Build and add new sub-section widget
        if node.expanded_content is not None:
            sub = self._build_section(node.expanded_content, top_level=False)
            layout.addWidget(sub)
            container._sub_section_widget = sub
            if hasattr(container, "_child_widgets"):
                container._child_widgets["expanded_content"] = sub

    def _make_override_section_for_module(self, mod_cfg: Any) -> "CfgSection":
        from zcu_tools.gui.adapter import module_cfg_to_section

        return module_cfg_to_section(mod_cfg)

    # ------------------------------------------------------------------
    # Read-back helpers
    # ------------------------------------------------------------------

    def _apply_to_section(
        self,
        section: "CfgSection",
        container: QWidget,
    ) -> None:
        """Recursively apply values from widget tree to CfgSchema."""
        from zcu_tools.gui.adapter import (
            CfgSection,
            ModuleRefField,
            WaveformRefField,
            MultiSweepField,
            ScalarField,
            SweepField,
        )

        child_widgets = getattr(container, "_child_widgets", {})

        for key in section.fields:
            w = child_widgets.get(key)
            if w is None:
                continue
            node = section.fields[key]

            if isinstance(node, ScalarField):
                node.value = read_scalar_widget(w, node)

            elif isinstance(node, SweepField):
                assert isinstance(w, _SweepRow)
                start, stop, expts, step = w.read_back()
                node.start = start
                node.stop = stop
                node.expts = expts
                node.step = step

            elif isinstance(node, MultiSweepField):
                rows: list[_SweepRow] = getattr(w, "_sweep_rows", [])
                for sf, row in zip(node.sweeps.values(), rows):
                    start, stop, expts, step = row.read_back()
                    sf.start = start
                    sf.stop = stop
                    sf.expts = expts
                    sf.step = step

            elif isinstance(node, CfgSection):
                self._apply_to_section(node, w)

            elif isinstance(node, ModuleRefField):
                combo = w
                node.module_name = combo.currentText() or None
                if node.module_name == "<Custom>":
                    node.module_name = None

                container_widget = getattr(combo, "_container", None)
                if node.expanded_content is not None and container_widget is not None:
                    sub_widget = getattr(container_widget, "_sub_section_widget", None)
                    if sub_widget is not None:
                        self._apply_to_section(node.expanded_content, sub_widget)

            elif isinstance(node, WaveformRefField):
                combo = w
                node.waveform_name = combo.currentText() or None
                if node.waveform_name == "<Custom>":
                    node.waveform_name = None

                container_widget = getattr(combo, "_container", None)
                if node.expanded_content is not None and container_widget is not None:
                    sub_widget = getattr(container_widget, "_sub_section_widget", None)
                    if sub_widget is not None:
                        self._apply_to_section(node.expanded_content, sub_widget)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _label_for(node: "CfgNode", key: str) -> str:
    label = getattr(node, "label", "")
    return label if label else key
