"""A dense Qt cfg tree presents and edits one published resource."""

from collections.abc import Callable, Iterator, Sequence

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QWidget,
)
from zcu_tools.experiment.cfg_editing.catalog import PROGRAM_SHAPES, ProgramSpecPolicy
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding.ports import ReferenceCatalog, ResolvedReference
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgResolution,
    CfgResource,
    CfgStaleError,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr
from zcu_tools.gui.widgets.cfg import ResourceCfgFormWidget
from zcu_tools.gui.widgets.cfg.fields.common import (
    CenteredSweepInputWidget,
    ScalarInputWidget,
    SweepInputWidget,
)
from zcu_tools.gui.widgets.cfg.fields.reference_shared import NONE_KEY

from tests.gui._dialog_fakes import DialogCall, RecordingDialogPresenter


class EmptyCatalog:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def resource(schema: CfgSchema, catalog: ReferenceCatalog | None = None) -> CfgResource:
    source = EmptyCatalog() if catalog is None else catalog
    return CfgResource(
        lambda: schema,
        resolution=lambda: CfgResolution(
            (),
            lambda expression: 0.0,
            lambda source_id: (),
            source,
            lambda name: 0.0,
            validate_scalar_expr,
        ),
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )


class OneGaussCatalog:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ("saved",) if "Gauss" in allowed_labels else ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        if key != "saved":
            return None
        return ResolvedReference(
            "Gauss",
            CfgSectionValue(
                {
                    "style": DirectValue("gauss"),
                    "length": DirectValue(1.0),
                    "sigma": DirectValue(0.2),
                }
            ),
        )


@pytest.fixture
def form(qapp: QApplication) -> Iterator[ResourceCfgFormWidget]:
    widget = ResourceCfgFormWidget(dialog_presenter=RecordingDialogPresenter())
    widget.resize(600, 450)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.detach()
    widget.deleteLater()
    qapp.processEvents()


def scalar_line(form: ResourceCfgFormWidget, path: str = "value") -> QLineEdit:
    widget = form.findChild(ScalarInputWidget, "cfgInput:" + path)
    assert widget is not None
    line = widget.findChild(QLineEdit)
    assert line is not None
    return line


def type_text(line: QLineEdit, text: str) -> None:
    line.setFocus()
    line.selectAll()
    for char in text:
        for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.sendEvent(
                line,
                QKeyEvent(event_type, 0, Qt.KeyboardModifier.NoModifier, char),
            )


def test_scalar_input_submits_one_batch(form: ResourceCfgFormWidget) -> None:
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
            CfgSectionValue({"value": DirectValue(2.0)}),
        )
    )
    form.attach(owner)
    base = form.current_ref()
    line = scalar_line(form)
    type_text(line, "3.5")
    assert form.has_pending()
    assert owner.observe().ref == base
    submitted = form.submit_pending()
    assert submitted == owner.observe().ref == form.current_ref()
    assert submitted.revision == base.revision + 1
    assert not form.has_pending()
    assert owner.accept(submitted.revision).values == {"value": 3.5}


@pytest.mark.parametrize(
    "shape_change", [False, True], ids=["stable-shape", "changed-shape"]
)
def test_external_update_and_stale_keep_text_focus_selection(
    form: ResourceCfgFormWidget,
    qapp: QApplication,
    shape_change: bool,
) -> None:
    a = ScalarSpec("A", float)
    b = ScalarSpec("B", float)
    choice = ChoiceSectionSpec(
        fields={"mode": ScalarSpec("Mode", str), "a": a, "b": b},
        bindings=(
            ChoiceBinding(
                "mode",
                {
                    "a": CfgSectionSpec(fields={"a": a}),
                    "b": CfgSectionSpec(fields={"b": b}),
                },
            ),
        ),
    )
    owner = resource(
        CfgSchema(
            CfgSectionSpec(
                fields={"value": ScalarSpec("Value", float), "choice": choice}
            ),
            CfgSectionValue(
                {
                    "value": DirectValue(2.0),
                    "choice": CfgSectionValue(
                        {
                            "mode": DirectValue("a"),
                            "a": DirectValue(1.0),
                            "b": DirectValue(2.0),
                        }
                    ),
                }
            ),
        )
    )
    form.attach(owner)
    qapp.processEvents()
    line = scalar_line(form)
    type_text(line, "3.51")
    line.setSelection(1, 2)
    assert line.hasFocus() and line.selectedText() == ".5"
    base = form.current_ref()
    path = ("choice", "mode") if shape_change else ("value",)
    current = owner.edit(
        base.revision, (CfgEdit(path, DirectValue("b" if shape_change else 8.0)),)
    )
    qapp.processEvents()

    assert form.current_ref() == current.ref
    assert line.text() == "3.51" and line.hasFocus() and line.selectedText() == ".5"
    assert any(
        "changed externally" in label.text() for label in form.findChildren(QLabel)
    )
    with pytest.raises(CfgStaleError):
        form.submit_pending()
    assert owner.observe().ref == current.ref
    assert form.has_pending()
    assert line.text() == "3.51" and line.hasFocus() and line.selectedText() == ".5"

    form.discard_pending()
    qapp.processEvents()
    assert not form.has_pending()
    assert owner.observe().ref == current.ref
    assert scalar_line(form).text() == ("2.0" if shape_change else "8.0")
    if shape_change:
        assert form.findChild(ScalarInputWidget, "cfgInput:choice.b") is not None


def test_invalid_input_reflects_published_status_and_reason(
    form: ResourceCfgFormWidget,
) -> None:
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
            CfgSectionValue({"value": DirectValue(2.0)}),
        )
    )
    form.attach(owner)
    input_widget = form.findChild(ScalarInputWidget, "cfgInput:value")
    assert input_widget is not None
    line = input_widget.findChild(QLineEdit)
    assert line is not None

    type_text(line, "-")
    assert owner.observe().status is CfgStatus.VALID
    form.submit_pending()

    assert owner.observe().status is CfgStatus.INVALID
    assert not form.is_valid()
    assert form.first_invalid_reason()
    assert line.text() == "-"


def test_reference_shape_switch_rebuilds_after_qt_signal(
    form: ResourceCfgFormWidget, qapp: QApplication
) -> None:
    policy = ProgramSpecPolicy()
    shapes = [shape.make_spec(policy) for shape in PROGRAM_SHAPES.waveforms()]
    owner = resource(
        CfgSchema(
            CfgSectionSpec(
                fields={
                    "ref": ReferenceSpec(
                        "waveform", shapes, discriminator="style", optional=True
                    )
                }
            ),
            CfgSectionValue(
                {
                    "ref": ReferenceValue(
                        "<Custom:Gauss>",
                        CfgSectionValue(
                            {
                                "style": DirectValue("gauss"),
                                "length": DirectValue(0.8),
                                "sigma": DirectValue(0.2),
                            }
                        ),
                    )
                }
            ),
        )
    )
    form.attach(owner)
    tree = form.findChild(QTreeWidget, "cfgTree")
    assert tree is not None and tree.topLevelItemCount() == 1
    ref_row = tree.topLevelItem(0)
    assert ref_row is not None and ref_row.isExpanded()
    combo = tree.findChild(QComboBox)
    assert combo is not None

    combo.setCurrentIndex(combo.findText("DRAG"))
    assert owner.observe().ref.revision == 1
    qapp.processEvents()
    assert form.findChild(ScalarInputWidget, "cfgInput:ref.delta") is not None
    contents = owner.accept(owner.observe().ref.revision).values["ref"]
    assert isinstance(contents, dict) and contents["length"] == 0.8
    combo = tree.findChild(QComboBox)
    assert combo is not None
    type_text(scalar_line(form, "ref.length"), "1.6")
    assert form.has_pending()
    combo.setCurrentIndex(combo.findData(NONE_KEY))
    qapp.processEvents()
    assert not form.has_pending()
    assert owner.observe().ref.revision == 3
    assert owner.observe().tree.children["ref"].value is None
    assert form.findChild(ScalarInputWidget, "cfgInput:ref.delta") is None


@pytest.mark.parametrize("centered", [False, True], ids=["endpoints", "center-span"])
def test_range_input_submits_subpath_to_owner(
    form: ResourceCfgFormWidget, centered: bool
) -> None:
    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(1.0, 2.0, 3) if centered else SweepValue(0.0, 2.0, 3)
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"range": spec}), CfgSectionValue({"range": value})
        )
    )
    form.attach(owner)
    kind = CenteredSweepInputWidget if centered else SweepInputWidget
    input_widget = form.findChild(kind, "cfgInput:range")
    assert input_widget is not None
    step = input_widget.findChild(QLineEdit, "step")
    assert step is not None

    type_text(step, "0.5")
    assert owner.observe().ref.revision == 0
    form.submit_pending()

    observed = owner.observe()
    assert observed.ref.revision == 1
    assert owner.accept(observed.ref.revision).values == {"range": (0, 2, 5)}


def test_choice_selection_changes_visible_rows_after_publication(
    form: ResourceCfgFormWidget, qapp: QApplication
) -> None:
    a = ScalarSpec("A", float)
    b = ScalarSpec("B", float)
    choice = ChoiceSectionSpec(
        label="Choice",
        fields={"mode": ScalarSpec("Mode", str), "a": a, "b": b},
        bindings=(
            ChoiceBinding(
                selector_key="mode",
                choices={
                    "a": CfgSectionSpec(fields={"a": a}),
                    "b": CfgSectionSpec(fields={"b": b}),
                },
            ),
        ),
    )
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"choice": choice}),
            CfgSectionValue(
                {
                    "choice": CfgSectionValue(
                        {
                            "mode": DirectValue("a"),
                            "a": DirectValue(1.0),
                            "b": DirectValue(2.0),
                        }
                    )
                }
            ),
        )
    )
    form.attach(owner)
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.a") is not None
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.b") is None
    mode = form.findChild(ScalarInputWidget, "cfgInput:choice.mode")
    assert mode is not None
    line = mode.findChild(QLineEdit)
    assert line is not None

    type_text(scalar_line(form, "choice.a"), "4.25")
    assert form.has_pending()
    type_text(line, "b")
    qapp.processEvents()

    assert not form.has_pending()
    assert owner.observe().ref.revision == 2
    assert owner.accept(form.current_ref().revision).values["choice"] == {
        "mode": "b",
        "a": 4.25,
        "b": 2.0,
    }
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.a") is None
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.b") is not None


def test_same_shape_library_relink_collapses_and_custom_expands(
    form: ResourceCfgFormWidget,
) -> None:
    policy = ProgramSpecPolicy()
    shapes = [shape.make_spec(policy) for shape in PROGRAM_SHAPES.waveforms()]
    owner = resource(
        CfgSchema(
            CfgSectionSpec(
                fields={"ref": ReferenceSpec("waveform", shapes, discriminator="style")}
            ),
            CfgSectionValue(
                {
                    "ref": ReferenceValue(
                        "<Custom:Gauss>",
                        CfgSectionValue(
                            {
                                "style": DirectValue("gauss"),
                                "length": DirectValue(0.8),
                                "sigma": DirectValue(0.2),
                            }
                        ),
                    )
                }
            ),
        ),
        OneGaussCatalog(),
    )
    form.attach(owner)
    tree = form.findChild(QTreeWidget, "cfgTree")
    assert tree is not None
    ref_row = tree.topLevelItem(0)
    assert ref_row is not None and ref_row.isExpanded()
    combo = tree.findChild(QComboBox)
    assert combo is not None

    combo.setCurrentIndex(combo.findText("Lib: saved"))
    assert owner.accept(owner.observe().ref.revision).values["ref"] == {
        "style": "gauss",
        "length": 1.0,
        "sigma": 0.2,
    }
    assert not ref_row.isExpanded()

    combo.setCurrentIndex(combo.findText("Gauss"))
    assert ref_row.isExpanded()
    assert owner.accept(owner.observe().ref.revision).values["ref"] == {
        "style": "gauss",
        "length": 1.0,
        "sigma": 0.2,
    }


def test_singleton_reference_wrapper_elision_keeps_real_edit_path(
    form: ResourceCfgFormWidget,
) -> None:
    inner = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Shape")
    outer = CfgSectionSpec(fields={"shape": inner}, label="Wrapper")
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"ref": ReferenceSpec("test", [outer])}),
            CfgSectionValue(
                {
                    "ref": ReferenceValue(
                        "<Custom:Wrapper>",
                        CfgSectionValue(
                            {"shape": CfgSectionValue({"x": DirectValue(2.0)})}
                        ),
                    )
                }
            ),
        )
    )
    form.attach(owner)
    tree = form.findChild(QTreeWidget, "cfgTree")
    assert tree is not None
    ref = tree.topLevelItem(0)
    assert ref is not None and ref.childCount() == 1
    child = ref.child(0)
    assert child is not None and child.text(0) == "X"
    input_widget = form.findChild(ScalarInputWidget, "cfgInput:ref.shape.x")
    assert input_widget is not None
    line = input_widget.findChild(QLineEdit)
    assert line is not None

    type_text(line, "5")
    form.submit_pending()

    assert owner.accept(owner.observe().ref.revision).values == {
        "ref": {"shape": {"x": 5.0}}
    }


class DeferredDialogs(RecordingDialogPresenter):
    def __init__(self) -> None:
        super().__init__()
        self.decide: Callable[[bool], None] | None = None

    def confirm_async(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        self.calls.append(DialogCall("confirm", title, message, default=default))
        self.decide = on_decision


@pytest.mark.parametrize(
    "race", [False, True], ids=["shown-revision", "newer-revision"]
)
@pytest.mark.parametrize("confirm", [False, True], ids=["cancel", "confirm"])
def test_reapply_uses_the_revision_shown_in_confirmation(
    qapp: QApplication,
    race: bool,
    confirm: bool,
) -> None:
    dialogs = DeferredDialogs()
    form = ResourceCfgFormWidget(dialog_presenter=dialogs)
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
            CfgSectionValue({"value": DirectValue(2.0)}),
        )
    )
    try:
        form.attach(owner)
        type_text(scalar_line(form), "3.5")
        shown = owner.edit(
            owner.observe().ref.revision, (CfgEdit(("value",), DirectValue(8.0)),)
        )
        button = form.findChild(QPushButton, "cfgReapplyPending")
        assert button is not None
        button.click()
        assert dialogs.decide is not None
        assert "8.0" in dialogs.calls[0].message and "3.5" in dialogs.calls[0].message
        assert str(shown.ref.revision) in dialogs.calls[0].message
        current = (
            owner.edit(shown.ref.revision, (CfgEdit(("value",), DirectValue(9.0)),))
            if race
            else shown
        )
        dialogs.decide(confirm)
        if confirm and not race:
            assert not form.has_pending()
            assert owner.accept(form.current_ref().revision).values == {"value": 3.5}
        else:
            assert owner.observe().ref == current.ref
            assert form.has_pending()
            assert scalar_line(form).text() == "3.5"
    finally:
        form.detach()
        form.deleteLater()
        qapp.processEvents()


def test_detach_reopen_discards_only_view_input_without_defaults(
    form: ResourceCfgFormWidget,
) -> None:
    calls: list[str] = []

    def defaults() -> CfgSchema:
        calls.append("defaults")
        return CfgSchema(
            CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
            CfgSectionValue({"value": DirectValue(2.0)}),
        )

    owner = CfgResource(
        defaults,
        resolution=lambda: CfgResolution(
            (),
            lambda expression: 0.0,
            lambda source_id: (),
            EmptyCatalog(),
            lambda name: 0.0,
            validate_scalar_expr,
        ),
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    form.attach(owner)
    type_text(scalar_line(form), "3.5")
    base = owner.observe().ref
    form.detach()
    assert owner.observe().ref == base
    current = owner.edit(base.revision, (CfgEdit(("value",), DirectValue(9.0)),))
    form.attach(owner)
    assert not form.has_pending()
    assert form.current_ref() == current.ref
    assert scalar_line(form).text() == "9.0"
    assert calls == ["defaults"]


def test_failed_selector_keeps_pending_and_restores_published_selection(
    form: ResourceCfgFormWidget,
    qapp: QApplication,
) -> None:
    owner = resource(
        CfgSchema(
            CfgSectionSpec(
                fields={
                    "value": ScalarSpec("Value", float),
                    "mode": ScalarSpec("Mode", str, choices=("a", "b")),
                }
            ),
            CfgSectionValue({"value": DirectValue(2.0), "mode": DirectValue("a")}),
        )
    )
    form.attach(owner)
    type_text(scalar_line(form), "3.5")
    current = owner.edit(
        owner.observe().ref.revision, (CfgEdit(("value",), DirectValue(8.0)),)
    )
    selector = form.findChild(ScalarInputWidget, "cfgInput:mode")
    assert selector is not None
    combo = selector.findChild(QComboBox)
    assert combo is not None
    combo.setCurrentIndex(combo.findText("b"))
    qapp.processEvents()
    assert owner.observe().ref == current.ref
    assert form.has_pending()
    assert scalar_line(form).text() == "3.5"
    assert combo.currentText() == "a"


@pytest.mark.parametrize("centered", [False, True], ids=["endpoints", "center-span"])
def test_pending_range_keeps_focus_and_selection_on_external_update(
    form: ResourceCfgFormWidget,
    centered: bool,
    qapp: QApplication,
) -> None:
    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(1.0, 2.0, 3) if centered else SweepValue(0.0, 2.0, 3)
    owner = resource(
        CfgSchema(
            CfgSectionSpec(fields={"range": spec}),
            CfgSectionValue({"range": value}),
        )
    )
    form.attach(owner)
    qapp.processEvents()
    widget = form.findChild(QWidget, "cfgInput:range")
    assert widget is not None
    step = widget.findChild(QLineEdit, "step")
    assert step is not None
    type_text(step, "0.51")
    step.setSelection(2, 1)
    current = owner.edit(
        owner.observe().ref.revision, (CfgEdit(("range", "expts"), DirectValue(7)),)
    )
    qapp.processEvents()
    assert step.text() == "0.51" and step.hasFocus() and step.selectedText() == "5"
    with pytest.raises(CfgStaleError):
        form.submit_pending()
    assert owner.observe().ref == current.ref
    assert step.text() == "0.51" and step.hasFocus() and step.selectedText() == "5"
