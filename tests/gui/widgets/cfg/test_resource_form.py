"""A dense Qt cfg tree presents and edits one published resource."""

from collections.abc import Iterator, Sequence

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit, QTreeWidget
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
from zcu_tools.gui.cfg.resource import CfgEdit, CfgResolution, CfgResource, CfgStatus
from zcu_tools.gui.session.expression import validate_scalar_expr
from zcu_tools.gui.widgets.cfg import ResourceCfgFormWidget
from zcu_tools.gui.widgets.cfg.fields.common import (
    CenteredSweepInputWidget,
    ScalarInputWidget,
    SweepInputWidget,
)
from zcu_tools.gui.widgets.cfg.fields.reference_shared import NONE_KEY


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
    widget = ResourceCfgFormWidget()
    yield widget
    widget.detach()
    widget.deleteLater()
    qapp.processEvents()


def test_scalar_edit_and_external_publication_preserve_focused_input(form) -> None:
    schema = CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue({"value": DirectValue(2.0)}),
    )
    owner = resource(schema)
    validity: list[bool] = []
    form.validity_changed.connect(validity.append)
    form.attach(owner)
    input_widget = form.findChild(ScalarInputWidget, "cfgInput:value")
    assert input_widget is not None
    line = input_widget.findChild(QLineEdit)
    assert line is not None
    line.setFocus()

    line.setText("3.5")
    assert owner.observe().ref.revision == 1
    assert owner.accept(owner.observe().ref.revision).values == {"value": 3.5}
    assert form.findChild(ScalarInputWidget, "cfgInput:value") is input_widget
    assert line.text() == "3.5"

    owner.edit(owner.observe().ref.revision, (CfgEdit(("value",), DirectValue(8.0)),))
    assert form.findChild(ScalarInputWidget, "cfgInput:value") is input_widget
    assert line.text() == "8.0"
    assert validity and all(validity)

    form.detach()
    owner.edit(owner.observe().ref.revision, (CfgEdit(("value",), DirectValue(9.0)),))
    assert owner.accept(owner.observe().ref.revision).values == {"value": 9.0}
    assert form.findChild(ScalarInputWidget, "cfgInput:value") is None


def test_invalid_input_reflects_published_status_and_reason(form) -> None:
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

    line.setText("-")

    assert owner.observe().status is CfgStatus.INVALID
    assert not form.is_valid()
    assert form.first_invalid_reason()
    assert line.text() == "-"


def test_reference_shape_switch_rebuilds_after_qt_signal(form, qapp) -> None:
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
    assert tree.topLevelItem(0).isExpanded()
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
    combo.setCurrentIndex(combo.findData(NONE_KEY))
    qapp.processEvents()
    assert owner.observe().tree.children["ref"].value is None
    assert form.findChild(ScalarInputWidget, "cfgInput:ref.delta") is None


@pytest.mark.parametrize("centered", [False, True], ids=["endpoints", "center-span"])
def test_range_input_submits_subpath_to_owner(form, centered: bool) -> None:
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

    step.setText("0.5")

    observed = owner.observe()
    assert observed.ref.revision == 1
    assert owner.accept(observed.ref.revision).values == {"range": (0, 2, 5)}


def test_choice_selection_changes_visible_rows_after_publication(form, qapp) -> None:
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

    line.setText("b")
    qapp.processEvents()

    assert owner.observe().ref.revision == 1
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.a") is None
    assert form.findChild(ScalarInputWidget, "cfgInput:choice.b") is not None


def test_same_shape_library_relink_collapses_and_custom_expands(form) -> None:
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
    assert tree is not None and tree.topLevelItem(0).isExpanded()
    combo = tree.findChild(QComboBox)
    assert combo is not None

    combo.setCurrentIndex(combo.findText("Lib: saved"))
    assert owner.accept(owner.observe().ref.revision).values["ref"] == {
        "style": "gauss",
        "length": 1.0,
        "sigma": 0.2,
    }
    assert not tree.topLevelItem(0).isExpanded()

    combo.setCurrentIndex(combo.findText("Gauss"))
    assert tree.topLevelItem(0).isExpanded()
    assert owner.accept(owner.observe().ref.revision).values["ref"] == {
        "style": "gauss",
        "length": 1.0,
        "sigma": 0.2,
    }


def test_singleton_reference_wrapper_elision_keeps_real_edit_path(form) -> None:
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
    assert ref.childCount() == 1
    assert ref.child(0).text(0) == "X"
    input_widget = form.findChild(ScalarInputWidget, "cfgInput:ref.shape.x")
    assert input_widget is not None
    line = input_widget.findChild(QLineEdit)
    assert line is not None

    line.setText("5")

    assert owner.accept(owner.observe().ref.revision).values == {
        "ref": {"shape": {"x": 5.0}}
    }
