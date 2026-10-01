"""Range presentation submits inputs to a real cfg resource, not a UI model."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgObservation,
    CfgResolution,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.widgets.cfg.fields.common import (
    CenteredSweepInputWidget,
    ScalarInputWidget,
    SweepInputWidget,
)


class NoReferences:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def make_range(start: float, stop: float, *, expts: int) -> object:
    return (start, stop, expts)


@dataclass
class RangeEditor:
    resource: CfgResource
    widget: SweepInputWidget | CenteredSweepInputWidget
    submitted: list[tuple[str, DirectValue | EvalValue]] = field(default_factory=list)

    def submit(self, edge: str, value: DirectValue | EvalValue) -> None:
        self.submitted.append((edge, value))
        self.resource.edit(
            self.resource.observe().ref.revision, (CfgEdit(("range", edge), value),)
        )

    def display(self, observation: CfgObservation) -> None:
        value = observation.tree.children["range"].value
        if isinstance(self.widget, SweepInputWidget):
            assert isinstance(value, SweepValue)
            self.widget.display(value)
        else:
            assert isinstance(value, CenteredSweepValue)
            self.widget.display(value)


@pytest.fixture(params=[False, True], ids=["endpoints", "center-span"])
def editor(qapp: QApplication, request) -> Iterator[RangeEditor]:
    centered = request.param
    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(1.0, 2.0, 3) if centered else SweepValue(0.0, 2.0, 3)
    schema = CfgSchema(
        CfgSectionSpec({"range": spec}), CfgSectionValue({"range": value})
    )
    resource = CfgResource(
        lambda: schema,
        resolution=lambda: CfgResolution(
            (),
            lambda expression: {"offset": 3.0}[expression],
            lambda source_id: (),
            NoReferences(),
            lambda _: 0,
            lambda _: None,
        ),
        make_range=make_range,
    )

    def submit(edge: str, value: DirectValue | EvalValue) -> None:
        mounted.submit(edge, value)

    if isinstance(spec, CenteredSweepSpec):
        assert isinstance(value, CenteredSweepValue)
        widget = CenteredSweepInputWidget(spec, value, submit=submit)
    else:
        assert isinstance(value, SweepValue)
        widget = SweepInputWidget(spec, value, submit=submit)
    mounted = RangeEditor(resource, widget)
    unsubscribe = resource.watch(mounted.display)
    yield mounted
    unsubscribe()
    resource.revoke()
    widget.deleteLater()
    qapp.processEvents()


def test_sampling_round_trip_uses_owner_normalization(editor: RangeEditor):
    step = editor.widget.findChild(QLineEdit, "step")
    points = editor.widget.findChild(QLineEdit, "expts")
    assert step is not None and points is not None
    step.setText("0.5")
    assert editor.submitted == [("step", DirectValue(None, raw="0.5"))]
    assert points.text() == "5"
    observation = editor.resource.observe()
    assert observation.ref.revision == 1
    assert editor.resource.accept(observation.ref.revision).values["range"] == (0, 2, 5)

    points.setText("-")
    observation = editor.resource.observe()
    assert observation.status is CfgStatus.INVALID
    assert points.text() == "-"
    points.setText("9")
    observation = editor.resource.observe()
    assert observation.status is CfgStatus.VALID
    assert observation.ref.revision == 3
    assert step.text() == "0.25"
    assert len(editor.submitted) == 3


def test_external_publication_updates_expression_without_resubmission(
    editor: RangeEditor,
):
    edge = "center" if isinstance(editor.widget, CenteredSweepInputWidget) else "start"
    changed = editor.resource.edit(
        editor.resource.observe().ref.revision,
        (CfgEdit(("range", edge), EvalValue("offset")),),
    )
    assert changed.ref.revision == 1
    assert editor.submitted == []
    scalar = editor.widget.findChildren(ScalarInputWidget)[0]
    line = scalar.findChild(QLineEdit)
    assert line is not None
    assert line.text() == "offset"
    line.setText("offset ")
    assert editor.submitted == [(edge, EvalValue("offset"))]
    assert editor.resource.observe().ref.revision == 2


def test_endpoint_raw_input_is_not_parsed_by_presentation(editor: RangeEditor):
    edge = "center" if isinstance(editor.widget, CenteredSweepInputWidget) else "start"
    scalar = editor.widget.findChildren(ScalarInputWidget)[0]
    line = scalar.findChild(QLineEdit)
    assert line is not None
    line.setText("1e-")
    assert editor.submitted == [(edge, DirectValue(None, raw="1e-"))]
    assert editor.resource.observe().status is CfgStatus.INVALID
    assert line.text() == "1e-"
    line.setText("1e-1")
    assert editor.resource.observe().status is CfgStatus.VALID
    assert editor.resource.observe().ref.revision == 2
