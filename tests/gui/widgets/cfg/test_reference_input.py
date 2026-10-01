"""Reference selection from Qt through the published cfg editing handle."""

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLabel
from zcu_tools.experiment.cfg_editing.catalog import PROGRAM_SHAPES, ProgramSpecPolicy
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
)
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgEditing,
    CfgObservation,
    CfgResolution,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.session.expression import validate_scalar_expr
from zcu_tools.gui.widgets.cfg.fields import (
    CustomReferenceSelection,
    ReferenceInputWidget,
    reference_library_keys,
)
from zcu_tools.gui.widgets.cfg.fields.reference_shared import (
    NONE_KEY,
    ReferenceSelection,
)


@dataclass
class Catalog:
    entries: dict[str, ResolvedReference] = field(default_factory=dict)

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return tuple(
            key for key, ref in self.entries.items() if ref.label in allowed_labels
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return self.entries.get(key)

    def snapshot(self) -> CfgResolution:
        frozen = Catalog(deepcopy(self.entries))
        return CfgResolution(
            (),
            lambda expression: 0.0,
            lambda source_id: (),
            frozen,
            lambda name: 0.0,
            validate_scalar_expr,
        )


@pytest.fixture
def widgets(qapp: QApplication) -> Iterator[list[ReferenceInputWidget]]:
    owned: list[ReferenceInputWidget] = []
    yield owned
    for widget in owned:
        widget.deleteLater()
    qapp.processEvents()


def waveform_resource(
    catalog: Catalog, *, linked: bool = False, optional: bool = False
) -> CfgResource:
    policy = ProgramSpecPolicy()
    allowed = [shape.make_spec(policy) for shape in PROGRAM_SHAPES.waveforms()]
    original = CfgSectionValue(
        {
            "style": DirectValue("gauss"),
            "length": DirectValue(0.8),
            "sigma": DirectValue(0.2),
        }
    )
    key = "saved" if linked else "<Custom:Gauss>"
    schema = CfgSchema(
        CfgSectionSpec(
            fields={
                "ref": ReferenceSpec(
                    "waveform", allowed, discriminator="style", optional=optional
                )
            }
        ),
        CfgSectionValue({"ref": ReferenceValue(key, original)}),
    )
    return CfgResource(
        lambda: schema,
        resolution=catalog.snapshot,
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )


def attach_header(
    resource: CfgResource,
) -> tuple[ReferenceInputWidget, Callable[[], None]]:
    editing: CfgEditing = resource
    observation = editing.observe()
    revision = observation.ref.revision
    node = observation.tree.children["ref"]
    assert isinstance(node.spec, ReferenceSpec)
    assert isinstance(node.value, (ReferenceValue, type(None)))

    def submit(choice: ReferenceSelection) -> None:
        nonlocal revision
        if isinstance(choice, CustomReferenceSelection):
            editing.select_custom_reference(revision, ("ref",), choice.label)
        else:
            editing.edit(revision, (CfgEdit(("ref", "__ref"), choice),))

    widget = ReferenceInputWidget(
        node.spec,
        node.value,
        library_keys=reference_library_keys(node),
        valid=node.valid,
        submit=submit,
    )

    def display(published: CfgObservation) -> None:
        nonlocal revision
        revision = published.ref.revision
        ref = published.tree.children["ref"]
        assert isinstance(ref.value, (ReferenceValue, type(None)))
        widget.display(
            ref.value, library_keys=reference_library_keys(ref), valid=ref.valid
        )

    unsubscribe = editing.watch(display)
    return widget, unsubscribe


def test_qt_custom_waveform_selection_inherits_length_without_extra_submit(
    widgets,
) -> None:
    resource = waveform_resource(Catalog())
    widget, unsubscribe = attach_header(resource)
    widgets.append(widget)
    combo = widget.findChild(QComboBox)
    assert combo is not None

    combo.setCurrentIndex(combo.findText("DRAG"))

    changed = resource.observe()
    assert changed.ref.revision == 1
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values["ref"] == {
        "style": "drag",
        "length": 0.8,
        "sigma": 0.2,
        "delta": 0.0,
        "alpha": 0.0,
    }
    assert combo.currentText() == "DRAG"
    unsubscribe()


def test_qt_library_revert_uses_relink_and_publication(widgets) -> None:
    catalog = Catalog(
        {
            "saved": ResolvedReference(
                "Gauss",
                CfgSectionValue(
                    {
                        "style": DirectValue("gauss"),
                        "length": DirectValue(1.0),
                        "sigma": DirectValue(0.2),
                    }
                ),
            )
        }
    )
    resource = waveform_resource(catalog, linked=True)
    widget, unsubscribe = attach_header(resource)
    widgets.append(widget)
    edited = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("ref", "length"), 3.0),)
    )
    combo = widget.findChild(QComboBox)
    assert combo is not None
    assert "Lib: saved (modified)" in (combo.itemText(i) for i in range(combo.count()))

    combo.setCurrentIndex(combo.findText("Revert to Lib: saved"))

    relinked = resource.observe()
    assert relinked.ref.revision == edited.ref.revision + 1
    selected = relinked.tree.children["ref"].value
    assert isinstance(selected, ReferenceValue)
    assert selected.chosen_key == "saved" and not selected.is_overridden
    values = resource.accept(relinked.ref.revision).values["ref"]
    assert isinstance(values, dict) and values["length"] == 1.0
    unsubscribe()


def test_qt_optional_none_then_custom_starts_complete_choice(widgets) -> None:
    resource = waveform_resource(Catalog(), optional=True)
    widget, unsubscribe = attach_header(resource)
    widgets.append(widget)
    combo = widget.findChild(QComboBox)
    assert combo is not None
    combo.setCurrentIndex(combo.findData(NONE_KEY))
    disabled = resource.observe()
    assert disabled.tree.children["ref"].value is None
    assert disabled.ref.revision == 1

    combo.setCurrentIndex(combo.findText("Cosine"))

    selected = resource.observe()
    assert selected.ref.revision == 2
    contents = selected.tree.children["ref"].value
    assert isinstance(contents, ReferenceValue)
    assert contents.chosen_key == "<Custom:Cosine>"
    assert contents.value.fields["length"] == DirectValue(0.0)
    unsubscribe()


def test_missing_library_hint_follows_publication_without_resubmit(widgets) -> None:
    catalog = Catalog(
        {
            "saved": ResolvedReference(
                "Gauss",
                CfgSectionValue(
                    {
                        "style": DirectValue("gauss"),
                        "length": DirectValue(1.0),
                        "sigma": DirectValue(0.2),
                    }
                ),
            )
        }
    )
    resource = waveform_resource(catalog, linked=True)
    widget, unsubscribe = attach_header(resource)
    widgets.append(widget)
    combo = widget.findChild(QComboBox)
    hint = widget.findChild(QLabel, "missingRefHint")
    assert combo is not None and hint is not None

    del catalog.entries["saved"]
    missing = resource.refresh(resource.observe().ref.revision)

    assert missing.status is CfgStatus.INVALID
    assert resource.observe().ref.revision == 1
    assert combo.currentText() == "Missing: saved"
    assert hint.isVisibleTo(widget)
    assert "saved" in hint.text()
    assert "red" in combo.styleSheet()
    unsubscribe()
