"""Focused cfg tree presentation tests (A1–A3, A5) for cfg-tree-presentation slice."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QLineEdit,
    QTreeWidgetItem,  # type: ignore[attr-defined]
)
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import (
    ReferenceField,
    ScalarField,
    SectionField,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg import (
    TREE_DEPTH_COLORS,
    CfgFormWidget,
    FieldRenderContext,
    TreeCfgWidget,
    TreeStructure,
    form_structure,
    tree_structure,
)
from zcu_tools.gui.widgets.cfg.fields import SectionWidget
from zcu_tools.meta_tool import MetaDict


@pytest.fixture()
def ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MetaDict()
    c.get_current_ml.return_value = MagicMock()
    c.get_current_ml.return_value.modules = {}
    c.get_current_ml.return_value.waveforms = {}
    c.list_arb_waveforms.return_value = []
    c.list_device_names.return_value = []
    return c


def _attach(widget: CfgFormWidget, schema: CfgSchema, ctrl) -> SectionField:
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    widget.attach(draft)
    return draft.root


def _simple_schema() -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )


# ---------------------------------------------------------------------------
# A1 — default is form
# ---------------------------------------------------------------------------


def test_default_is_form_preserves_existing_behavior(qapp, ctrl):
    w = CfgFormWidget()
    schema = _simple_schema()
    _attach(w, schema, ctrl)
    assert isinstance(w._root_widget, SectionWidget)
    # form uses QFormLayout style containers, not tree
    assert w._root_widget is not None
    # no tree
    assert not isinstance(w._root_widget, TreeCfgWidget)


def test_tree_structure_creates_tree(qapp, ctrl):
    w = CfgFormWidget(structure=tree_structure)
    schema = _simple_schema()
    _attach(w, schema, ctrl)
    assert isinstance(w._root_widget, TreeCfgWidget)
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    assert tree.isHeaderHidden() is True
    assert tree.rootIsDecorated() is False
    assert tree.indentation() == 10
    assert tree.columnCount() == 2
    assert tree.font().pixelSize() == 13
    assert w._root_widget.font().pixelSize() == 13


# ---------------------------------------------------------------------------
# A2 — renders nested sections, references, scalars, sweeps etc via same field behavior
# ---------------------------------------------------------------------------


def _complex_schema() -> CfgSchema:
    gauss_spec = CfgSectionSpec(
        label="Gaussian",
        fields={
            "sigma": ScalarSpec(label="Sigma", type=float, decimals=3),
            "length": ScalarSpec(label="Length", type=float, decimals=3),
        },
    )
    waveform_ref = ReferenceSpec(
        kind="waveform", label="Waveform", allowed=[gauss_spec], optional=True
    )
    sweep_spec = SweepSpec(label="Sweep")
    csweep_spec = CenteredSweepSpec(label="CSweep", decimals=2)
    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={
            "freq": ScalarSpec(label="Freq", type=float, decimals=2),
            "waveform": waveform_ref,
            "sweep": sweep_spec,
            "csweep": csweep_spec,
            "flag": ScalarSpec(label="Flag", type=bool),
            "choice": ScalarSpec(label="Mode", type=str, choices=["a", "b", "c"]),
            "eval": ScalarSpec(label="Eval", type=float),
        },
    )
    root_spec = CfgSectionSpec(
        label="Root",
        fields={
            "reps": ScalarSpec(label="Reps", type=int),
            "eval_direct": ScalarSpec(label="EvalDirect", type=float),
            "nested": pulse_spec,
            "ref": ReferenceSpec(kind="module", label="Ref", allowed=[pulse_spec]),
        },
    )
    gauss_val = CfgSectionValue(
        fields={
            "sigma": DirectValue(0.03),
            "length": EvalValue("2*sigma", resolved=0.06),
        }
    )
    pulse_val = CfgSectionValue(
        fields={
            "freq": EvalValue("r_f", resolved=6000.0),
            "waveform": ReferenceValue(chosen_key="<Custom:Gaussian>", value=gauss_val),
            "sweep": SweepValue(start=0.0, stop=1.0, expts=11),
            "csweep": CenteredSweepValue(center=0.5, span=1.0, expts=11),
            "flag": DirectValue(True),
            "choice": DirectValue("b"),
            "eval": DirectValue(3.14),
        }
    )
    root_val = CfgSectionValue(
        fields={
            "reps": DirectValue(42),
            "eval_direct": DirectValue(1.23),
            "nested": pulse_val,
            "ref": ReferenceValue(chosen_key="<Custom:Pulse>", value=pulse_val),
        }
    )
    return CfgSchema(spec=root_spec, value=root_val)


def test_tree_renders_and_edits_same_observable_behavior(qapp, ctrl):
    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    schema = _complex_schema()
    for struct in (None, tree_structure):
        w = CfgFormWidget(structure=struct)
        draft = MeasureCfgBindings(ctrl).new_draft(schema)
        w.attach(draft)
        qapp.processEvents()
        # read back
        out = w.read_values()
        reps_val = out.fields["reps"]
        assert isinstance(reps_val, DirectValue)
        assert reps_val.value == 42
        nested = out.fields["nested"]
        assert isinstance(nested, CfgSectionValue)
        freq_val = nested.fields["freq"]
        assert isinstance(freq_val, EvalValue)
        assert freq_val.resolved == 6000.0
        assert isinstance(nested.fields["sweep"], SweepValue)
        # edit via draft leaf should reflect in read_values
        reps_field = draft.root.fields["reps"]
        assert isinstance(reps_field, ScalarField)
        reps_field.set_value(999)
        qapp.processEvents()
        reps_val2 = w.read_values().fields["reps"]
        assert isinstance(reps_val2, DirectValue)
        assert reps_val2.value == 999
        # dropdown edit
        choice_field = cast(SectionField, draft.root.fields["nested"]).fields["choice"]
        assert isinstance(choice_field, ScalarField)
        choice_field.set_value(DirectValue("c"))
        qapp.processEvents()
        choice_val = cast(
            ScalarField,
            cast(SectionField, draft.root.fields["nested"]).fields["choice"],
        ).get_value()
        assert isinstance(choice_val, DirectValue)
        assert choice_val.value == "c"
        # boolean
        flag_field = cast(SectionField, draft.root.fields["nested"]).fields["flag"]
        assert isinstance(flag_field, ScalarField)
        flag_field.set_value(False)
        qapp.processEvents()
        nested_flag = w.read_values().fields["nested"]
        assert isinstance(nested_flag, CfgSectionValue)
        flag_val = nested_flag.fields["flag"]
        assert isinstance(flag_val, DirectValue)
        assert flag_val.value is False
        # sweep
        sweep_field = cast(SectionField, draft.root.fields["nested"]).fields["sweep"]
        sweep_field.update_expts(5)  # type: ignore[union-attr]
        qapp.processEvents()
        nested_sweep = w.read_values().fields["nested"]
        assert isinstance(nested_sweep, CfgSectionValue)
        sv = nested_sweep.fields["sweep"]
        assert isinstance(sv, SweepValue)
        assert sv.expts == 5
        # centered sweep span
        cs_field = cast(SectionField, draft.root.fields["nested"]).fields["csweep"]
        cs_field.update_span(2.0)  # type: ignore[union-attr]
        qapp.processEvents()
        nested_cs = w.read_values().fields["nested"]
        assert isinstance(nested_cs, CfgSectionValue)
        csv = nested_cs.fields["csweep"]
        assert isinstance(csv, CenteredSweepValue)
        assert csv.span == pytest.approx(2.0)
        # reference elided but still editable
        ref_field = draft.root.fields["ref"]
        assert isinstance(ref_field, ReferenceField)
        before = w.read_values().fields["ref"]
        assert isinstance(before, ReferenceValue)
        # detach to clean
        w.detach()
        draft.close()


def test_tree_whole_row_folding_is_view_only(qapp, ctrl):
    schema = _complex_schema()
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    tree_w = cast(TreeCfgWidget, w._root_widget)
    tree = tree_w._tree
    # find a foldable item (Root or nested Pulse)
    top = tree.topLevelItem(0)
    assert top is not None
    assert top.childCount() > 0
    before = w.read_values()
    expanded_before = top.isExpanded()
    # simulate whole-row click toggling
    tree_w._on_item_clicked(top, 0)
    qapp.processEvents()
    assert top.isExpanded() != expanded_before
    # toggling again
    tree_w._on_item_clicked(top, 1)
    qapp.processEvents()
    assert top.isExpanded() == expanded_before
    after = w.read_values()
    assert before == after
    # leaf items should not toggle
    # find leaf Reps
    reps_item: QTreeWidgetItem | None = None
    for i in range(tree.topLevelItemCount()):
        t = tree.topLevelItem(i)
        assert t is not None
        # search
        stack: list[QTreeWidgetItem] = [t]
        while stack:
            cur = stack.pop()
            if "Reps" in cur.text(0):
                reps_item = cur
                break
            for j in range(cur.childCount()):
                child = cur.child(j)
                assert child is not None
                stack.append(child)
        if reps_item:
            break
    assert reps_item is not None
    assert reps_item.childCount() == 0
    before_leaf = reps_item.isExpanded()
    tree_w._on_item_clicked(reps_item, 0)
    assert reps_item.isExpanded() == before_leaf
    w.detach()
    draft.close()


# ---------------------------------------------------------------------------
# A3 — visual specifics : root alignment, indentation, connectors, depth colors, elision
# ---------------------------------------------------------------------------


def test_tree_indentation_and_header_and_connectors(qapp, ctrl):
    from qtpy.QtWidgets import QProxyStyle  # type: ignore[attr-defined]

    w = CfgFormWidget(structure=tree_structure)
    schema = _simple_schema()
    _attach(w, schema, ctrl)
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    assert tree.indentation() == 10
    assert tree.isHeaderHidden() is True
    assert tree.rootIsDecorated() is False
    # connector style is our custom proxy (classic branch lines without triangles)
    style = tree.style()
    assert (
        isinstance(style, QProxyStyle) or style.__class__.__name__ == "_TreeBranchStyle"
    )
    # also verify our stored branch style instance
    assert hasattr(cast(TreeCfgWidget, w._root_widget), "_branch_style")
    assert isinstance(cast(TreeCfgWidget, w._root_widget)._branch_style, QProxyStyle)
    # root item has no extra indentation beyond 0 (checked via indentation property alone)
    # ensure top-level item's parent is invisibleRootItem
    top = tree.topLevelItem(0)
    assert top is not None
    assert top.parent() is None


def test_tree_depth_color_cycling_and_own_depth(qapp, ctrl):
    # Build a 6-deep nested section chain to test cycling (0..5 should wrap)
    spec = CfgSectionSpec(label="L0", fields={"a": ScalarSpec(label="A", type=int)})
    # Nest 6 levels: each level contains a child section L{n}
    cur_spec = spec
    for i in range(1, 7):
        child = CfgSectionSpec(
            label=f"L{i}", fields={"a": ScalarSpec(label="A", type=int)}
        )
        cur_spec = CfgSectionSpec(
            label=f"L{6 - i}", fields={"child": cur_spec if i == 1 else child}
        )
        # Actually simplify: create chain via loop, but easier: directly build nested spec chain
    # For deterministic, build chain manually
    l6 = CfgSectionSpec(label="L6", fields={"leaf": ScalarSpec(label="Leaf", type=int)})
    l5 = CfgSectionSpec(label="L5", fields={"c": l6})
    l4 = CfgSectionSpec(label="L4", fields={"c": l5})
    l3 = CfgSectionSpec(label="L3", fields={"c": l4})
    l2 = CfgSectionSpec(label="L2", fields={"c": l3})
    l1 = CfgSectionSpec(label="L1", fields={"c": l2})
    root_spec = CfgSectionSpec(label="Root", fields={"c": l1})

    # Build value chain
    def leaf_val(v: int) -> CfgSectionValue:
        return CfgSectionValue(fields={"leaf": DirectValue(v)})

    v6 = leaf_val(1)
    v5 = CfgSectionValue(fields={"c": v6})
    v4 = CfgSectionValue(fields={"c": v5})
    v3 = CfgSectionValue(fields={"c": v4})
    v2 = CfgSectionValue(fields={"c": v3})
    v1 = CfgSectionValue(fields={"c": v2})
    root_val = CfgSectionValue(fields={"c": v1})
    schema = CfgSchema(spec=root_spec, value=root_val)

    w = CfgFormWidget(structure=tree_structure)
    _attach(w, schema, ctrl)
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    # Walk depth chain and verify background colors cycle
    # Root at depth 0 => TREE_DEPTH_COLORS[0]
    # Its child L1 at depth 1 => TREE_DEPTH_COLORS[1], etc.
    expected = list(TREE_DEPTH_COLORS)  # 0..4
    # collect items in order of nesting
    items: list[QTreeWidgetItem] = []
    cur = tree.topLevelItem(0)
    assert cur is not None
    items.append(cur)
    while cur.childCount() > 0:
        nxt = cur.child(0)
        assert nxt is not None
        cur = nxt
        items.append(cur)
    # items should be Root, L1, L2, L3, L4, L5, L6, leaf
    # background for each foldable node row at its displayed depth, leaves at their depth
    for idx, it in enumerate(items):
        bg = it.background(0).color().name().lower()
        # depth for item: for Root 0, L1 1, L2 2, etc., leaf depth 7?
        # Our tree depth increments per section: Root 0, L1 1, L2 2...
        exp = TREE_DEPTH_COLORS[idx % len(TREE_DEPTH_COLORS)].lower()
        assert bg == exp, f"depth {idx} item {it.text(0)!r} bg {bg} != {exp}"
    # own-depth: child's color should be next, not same as parent
    for i in range(1, len(items)):
        assert (
            items[i].background(0).color().name().lower()
            != items[i - 1].background(0).color().name().lower()
            or len(TREE_DEPTH_COLORS) == 1
        )


def test_tree_reference_shape_elision(qapp, ctrl):
    gauss = CfgSectionSpec(
        label="Gauss", fields={"sigma": ScalarSpec(label="Sigma", type=float)}
    )
    ref_spec = ReferenceSpec(kind="waveform", label="Waveform", allowed=[gauss])
    root_spec = CfgSectionSpec(
        label="Root",
        fields={"ref": ref_spec, "other": ScalarSpec(label="Other", type=int)},
    )
    root_val = CfgSectionValue(
        fields={
            "ref": ReferenceValue(
                chosen_key="<Custom:Gauss>",
                value=CfgSectionValue(fields={"sigma": DirectValue(0.1)}),
            ),
            "other": DirectValue(1),
        }
    )
    schema = CfgSchema(spec=root_spec, value=root_val)
    w = CfgFormWidget(structure=tree_structure)
    _attach(w, schema, ctrl)
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    # find ref item
    ref_item: QTreeWidgetItem | None = None
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        assert top is not None
        # top is Root
        for j in range(top.childCount()):
            child = top.child(j)
            assert child is not None
            if "Waveform" in child.text(0):
                ref_item = child
                break
        if ref_item:
            break
    assert ref_item is not None
    # Elision: ref's children should be sigma directly, not an intermediate Gauss wrapper
    child_texts: list[str] = []
    for k in range(ref_item.childCount()):
        ch = ref_item.child(k)
        assert ch is not None
        child_texts.append(ch.text(0))
    assert "Sigma" in child_texts
    assert "Gauss" not in child_texts
    # guaranteed single shape row elided => childCount == number of fields in shape (1)
    assert ref_item.childCount() == 1


# ---------------------------------------------------------------------------
# A5 — editing lock, detach/attach, validation, section-local refresh
# ---------------------------------------------------------------------------


def test_tree_editing_lock_disables_editors(qapp, ctrl):
    schema = _simple_schema()
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    assert w._root_widget is not None
    assert w._root_widget.isEnabled() is True
    w.set_editing_enabled(False)
    qapp.processEvents()
    assert w._root_widget.isEnabled() is False
    # child editor also disabled via parent
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    # find leaf widget
    leaf: QTreeWidgetItem | None = None
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        assert top is not None
        if top.childCount() == 0 and "Reps" in top.text(0):
            leaf = top
            break
        for j in range(top.childCount()):
            child = top.child(j)
            assert child is not None
            if "Reps" in child.text(0):
                leaf = child
                break
    # leaf widget may be under root; fallback search exhaustive
    if leaf is None:
        # search all
        root_item = tree.invisibleRootItem()
        assert root_item is not None
        stack: list[QTreeWidgetItem] = [root_item]
        while stack:
            cur = stack.pop()
            for j in range(cur.childCount()):
                ch = cur.child(j)
                assert ch is not None
                if "Reps" in ch.text(0):
                    leaf = ch
                    break
                stack.append(ch)
            if leaf:
                break
    assert leaf is not None
    wg = tree.itemWidget(leaf, 1)
    assert wg is not None
    assert wg.isEnabled() is False
    w.set_editing_enabled(True)
    assert w._root_widget.isEnabled() is True
    assert wg.isEnabled() is True
    w.detach()
    draft.close()


def test_tree_detach_attach_preserves_draft(qapp, ctrl):
    schema = _simple_schema()
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    # draft callbacks should be subscribed
    assert draft.on_change._callbacks
    assert draft.on_validity_changed._callbacks
    w.detach()
    assert draft.on_change._callbacks == []
    assert draft.on_validity_changed._callbacks == []
    # reattach same draft
    w.attach(draft)
    assert draft.on_change._callbacks
    # draft not closed
    reps_check = w.read_values().fields["reps"]
    assert isinstance(reps_check, DirectValue)
    assert reps_check.value == 10
    w.detach()
    draft.close()


def test_tree_validation_propagation(qapp, ctrl):
    spec = CfgSectionSpec(fields={"v": ScalarSpec(label="V", type=int, required=True)})
    val = CfgSectionValue(fields={"v": DirectValue(1)})
    schema = CfgSchema(spec=spec, value=val)
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    events: list[bool] = []
    w.validity_changed.connect(events.append)
    w.attach(draft)
    qapp.processEvents()
    assert events == [True]
    assert w.is_valid() is True
    cast(ScalarField, draft.root.fields["v"]).set_value(None)
    qapp.processEvents()
    assert events[-1] is False
    assert w.is_valid() is False
    assert w.first_invalid_reason() is not None
    cast(ScalarField, draft.root.fields["v"]).set_value(2)
    qapp.processEvents()
    assert events[-1] is True
    assert w.is_valid() is True
    w.detach()
    draft.close()


def test_tree_section_local_refresh_choice(qapp, ctrl):
    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half": ScalarSpec(label="Half", type=float),
        "manual": ScalarSpec(label="Manual", type=float),
    }
    spec = ChoiceSectionSpec(
        label="Choice",
        fields=fields,
        bindings=(
            ChoiceBinding(
                "mode",
                {
                    "auto": CfgSectionSpec(fields={"half": fields["half"]}),
                    "fixed": CfgSectionSpec(fields={"manual": fields["manual"]}),
                },
            ),
        ),
    )
    root_spec = CfgSectionSpec(
        fields={"choice": spec, "stable": ScalarSpec(label="Stable", type=float)}
    )
    root_val = CfgSectionValue(
        fields={
            "choice": CfgSectionValue(
                fields={
                    "mode": DirectValue("auto"),
                    "half": DirectValue(1.0),
                    "manual": DirectValue(2.0),
                }
            ),
            "stable": DirectValue(3.0),
        }
    )
    schema = CfgSchema(spec=root_spec, value=root_val)
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    assert "choice.half" in w.decoration_paths()
    assert "choice.manual" not in w.decoration_paths()
    # switch mode
    cast(
        ScalarField, cast(SectionField, draft.root.fields["choice"]).fields["mode"]
    ).set_value(DirectValue("fixed"))
    # flush via decoration_paths
    paths = set(w.decoration_paths())
    assert "choice.half" not in paths
    assert "choice.manual" in paths
    # stable should still exist
    assert "stable" in paths
    w.detach()
    draft.close()


def test_tree_shares_same_draft_binding_ref_identity(qapp, ctrl):
    # Ensure tree does not alter draft reference identity / paths
    gauss = CfgSectionSpec(
        label="Gauss", fields={"sigma": ScalarSpec(label="Sigma", type=float)}
    )
    ref_spec = ReferenceSpec(kind="waveform", label="Waveform", allowed=[gauss])
    root_spec = CfgSectionSpec(fields={"ref": ref_spec})
    root_val = CfgSectionValue(
        fields={
            "ref": ReferenceValue(
                chosen_key="<Custom:Gauss>",
                value=CfgSectionValue(fields={"sigma": DirectValue(0.1)}),
            )
        }
    )
    schema = CfgSchema(spec=root_spec, value=root_val)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w_form = CfgFormWidget()
    w_tree = CfgFormWidget(structure=tree_structure)
    w_form.attach(draft)
    w_tree.attach(
        draft
    )  # both attach same draft sequentially via separate widgets? Actually CfgFormWidget supports one writer at a time; attach sequentially will detach previous, but draft shared
    w_form.detach()
    w_tree.attach(draft)
    qapp.processEvents()
    # path resolve still works
    target = draft.resolve_target("ref.sigma")
    assert target is not None
    # set via draft path should be visible in both widgets' read_values after reattach
    draft.set_target("ref.sigma", 0.99)
    qapp.processEvents()
    ref_val_tree = w_tree.read_values().fields["ref"]
    assert isinstance(ref_val_tree, ReferenceValue)
    sigma_val_tree = ref_val_tree.value.fields["sigma"]
    assert isinstance(sigma_val_tree, DirectValue)
    assert sigma_val_tree.value == pytest.approx(0.99)
    w_tree.detach()
    w_form.attach(draft)
    ref_val_form = w_form.read_values().fields["ref"]
    assert isinstance(ref_val_form, ReferenceValue)
    sigma_val_form = ref_val_form.value.fields["sigma"]
    assert isinstance(sigma_val_form, DirectValue)
    assert sigma_val_form.value == pytest.approx(0.99)
    w_form.detach()
    draft.close()


def test_tree_outer_reenable_preserves_nested_reference_and_decoration_parity(
    qapp, ctrl
):
    """Parity regression: re-enabling outer optional ref must not overwrite
    nested optional (still disabled) or decoration-disabled child (S1/A4, form/tree A2 parity)."""
    from zcu_tools.gui.widgets.cfg import FieldDecorationPatch

    inner_shape = CfgSectionSpec(
        label="InnerShape",
        fields={"inner_leaf": ScalarSpec(label="InnerLeaf", type=int)},
    )
    inner_optional = ReferenceSpec(
        kind="module", label="Inner", allowed=[inner_shape], optional=True
    )
    outer_shape = CfgSectionSpec(
        label="OuterShape",
        fields={
            "inner_ref": inner_optional,
            "deco_leaf": ScalarSpec(label="Deco", type=int),
            "normal_leaf": ScalarSpec(label="Normal", type=int),
        },
    )
    outer_optional = ReferenceSpec(
        kind="module", label="Outer", allowed=[outer_shape], optional=True
    )
    root_spec = CfgSectionSpec(fields={"outer": outer_optional})
    outer_shape_val = CfgSectionValue(
        fields={
            "deco_leaf": DirectValue(10),
            "normal_leaf": DirectValue(20),
        }
    )
    root_val = CfgSectionValue(
        fields={
            "outer": ReferenceValue(
                chosen_key="<Custom:OuterShape>", value=outer_shape_val
            ),
        }
    )
    schema = CfgSchema(spec=root_spec, value=root_val)

    class DecoProvider:
        def decoration_for(self, path: str, spec: object, value: object):
            del spec, value
            if path == "outer.deco_leaf":
                return FieldDecorationPatch(enabled=False, badge="deco")
            return None

    w = CfgFormWidget(structure=tree_structure, decoration_provider=DecoProvider())
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    outer_field = cast(ReferenceField, draft.root.fields["outer"])
    assert outer_field.is_enabled is True
    outer_sub = outer_field.sub_field
    assert outer_sub is not None
    inner_raw = outer_sub.fields["inner_ref"]
    assert isinstance(inner_raw, ReferenceField)
    inner_field = inner_raw
    assert inner_field.is_enabled is False

    def find_item(path: str):
        # Leaves are not in _path_to_item; search via UserRole data.
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]

        stack = [tree.invisibleRootItem()]
        while stack:
            cur = stack.pop()
            if cur is None:
                continue
            for idx in range(cur.childCount()):
                ch = cur.child(idx)
                if ch is None:
                    continue
                data = ch.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
                if data == path:
                    return ch
                stack.append(ch)
        return None

    def is_enabled(path: str) -> bool:
        item = find_item(path)
        assert item is not None, f"missing item {path!r}"
        wg = tree.itemWidget(item, 1)
        # item disabled mirrors decoration + ancestor gating; widget enabled same.
        if wg is not None:
            return not item.isDisabled() and wg.isEnabled()  # type: ignore[attr-defined]
        return not item.isDisabled()  # type: ignore[attr-defined]

    # Initial: inner leaf disabled (nested ref), deco leaf disabled (decoration), normal enabled, inner header enabled.
    assert (
        is_enabled("outer.inner_ref") is True
    )  # header stays enabled so combo can re-select
    assert is_enabled("outer.inner_ref.inner_leaf") is False
    assert is_enabled("outer.deco_leaf") is False
    assert is_enabled("outer.normal_leaf") is True

    # Disable outer -> all descendants disabled.
    outer_field.set_enabled(False)
    qapp.processEvents()
    assert outer_field.is_enabled is False
    assert is_enabled("outer.inner_ref") is False
    assert is_enabled("outer.inner_ref.inner_leaf") is False
    assert is_enabled("outer.deco_leaf") is False
    assert is_enabled("outer.normal_leaf") is False

    # Re-enable outer -> inner leaf must stay disabled (nested optional still disabled),
    # deco leaf must stay disabled (decoration), normal leaf must become enabled,
    # and inner header must be re-enabled.
    outer_field.set_enabled(True)
    qapp.processEvents()
    assert outer_field.is_enabled is True
    assert inner_field.is_enabled is False  # still disabled
    assert is_enabled("outer.inner_ref") is True
    assert is_enabled("outer.inner_ref.inner_leaf") is False, (
        "nested optional child was incorrectly re-enabled"
    )
    assert is_enabled("outer.deco_leaf") is False, (
        "decoration-disabled child was incorrectly re-enabled"
    )
    assert is_enabled("outer.normal_leaf") is True

    # Form parity: same draft sequence on form presentation keeps identical effective enabled.
    w_form = CfgFormWidget(decoration_provider=DecoProvider())
    draft2 = MeasureCfgBindings(ctrl).new_draft(schema)
    w_form.attach(draft2)
    qapp.processEvents()
    outer2 = cast(ReferenceField, draft2.root.fields["outer"])
    outer2_sub = outer2.sub_field
    assert outer2_sub is not None
    inner2_raw = outer2_sub.fields["inner_ref"]
    assert isinstance(inner2_raw, ReferenceField)
    inner2 = inner2_raw
    assert inner2.is_enabled is False
    outer2.set_enabled(False)
    qapp.processEvents()
    outer2.set_enabled(True)
    qapp.processEvents()
    assert inner2.is_enabled is False
    # decoration for deco leaf stays disabled, inner leaf effectively disabled via ancestor
    assert w_form.decoration_for_path("outer.deco_leaf").enabled is False
    assert w_form.decoration_for_path("outer.normal_leaf").enabled is True
    w.detach()
    draft.close()
    w_form.detach()
    draft2.close()


def test_tree_outer_reenable_preserves_decoration_disabled_container_parity(qapp, ctrl):
    """Parity for decoration-disabled ancestor containers (Section/Reference).
    Outer disable/re-enable must not re-enable children of a decoration-disabled
    Section or non-optional Reference inside the outer shape."""
    from zcu_tools.gui.widgets.cfg import FieldDecorationPatch

    inner_section = CfgSectionSpec(
        label="InnerSection", fields={"sec_leaf": ScalarSpec(label="SecLeaf", type=int)}
    )
    inner_ref_shape = CfgSectionSpec(
        label="InnerRefShape",
        fields={"ref_leaf": ScalarSpec(label="RefLeaf", type=int)},
    )
    inner_ref = ReferenceSpec(
        kind="module", label="InnerRef", allowed=[inner_ref_shape], optional=False
    )
    outer_shape = CfgSectionSpec(
        label="OuterShape",
        fields={
            "inner_section": inner_section,
            "inner_ref": inner_ref,
            "normal_leaf": ScalarSpec(label="Normal", type=int),
        },
    )
    outer_optional = ReferenceSpec(
        kind="module", label="Outer", allowed=[outer_shape], optional=True
    )
    root_spec = CfgSectionSpec(fields={"outer": outer_optional})
    outer_shape_val = CfgSectionValue(
        fields={
            "inner_section": CfgSectionValue(fields={"sec_leaf": DirectValue(7)}),
            "inner_ref": ReferenceValue(
                chosen_key="<Custom:InnerRefShape>",
                value=CfgSectionValue(fields={"ref_leaf": DirectValue(5)}),
            ),
            "normal_leaf": DirectValue(1),
        }
    )
    root_val = CfgSectionValue(
        fields={
            "outer": ReferenceValue(
                chosen_key="<Custom:OuterShape>", value=outer_shape_val
            )
        }
    )
    schema = CfgSchema(spec=root_spec, value=root_val)

    class ContainerDecoProvider:
        def decoration_for(self, path: str, spec: object, value: object):
            del spec, value
            if path in ("outer.inner_section", "outer.inner_ref"):
                return FieldDecorationPatch(enabled=False, badge="disabled")
            return None

    w = CfgFormWidget(
        structure=tree_structure, decoration_provider=ContainerDecoProvider()
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    tree = cast(TreeCfgWidget, w._root_widget)._tree
    outer_field = cast(ReferenceField, draft.root.fields["outer"])

    def find_item(path: str):
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]

        stack = [tree.invisibleRootItem()]
        while stack:
            cur = stack.pop()
            if cur is None:
                continue
            for idx in range(cur.childCount()):
                ch = cur.child(idx)
                if ch is None:
                    continue
                data = ch.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
                if data == path:
                    return ch
                stack.append(ch)
        return None

    def is_enabled(path: str) -> bool:
        item = find_item(path)
        assert item is not None, f"missing item {path!r}"
        wg = tree.itemWidget(item, 1)
        if wg is not None:
            return not item.isDisabled() and wg.isEnabled()  # type: ignore[attr-defined]
        return not item.isDisabled()  # type: ignore[attr-defined]

    # Initially the decoration-disabled containers and their children must be disabled.
    assert is_enabled("outer.inner_section") is False
    assert is_enabled("outer.inner_section.sec_leaf") is False
    assert is_enabled("outer.inner_ref") is False
    assert is_enabled("outer.inner_ref.ref_leaf") is False
    assert is_enabled("outer.normal_leaf") is True

    outer_field.set_enabled(False)
    qapp.processEvents()
    assert is_enabled("outer.inner_section") is False
    assert is_enabled("outer.inner_section.sec_leaf") is False
    assert is_enabled("outer.inner_ref") is False
    assert is_enabled("outer.inner_ref.ref_leaf") is False
    assert is_enabled("outer.normal_leaf") is False

    outer_field.set_enabled(True)
    qapp.processEvents()
    # Re-enabling outer must NOT re-enable children of decoration-disabled containers.
    assert is_enabled("outer.inner_section") is False, (
        "decoration-disabled Section was incorrectly re-enabled"
    )
    assert is_enabled("outer.inner_section.sec_leaf") is False, (
        "child of decoration-disabled Section was incorrectly re-enabled"
    )
    assert is_enabled("outer.inner_ref") is False, (
        "decoration-disabled Reference was incorrectly re-enabled"
    )
    assert is_enabled("outer.inner_ref.ref_leaf") is False, (
        "child of decoration-disabled Reference was incorrectly re-enabled"
    )
    assert is_enabled("outer.normal_leaf") is True

    # Form parity: same provider keeps containers disabled after same sequence.
    w_form = CfgFormWidget(decoration_provider=ContainerDecoProvider())
    draft2 = MeasureCfgBindings(ctrl).new_draft(schema)
    w_form.attach(draft2)
    qapp.processEvents()
    assert w_form.decoration_for_path("outer.inner_section").enabled is False
    assert w_form.decoration_for_path("outer.inner_ref").enabled is False
    assert w_form.decoration_for_path("outer.inner_section.sec_leaf").enabled is True
    assert w_form.decoration_for_path("outer.inner_ref.ref_leaf").enabled is True
    outer2 = cast(ReferenceField, draft2.root.fields["outer"])
    outer2.set_enabled(False)
    qapp.processEvents()
    outer2.set_enabled(True)
    qapp.processEvents()
    assert w_form.decoration_for_path("outer.inner_section").enabled is False
    assert w_form.decoration_for_path("outer.inner_ref").enabled is False
    w.detach()
    draft.close()
    w_form.detach()
    draft2.close()
