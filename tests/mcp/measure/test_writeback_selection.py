"""Captured question selection is detached from later GUI drafts."""

from copy import deepcopy

import pytest
from zcu_tools.mcp.measure.recipe import RecipeWritebackPreview
from zcu_tools.mcp.measure.writeback import select_writeback_items


def captured_preview() -> RecipeWritebackPreview:
    return RecipeWritebackPreview(
        primary={
            "has_draft": True,
            "items": [
                {
                    "id": "md-old",
                    "target_name": "frequency",
                    "selected": False,
                    "proposed_value": {"value": 2},
                },
                {"id": "wf-old", "target_name": "pulse", "selected": True},
            ],
            "destination_context": {"project": {"name": "captured"}},
        },
        post={
            "has_draft": True,
            "items": [
                {"id": "post-old", "target_name": "classifier", "selected": True}
            ],
            "destination_context": {"project": {"name": "captured"}},
        },
    )


def test_captured_selection_uses_stable_names_and_preserves_original_order():
    preview = captured_preview()
    original = deepcopy(preview)
    selection = select_writeback_items(preview, ["classifier", "frequency"])

    assert selection.items == ("frequency", "classifier")
    assert selection.preview.primary is not None
    assert selection.preview.post is not None
    assert [item["id"] for item in selection.preview.primary["items"]] == ["md-old"]
    assert [item["id"] for item in selection.preview.post["items"]] == ["post-old"]
    assert selection.preview.primary["items"][0]["selected"] is False
    assert selection.preview.primary["destination_context"] == {
        "project": {"name": "captured"}
    }
    selection.preview.primary["items"][0]["target_name"] = "later_change"
    selection.preview.primary["destination_context"]["project"] = {"name": "later"}
    assert preview == original


@pytest.mark.parametrize("items", [None, []])
def test_captured_selection_distinguishes_all_from_zero_items(items):
    selection = select_writeback_items(captured_preview(), items)
    assert selection.items == (
        ("frequency", "pulse", "classifier") if items is None else ()
    )
    assert selection.preview.primary is not None
    assert len(selection.preview.primary["items"]) == (2 if items is None else 0)
    assert selection.preview.post is not None
    assert len(selection.preview.post["items"]) == (1 if items is None else 0)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["absent"], "Unknown"),
        (["frequency", "frequency"], "Duplicate"),
        ("frequency", "items"),
        ([""], "items"),
    ],
)
def test_captured_selection_rejects_invalid_stable_names(items, message):
    with pytest.raises(ValueError, match=message):
        select_writeback_items(captured_preview(), items)


def test_captured_selection_rejects_ambiguous_names_even_for_zero_items():
    preview = captured_preview()
    assert preview.post is not None
    preview.post["items"][0]["target_name"] = "frequency"
    with pytest.raises(ValueError, match="Ambiguous"):
        select_writeback_items(preview, [])


def test_captured_selection_keeps_absence_and_confirmed_empty_draft_distinct():
    preview = RecipeWritebackPreview(
        primary=None,
        post={"has_draft": False, "items": [], "destination_context": {}},
    )
    selection = select_writeback_items(preview)
    assert selection.items == ()
    assert selection.preview == preview
    assert selection.preview is not preview
