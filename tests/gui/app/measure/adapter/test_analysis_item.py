"""Historical analysis documents for adapters that produce figures only."""

import pytest
from tests.gui.app.measure._native_save_support import SavingAdapter
from zcu_tools.resources.entry import ArtifactKey


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_none_result_has_empty_summary_and_inputs(pane):
    key = ArtifactKey(pane, "results", "analysis", pane)
    item = SavingAdapter.analysis_item(key, None, None, pane=pane)
    assert item.content.model_dump() == {"pane": pane, "parameters": {}, "summary": {}}
