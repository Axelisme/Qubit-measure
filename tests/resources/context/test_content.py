"""Memory publication is independent of the stores' file synchronization."""

import pytest
from zcu_tools.resources.context import MetaDict, ModuleLibrary
from zcu_tools.resources.context.content import (
    replace_context_contents,
    snapshot_context_contents,
)


def test_snapshot_uses_memory_without_reloading_external_file_changes(tmp_path):
    path = tmp_path / "metadata.json"
    md = MetaDict(path)
    md.update(offset=1.0)
    path.write_text('{"offset": 9.0}')
    copied_md, copied_ml = snapshot_context_contents(md, ModuleLibrary())
    assert copied_md.get("offset") == 1.0
    assert not copied_md.has_persistence
    assert not copied_ml.has_persistence
    copied_md.update(offset=3.0)
    again, _ = snapshot_context_contents(md, ModuleLibrary())
    assert again.get("offset") == 1.0


def test_replacement_preserves_store_identity_and_does_not_share_candidate_data():
    md, ml = MetaDict(), ModuleLibrary()
    candidate_md, candidate_ml = snapshot_context_contents(md, ml)
    candidate_md.update(offset=2.0)
    candidate_ml.register_waveform(tone={"style": "const", "length": 0.1})
    replace_context_contents(md, ml, metadata=candidate_md, library=candidate_ml)
    candidate_md.update(offset=8.0)
    candidate_ml.waveforms.clear()
    assert md.get("offset") == 2.0
    assert "tone" in ml.waveforms


@pytest.mark.parametrize("readonly_store", ["metadata", "library"])
def test_readonly_target_rejects_whole_replacement(readonly_store):
    md = MetaDict(readonly=readonly_store == "metadata")
    ml = ModuleLibrary(readonly=readonly_store == "library")
    candidate_md, candidate_ml = snapshot_context_contents(md, ml)
    candidate_md.update(offset=2.0)
    candidate_ml.register_waveform(tone={"style": "const", "length": 0.1})
    with pytest.raises(RuntimeError, match="read-only"):
        replace_context_contents(md, ml, metadata=candidate_md, library=candidate_ml)
    assert list(md.items()) == []
    assert ml.waveforms == {}
