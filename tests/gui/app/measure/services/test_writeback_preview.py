"""Complete live writeback previews through the service's public boundary."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    SessionEnv,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary


def test_values_follow_live_context_retarget_and_do_not_alias_mutable_md():
    context = SessionEnv(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    context.md.value = [1, 2]
    service = WritebackService(MagicMock(), MagicMock())
    draft = service.create_draft(
        [
            MetaDictWriteback(
                target_name="value", description="d", proposed_value=[3, 4]
            ),
        ]
    )
    first = service.preview_values(draft, context)["md-1"]
    assert first.current == [1, 2]
    assert first.proposed == [3, 4]
    context.md.value.append(5)
    assert first.current == [1, 2]
    assert isinstance(first.proposed, list)
    first.proposed.append(99)
    assert service.preview_values(draft, context)["md-1"].proposed == [3, 4]
    draft.edit("md-1", target_name="other")
    assert service.preview_values(draft, context)["md-1"].current is None
    context.md.other = 7
    assert service.preview_values(draft, context)["md-1"].current == 7
    replacement = replace(context, md=MetaDict())
    replacement.md.other = 8
    assert service.preview_values(draft, replacement)["md-1"].current == 8
    assert service.get_all_applied(draft) == {"md-1": False}
    assert draft.items[0].selected is True


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_cfg_preview_contains_complete_current_and_live_draft_values(kind):
    context = SessionEnv(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    if kind == "module":
        cfg = ModuleCfgFactory.from_raw(
            {
                "type": "pulse",
                "ch": 0,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.5,
                "phase": 0.0,
                "pre_delay": 0.0,
                "post_delay": 0.0,
                "waveform": {"style": "const", "length": 0.1},
            }
        )
        context.ml.modules["target"] = cfg
        spec, value = module_cfg_to_value(cfg)
        item_type, item_id = ModuleWriteback, "ml-1"
    else:
        cfg = WaveformCfgFactory.from_raw({"style": "const", "length": 0.1})
        context.ml.waveforms["target"] = cfg
        spec, value = waveform_cfg_to_value(cfg)
        item_type, item_id = WaveformWriteback, "wf-1"
    schema = CfgSchema(spec, value)
    editor = MagicMock()
    editor.open_seeded.return_value = ("editor", ())
    editor.get_draft.return_value.snapshot.return_value = schema
    service = WritebackService(editor, MagicMock())
    draft = service.create_draft(
        [
            item_type(target_name="target", description="d", edit_schema=schema),
        ]
    )

    preview = service.preview_values(draft, context)[item_id]

    assert preview.current == cfg.to_dict()
    assert preview.proposed == cfg.to_dict()
    draft.edit(item_id, target_name="new")
    missing = service.preview_values(draft, context)[item_id]
    assert missing.current is None
    assert missing.proposed == preview.proposed
    error = FailedPreconditionError("draft unavailable")
    editor.get_draft.return_value.snapshot.side_effect = error
    with pytest.raises(FailedPreconditionError) as caught:
        service.preview_values(draft, context)
    assert caught.value is error


def test_preview_without_context_fails_instead_of_claiming_missing_targets():
    service = WritebackService(MagicMock(), MagicMock())
    draft = service.create_draft(
        [
            MetaDictWriteback(target_name="value", description="d", proposed_value=1),
        ]
    )
    with pytest.raises(FailedPreconditionError):
        service.preview_values(
            draft, SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
        )
