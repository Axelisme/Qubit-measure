"""Form refusal projection keeps the existing taxonomy and readonly details."""

from dataclasses import replace
from typing import Annotated

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.app.measure.remote.form_errors import input_rejected_to_remote
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormInputRejected,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext
from zcu_tools.gui.remote.errors import ErrorCode, ErrorEnvelope

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


class ReadonlyForm(FormModel):
    value: Annotated[list[int] | None, Display(readonly=True)]


@pytest.mark.parametrize("current", [[1, 2], None])
def test_readonly_refusal_wire_preserves_path_reason_and_json_current(
    current: JsonValue,
) -> None:
    state = FormState(FormInput(ReadonlyForm, {"value": current}))
    with pytest.raises(FormInputRejected) as caught:
        state.write({("value",): [3]}, FormEnv())

    wire = ErrorEnvelope.from_remote_error(
        input_rejected_to_remote(caught.value)
    ).to_wire()

    assert wire["code"] == ErrorCode.INVALID_PARAMS.value
    assert wire["reason"] == "form_input_rejected"
    assert wire["data"] == {
        "form_path": ["value"],
        "reason": "readonly",
        "current": current,
    }


def test_other_refusals_keep_the_original_wire_data_without_readonly_details() -> None:
    wire = ErrorEnvelope.from_remote_error(
        input_rejected_to_remote(FormInputRejected(("items", 1, "name"), "identity"))
    ).to_wire()

    assert wire["reason"] == "form_input_rejected"
    assert wire["data"] == {"form_path": ["items", 1, "name"]}


class RefusalTone(ConfigBase):
    name: Annotated[str, FieldMeta(readonly=True)]
    gain: float


class EditableSlot(ConfigBase):
    gain: float


class RefusalCfg(SampleCfg):
    tones: list[RefusalTone]
    slots: list[EditableSlot]


def RefusalAdapter_make_form(ctx: FormContext) -> Form[RefusalCfg]:
    del ctx
    return Form(
        RefusalCfg,
        {
            "sweep": Range.between(1, 2, expts=3),
            "tones": [{"name": "A", "gain": 0.1}, {"name": "B", "gain": 0.2}],
            "slots": [{"gain": 0.1}],
        },
        fixed_elements=(FixedElements(("slots",)),),
    )


RefusalAdapter = Adapter(
    spec=replace(SampleAdapter.spec, cfg_type=RefusalCfg),
    make_form=RefusalAdapter_make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)


@pytest.mark.parametrize(
    ("path", "value", "details"),
    [
        (
            ["tones", 0, "name"],
            "renamed",
            {"form_path": ["tones", 0, "name"], "reason": "readonly", "current": "A"},
        ),
        (["slots"], [], {"form_path": ["slots"]}),
    ],
    ids=["readonly", "fixed-list"],
)
def test_cfg_refusal_includes_usable_canonical_paths_without_partial_write(
    form_rpc, path, value, details
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("refusal", RefusalAdapter)
    tab = fixture.ctrl.new_tab("refusal")
    try:
        before = rpc.call("tab.get_cfg", {"tab_id": tab})["result"]
        refused = rpc.call(
            "tab.edit_cfg",
            {
                "tab_id": tab,
                "edits": [
                    {"path": ["gain"], "value": 0.75},
                    {"path": path, "value": value},
                ],
            },
        )
        assert not refused["ok"], refused
        error = refused["error"]
        assert error["code"] == "invalid_params"
        assert error["reason"] == "form_input_rejected"
        assert {
            key: value
            for key, value in error["data"].items()
            if key not in {"editable_paths", "selection_paths"}
        } == details
        if path == ["slots"]:
            assert "Fixed slot count changed" in error["message"]
        assert rpc.call("tab.get_cfg", {"tab_id": tab})["result"] == before

        editable = error["data"].get("editable_paths", [])
        assert ["tones", 1, "gain"] in editable, error
        assert path not in editable
        assert ["tones", 0, "name"] not in editable
        assert ["sweep"] in error["data"]["selection_paths"]
        leaf = rpc.call(
            "tab.describe_cfg", {"tab_id": tab, "path": ["tones", 1, "gain"]}
        )
        assert leaf["ok"], leaf
        assert leaf["result"]["description"]["path"] == ["tones", 1, "gain"]
        edited = rpc.call(
            "tab.edit_cfg",
            {
                "tab_id": tab,
                "edits": [{"path": ["tones", 1, "gain"], "value": 0.3}],
            },
        )
        assert edited["ok"], edited
        assert edited["result"]["input"]["tones"][1]["gain"] == 0.3
    finally:
        fixture.ctrl.tab_control.close_tab(tab)
