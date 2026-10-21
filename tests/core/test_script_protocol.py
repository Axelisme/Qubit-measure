"""Authored decision models reject undeclared data and mutation."""

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("model_kind", ["answer", "question"])
def test_decision_models_reject_unknown_fields(model_kind: str) -> None:
    from zcu_tools.core.script_protocol import Answer, Question

    class Approval(Answer):
        accepted: bool

    class Confirm(Question[Approval]):
        evidence: str

    model = Approval if model_kind == "answer" else Confirm
    fields = {"accepted": True} if model_kind == "answer" else {"evidence": "ready"}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({**fields, "undeclared": "not authored"})


@pytest.mark.parametrize("model_kind", ["answer", "question"])
def test_decision_models_cannot_change_after_validation(model_kind: str) -> None:
    from zcu_tools.core.script_protocol import Answer, Question

    class Approval(Answer):
        accepted: bool

    class Confirm(Question[Approval]):
        evidence: str

    model = (
        Approval(accepted=True) if model_kind == "answer" else Confirm(evidence="ready")
    )
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(model, "accepted" if model_kind == "answer" else "evidence", "changed")
