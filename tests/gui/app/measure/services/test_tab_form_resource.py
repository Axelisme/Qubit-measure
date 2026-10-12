"""Headless form resources fix sources until explicit Reset and guard lifetime."""

from collections.abc import Mapping

import pytest
from zcu_tools.cfg_model import ConfigBase, Expr
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.services import tab_cfg as resource_api
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class ScalarCore(ConfigBase):
    """A concrete experiment cfg accepts one positive measurement setting."""

    amount: float = 1.0


def captured_env(amount: float) -> FormEnv:
    """Construct a detached source with a known arithmetic value."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(
                created_at="2026-10-13T00:00:00Z", ext={"amount": amount}
            ),
        ),
        library=library,
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
    )
    return FormEnv(parameters=parameters, library=library)


def test_resource_edit_preview_run_use_capture_until_reset() -> None:
    assert hasattr(resource_api, "TabFormResource"), (
        "Missing tab-owned FormState resource"
    )
    assert hasattr(resource_api, "TabFormSources"), "Missing fixed form sources"
    latest = 2.0
    captures: list[Mapping[str, str] | None] = []

    def capture(aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        captures.append(aliases)
        return resource_api.TabFormSources(env=captured_env(latest))

    resource = resource_api.TabFormResource(
        Form(ScalarCore, {"amount": Expr("general.ext.amount * 2")}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    assert resource.result() == ScalarCore(amount=4.0)
    latest = 5.0
    resource.write({("amount",): {"expr": "general.ext.amount * 3"}})
    assert resource.preview().values[("amount",)] == 6.0
    assert resource.result() == ScalarCore(amount=6.0)
    assert resource.accept().values == {"amount": 6.0}
    assert captures == [None]
    resource.reset(aliases={"selected": "Q2"})
    assert resource.result() == ScalarCore(amount=10.0)
    assert captures == [None, {"selected": "Q2"}]
    resource.write({("amount",): {"expr": "unfinished +"}})
    assert isinstance(resource.result(), FormErrors)
    with pytest.raises(CfgPreconditionError) as caught:
        resource.accept()
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID
    assert resource.input.data["amount"] == {"expr": "unfinished +"}
