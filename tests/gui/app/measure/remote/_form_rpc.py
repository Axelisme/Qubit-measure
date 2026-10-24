"""In-process Form RPC collaborator and registered module adapter."""

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pydantic import Field, create_model
from zcu_tools.cfg_model import PulseCfg, PulseSpec, Range
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter import Form, FormContext, ModuleInputField
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


class ImmediateScheduler:
    """Dispatch on the test's current owner thread."""

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call[T](self, callback: Callable[[], T]) -> T:
        return callback()


class FormRpc:
    """Record real dispatch replies without a socket or a rendered window."""

    def __init__(self, remote: RemoteControlAdapter) -> None:
        self.remote = remote
        self.link = ClientLink("form-test", token_required=False)
        remote.on_client_open(self.link)
        self.call("control.connect", {})

    def call(self, method: str, params: dict[str, object]) -> dict[str, Any]:
        self.remote.route(self.link, Request("form-request", method, params))
        return json.loads(self.link.outbound.get(timeout=2.0))


class ModuleRpcCfg(SampleCfg):
    pulse: PulseCfg | None = None


class OrdinaryRecipeSection(FormModel):
    value: int = 1


_recipe_drive = ModuleInputField(PulseSpec)
RecipeGroup = create_model(
    "RecipeGroup", __base__=FormModel, drive=(_recipe_drive.annotation, ...)
)
RecipeExtras = create_model(
    "RecipeExtras",
    __base__=FormModel,
    real=(OrdinaryRecipeSection, OrdinaryRecipeSection()),
    group=(RecipeGroup | None, None),
)


ContainerRpcInput = create_model(
    "ContainerRpcInput",
    __base__=SampleCfg,
    extras=(RecipeExtras, Field(default_factory=RecipeExtras)),
)


def ContainerRpcAdapter_make_form(ctx: FormContext) -> Form[SampleCfg]:
    del ctx
    return Form(
        ContainerRpcInput,
        {"sweep": Range.between(1.0, 2.0, expts=3)},
    )


ContainerRpcAdapter = Adapter(
    spec=replace(SampleAdapter.spec, cfg_type=SampleCfg),
    make_form=ContainerRpcAdapter_make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)


def IndexedRpcAdapter_make_form(ctx: FormContext):
    from zcu_tools.gui.form_adapter import FixedElements

    from tests.gui.form_adapter._indexed_models import NativeTone, indexed_tree

    del ctx
    cfg = create_model(
        "IndexedRpcCfg",
        __base__=SampleCfg,
        tones=(list[NativeTone], ...),
        participants=(list[int], ...),
    )
    return Form(
        cfg,
        {"sweep": Range.between(1, 2, expts=3), **indexed_tree()},
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )


IndexedRpcAdapter = Adapter(
    spec=replace(SampleAdapter.spec, cfg_type=SampleCfg),
    make_form=IndexedRpcAdapter_make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)


def ModuleRpcAdapter_make_form(ctx: FormContext) -> Form[ModuleRpcCfg]:
    del ctx
    return Form(ModuleRpcCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})


ModuleRpcAdapter = Adapter(
    spec=replace(SampleAdapter.spec, cfg_type=ModuleRpcCfg),
    make_form=ModuleRpcAdapter_make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)
