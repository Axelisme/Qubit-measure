"""Execution module observations and guarded conversion through the Form owner."""

import pytest
from pydantic import BaseModel
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
)
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv, FormPathError
from zcu_tools.gui.form_adapter import Form
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleCore(ConfigBase):
    pulse: PulseCfg | None = None
    locked: PulseCfg | None = None
    amount: float = 1.0


def module_env(freq: float) -> FormEnv:
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={
                            "pi": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=freq,
                                gain=0.2,
                                waveform=ConstWaveformCfg(length=0.1),
                            )
                        }
                    )
                },
            )
        )
    )


def test_module_read_and_conversion_use_capture_until_reset() -> None:
    latest = 100.0
    captures: list[float] = []

    def capture(_aliases: object) -> TabFormSources:
        captures.append(latest)
        return TabFormSources(module_env(latest))

    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert hasattr(resource, "read_module"), "Missing resource execution module read"
    before = resource.input.data
    latest = 200.0
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    assert resource.input.data == before
    resource.run_command(("pulse",), "to_inline")
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    assert captures == [100.0]
    resource.reset()
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert resource.read_module(("pulse",)).value["freq"] == 200.0
    assert captures == [100.0, 200.0]


class ModuleExtras(BaseModel):
    shift: float = 5.0


def shift_frequency(extras: ModuleExtras, cfg: ModuleCore) -> ModuleCore:
    result = cfg.model_copy(deep=True)
    if result.pulse is not None:
        result.pulse.freq += extras.shift
    return result


def test_module_read_is_field_local_and_excludes_extras_lowering() -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}, extras=ModuleExtras, lower=shift_frequency),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert hasattr(resource, "read_module"), "Missing field-local module observation"
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    resource.run_command(("pulse",), "to_inline")
    assert resource.read_module(("pulse",)).value["freq"] == 100.0


def test_module_read_allows_other_invalid_input_but_rejects_own_errors() -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    resource.write({("amount",): {"expr": "unfinished +"}})
    assert hasattr(resource, "read_module"), "Missing field-local module read"
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    resource.run_command(("pulse",), "to_inline")
    resource.write({("pulse", "module", "freq"): {"expr": "1 / 0"}})
    with pytest.raises(CfgPreconditionError) as invalid:
        resource.read_module(("pulse",))
    assert invalid.value.reason is CfgPreconditionReason.NOT_VALID
    assert "pulse" in str(invalid.value)


def test_option_command_copies_fixed_capture_and_publishes_once() -> None:
    latest = 100.0
    captures: list[float] = []

    def capture(_aliases: object) -> TabFormSources:
        captures.append(latest)
        return TabFormSources(module_env(latest))

    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    notifications: list[str] = []
    state_notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    resource.editor_state.subscribe(lambda: state_notifications.append("changed"))
    latest = 200.0
    assert hasattr(resource, "run_command"), "Missing guarded option command host"
    resource.run_command(("pulse",), "to_inline")
    module = resource.read_module(("pulse",))
    assert module.mode == "inline"
    assert module.source == "pulse"
    assert module.value["freq"] == 100.0
    assert notifications == ["changed"]
    assert state_notifications == ["changed"]
    assert captures == [100.0]
    module.value["freq"] = -1.0
    resource.write({("pulse", "module", "freq"): 125.0})
    assert resource.read_module(("pulse",)).value["freq"] == 125.0
    assert resource.accept().values["pulse"] is not None
    resource.reset()
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    resource.run_command(("pulse",), "to_inline")
    assert resource.read_module(("pulse",)).value["freq"] == 200.0
    assert captures == [100.0, 200.0]


def test_option_command_guards_busy_lifetime_and_peer_reentry() -> None:
    from zcu_tools.gui.app.measure.services.tab_form import TabFormResources

    allowed = True
    pool = TabFormResources(mutation_allowed=lambda _tab: allowed)
    form = Form(ModuleCore, {})

    def capture(_aliases: object) -> TabFormSources:
        return TabFormSources(module_env(100.0))

    resource = pool.create("target", lambda _ctx: form, capture=capture)
    peer = pool.create("peer", lambda _ctx: form, capture=capture)
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    assert hasattr(resource, "run_command"), "Missing guarded option command host"
    before = resource.input.data
    notifications: list[str] = []
    failures: list[CfgPreconditionReason] = []
    resource.subscribe(lambda: notifications.append("changed"))
    allowed = False
    with pytest.raises(CfgPreconditionError) as busy:
        resource.run_command(("pulse",), "to_inline")
    assert busy.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    allowed = True

    def reenter() -> None:
        try:
            resource.run_command(("pulse",), "to_inline")
        except CfgPreconditionError as error:
            failures.append(error.reason)

    peer.subscribe(reenter)
    peer.write({("amount",): 2.0})
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert resource.input.data == before
    assert notifications == []
    resource.subscribe(reenter)
    resource.run_command(("pulse",), "to_inline")
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 2
    assert notifications == ["changed"]
    pool.retire("target")
    with pytest.raises(CfgPreconditionError) as gone:
        resource.run_command(("pulse",), "to_inline")
    assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (("missing",), "to_inline"),
        (("amount",), "to_inline"),
        (("pulse",), "missing"),
        (("pulse", "module", "freq"), "to_inline"),
    ],
)
def test_option_command_rejection_is_atomic(path: tuple[str, ...], name: str) -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    before = resource.input.data
    notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    assert hasattr(resource, "run_command"), "Missing guarded option command host"
    if path in (("missing",), ("pulse", "module", "freq")):
        with pytest.raises(FormPathError) as invalid_path:
            resource.run_command(path, name)
        assert invalid_path.value.path == path
    else:
        with pytest.raises(ValueError, match="Unavailable option command"):
            resource.run_command(path, name)
    assert resource.input.data == before
    assert notifications == []
    resource.run_command(("pulse",), "to_inline")
    before = resource.input.data
    with pytest.raises(ValueError, match="Unavailable"):
        resource.run_command(("pulse",), "to_inline")
    assert resource.input.data == before
    assert notifications == ["changed"]


def test_recipe_ref_read_is_detached_and_does_not_authorize_descendant_write() -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "select_recipe_module"), "Missing logical selector command"
    resource.select_recipe_module(("pulse",), "use_library", key="Q1.pulses.pi")
    before = resource.input.data
    base = resource.editor_state.base.data
    notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    observed = resource.read_recipe_field(("pulse", "waveform", "length"))
    assert observed.value_status == "available"
    assert observed.value == 0.1
    assert observed.input_present is False
    assert observed.input is None
    assert observed.readonly is True
    assert observed.selector is None
    assert observed.source is not None
    assert observed.source.selector_path == ("pulse",)
    assert observed.source.key == "Q1.pulses.pi"
    assert observed.source.spec_path == ("waveform", "length")
    assert resource.input.data == before
    assert resource.editor_state.base.data == base
    assert notifications == []
    from zcu_tools.gui.form_adapter import LogicalEdit, LogicalEditError

    with pytest.raises(LogicalEditError) as rejected:
        resource.edit_recipe_fields(
            [
                LogicalEdit(("amount",), 2.0),
                LogicalEdit(("pulse", "freq"), 125.0),
            ]
        )
    assert rejected.value.reason_code == "ref_ancestor"
    assert rejected.value.edit_index == 1
    assert resource.input.data == before
    assert notifications == []


def test_recipe_copy_and_batch_install_once_without_replacing_base() -> None:
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "edit_recipe_fields"), "Missing logical batch command"
    from zcu_tools.gui.form_adapter import LogicalEdit

    base = resource.editor_state.base.data
    resource.select_recipe_module(("pulse",), "use_library", key="Q1.pulses.pi")
    assert resource.select_recipe_module(("pulse",), "use_inline") is True
    notifications: list[str] = []
    state_notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    resource.editor_state.subscribe(lambda: state_notifications.append("changed"))
    assert resource.select_recipe_module(("pulse",), "use_inline") is False
    assert (
        resource.edit_recipe_fields(
            [
                LogicalEdit(("amount",), 2.0),
                LogicalEdit(("pulse", "freq"), {"expr": "125 + 5"}),
            ]
        )
        is True
    )
    assert resource.read_recipe_field(("pulse", "freq")).value == 130.0
    assert resource.read_recipe_field(("pulse", "freq")).input == {"expr": "125 + 5"}
    assert resource.editor_state.base.data == base
    assert notifications == ["changed"]
    assert state_notifications == ["changed"]
    resource.edit_recipe_fields([LogicalEdit(("amount",), {"expr": "1 / 0"})])
    invalid = resource.read_recipe_field(("amount",))
    assert invalid.value_status == "unavailable"
    assert invalid.errors is not None
    assert invalid.input == {"expr": "1 / 0"}


def test_recipe_inline_noop_still_takes_busy_notification_and_lifetime_guards() -> None:
    allowed = True
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: allowed,
    )
    assert hasattr(resource, "select_recipe_module"), "Missing guarded selector command"
    resource.select_recipe_module(("pulse",), "use_library", key="Q1.pulses.pi")
    resource.select_recipe_module(("pulse",), "use_inline")
    before = resource.input.data
    allowed = False
    with pytest.raises(CfgPreconditionError) as busy:
        resource.select_recipe_module(("pulse",), "use_inline")
    assert busy.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    assert resource.read_recipe_field(("pulse", "freq")).value == 100.0
    allowed = True
    failures: list[CfgPreconditionReason] = []

    def during_notification() -> None:
        assert resource.read_recipe_field(("pulse", "freq")).value == 100.0
        try:
            resource.select_recipe_module(("pulse",), "use_inline")
        except CfgPreconditionError as error:
            failures.append(error.reason)

    resource.editor_state.subscribe(during_notification)
    resource.write({("amount",): 2.0})
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert resource.input.data["pulse"] == before["pulse"]
    resource.revoke()
    with pytest.raises(CfgPreconditionError) as gone:
        resource.select_recipe_module(("pulse",), "use_inline")
    assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE


def test_recipe_nested_and_extra_selectors_require_explicit_ancestor_copies() -> None:
    from pydantic import create_model
    from zcu_tools.gui.form import FormModel
    from zcu_tools.gui.form_adapter import LogicalEdit, LogicalEditError, ModuleInputField

    field = ModuleInputField(PulseSpec, nullable=True)

    class ExtraSelectors(FormModel):
        @classmethod
        def options(cls, env: FormEnv):
            return {("reference",): field.options(env)}

    extras = create_model("ExtraSelectorFields", __base__=ExtraSelectors,
                          reference=(field.annotation, None))

    def lower(_extras: BaseModel, cfg: ModuleCore) -> ModuleCore:
        return cfg

    library = LibrarySnapshot(ModuleLibraryDocument(
        format="zcu.module-library", format_version="2.0", components={
            "Q1": ComponentLibrary(pulses={"pi": PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.const",
            )}, waveforms={"const": ConstWaveformCfg(length=0.1)}),
        },
    ))
    resource = TabFormResource(
        lambda _ctx: Form(ModuleCore, {}, extras=extras, lower=lower),
        capture=lambda _aliases: TabFormSources(FormEnv(library=library)),
        mutation_allowed=lambda: True,
    )
    resource.select_recipe_module(("pulse",), "use_library", key="Q1.pulses.pi")
    before = resource.input.data
    observed = resource.read_recipe_field(("pulse", "waveform", "length"))
    assert observed.value == 0.1 and observed.readonly
    assert observed.source is not None
    assert observed.source.key == "Q1.waveforms.const"
    assert observed.source.selector_path == ("pulse", "waveform")
    assert observed.source.source_kind == "waveform"
    assert observed.source.spec_path == ("length",)
    with pytest.raises(LogicalEditError) as ancestor:
        resource.select_recipe_module(("pulse", "waveform"), "use_inline")
    assert ancestor.value.reason_code == "ref_ancestor"
    assert ancestor.value.selector_path == ("pulse",)
    assert resource.input.data == before
    resource.select_recipe_module(("pulse",), "use_inline")
    assert resource.read_recipe_field(("pulse", "waveform")).selector.key == "Q1.waveforms.const"
    resource.select_recipe_module(("pulse", "waveform"), "use_inline")
    resource.edit_recipe_fields([LogicalEdit(("pulse", "waveform", "length"), 0.08)])
    assert resource.read_recipe_field(("pulse", "waveform", "length")).value == 0.08
    resource.select_recipe_module(("extras", "reference"), "use_library", key="Q1.pulses.pi")
    assert resource.read_recipe_field(("extras", "reference", "freq")).value == 100.0
    resource.select_recipe_module(("extras", "reference"), "use_inline")
    resource.edit_recipe_fields([LogicalEdit(("extras", "reference", "freq"), 125.0)])
    assert resource.read_recipe_field(("extras", "reference", "freq")).value == 125.0
    assert library.module_spec("Q1.pulses.pi").freq == 100.0
