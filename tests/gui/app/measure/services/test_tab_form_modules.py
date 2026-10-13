"""Execution module observations and guarded conversion through the Form owner."""

import pytest
from pydantic import BaseModel
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv
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
                format_version="1.0",
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
        Form(ModuleCore, {}), capture=capture, mutation_allowed=lambda: True
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert hasattr(resource, "read_module"), "Missing resource execution module read"
    before = resource.input.data
    latest = 200.0
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    assert resource.input.data == before
    resource.use_inline(("pulse",))
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    assert captures == [100.0]
    resource.reset()
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert resource.read_module(("pulse",)).value["freq"] == 200.0
    assert captures == [100.0, 200.0]


@pytest.mark.parametrize("fallback", [True, False])
def test_missing_library_conversion_uses_template_or_editable_blank(
    fallback: bool,
) -> None:
    form = Form(
        ModuleCore,
        {
            "pulse": Slot(
                "$qubit.module.pulse", fallback=Template("pulse") if fallback else None
            )
        },
    )
    form.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Pulse",
                {
                    "ch": 0,
                    "nqz": 1,
                    "freq": 90.0,
                    "gain": 0.3,
                    "waveform.style": "const",
                    "waveform.length": 0.2,
                },
            )
        },
        adapter="synthetic",
    )
    resource = TabFormResource(
        form,
        capture=lambda _aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "use_inline"), (
        "Missing owner inline conversion with fallback"
    )
    resource.use_inline(("pulse",))
    data = resource.input.data["pulse"]
    assert isinstance(data, dict) and data["kind"] == "inline"
    module = data["module"]
    assert isinstance(module, dict)
    if fallback:
        assert resource.read_module(("pulse",)).value["freq"] == 90.0
    else:
        assert module["type"] == "pulse"
        resource.write({("pulse", "module", "freq"): 80.0})
        assert resource.preview().values[("pulse", "module", "freq")] == 80.0


class ModuleExtras(BaseModel):
    shift: float = 5.0


def shift_frequency(extras: ModuleExtras, cfg: ModuleCore) -> ModuleCore:
    result = cfg.model_copy(deep=True)
    if result.pulse is not None:
        result.pulse.freq += extras.shift
    return result


def test_module_read_is_field_local_and_excludes_extras_lowering() -> None:
    resource = TabFormResource(
        Form(ModuleCore, {}, extras=ModuleExtras, lower=shift_frequency),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert hasattr(resource, "read_module"), "Missing field-local module observation"
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    resource.use_inline(("pulse",))
    assert resource.read_module(("pulse",)).value["freq"] == 100.0


def test_conversion_guards_and_nonmodule_rejections_preserve_input() -> None:
    allowed = True
    resource = TabFormResource(
        Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: allowed,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    assert hasattr(resource, "use_inline"), "Missing guarded inline conversion"
    before = resource.input.data
    notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    allowed = False
    with pytest.raises(CfgPreconditionError) as caught:
        resource.use_inline(("pulse",))
    assert caught.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    allowed = True
    for path in [("amount",), ("unknown",), ("pulse", "module", "freq")]:
        with pytest.raises(ValueError, match="core module field"):
            resource.use_inline(path)
        with pytest.raises(ValueError, match="core module field"):
            resource.read_module(path)
    assert resource.input.data == before
    assert notifications == []

    def during_notification() -> None:
        assert resource.read_module(("pulse",)).value["freq"] == 100.0
        with pytest.raises(CfgPreconditionError) as reentrant:
            resource.use_inline(("pulse",))
        assert reentrant.value.reason is CfgPreconditionReason.REENTRANT_MUTATION

    resource.subscribe(during_notification)
    resource.use_inline(("pulse",))
    assert notifications == ["changed"]
    resource.revoke()
    for operation in [resource.read_module, resource.use_inline]:
        with pytest.raises(CfgPreconditionError) as gone:
            operation(("pulse",))
        assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE


def test_module_read_allows_other_invalid_input_but_rejects_own_errors() -> None:
    resource = TabFormResource(
        Form(ModuleCore, {}),
        capture=lambda _aliases: TabFormSources(module_env(100.0)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    resource.write({("amount",): {"expr": "unfinished +"}})
    assert hasattr(resource, "read_module"), "Missing field-local module read"
    assert resource.read_module(("pulse",)).value["freq"] == 100.0
    resource.use_inline(("pulse",))
    resource.write({("pulse", "module", "freq"): {"expr": "1 / 0"}})
    with pytest.raises(CfgPreconditionError) as invalid:
        resource.read_module(("pulse",))
    assert invalid.value.reason is CfgPreconditionReason.NOT_VALID
    assert "pulse" in str(invalid.value)


class RequiredModuleCore(ConfigBase):
    pulse: PulseCfg


@pytest.mark.parametrize("captured", [True, False], ids=["captured-ref", "blank"])
def test_required_module_conversion_uses_public_write_once(captured: bool) -> None:
    resource = TabFormResource(
        Form(RequiredModuleCore, {}),
        capture=lambda _aliases: TabFormSources(
            module_env(100.0) if captured else FormEnv()
        ),
        mutation_allowed=lambda: True,
    )
    if captured:
        resource.write({("pulse", "kind"): "ref", ("pulse", "name"): "Q1.pulses.pi"})
    notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    resource.use_inline(("pulse",))
    assert notifications == ["changed"]
    assert resource.input.data["pulse"]["kind"] == "inline"
    if captured:
        assert resource.read_module(("pulse",)).value["freq"] == 100.0
        resource.write({("pulse", "module", "freq"): 125.0})
    else:
        resource.write({("pulse", "module", "freq"): 125.0})
        assert resource.preview().values[("pulse", "module", "freq")] == 125.0
    before = resource.input.data
    changed = len(notifications)
    resource.use_inline(("pulse",))
    assert resource.input.data == before
    assert len(notifications) == changed
