"""Editor and resource subscribers share one synchronous guarded publication."""

import pytest
from zcu_tools.cfg_model import ConfigBase, ConstWaveformCfg, PulseCfg, PulseSpec
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.version_table import VersionTable
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class SubscriberCfg(ConfigBase):
    pulse: PulseCfg | None = None
    amount: float = 1.0


@pytest.fixture
def notification_resource() -> tuple[TabFormResource, VersionTable]:
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "pi": PulseSpec(
                            ch=0,
                            nqz=1,
                            freq=100.0,
                            gain=0.2,
                            waveform=ConstWaveformCfg(length=0.1),
                        )
                    }
                )
            },
        )
    )
    resource = TabFormResource(
        lambda _ctx: Form(SubscriberCfg, {}),
        capture=lambda _aliases: TabFormSources(FormEnv(library=library)),
        mutation_allowed=lambda: True,
    )
    resource.write({("pulse",): {"kind": "ref", "name": "Q1.pulses.pi"}})
    version = VersionTable()

    def cfg_changed() -> None:
        version.bump("tab:synthetic:cfg")

    resource.subscribe(cfg_changed)
    return resource, version


def outer_write(resource: TabFormResource, operation: str) -> None:
    """Submit the selected public editing operation, without a state side channel."""
    if operation == "command":
        resource.run_command(("pulse",), "to_inline")
    else:
        resource.write({("amount",): 2.0})


@pytest.mark.parametrize("outer", ["write", "command"])
@pytest.mark.parametrize("inner", ["write", "command"])
def test_editor_notification_rejects_reentry_before_mutation(
    notification_resource: tuple[TabFormResource, VersionTable],
    outer: str,
    inner: str,
) -> None:
    resource, version = notification_resource
    attempts: list[CfgPreconditionReason] = []
    entered = False

    def reenter() -> None:
        nonlocal entered
        if entered:
            return
        entered = True
        assert resource.preview().values[("amount",)] == (
            2.0 if outer == "write" else 1.0
        )
        try:
            outer_write(resource, inner)
        except CfgPreconditionError as error:
            attempts.append(error.reason)

    resource.editor_state.subscribe(reenter)
    outer_write(resource, outer)
    assert attempts == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert resource.input.data["amount"] == (2.0 if outer == "write" else 1.0)
    assert resource.read_module(("pulse",)).mode == (
        "key" if outer == "write" else "inline"
    )
    assert version.get("tab:synthetic:cfg") == 1


@pytest.mark.parametrize("outer", ["write", "command"])
def test_editor_subscriber_fault_does_not_drop_publication(
    notification_resource: tuple[TabFormResource, VersionTable],
    caplog: pytest.LogCaptureFixture,
    outer: str,
) -> None:
    resource, version = notification_resource
    received: list[str] = []

    def fail() -> None:
        raise RuntimeError("editor observer failed")

    unsubscribe = resource.editor_state.subscribe(fail)
    resource.editor_state.subscribe(lambda: received.append("editor"))
    resource.subscribe(lambda: received.append("resource"))
    try:
        outer_write(resource, outer)
    except RuntimeError as exc:
        raise AssertionError(
            "subscriber fault must not reject committed write"
        ) from exc
    assert received == ["editor", "resource"]
    assert resource.input.data["amount"] == (2.0 if outer == "write" else 1.0)
    assert resource.read_module(("pulse",)).mode == (
        "key" if outer == "write" else "inline"
    )
    assert version.get("tab:synthetic:cfg") == 1
    assert any(record.exc_info for record in caplog.records)
    assert "editor observer failed" in caplog.text
    unsubscribe()
    resource.write({("amount",): 3.0})
    assert version.get("tab:synthetic:cfg") == 2
