"""Create persistent templates through the controller and real library owners."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    Expr,
    PulseResetSpec,
    PulseSpec,
    Slot,
    TemplateDef,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import component_registry

from tests.gui._component_fakes import Qubit


class TemplateController(Controller):
    """Join the concrete background executor during fixture teardown."""

    def quiesce(self) -> None:
        self._background_svc.quiesce()


@pytest.fixture
def controller(qapp, tmp_path: Path):
    templates = {
        "probe": TemplateDef(
            PulseSpec,
            "Probe",
            {
                "ch": 2,
                "nqz": 1,
                "freq": Expr("{qubit}.ext.freq"),
                "gain": 0.25,
                "waveform.style": "const",
                "waveform.length": 0.4,
            },
            "{qubit}.pulses.probe",
        ),
        "reset": TemplateDef(
            PulseResetSpec,
            "Reset",
            {
                "pulse_cfg": Slot("{qubit}.pulses.probe"),
            },
            "{qubit}.reset.probe",
        ),
        "envelope": TemplateDef(ConstWaveformCfg, "Envelope", {"length": 0.5}, "base"),
    }
    ctrl = TemplateController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(templates=templates),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    ctrl.setup_control.new_entry(EntryRequest("templates"))
    before = component_registry.kinds()
    component_registry.register("template/qubit", Qubit)
    try:
        ctrl.context_control.add_component("Q1", "template/qubit", {}, setup=True)
        ctrl.setup_control.new_point(
            PointCreateRequest("working", "", CommandOrigin("gui"))
        )
        ctrl.context_control.set_alias_rows((AliasRow("qubit", "Q1"),))
        capture = ctrl.read_context()
        ctrl.context_control.set_parameter(capture.target, "Q1.ext.freq", 6123.0)
        yield ctrl
    finally:
        ctrl.quiesce()
        component_registry.unregister("template/qubit")
        assert component_registry.kinds() == before


def test_registry_template_creation_evaluates_placeholder_and_keeps_bound_slot(
    controller: Controller,
):
    controller.create_from_template("module", "probe", "Q1.pulses.probe")
    library = controller.read_context().library
    pulse = library.module_spec("Q1.pulses.probe")
    assert pulse == PulseSpec(
        ch=2, nqz=1, freq=6123.0, gain=0.25, waveform=ConstWaveformCfg(length=0.4)
    )
    controller.create_from_template("waveform", "envelope", "Q1.waveforms.base")
    controller.create_from_template("module", "reset", "Q1.reset.probe")
    reset = controller.read_context().library.module_spec("Q1.reset.probe")
    assert isinstance(reset, PulseResetSpec)
    assert reset.pulse_cfg == "Q1.pulses.probe"
    assert controller.read_context().library.waveform_spec(
        "Q1.waveforms.base"
    ) == ConstWaveformCfg(length=0.5)


@pytest.mark.parametrize(
    "kind,template,name,error",
    [
        ("module", "missing", "Q1.pulses.new", KeyError),
        ("module", "probe", "Q1.pulses.occupied", FailedPreconditionError),
        ("waveform", "probe", "Q1.waveforms.new", FailedPreconditionError),
        ("module", "probe", "", FailedPreconditionError),
    ],
)
def test_rejected_creation_leaves_library_unchanged(
    controller: Controller, kind: str, template: str, name: str, error: type[Exception]
):
    capture = controller.read_context()
    controller.context_control.set_module(
        capture.target,
        "Q1.pulses.occupied",
        PulseSpec(
            ch=0, nqz=1, freq=1.0, gain=0.1, waveform=ConstWaveformCfg(length=1.0)
        ),
    )
    before = controller.read_context().library
    with pytest.raises(error):
        controller.create_from_template(kind, template, name)
    after = controller.read_context().library
    assert after.module_keys() == before.module_keys()
    assert after.waveform_keys() == before.waveform_keys()
    assert after.module_spec("Q1.pulses.occupied") == before.module_spec(
        "Q1.pulses.occupied"
    )


def test_invalid_template_input_is_rejected_before_write(controller: Controller):
    capture = controller.read_context()
    controller.context_control.set_parameter(capture.target, "Q1.ext.freq", None)
    with pytest.raises(LibraryEntryInvalid, match="freq"):
        controller.create_from_template("module", "probe", "Q1.pulses.new")
    assert controller.read_context().library.module_keys() == ()


def test_wire_discovery_and_creation_keep_reply_shapes(controller: Controller):
    from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
    from zcu_tools.gui.app.measure.remote.handlers.context import (
        h_context_ml_create_from_role,
        h_context_ml_list_roles,
    )

    adapter = RemoteControlAdapter(
        controller=controller,
        opts=ControlOptions(port=0),
        owner_scheduler=controller.owner_scheduler,
    )
    assert h_context_ml_list_roles(adapter, {}) == {
        "roles": [
            {
                "role_id": "probe",
                "label": "Probe",
                "item_kind": "module",
                "default_name": "{qubit}.pulses.probe",
            },
            {
                "role_id": "reset",
                "label": "Reset",
                "item_kind": "module",
                "default_name": "{qubit}.reset.probe",
            },
            {
                "role_id": "envelope",
                "label": "Envelope",
                "item_kind": "waveform",
                "default_name": "base",
            },
        ]
    }
    assert h_context_ml_create_from_role(
        adapter, {"role_id": "probe", "name": "Q1.pulses.rpc"}
    ) == {"created": "Q1.pulses.rpc"}
    assert (
        controller.read_context()
        .library.module_spec("Q1.pulses.rpc")
        .model_dump()["freq"]
        == 6123.0
    )


def test_wire_missing_id_is_invalid_params_without_write(controller: Controller):
    from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
    from zcu_tools.gui.app.measure.remote.handlers.context import (
        h_context_ml_create_from_role,
    )
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

    adapter = RemoteControlAdapter(
        controller=controller,
        opts=ControlOptions(port=0),
        owner_scheduler=controller.owner_scheduler,
    )
    with pytest.raises(RemoteError) as exc:
        h_context_ml_create_from_role(
            adapter, {"role_id": "missing", "name": "Q1.pulses.rpc"}
        )
    assert exc.value.code == ErrorCode.INVALID_PARAMS
    assert controller.read_context().library.module_keys() == ()
