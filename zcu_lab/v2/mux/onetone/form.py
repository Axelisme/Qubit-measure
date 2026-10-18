"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def _validate_qubit_bindings(parameters: ParameterContext | None) -> None:
    if parameters is None:
        return
    try:
        name = parameters.resolve_path("$mux")
    except RoleResolutionError as exc:
        if exc.reason in ("Unknown alias", "Alias is not bound", "Role is not bound"):
            return
        raise
    components = parameters.document().components
    group = components.get(name)
    if not isinstance(group, MuxGroupSchema) or group.qubits is None:
        return
    for index, (qubit, resonator) in enumerate(
        zip(group.qubits, group.resonators, strict=True)
    ):
        for field, target, expected in (
            ("qubits", qubit, "qubit"),
            ("resonators", resonator, "resonator"),
        ):
            component = components.get(target)
            if component is None:
                raise ValueError(
                    f"$mux.{field}[{index}]={target!r}: component is missing"
                )
            if component.kind.split("/")[0] != expected:
                raise ValueError(
                    f"$mux.{field}[{index}]={target!r}: kind={component.kind!r}; expected {expected!r}"
                )
        actual = parameters.lookup_point(f"{qubit}.resonator")
        if not actual.present or actual.value != resonator:
            raise ValueError(
                f"$mux.qubits[{index}]={qubit!r}: {qubit}.resonator="
                f"{actual.value!r} (present={actual.present}); "
                f"expected $mux.resonators[{index}]={resonator!r}"
            )


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Use the captured mux slot, or a blank editable readout without tones.

    With qubits, reject missing/wrong-kind components or mismatched resonators
    from the fixed capture at Open/Reset. Group structure is model-validated.
    Without qubits, missing bindings do not gate Run: manual readout suffices.
    Active tones defaults to None (all), as declared by the core cfg.
    Source and alias faults propagate; no live files or devices are read.
    """
    _validate_qubit_bindings(ctx.parameters)
    return Form(
        MuxOnetoneCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "sweep": {"df": Range.between(-1.0, 1.0, expts=101)},
            "initial_delay": Hidden(),
        },
    )
