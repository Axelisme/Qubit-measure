"""Native library mutations leave tab Form sources fixed until Reset."""

import pytest
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    Range,
    Slot,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import LibraryChange

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg

from ._helpers import Fixture, mcp_client


class PulseReferenceCfg(SampleCfg):
    linked: PulseCfg | None = None
    inline: PulseCfg | None = None


class PulseReferenceAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[PulseReferenceCfg]:
        del ctx
        return Form(
            PulseReferenceCfg,
            {
                "sweep": Range.between(1.0, 2.0, expts=3),
                "linked": Slot("Q1.pulses.seed"),
                "inline": Slot("Q1.pulses.seed", copy=True, set={"gain": 0.75}),
            },
        )


class WaveformReferenceAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[PulseReferenceCfg]:
        del ctx
        return Form(
            PulseReferenceCfg,
            {
                "sweep": Range.between(1.0, 2.0, expts=3),
                "linked": Slot("Q1.pulses.carrier", copy=True),
                "inline": Slot("Q1.pulses.carrier", copy=True),
            },
        )


@pytest.fixture(params=["module", "waveform"])
def reference_library(request):
    kind = request.param
    if kind == "module":
        path = "Q1.pulses.seed"
        module = PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
        )
        change = LibraryChange(kind, "put", path, module=module)
        return kind, change, "gain", path, PulseReferenceAdapter
    path = "Q1.waveforms.seed"
    change = LibraryChange(kind, "put", path, waveform=ConstWaveformCfg(length=0.1))
    return kind, change, "length", path, WaveformReferenceAdapter


@pytest.mark.parametrize("action", ["rename", "del"])
def test_library_mutation_refreshes_ref_and_explicit_inline_tab_cfg(
    qapp,
    tmp_path,
    reference_library,
    action,
    point_models,
    point_registry_guard,
):
    kind, put, field, path, adapter_type = reference_library
    fx = Fixture(project_root=str(tmp_path), active_label="ctx001")
    fx.ctrl.context_control.add_component("Q1", "fake/drive/a", {})
    fx.ctrl.apply_library_change(fx.ctrl.read_context(), put)
    if kind == "waveform":
        fx.ctrl.context_control.set_module(
            fx.ctrl.read_context().target,
            "Q1.pulses.carrier",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.5, waveform=path),
        )
    fx.registry.register("references", adapter_type)
    tab_id = fx.ctrl.new_tab("references")
    owner = fx.ctrl.cfg_resources.lookup(tab_id)
    if kind == "waveform":
        owner.write(
            {
                ("inline", "module", "waveform"): {
                    "kind": "inline",
                    "module": {"style": "const", "length": 0.75},
                }
            }
        )
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        before = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )
        linked = before["input"]["linked"]
        if kind == "module":
            assert linked == {"kind": "ref", "name": path}
        else:
            assert linked["module"]["waveform"] == {"kind": "ref", "name": path}
        inline_path = (
            ("inline", "module", field)
            if kind == "module"
            else ("inline", "module", "waveform", "module", field)
        )
        leaves = {tuple(item["path"]): item["value"] for item in before["values"]}
        assert leaves[inline_path] == 0.75
        fx.ctrl.apply_library_change(
            fx.ctrl.read_context(),
            LibraryChange(
                kind,
                "rename" if action == "rename" else "delete",
                path,
                new_path=path.replace("seed", "moved") if action == "rename" else None,
            ),
        )
        after = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )
        assert after == before
        captured = owner.env.library
        if captured is None:
            raise TypeError("Expected captured library")
        keys = captured.module_keys() if kind == "module" else captured.waveform_keys()
        assert path in keys
        current = fx.ctrl.read_context().library
        current_keys = (
            current.module_keys() if kind == "module" else current.waveform_keys()
        )
        assert current_keys == (
            (path.replace("seed", "moved"),) if action == "rename" else ()
        )
        reset = invoke(
            "rpc_call", {"method": "tab.reset_cfg", "params": {"tab_id": tab_id}}
        )
        assert reset != before
        captured = owner.env.library
        if captured is None:
            raise TypeError("Expected Reset library")
        reset_keys = (
            captured.module_keys() if kind == "module" else captured.waveform_keys()
        )
        assert reset_keys == current_keys
        after["input"]["inline"] = None
        again = invoke(
            "rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}}
        )
        assert again == reset
    finally:
        bridge.disconnect()
        fx.stop()
