"""Recording GUI collaborator for public recipe contract tests."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, ClassVar

from zcu_tools.cfg_model import (
    ConfigBase,
    DirectReadoutCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseReadoutSpec,
    Range,
    ResetCfg,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.remote.form_description import describe_form
from zcu_tools.gui.app.measure.remote.form_observation import form_writes, observe_form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv, FormPathError
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import (
    ComponentSchema,
    PointDocument,
    PointGeneral,
)

from ._support import make_client


class FormulaComponent(ComponentSchema):
    low: float | None = None
    high: float | None = None
    threshold: float | None = None
    freq: float | None = None
    kappa: float | None = None
    flux_half: float | None = None
    flux_int: float | None = None


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


class RecipeModules(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    readout: PulseReadoutCfg


class RecipeDevices(ConfigBase):
    flux_dev: str = "coil"


class RecipeControlsBase(ConfigBase):
    rounds: int = 3
    fixed: int = 2
    modules: RecipeModules
    dev: RecipeDevices = RecipeDevices()
    flux_unit: str = "A"


class RecipeControls(RecipeControlsBase):
    range: SweepCfg
    sweep: SweepCfg


class ExecutionModules(ConfigBase):
    """Nullable module collaborators, not native reset or init pulse execution."""

    reset: DirectReadoutCfg | None = None
    init_pulse: DirectReadoutCfg | None = None
    readout: PulseReadoutCfg


class ExecutionControls(ConfigBase):
    """Execution collaborator values; reset is only a disabled-null module seam."""

    rounds: int = 3
    fixed: int = 2
    modules: ExecutionModules
    dev: RecipeDevices = RecipeDevices()
    flux_unit: str = "A"
    range: SweepCfg
    sweep: SweepCfg


_READOUT_SPEC = PulseReadoutSpec.model_validate(
    {
        "type": "readout/pulse",
        "pulse_cfg": {
            "type": "pulse",
            "ch": 0,
            "nqz": 1,
            "freq": 5000.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 0.1},
        },
        "ro_cfg": {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 5000.0,
            "ro_length": 2.0,
            "trig_offset": 0.1,
        },
    }
)


READOUT_TEMPLATE = TemplateDef(
    PulseReadoutSpec,
    "Readout",
    {
        "pulse_cfg.type": "pulse",
        "pulse_cfg.ch": 0,
        "pulse_cfg.nqz": 1,
        "pulse_cfg.freq": 5000.0,
        "pulse_cfg.gain": 0.2,
        "pulse_cfg.waveform.style": "const",
        "pulse_cfg.waveform.length": 0.1,
        "ro_cfg.ro_ch": 0,
        "ro_cfg.ro_freq": 5000.0,
        "ro_cfg.ro_length": 2.0,
        "ro_cfg.trig_offset": 0.1,
    },
)


class LookbackGui:
    """GUI collaborator with distinct Run, save and analysis wire identities."""

    cfg_cls: ClassVar[type[ConfigBase]] = RecipeControls

    def __init__(self):
        self.ran = False
        self.raw_saved = False
        self.md: dict[str, Any] = {}
        self.library: dict[str, Any] = {}
        self._owner: TabFormResource | None = None
        self.publication: dict[str, Any] = {}
        self.fixed_values: dict[str, object] = {}

    @property
    def owner(self) -> TabFormResource:
        if self._owner is None:
            raise RuntimeError("Reset before accessing Form")
        return self._owner

    @owner.setter
    def owner(self, resource: TabFormResource) -> None:
        self._owner = resource

    def capture_parameters(self, aliases=None):
        resonator = FormulaComponent.model_validate(
            {
                "kind": "resonator",
                **{
                    name: self.md[f"$resonator.{name}"]
                    for name in ("freq", "kappa")
                    if f"$resonator.{name}" in self.md
                },
            }
        )
        qubit = FormulaComponent.model_validate(
            {
                "kind": "qubit/transmon",
                **{
                    name: self.md[f"$qubit.{name}"]
                    for name in ("freq", "kappa", "flux_half", "flux_int")
                    if f"$qubit.{name}" in self.md
                },
            }
        )
        custom = FormulaComponent.model_validate(
            {
                "kind": "qubit/transmon",
                **{
                    name: self.md[f"$custom.{name}"]
                    for name in ("low", "high", "threshold")
                    if f"$custom.{name}" in self.md
                    and isinstance(self.md[f"$custom.{name}"], (int, float))
                },
            }
        )
        parameters = ParameterContext(
            point=PointDocument.model_construct(
                format="zcu.parameter-container",
                format_version="2.0",
                general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
                components={
                    "R1": resonator,
                    "R2": resonator,
                    "Q1": qubit,
                    "Q2": qubit,
                    "C1": custom,
                },
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="2.0",
                    components={
                        "R1": ComponentLibrary(readout={"main": _READOUT_SPEC}),
                        "R2": ComponentLibrary(readout={"main": _READOUT_SPEC}),
                    },
                )
            ),
            target=SavedPoint(entry_id="recipe-entry", point="working"),
            roles=None,
        )
        table = AliasTable(
            (
                AliasRow("resonator", "R1"),
                AliasRow("qubit", "Q1"),
                AliasRow("custom", "C1"),
            )
        )
        for alias, component in (aliases or {}).items():
            table = table.with_override(alias, component)
        return parameters.bind_aliases(table.rows, {})

    def make_form(self):
        form = Form(
            self.cfg_cls,
            {
                "range": Range.between(1.0, 2.0, expts=11),
                "sweep": Range.between(1.0, 2.0, expts=51),
                "modules": {
                    "reset": Slot("$qubit.module.reset"),
                    "init_pulse": Slot("$qubit.module.init_pulse"),
                    "readout": Template("readout"),
                },
            },
        )
        form.bind_templates({"readout": READOUT_TEMPLATE}, adapter="lookback")
        return form

    def reset_form(self, aliases=None):
        self.fixed_values = deepcopy(self.md)

        def sources(overrides):
            parameters = self.capture_parameters(overrides)
            return TabFormSources(
                FormEnv(parameters=parameters, library=parameters.library)
            )

        if self._owner is None:
            self.owner = TabFormResource(
                lambda _ctx: self.make_form(),
                capture=sources,
                mutation_allowed=lambda: True,
            )
        self.owner.reset(aliases=aliases)
        self.publication = dict(observe_form(self.owner))

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            "context.snapshot": {
                "label": "sample",
                "md": self.md,
                "ml": {"modules": {}, "waveforms": {}},
            },
            "tab.new": {"tab_id": "t"},
            "soc.info": {"connected": True, "cfg": {}},
            "device.list": {"devices": [{"name": "bias"}]},
            "device.snapshot": {"snapshot": {"name": "bias", "info": {"value": 0.0}}},
            "tab.snapshot": {
                "tabs": [
                    {
                        "tab_id": "t",
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                        },
                        "result_state": {
                            "available": self.ran,
                            "revision": 1,
                            "source_operation_id": 71 if self.ran else None,
                        },
                    }
                ]
            },
        }

    def _edit(self, params: dict[str, object]) -> None:
        self.owner.write(form_writes(params["edits"]))
        self.publication = dict(observe_form(self.owner))

    def _describe(self, path: tuple[str, ...]) -> dict[str, Any]:
        try:
            return dict(describe_form(self.owner, path))
        except FormPathError as error:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                str(error),
                reason="form_path",
                data={"form_path": list(error.path)},
            ) from error

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        observations = self._observations()
        if method == "tab.describe_cfg":
            observations[method] = self._describe(tuple(params["path"]))
        if method == "tab.parameter_read":
            observations[method] = {
                "values": {
                    path: deepcopy(self.fixed_values.get(path))
                    for path in params["paths"]
                },
                "modules": {
                    reference: reference if reference in self.library else None
                    for reference in params["module_refs"]
                },
            }
        if method in observations:
            expected = {
                "tab.new": {"adapter_name": "lookback"},
                "soc.info": {"include_cfg": True},
                "device.snapshot": {"name": "bias"},
            }
            if method in expected:
                assert params == expected[method]
            return deepcopy(observations[method])
        if method in ("tab.get_cfg", "tab.edit_cfg", "tab.reset_cfg"):
            if method == "tab.reset_cfg":
                self.reset_form(params.get("aliases"))
            elif method == "tab.edit_cfg":
                self._edit(params)
            return deepcopy(self.publication)
        if method == "tab.run_start":
            assert params == {
                "tab_id": "t",
            }
            assert not self.ran
            self.ran = True
            return {"operation_id": 71}
        if method == "tab.save_data":
            assert params == {"tab_id": "t", "run_operation_id": 71}
            self.raw_saved = True
            return {
                "run_id": "run-a",
                "artifacts": {
                    "raw": {
                        "data": {
                            "status": "Saved",
                            "members": {
                                "data": [
                                    {
                                        "format": "data_h5",
                                        "path": "/runs/run-a/data.h5",
                                        "status": "saved",
                                        "error": None,
                                    },
                                    {
                                        "format": "labber",
                                        "path": "/labber/raw.hdf5",
                                        "status": "saved",
                                        "error": None,
                                    },
                                ]
                            },
                        }
                    }
                },
            }
        if method == "operation.await":
            assert 0 < params["timeout"] <= 0.25
            assert params["operation_id"] in (71, 82, 93)
            if params["operation_id"] == 82:
                self.raw_saved = True
            return {"reason": "completed", "status": "finished"}
        return self._analysis(method, params)

    def _analysis(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.analyze":
            assert self.raw_saved
            assert params == {"tab_id": "t", "updates": {}, "run_operation_id": 71}
            return {
                "operation_id": 93,
                "interactive": False,
                "params": {"threshold": 0.5},
                "invalidated_on_success": [],
            }
        if method == "tab.get_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 93}
            return {
                "summary": {"offset": 0.24},
                "invalid": [],
                "params": {"threshold": 0.5},
                "operation_state": {"analysis_state": {"figure_names": ["trace"]}},
            }
        if method == "tab.save_artifacts":
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": "analysis", "name": "results"}],
                "operation_id": 93,
            }
            return {
                "run_id": "run-a",
                "artifacts": {
                    "analysis": {
                        "results": {
                            "status": "Saved",
                            "members": {
                                "analysis:primary": [
                                    {
                                        "format": "json",
                                        "path": "/runs/run-a/analysis/results.json",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                                "figure:trace": [
                                    {
                                        "format": "png",
                                        "path": "/runs/run-a/analysis/trace.png",
                                        "status": "saved",
                                        "error": None,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
        if method == "tab.get_figure":
            assert params["operation_id"] == 93
            return {"png_b64": base64.b64encode(PNG).decode()}
        if method == "tab.writeback_preview":
            assert params == {
                "tab_id": "t",
                "subtab_id": "analysis",
                "operation_id": 93,
            }
            return {
                "has_draft": True,
                "items": [
                    {
                        "id": "param-1",
                        "kind": "parameter",
                        "target_name": "trigger_offset",
                        "selected": False,
                        "proposed": 0.24,
                        "current": 0.1,
                    }
                ],
                "destination_context": {"active_label": "sample"},
            }
        raise AssertionError(method)


class ExecutionGui(LookbackGui):
    """Record execution phases without unrelated native reset family schemas."""

    cfg_cls: ClassVar[type[ConfigBase]] = ExecutionControls
