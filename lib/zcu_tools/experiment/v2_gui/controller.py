from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
from numpy import ndarray

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.v2.fake import FakeExp
from zcu_tools.experiment.v2.runner import task_manager
from zcu_tools.meta_tool import ExperimentManager, MetaDict, ModuleLibrary
from zcu_tools.progress_bar import progress_backend_scope, qt_progress_callbacks_scope
from zcu_tools.utils import format_dict

from .experiment_group import (
    AnalyzeResult,
    ExperimentGroup,
    ExperimentPort,
    FakeRunResult,
    RunRequest,
)
from .state import BufferDescriptor, BufferKind, GuiState

class ExperimentAdapter:
    """Bridge `FakeExp` to `ExperimentPort` used by GUI."""

    def __init__(self, exp: FakeExp, scope_name: str = "v2_gui_run") -> None:
        self.exp = exp
        self.scope_name = scope_name

    def run(
        self,
        soc: Any,
        soccfg: Any,
        cfg: Dict[str, Any],
        on_progress=None,
        should_cancel=None,
    ) -> FakeRunResult:
        total_ref = {"total": 0}
        progress_seen = {"value": False}

        def _on_task_pbar_start(total: Optional[int], desc: str) -> None:
            del desc
            total_ref["total"] = int(total or 0)

        def _on_task_pbar_update(n: int) -> None:
            if on_progress is None:
                return
            progress_seen["value"] = True
            total = total_ref["total"] if total_ref["total"] > 0 else max(n, 1)
            on_progress(int(n), int(total))

        with task_manager.scope(self.scope_name) as scope:
            if should_cancel is not None and should_cancel():
                task_manager.cancel_current()
            with progress_backend_scope("qt"):
                with qt_progress_callbacks_scope(
                    on_start=_on_task_pbar_start,
                    on_update_to=_on_task_pbar_update,
                ):
                    freqs, signals = self.exp.run()
        y = self._signals_to_real(signals)
        if on_progress is not None and not progress_seen["value"]:
            on_progress(len(y), len(y))
        return FakeRunResult(
            x=[float(v) for v in freqs],
            y=[float(v) for v in y],
            partial=scope.stop_event.is_set(),
        )

    def analyze(self, result: Optional[FakeRunResult] = None) -> Dict[str, Any]:
        if result is None:
            result = self._last_result()
        if result is None or not result.x:
            raise RuntimeError("No run result available to analyze")
        peak_idx = max(range(len(result.y)), key=lambda i: result.y[i])
        return {
            "peak_freq_mhz": result.x[peak_idx],
            "peak_amp": result.y[peak_idx],
            "points": len(result.x),
            "partial": result.partial,
        }

    def save_run(self, filepath: Path, cfg: Dict[str, Any], result: FakeRunResult) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": cfg,
            "result": {"x": result.x, "y": result.y, "partial": result.partial},
        }
        filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return filepath

    def save_analysis_figure(self, filepath: Path, result: FakeRunResult) -> Path:
        source = self.exp.last_result
        if source is None:
            source = (self._to_float_array(result.x), self._to_complex_array(result.y))
        fig = self.exp.analyze(source)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=120)
        return filepath

    def _last_result(self) -> Optional[FakeRunResult]:
        if self.exp.last_result is None:
            return None
        freqs, signals = self.exp.last_result
        y = self._signals_to_real(signals)
        return FakeRunResult(
            x=[float(v) for v in freqs],
            y=[float(v) for v in y],
            partial=False,
        )

    def _signals_to_real(self, signals: ndarray) -> ndarray:
        return np.abs(signals)

    def _to_float_array(self, values: list[float]) -> ndarray:
        import numpy as np

        return np.asarray(values, dtype=np.float64)

    def _to_complex_array(self, values: list[float]) -> ndarray:
        import numpy as np

        return np.asarray(values, dtype=np.complex128)


class GuiController:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.state = GuiState()
        self.group_models: Dict[str, ExperimentGroup] = {}

        self.soc = None
        self.soccfg = None
        self._ensure_fake_device_registered()
        self.exp_manager = ExperimentManager(
            self.project_root / "result" / "mock_gui" / "exps"
        )
        self.meta_dict: MetaDict
        self.module_library: ModuleLibrary
        self._initialize_context()

    def _ensure_fake_device_registered(self) -> None:
        devices = GlobalDeviceManager.get_all_devices()
        if devices:
            return
        fake1 = FakeDevice(address="FAKE::DEVICE1")
        fake1.set_field("value", 0.0)
        GlobalDeviceManager.register_device("fakedevice1", fake1)

        fake2 = FakeDevice(address="FAKE::DEVICE2")
        fake2.set_field("value", 1e-3)
        GlobalDeviceManager.register_device("fakedevice2", fake2)

    def _initialize_context(self) -> None:
        labels = self.exp_manager.list_contexts()
        if not labels:
            default_label = "20260411"
            self.exp_manager.new_flux(label=default_label)
            labels = self.exp_manager.list_contexts()
        self.set_context(labels[0])

    def bootstrap_groups(self) -> None:
        group_id = "exp:onetone_mock"
        self.create_experiment_group(
            group_id=group_id,
            title="OneTone Mock",
            experiment=ExperimentAdapter(FakeExp(), scope_name=group_id),
        )
        self.state.current_group_id = group_id

    def create_experiment_group(
        self,
        group_id: str,
        title: str,
        experiment: ExperimentPort,
        default_cfg: Optional[RunRequest] = None,
    ) -> ExperimentGroup:
        model = ExperimentGroup.create(
            group_id=group_id,
            title=title,
            experiment=experiment,
            soc=self.soc,
            soccfg=self.soccfg,
            default_cfg=default_cfg,
        )
        group, buffers = model.to_descriptors()
        self.state.groups[group.group_id] = group
        for buffer in buffers:
            self.state.buffers[buffer.buffer_id] = buffer
        self.group_models[group.group_id] = model
        return model

    def list_contexts(self) -> List[str]:
        return self.exp_manager.list_contexts()

    def set_context(self, label: str) -> None:
        self.module_library, self.meta_dict = self.exp_manager.use_flux(label)

    def create_context(
        self, label: str, clone_from: Optional[str] = None
    ) -> tuple[ModuleLibrary, MetaDict]:
        if clone_from:
            ml, md = self.exp_manager.new_flux(label=label, clone_from=clone_from)
        else:
            ml, md = self.exp_manager.new_flux(label=label)
        self.module_library, self.meta_dict = ml, md
        return ml, md

    def get_supported_label_devices(self) -> list[dict[str, Any]]:
        supported: list[dict[str, Any]] = []
        for name, info in GlobalDeviceManager.get_all_info().items():
            if info.get("type") != "FakeDevice":
                continue
            value = info.get("value")
            if isinstance(value, (int, float)):
                supported.append({"name": name, "value": float(value)})
        return supported

    def suggest_auto_label(self, device_name: Optional[str] = None) -> str:
        devices = self.get_supported_label_devices()
        if not devices:
            return self.exp_manager.auto_label()
        selected = devices[0]
        if device_name is not None:
            for item in devices:
                if item["name"] == device_name:
                    selected = item
                    break
        return self.exp_manager.auto_label(selected["value"])

    def active_dir(self) -> Path:
        return self.exp_manager.flux_dir

    def save_meta_dict(self) -> Path:
        self.meta_dict.sync()
        return self.active_dir() / "meta_info.json"

    def save_module_library(self) -> Path:
        self.module_library.sync()
        return self.active_dir() / "module_cfg.yaml"

    def open_file_buffer(self, path: Path) -> BufferDescriptor:
        normalized_path = str(path.resolve())
        for buffer in self.state.buffers.values():
            if buffer.payload.get("path") == normalized_path:
                group = self.state.groups.get(buffer.group_id)
                if group is not None and buffer.buffer_id in group.buffer_ids:
                    group.current_index = group.buffer_ids.index(buffer.buffer_id)
                self.state.current_group_id = buffer.group_id
                return buffer

        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            kind = BufferKind.FILE_IMAGE
        elif ext in {".csv"}:
            kind = BufferKind.FILE_CSV
        else:
            kind = BufferKind.FILE_TEXT

        group_id = f"file:{normalized_path}"
        self.state.ensure_group(group_id, f"File:{path.name}")

        buffer_id = f"buf:file:{abs(hash(normalized_path))}"
        buffer = BufferDescriptor(
            buffer_id=buffer_id,
            group_id=group_id,
            title=path.name,
            kind=kind,
            payload={"path": normalized_path},
        )
        self.state.add_buffer(buffer)
        self.state.current_group_id = group_id
        return buffer

    def _active_group_model(self) -> ExperimentGroup:
        group = self.state.current_group()
        if group is not None:
            model = self.group_models.get(group.group_id)
            if model is not None:
                return model
        if self.group_models:
            return next(iter(self.group_models.values()))
        raise RuntimeError("No experiment group available")

    @property
    def exp_cfg(self) -> Dict[str, Any]:
        return self._active_group_model().exp_cfg

    @property
    def last_run_result(self) -> Optional[FakeRunResult]:
        return self._active_group_model().last_result

    @property
    def last_analysis(self) -> Optional[AnalyzeResult]:
        return self._active_group_model().last_analysis

    def get_exp_cfg_text(self) -> str:
        return json.dumps(self._active_group_model().exp_cfg, indent=2)

    def update_exp_cfg_from_text(self, text: str) -> None:
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("exp_cfg must be a JSON object")
        model = self._active_group_model()
        model.exp_cfg = cfg
        run_buffer = self.state.buffers.get(model.run_buffer_id)
        if run_buffer is not None:
            run_buffer.payload["cfg"] = dict(cfg)

    def get_device_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for dev_name, info in GlobalDeviceManager.get_all_info().items():
            for field, value in info.items():
                rows.append({"device": dev_name, "field": field, "value": value})
        return rows

    def get_device_infos(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: dict(info) for name, info in GlobalDeviceManager.get_all_info().items()
        }

    def update_device_field(
        self, device: str, field: str, value: Any
    ) -> Dict[str, Any]:
        try:
            dev = GlobalDeviceManager.get_device(device)
        except Exception as exc:
            return {"ok": False, "message": str(exc), "value": None}
        old_info = dev.get_info()
        old_value = old_info.get(field)
        setter = cast(Optional[Any], getattr(dev, "set_field", None))
        try:
            if callable(setter):
                setter(field, value)
            else:
                raise RuntimeError(
                    f"Device '{device}' does not support GUI field updates for now."
                )
            new_value = dev.get_info().get(field)
            return {"ok": True, "message": "ok", "value": new_value}
        except Exception as exc:
            if callable(setter):
                try:
                    setter(field, old_value)
                except Exception:
                    pass
            return {"ok": False, "message": str(exc), "value": old_value}

    def get_meta_rows(self) -> List[Dict[str, Any]]:
        self.meta_dict.sync()
        return [{"key": k, "value": v} for k, v in self.meta_dict._data.items()]

    def set_meta_value(self, key: str, value: Any) -> None:
        setattr(self.meta_dict, key, value)

    def delete_meta_key(self, key: str) -> None:
        self.meta_dict.sync()
        if key in self.meta_dict._data:
            delattr(self.meta_dict, key)

    def replace_meta_dict(self, updated: Dict[str, Any]) -> None:
        self.meta_dict.sync()
        for key in list(self.meta_dict._data.keys()):
            delattr(self.meta_dict, key)
        for key, value in updated.items():
            setattr(self.meta_dict, key, value)

    def get_library_rows(self) -> List[Dict[str, Any]]:
        self.module_library.sync()
        rows: List[Dict[str, Any]] = []
        for name, cfg in self.module_library.modules.items():
            rows.append(
                {
                    "name": f"module:{name}",
                    "cfg": format_dict({"cfg": cfg}).get("cfg", {}),
                }
            )
        for name, cfg in self.module_library.waveforms.items():
            rows.append(
                {
                    "name": f"waveform:{name}",
                    "cfg": format_dict({"cfg": cfg}).get("cfg", {}),
                }
            )
        return rows

    def get_library_snapshot(self) -> Dict[str, Dict[str, Any]]:
        self.module_library.sync()
        return {
            "modules": format_dict(self.module_library.modules),
            "waveforms": format_dict(self.module_library.waveforms),
        }

    def set_library_item(self, name: str, cfg: Dict[str, Any]) -> None:
        if name.startswith("waveform:"):
            self.module_library.register_waveform(**{name.split(":", 1)[1]: cfg})
            return
        module_name = name.split(":", 1)[1] if name.startswith("module:") else name
        self.module_library.register_module(**{module_name: cfg})

    def replace_library_items(self, updated: Dict[str, Dict[str, Any]]) -> None:
        self.module_library.sync()
        self.module_library.waveforms = {}
        self.module_library.modules = {}
        self.module_library._dirty = True
        for name, cfg in updated.items():
            self.set_library_item(name, cfg)
        self.module_library.sync()

    def run_mock_experiment(self, on_progress, should_cancel) -> FakeRunResult:
        model = self._active_group_model()
        result = model.run(on_progress=on_progress, should_cancel=should_cancel)

        run_buffer = self.state.buffers.get(model.run_buffer_id)
        if run_buffer is not None:
            run_buffer.payload["cfg"] = dict(model.exp_cfg)
        return result

    def request_stop_active_run(self) -> bool:
        model = self._active_group_model()
        return task_manager.cancel_scope(model.group_id)

    def analyze_last_result(self) -> Dict[str, Any]:
        model = self._active_group_model()
        analysis = dict(model.analyze())
        figure_path = self.save_analysis_figure()
        analysis["figure_path"] = str(figure_path)
        if model.last_analysis is not None:
            model.last_analysis["figure_path"] = str(figure_path)

        analyze_buffer = self.state.buffers.get(model.analyze_buffer_id)
        if analyze_buffer is None:
            raise RuntimeError("Analyze buffer not found for active group")
        analyze_buffer.payload["analysis"] = dict(analysis)

        group = self.state.groups.get(model.group_id)
        if group is not None:
            group.current_index = group.buffer_ids.index(model.analyze_buffer_id)
        return analysis

    def save_run_payload(self) -> Path:
        model = self._active_group_model()
        dst = self.active_dir() / "mock_run_result.json"
        return model.save_run(dst)

    def save_analysis_figure(self) -> Path:
        model = self._active_group_model()
        dst = self.active_dir() / "mock_analysis.png"
        path = model.save_analysis_figure(dst)

        analyze_buffer = self.state.buffers.get(model.analyze_buffer_id)
        if analyze_buffer is not None:
            analysis = dict(analyze_buffer.payload.get("analysis", {}))
            analysis["figure_path"] = str(path)
            analyze_buffer.payload["analysis"] = analysis
        return path

    def apply_analysis_to_context(self) -> tuple[Path, Path]:
        if self.last_analysis is None:
            raise RuntimeError("No analysis result available")

        peak_raw = self.last_analysis.get("peak_freq_mhz")
        if peak_raw is None:
            raise RuntimeError("Analysis result missing 'peak_freq_mhz'")

        peak = float(peak_raw)
        self.set_meta_value("r_f", peak)

        self.module_library.sync()
        if "readout_rf" in self.module_library.modules:
            self.module_library.update_module("readout_rf", {"freq": peak})

        return self.save_meta_dict(), self.save_module_library()
