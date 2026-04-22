from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from zcu_tools.meta_tool import ExperimentManager, MetaDict, ModuleLibrary
from zcu_tools.utils import format_dict

from .experiment_group import AnalyzeResult, ExperimentGroup, RunRequest
from .fake_backend import FakeExperiment, FakeRunResult, FakeSOC
from .state import BufferDescriptor, BufferKind, GuiState


class FakeSOCCfg:
    def __init__(self) -> None:
        self.sample_rates = {"dac": 9.8304e9, "adc": 4.9152e9}


class FakeDevice:
    def __init__(self, name: str, info: Dict[str, Any]) -> None:
        self.name = name
        self.info = info

    def set_field(self, field: str, value: Any) -> None:
        if field not in self.info:
            raise KeyError(f"Unknown field: {field}")
        if field == "power_dBm" and not (-120.0 <= float(value) <= 25.0):
            raise ValueError("power_dBm out of range")
        if field == "freq_Hz" and not (1e6 <= float(value) <= 20e9):
            raise ValueError("freq_Hz out of range")
        self.info[field] = value


class FakeDeviceManager:
    def __init__(self) -> None:
        self._devices: Dict[str, FakeDevice] = {
            "jpa_sgs": FakeDevice(
                name="jpa_sgs",
                info={
                    "type": "RohdeSchwarzSGS100A",
                    "address": "TCPIP0::192.168.10.89::inst0::INSTR",
                    "output": "on",
                    "freq_Hz": 11_800_000_000.0,
                    "power_dBm": -15.0,
                },
            ),
            "flux_yoko": FakeDevice(
                name="flux_yoko",
                info={
                    "type": "YOKOGS200",
                    "address": "USB0::FAKE::YOKO::INSTR",
                    "mode": "current",
                    "value": 0.0,
                    "output": "on",
                },
            ),
        }

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(dev.info) for name, dev in self._devices.items()}

    def update_field(
        self, device_name: str, field: str, value: Any
    ) -> tuple[bool, str, Any]:
        if device_name not in self._devices:
            return False, f"Device not found: {device_name}", None
        dev = self._devices[device_name]
        old_value = dev.info.get(field)
        try:
            dev.set_field(field, value)
        except Exception as exc:  # rollback only this field
            dev.info[field] = old_value
            return False, str(exc), old_value
        return True, "ok", dev.info[field]


class FakeStorage:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.base = project_root / "result" / "mock_gui"
        self.contexts = ["20260411", "20260412", "20260413"]
        self.active_label = self.contexts[0]
        self._ensure_demo_files()

    @property
    def active_dir(self) -> Path:
        return self.base / "exps" / self.active_label

    def list_contexts(self) -> list[str]:
        return list(self.contexts)

    def set_active_context(self, label: str) -> None:
        if label not in self.contexts:
            self.contexts.append(label)
        self.active_label = label
        self._ensure_context_files(self.active_dir)

    def _ensure_demo_files(self) -> None:
        for label in self.contexts:
            self._ensure_context_files(self.base / "exps" / label)

    def _ensure_context_files(self, context_dir: Path) -> None:
        context_dir.mkdir(parents=True, exist_ok=True)
        image_path = context_dir / "demo_plot.png"
        json_path = context_dir / "meta_info.json"
        yaml_path = context_dir / "module_cfg.yaml"
        csv_path = context_dir / "samples.csv"

        if not image_path.exists():
            import numpy as np
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 3))
            x = np.linspace(0, 1, 100)
            ax.plot(x, np.sin(6 * np.pi * x))
            ax.set_title("Mock Plot")
            fig.tight_layout()
            fig.savefig(image_path, dpi=110)
            plt.close(fig)

        if not json_path.exists():
            json_path.write_text(
                json.dumps({"r_f": 6812.3, "q_f": 5120.2}, indent=2), encoding="utf-8"
            )

        yaml_payload = {"waveforms": {}, "modules": {}}
        rewrite_yaml = True
        if yaml_path.exists():
            try:
                existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            if (
                isinstance(existing, dict)
                and isinstance(existing.get("waveforms"), dict)
                and isinstance(existing.get("modules"), dict)
            ):
                rewrite_yaml = False

        if rewrite_yaml:
            yaml_path.write_text(
                yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8"
            )

        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["calibrated mA", "Freq (MHz)", "T1 (us)"])
                writer.writerow([0.0, 5120.2, 34.1])
                writer.writerow([1.8, 5088.6, 29.3])


BackendName = Literal["mock", "real"]


class GuiController:
    def __init__(self, project_root: Path, backend: BackendName = "mock") -> None:
        self.project_root = project_root
        self.backend: BackendName = backend
        self.state = GuiState()
        self.group_models: Dict[str, ExperimentGroup] = {}

        self._setup_backend(backend)
        self.exp_manager = ExperimentManager(self.storage.base / "exps")
        self.meta_dict: MetaDict
        self.module_library: ModuleLibrary
        self._initialize_context()

    def _setup_backend(self, backend: BackendName) -> None:
        if backend == "mock":
            self.fake_soc = FakeSOC()
            self.fake_soccfg = FakeSOCCfg()
            self.storage = FakeStorage(self.project_root)
            self.device_manager = FakeDeviceManager()
            return

        raise NotImplementedError("backend='real' 尚未實作，請先使用 backend='mock'。")

    def _initialize_context(self) -> None:
        labels = self.exp_manager.list_contexts()
        if not labels:
            default_label = self.storage.contexts[0]
            self.exp_manager.new_flux(label=default_label)
            labels = self.exp_manager.list_contexts()
        self.set_context(labels[0])

    def bootstrap_groups(self) -> None:
        self.create_experiment_group(
            group_id="exp:onetone_mock",
            title="OneTone Mock",
            experiment=FakeExperiment(),
        )
        self.state.current_group_id = "exp:onetone_mock"

    def create_experiment_group(
        self,
        group_id: str,
        title: str,
        experiment: FakeExperiment,
        default_cfg: Optional[RunRequest] = None,
    ) -> ExperimentGroup:
        model = ExperimentGroup.create(
            group_id=group_id,
            title=title,
            experiment=experiment,
            soc=self.fake_soc,
            soccfg=self.fake_soccfg,
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
        self.storage.set_active_context(label)
        self.module_library, self.meta_dict = self.exp_manager.use_flux(label)

    def active_dir(self) -> Path:
        return self.exp_manager.flux_dir

    def save_meta_dict(self) -> Path:
        self.meta_dict.sync()
        return self.active_dir() / "meta_info.json"

    def save_module_library(self) -> Path:
        self.module_library.sync()
        return self.active_dir() / "module_cfg.yaml"

    def open_file_buffer(self, path: Path) -> BufferDescriptor:
        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            kind = BufferKind.FILE_IMAGE
        elif ext in {".csv"}:
            kind = BufferKind.FILE_CSV
        else:
            kind = BufferKind.FILE_TEXT

        group_id = f"file:{path.parent.name}"
        self.state.ensure_group(group_id, f"File:{path.parent.name}")

        buffer_id = f"buf:file:{uuid.uuid4().hex[:8]}"
        buffer = BufferDescriptor(
            buffer_id=buffer_id,
            group_id=group_id,
            title=path.name,
            kind=kind,
            payload={"path": str(path)},
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
        for dev_name, info in self.device_manager.get_all_info().items():
            for field, value in info.items():
                rows.append({"device": dev_name, "field": field, "value": value})
        return rows

    def update_device_field(
        self, device: str, field: str, value: Any
    ) -> Dict[str, Any]:
        ok, msg, actual = self.device_manager.update_field(device, field, value)
        return {"ok": ok, "message": msg, "value": actual}

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
            run_buffer.payload["log"] = [
                json.dumps({"points": len(result.x), "partial": result.partial})
            ]
        return result

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
        dst = self.storage.active_dir / "mock_run_result.json"
        return model.save_run(dst)

    def save_analysis_figure(self) -> Path:
        model = self._active_group_model()
        dst = self.storage.active_dir / "mock_analysis.png"
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
