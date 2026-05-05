from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".tmp" / "matplotlib"))

import matplotlib
import numpy as np
import zcu_tools.experiment.v2 as ze
from qick import QickConfig
from zcu_tools.debug import debug_scope
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice
from zcu_tools.meta_tool import ExperimentManager, MetaDict, ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import make_mock_soc
from zcu_tools.program.v2.ir.pipeline import DEFAULT_PIPELINE_CONFIG, PipeLineConfig

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DEFAULT_CHIP = "purcell_tmon"
DEFAULT_QUBIT = "Q3"
DEFAULT_LABEL = "20260411"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "result" / "debug_opt_compare"


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    build_cfg: Callable[[ModuleLibrary, MetaDict], Any]
    run_experiment: Callable[[Any, Any, Any, Any], Any]


@dataclass(frozen=True)
class RunArtifacts:
    log_path: Path
    summary_path: Path


def parse_args(experiment_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Compare optimized vs non-optimized IR logs for {experiment_name}."
    )
    parser.add_argument("--chip", default=DEFAULT_CHIP)
    parser.add_argument("--qubit", default=DEFAULT_QUBIT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / experiment_name,
    )
    parser.add_argument(
        "--mock-gens",
        type=int,
        default=15,
        help="Number of mock generator channels for make_mock_soc().",
    )
    parser.add_argument(
        "--mock-readouts",
        type=int,
        default=1,
        help="Number of mock readout channels for make_mock_soc().",
    )
    return parser.parse_args()


def ensure_fake_device_registered() -> None:
    if "fake_device" not in GlobalDeviceManager.get_all_info():
        GlobalDeviceManager.register_device("fake_device", FakeDevice())


def load_context(chip: str, qubit: str, label: str) -> tuple[ModuleLibrary, MetaDict]:
    exp_dir = ROOT_DIR / "result" / chip / qubit / "exps"
    em = ExperimentManager(exp_dir)
    return em.use_flux(label=label, readonly=True)


def build_mock_soc(
    n_gens: int,
    n_readouts: int,
) -> tuple[Any, QickConfig]:
    soc = make_mock_soc(n_gens=n_gens, n_readouts=n_readouts)
    soccfg = QickConfig(soc.get_cfg())
    return soc, soccfg


def _config_to_dict(config: PipeLineConfig) -> dict[str, Any]:
    return asdict(config)


@contextmanager
def pipeline_config_scope(*, enable_opt: bool) -> Iterator[PipeLineConfig]:
    original = PipeLineConfig(**_config_to_dict(DEFAULT_PIPELINE_CONFIG))
    try:
        if enable_opt:
            DEFAULT_PIPELINE_CONFIG.disable_all_opt = False
            DEFAULT_PIPELINE_CONFIG.enable_unroll_loop = True
            DEFAULT_PIPELINE_CONFIG.enable_dead_write = True
            DEFAULT_PIPELINE_CONFIG.enable_dead_label = True
            DEFAULT_PIPELINE_CONFIG.enable_zero_delay_dce = True
            DEFAULT_PIPELINE_CONFIG.enable_timed_instruction_merge = True
        else:
            DEFAULT_PIPELINE_CONFIG.disable_all_opt = True
        yield PipeLineConfig(**_config_to_dict(DEFAULT_PIPELINE_CONFIG))
    finally:
        for key, value in _config_to_dict(original).items():
            setattr(DEFAULT_PIPELINE_CONFIG, key, value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _extract_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _build_diff_summary(
    *,
    experiment_name: str,
    opt_log_path: Path,
    noopt_log_path: Path,
    output_dir: Path,
) -> Path:
    opt_lines = _extract_lines(opt_log_path)
    noopt_lines = _extract_lines(noopt_log_path)
    diff_lines = list(
        difflib.unified_diff(
            noopt_lines,
            opt_lines,
            fromfile=noopt_log_path.name,
            tofile=opt_log_path.name,
            lineterm="",
        )
    )

    diff_path = output_dir / f"{experiment_name}-opt.diff"
    diff_path.write_text("\n".join(diff_lines) + ("\n" if diff_lines else ""))

    summary = {
        "experiment": experiment_name,
        "no_opt_log": str(noopt_log_path),
        "opt_log": str(opt_log_path),
        "diff_path": str(diff_path),
        "no_opt_lines": len(noopt_lines),
        "opt_lines": len(opt_lines),
        "diff_lines": len(diff_lines),
        "first_diff_preview": diff_lines[:40],
        "has_diff": bool(diff_lines),
    }
    summary_path = output_dir / f"{experiment_name}-summary.json"
    _write_json(summary_path, summary)
    return summary_path


def run_case(
    case: ExperimentCase, args: argparse.Namespace
) -> tuple[RunArtifacts, RunArtifacts]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_fake_device_registered()
    ml, md = load_context(args.chip, args.qubit, args.label)
    soc, soccfg = build_mock_soc(args.mock_gens, args.mock_readouts)
    cfg = case.build_cfg(ml, md)

    artifacts: list[RunArtifacts] = []
    for mode_name, enable_opt in (("noopt", False), ("opt", True)):
        log_path = output_dir / f"{case.name}-{mode_name}.log"
        summary_path = output_dir / f"{case.name}-{mode_name}-pipeline.json"

        np.random.seed(0)
        with pipeline_config_scope(enable_opt=enable_opt) as pipeline_cfg:
            _write_json(
                summary_path,
                {
                    "experiment": case.name,
                    "mode": mode_name,
                    "pipeline_config": _config_to_dict(pipeline_cfg),
                },
            )
            with log_path.open("w") as stream:
                with debug_scope("zcu_tools.program.v2", stream=stream):
                    case.run_experiment(soc, soccfg, cfg, ml)

        artifacts.append(RunArtifacts(log_path=log_path, summary_path=summary_path))

    diff_summary_path = _build_diff_summary(
        experiment_name=case.name,
        opt_log_path=artifacts[1].log_path,
        noopt_log_path=artifacts[0].log_path,
        output_dir=output_dir,
    )
    logger.warning("Diff summary written to %s", diff_summary_path)
    return artifacts[0], artifacts[1]


def build_allxy_cfg(ml: ModuleLibrary, _md: MetaDict) -> ze.twotone.AllXYCfg:
    exp_cfg = {
        "modules": {
            "reset": "reset_bath",
            "X180_pulse": "pi_amp",
            "X90_pulse": "pi2_amp",
            "readout": "readout_dpm",
        },
        "relax_delay": 10.5,
    }
    return ml.make_cfg(exp_cfg, ze.twotone.AllXYCfg, reps=100, rounds=10)


def run_allxy(soc: Any, soccfg: QickConfig, cfg: Any, _ml: ModuleLibrary) -> Any:
    return ze.twotone.AllXY_Exp().run(soc, soccfg, cfg)


def build_cpmg_cfg(ml: ModuleLibrary, md: MetaDict) -> ze.twotone.time_domain.CPMG_Cfg:
    times = list(range(10, 0, -5))
    exp_cfg = {
        "modules": {
            "pi_pulse": "pi_len",
            "pi2_pulse": ml.get_module("pi2_len", {"phase": 90}),
            "readout": "readout_dpm",
        },
        "sweep": {"times": times},
        "length_expts": 11,
        "length_range": [(0.1 * t, 5.0 * t) for t in times],
        "relax_delay": 0.05 * md.t1,
    }
    return ml.make_cfg(exp_cfg, ze.twotone.time_domain.CPMG_Cfg, reps=100, rounds=10)


def run_cpmg(soc: Any, soccfg: QickConfig, cfg: Any, _ml: ModuleLibrary) -> Any:
    return ze.twotone.time_domain.CPMG_Exp().run(
        soc,
        soccfg,
        cfg,
        detune_ratio=0.1,
        earlystop_snr=10.0,
    )


def build_rb_cfg(ml: ModuleLibrary, _md: MetaDict) -> ze.twotone.RBCfg:
    exp_cfg = {
        "modules": {
            "reset": "reset_bath",
            "X90_pulse": "pi2_amp",
            "X180_pulse": "pi_amp",
            "readout": "readout_dpm",
        },
        "sweep": make_sweep(0, 500, 3, force_int=True),
        "seed": 0,
        "n_seeds": 5,
        "relax_delay": 0.5,
    }
    return ml.make_cfg(exp_cfg, ze.twotone.RBCfg, reps=100, rounds=1)


def run_rb(soc: Any, soccfg: QickConfig, cfg: Any, _ml: ModuleLibrary) -> Any:
    return ze.twotone.RB_Exp().run(soc, soccfg, cfg)
