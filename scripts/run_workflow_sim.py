"""Run one current workflow callback on a simulated board, without saving.

Provide --sim with current SimParams JSON and --cfg with the selected callback's
current native cfg JSON (dev must be None; this host has no device adapter).
Completed zero-detune echo also reports its native decay fit in microseconds.
Other callbacks report acquisition metadata only. This is not a full workflow.
No instruments are connected and no old-format alias/fallback is supported.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from qick import QickConfig
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import Run
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.sim import SimParams

from zcu_lab.v2.twotone.freq.core import FreqCfg
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg
from zcu_lab.workflows.acquire import (
    FrequencyAcquisition,
    RabiAcquisition,
    ReadoutAcquisition,
    T1Acquisition,
    T2EchoAcquisition,
    run_len_rabi,
    run_qubit_freq,
    run_ro_landscape,
    run_t1,
    run_t2echo,
)
from zcu_lab.workflows.analysis import analyze_t2echo


class _Arguments(argparse.Namespace):
    experiment: Literal["qubit_freq", "len_rabi", "ro_landscape", "t1", "t2echo"]
    cfg: Path
    sim: Path


def _execute[
    C,
    R: FrequencyAcquisition
    | RabiAcquisition
    | ReadoutAcquisition
    | T1Acquisition
    | T2EchoAcquisition,
](experiment: Callable[[Run[C]], R], cfg: C, soc: object, soccfg: QickConfig) -> None:
    run = Run(cfg, StopSignal(), soc=soc, soccfg=soccfg)
    settings = run.device_setup
    if settings is not None:
        run.devices.setup(settings, run.cancel_signal)
    result = experiment(run)
    run.cancel_signal.raise_if_error()
    fit_metrics: dict[str, float] = {}
    if isinstance(result, T2EchoAcquisition) and run.outcome.status == "completed":
        fit = analyze_t2echo(result)
        fit_metrics = {
            "lifetime_us": fit.lifetime_us,
            "lifetime_error_us": fit.error_us,
        }
    print(
        json.dumps(
            {
                "run_id": result.evidence.run_id,
                "completion": result.evidence.completion,
                "started_at": result.evidence.started_at.isoformat(),
                "finished_at": result.evidence.finished_at.isoformat(),
                "shape": list(result.signals.shape),
                **fit_metrics,
            }
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Acquire from current cfg/simulator JSON and report metadata or echo fit.

    argv uses the process arguments when None. Completed echo also reports
    lifetime_us/lifetime_error_us; other callbacks or stops omit those fields.
    File, validation, acquisition and analysis errors propagate. No output
    files or hardware connections; argparse rejects unknown experiments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", choices=("qubit_freq", "len_rabi", "ro_landscape", "t1", "t2echo")
    )
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    args = parser.parse_args(argv, namespace=_Arguments())
    soc, soccfg = make_mock_soc(
        sim=SimParams.model_validate_json(args.sim.read_text(encoding="utf-8"))
    )
    cfg_json = args.cfg.read_text(encoding="utf-8")
    if args.experiment == "qubit_freq":
        _execute(run_qubit_freq, FreqCfg.model_validate_json(cfg_json), soc, soccfg)
    elif args.experiment == "len_rabi":
        _execute(run_len_rabi, LenRabiCfg.model_validate_json(cfg_json), soc, soccfg)
    elif args.experiment == "ro_landscape":
        _execute(
            run_ro_landscape, FreqGainCfg.model_validate_json(cfg_json), soc, soccfg
        )
    elif args.experiment == "t1":
        _execute(run_t1, T1Cfg.model_validate_json(cfg_json), soc, soccfg)
    else:
        _execute(run_t2echo, T2EchoCfg.model_validate_json(cfg_json), soc, soccfg)


if __name__ == "__main__":
    main()
