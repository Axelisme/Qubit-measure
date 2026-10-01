"""GUI SNR control enters the captured typed acquisition config."""

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters.onetone.power_dep import (
    OneTonePowerDepAdapter,
)
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.experiment.v2.onetone._support import make_power_cfg


@pytest.mark.parametrize(
    ("control", "expected"), [(2.5, 2.5), (0.0, None), (-1.0, None), (None, None)]
)
def test_run_captures_typed_snr_cfg_and_the_same_source_context(
    monkeypatch: pytest.MonkeyPatch, control: float | None, expected: float | None
) -> None:
    cfg = make_power_cfg()
    raw_cfg = cfg.to_dict()
    raw_cfg["earlystop_snr"] = control
    data = PowerDepResult(np.array([0.1]), np.array([6000.0]), np.array([[1 + 2j]]))
    observed: list[tuple[PowerDepCfg, RunContext]] = []

    def run(
        _core: PowerDepExp, config: PowerDepCfg, *, context: RunContext
    ) -> PowerDepResult:
        observed.append((config, context))
        return data

    monkeypatch.setattr(PowerDepExp, "run", run)
    soc, soccfg = make_mock_soc()
    req = RunRequest(soc=soc, soccfg=soccfg, device_snapshot={})
    plots = Plots(NonPresentingHost())
    source = OneTonePowerDepAdapter().run(
        req,
        raw_cfg,
        context=RunContext(
            req.soc, req.soccfg, plots, devices={}, cancel_signal=StopSignal()
        ),
    )
    config, context = observed[0]
    assert config.earlystop_snr == expected
    assert source.cfg == config and source.result is data
    assert context.soc is soc and context.soccfg is soccfg and context.plots is plots
    config.earlystop_snr = 99.0
    raw_cfg["earlystop_snr"] = 100.0
    assert source.cfg is not None and source.cfg.earlystop_snr == expected
    plots.finish()
    plots.release()
