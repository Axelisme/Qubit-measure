"""Run-stage contracts through caller-owned synthetic experiments."""

from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import image as mpl_image
from zcu_tools.experiment import probe
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.simulation import white_noise

from tests.experiment._probe_support import ProbeExperiment
from tests.experiment._saving_support import SavingCfg, SavingResult


def test_run_returns_record_and_in_memory_live_png_without_save_or_analysis(
    tmp_path: Path, monkeypatch,
):
    class RunOnly(ProbeExperiment):
        def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
            assert context.soc is environment.soc
            assert context.soccfg is environment.soccfg
            result = super().run(config, context=context)
            config.gain = 99
            context.plots.draw(
                "signal/trace", lambda figure: figure.set_size_inches(9, 3)
            )
            return result

    environment = white_noise(seed=17)
    cfg = SavingCfg(gain=2)
    monkeypatch.chdir(tmp_path)
    try:
        with matplotlib.rc_context({"savefig.bbox": "tight"}):
            outcome = probe.run_stage(RunOnly(), cfg, environment)
        assert outcome.record.cfg == cfg
        assert cfg.gain == 2
        np.testing.assert_array_equal(outcome.record.result.signal[:2], [2 + 4j, 6 - 8j])
        assert outcome.record.metadata["experiment"] == "test/saving"
        assert outcome.elapsed_seconds >= 0
        assert set(outcome.pngs) == {"live-signal%2Ftrace.png"}
        assert mpl_image.imread(BytesIO(outcome.pngs["live-signal%2Ftrace.png"])).shape[:2] == (300, 900)
        assert list(tmp_path.iterdir()) == []
        # The same caller environment remains usable and is not closed by run_stage.
        assert environment.soc is not None
    finally:
        environment.devices.close_all_devices()
