"""Package loading keeps numerical tools independent of stage dependencies."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.requires_subprocess
def test_numeric_import_defers_stage_dependencies_until_selected():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import numpy as np
                from zcu_tools.experiment.probe.generators import synthesize
                from zcu_tools.experiment import probe

                specialized = (
                    "qick",
                    "zcu_tools.experiment.simulation",
                    "zcu_tools.experiment.acquisition",
                    "zcu_tools.program",
                )
                loaded = [
                    name for name in sys.modules
                    if any(name == root or name.startswith(root + ".") for root in specialized)
                ]
                assert not loaded, f"numeric import loaded stage dependencies: {loaded}"
                np.testing.assert_array_equal(
                    synthesize(lambda xs: xs, np.array([1.0]), [], noise_std=0.0, seed=1),
                    [1.0],
                )

                from zcu_tools.experiment.probe import (
                    AnalysisStageResult, RunStageResult, SaveStageResult,
                    analysis_stage, compare_roundtrip, run_stage, save_stage,
                )
                from zcu_tools.experiment.probe.analysis import (
                    AnalysisStageResult as DirectAnalysisResult,
                    analysis_stage as direct_analysis,
                )
                from zcu_tools.experiment.probe.run_save import (
                    RunStageResult as DirectRunResult,
                    SaveStageResult as DirectSaveResult,
                    run_stage as direct_run, save_stage as direct_save,
                )
                assert run_stage is direct_run is probe.run_stage
                assert save_stage is direct_save is probe.save_stage
                assert analysis_stage is direct_analysis is probe.analysis_stage
                assert RunStageResult is DirectRunResult is probe.RunStageResult
                assert SaveStageResult is DirectSaveResult is probe.SaveStageResult
                assert AnalysisStageResult is DirectAnalysisResult is probe.AnalysisStageResult
                assert callable(compare_roundtrip)
                try:
                    getattr(probe, "unknown_public_name")
                except AttributeError:
                    pass
                else:
                    raise AssertionError("unknown name did not raise AttributeError")
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
