"""Native analysis values handed to artifact persistence without serializers."""

from dataclasses import dataclass

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.form_adapter import Form
from zcu_tools.resources.entry import ArtifactKey

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


def saving_run(
    cfg: SavingCfg, *, context: RunContext
) -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(cfg, SavingCore().run(cfg, context=context))


@dataclass(frozen=True)
class NativeFit:
    pi_amp: float
    errors: np.ndarray
    center: complex


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_native_result_handover_keeps_fields_arrays_and_complex(pane):
    definition = Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=lambda context: Form(SavingCfg, {}),
        run=saving_run,
        guide=None,
    )
    result = NativeFit(0.25, np.array([0.0, np.inf]), 1 + 2j)
    item = definition.analysis_item(
        ArtifactKey(pane, "results", "analysis", pane), result, None, pane=pane
    )
    assert item.content.model_dump() == {
        "pane": pane,
        "parameters": {},
        "summary": {
            "pi_amp": 0.25,
            "errors": [0.0, None],
            "center": {"real": 1.0, "imag": 2.0},
        },
    }
    assert np.isinf(result.errors[1])
