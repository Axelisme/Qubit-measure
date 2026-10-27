"""Receiver background wiring through public single-cavity ADC preparation."""

import numpy as np
import pytest
from zcu_tools.core.simulation import AcquireRequest, ReadoutWindow
from zcu_tools.experiment.single_fluxonium.readout import prepare_readout
from zcu_tools.experiment.single_fluxonium.template import plan_template
from zcu_tools.resources.single_fluxonium import parse_single_fluxonium
from zcu_tools.simulate.adc import BackgroundRoughness, ReadoutModel

from ._inputs import settings, snapshot, template

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("noise_seed", [None, 123, 999])
def test_preparation_uses_fixed_background_seed_independent_of_noise(noise_seed):
    values = settings().model_dump()
    values["numerics"]["seed"] = noise_seed
    values["readout"]["adc"]["background"] = {
        "extra_ripples": [{"depth": 0.2, "period_mhz": 2000.0, "phase_deg": 90.0}],
        "roughness": {"amplitude": 0.02, "correlation_mhz": 50.0, "seed": 82},
        "band_edges": {"low": {"corner_ghz": 7.5, "rolloff_db_per_ghz": 20.0}},
    }
    config = parse_single_fluxonium(values)
    original = template()
    plan = plan_template(config, snapshot(), original)
    request = AcquireRequest(
        "integrated",
        (1,),
        0,
        (ReadoutWindow(0, 0, 4, 1000.0, (0.018,)),),
        22,
        0,
        "synthetic",
        point_dims=(4,),
        point_names=("sweep",),
        reps=1,
        templates=(original,),
    )
    physical = ReadoutModel(
        "background-wiring",
        ("g",),
        np.array([[7.0]]),
        7.0,
        np.array([0.01]),
        np.array([0.0]),
        np.zeros((1, 1)),
        mismatch_phase_rad=np.zeros(1),
    )
    _, batch = prepare_readout(config, physical, (plan,), request)
    # At 7 GHz the wave is 1-0.2j, and the low edge contributes -10 dB.
    expected = (1 - 0.2j) * 10**-0.5 * BackgroundRoughness(0.02, 50.0, 82).factor(7.0)
    np.testing.assert_allclose(batch.model.receiver.background(7.0), expected)
