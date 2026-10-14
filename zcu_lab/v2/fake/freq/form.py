"""Synthetic spectrum defaults using the same captured readout templates."""

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.fake.freq.core import FakeFreqCfg

FORM = Form(
    FakeFreqCfg,
    tree={
        "modules": {"readout": Template("readout")},
        "sweep": {
            "freq": Range.around(
                "$resonator.freq", "5.0 * $resonator.kappa", expts=201
            ),
            # The native sweep record inherits program defaults, not extra inputs.
            "reps": Hidden(),
            "rounds": Hidden(),
            "relax_delay": Hidden(),
            "initial_delay": Hidden(),
        },
        "reps": 100,
        "rounds": 100,
        "relax_delay": 1.0,
        "initial_delay": Hidden(),
        "fast_mode": Hidden(),
    },
)
