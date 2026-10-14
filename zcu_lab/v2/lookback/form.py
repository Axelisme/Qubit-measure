"""Fresh lookback defaults from the captured resonator and persistent templates."""

from zcu_tools.cfg_model import Expr, Hidden, Locked, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.lookback.core import LookbackCfg

FORM = Form(
    LookbackCfg,
    tree={
        "modules": {
            "reset": None,
            "init_pulse": None,
            "readout": Template(
                "readout",
                set={
                    "pulse_cfg.gain": 1.0,
                    "ro_cfg.ro_length": 1.5,
                    "ro_cfg.trig_offset": Expr("$resonator.ext.time_of_fly - 0.1"),
                },
            ),
        },
        "relax_delay": 0.0,
        "reps": Locked(1),
        "rounds": 500,
        "initial_delay": Hidden(),
    },
)
