"""Starting values and form deviations for twotone/freq; Qt-free data only."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.twotone.freq.core import FreqCfg

FORM = Form(
    FreqCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Hidden(),
            # A fresh inline probe; the sweep drives freq, so it stays locked at 0.
            "qub_pulse": Template(
                "qub_probe",
                set={"gain": 0.1, "waveform.length": 5.0},
                lock={"freq": 0.0},
            ),
            # This experiment's unbound readout keeps its fixed 0.55 us trigger
            # offset instead of the template's time-of-flight policy.
            "readout": Slot(
                "$qubit.module.readout",
                fallback=Template("readout", set={"ro_cfg.trig_offset": 0.55}),
            ),
        },
        "sweep": {"freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=301)},
        "relax_delay": 1.0,
        "reps": 1000,
        "rounds": 100,
        "initial_delay": Hidden(),
    },
)
