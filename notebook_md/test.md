```python
%load_ext autoreload
import os

%autoreload 2
from zcu_tools.notebook.utils import gc_collect
import zcu_tools.experiment.v2 as ze
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.experiment.cfg_assembler import CfgEnv, make_cfg
from zcu_tools.resources.context import ModuleLibrary, MetaDict, ContextManager
from zcu_tools.datafile import create_datafolder
import zcu_tools.program.v2.base as zp2b
from zcu_tools.utils.debug import debug_scope
from zcu_tools.notebook.utils import make_sweep
```

```python
chip_name = "purcell_tmon"

res_name = "R3"
qub_name = "Q3"

result_dir = os.path.join("..", "result", chip_name, qub_name)
database_path = create_datafolder(
    database_dir=os.path.join("..", "Database"),
    name=os.path.join(chip_name, qub_name),
)

em = ContextManager(os.path.join(result_dir, "exps"))
ml = ModuleLibrary()
md = MetaDict()
```

```python
from zcu_tools.program.v2 import make_mock_soc
from qick import QickConfig

soc = make_mock_soc(n_gens=15)
soccfg = QickConfig(soc.get_cfg())
```

```python
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDevice

device_manager = DeviceManager()
fake_device = FakeDevice()
device_manager.register_device("fake_device", fake_device)
nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)

fake_device.set_value(1.0)
```

```python
ml, md = em.use_flux(label="20260411", readonly=True)
env = CfgEnv(md=md, ml=ml, device_manager=device_manager)
ml, md
```

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        "X180_pulse": "pi_amp",
        "X90_pulse": "pi2_amp",
        # "readout": "readout_rf",
        "readout": "readout_dpm",
    },
    "relax_delay": 10.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.AllXYCfg, env, overrides={"reps": 100, "rounds": 10})
print(cfg)

allxy_exp = nb_adapter(ze.twotone.AllXY_Exp())
with open("allxy-opt2.log", "w") as f:
    with debug_scope(zp2b, stream=f):
        _ = allxy_exp.run(cfg)
```

```python
gc_collect()
```

```python
%matplotlib widget
times = list(range(10, 0, -5))

exp_cfg = {
    "modules": {
        # "reset": "reset_120",
        "pi_pulse": "pi_len",
        "pi2_pulse": ml.get_module("pi2_len", {"phase": 90}),  # Y/2 gate
        "readout": "readout_dpm",
    },
    "sweep": {"times": times},
    "length_expts": 11,
    "length_range": [(0.1 * t, 5.0 * t) for t in times],
    # "relax_delay": 30.0,  # us
    "relax_delay": 0.05 * md.t1,  # us
}
cfg = make_cfg(
    exp_cfg, ze.twotone.time_domain.CPMG_Cfg, env,
    overrides={"reps": 100, "rounds": 10, "detune_ratio": 0.1, "earlystop_snr": 10.0},
)
print(cfg)

cpmg_exp = nb_adapter(ze.twotone.time_domain.CPMG_Exp())
with open("cpmg-opt2.log", "w") as f:
    with debug_scope(zp2b, stream=f):
        _ = cpmg_exp.run(cfg)
```

```python
gc_collect()
```

```python
%matplotlib widget
exp_cfg = {
    "modules": {
        "reset": "reset_bath",
        # "X90_pulse": "pi2_amp",
        # "X180_pulse": "pi_amp",
        "X90_pulse": "pi2_amp",
        "X180_pulse": "pi_amp",
        "readout": "readout_dpm",
    },
    "sweep": make_sweep(0, 500, 3, force_int=True),
    "seed": 0,
    "n_seeds": 5,
    "relax_delay": 0.5,  # us
}
cfg = make_cfg(exp_cfg, ze.twotone.RBCfg, env, overrides={"reps": 100, "rounds": 1})
print(cfg)

rb_exp = nb_adapter(ze.twotone.RB_Exp())
with open("rb-opt2.log", "w") as f:
    with debug_scope(zp2b, stream=f):
        _ = rb_exp.run(cfg)
```
