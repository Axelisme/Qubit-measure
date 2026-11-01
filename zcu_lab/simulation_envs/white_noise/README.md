# White-noise environment

**Last updated:** 2026-11-01

**Readers:** lab-author, framework-dev. Build an explicit Gaussian-noise environment without physical models or hardware.

`profile.yaml` supplies two ordinary generators and one ADC. Load it with
`program.build_soccfg`, then supply every white-noise argument:

```python
from pathlib import Path

import yaml
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import build_soccfg

profile_path = Path("zcu_lab/simulation_envs/white_noise/profile.yaml")
soccfg = build_soccfg(yaml.safe_load(profile_path.read_text()))
env = white_noise(soccfg, std_adc=1.0, seed=27, routes="all")
```

`std_adc` is the nonnegative per-sample I/Q standard deviation. An integer seed
reproduces a call sequence; `None` chooses a random root once. `routes="all"`
connects every generator to every ADC. Supply a tuple of `(generator, ADC)` pairs
to model only explicit connections instead. No pulse response is modeled.
The caller closes `env.devices` after acquisition has drained. The board profile
sets compile capabilities; loading it does not connect to hardware.
