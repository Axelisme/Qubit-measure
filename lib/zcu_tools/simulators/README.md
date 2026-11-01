# Physical simulator models

**Last updated:** 2026-11-01

**Readers:** framework-dev, lab-author, agent. Locate physical model configuration, numerics and environment builders.

`simulators` composes framework mechanisms into three physical environments.
Each model owns its topology, YAML schema and environment assembly.
Import the builder from its `environment` package and the YAML loader from `config`:

```python
from zcu_tools.simulators.ftf.config import load_ftf
from zcu_tools.simulators.ftf.environment import ftf

settings = load_ftf(model_path)
environment = ftf(settings, soccfg=soccfg)
```

The caller supplies `model_path` and a `QickConfig` built from independent board YAML.
After acquisitions drain, the host closes the backend and then its devices.
The builders retain the framework's `SimulationEnvironment` and public acquisition contracts.

| Model | Configuration | Environment | Model numerics |
| --- | --- | --- | --- |
| FTF | [config](ftf/config/README.md) | [environment](ftf/environment/README.md) | [physics](ftf/physics/README.md) |
| Single fluxonium | [config](single_fluxonium/config/README.md) | [environment](single_fluxonium/environment/README.md) | Reuses `simulate.fluxonium` |
| Six fluxonium | [config](six_fluxonium/config/README.md) | [environment](six_fluxonium/environment/README.md) | Composes six independent single sources |

Model roots do not load environments. Configuration can be imported without loading runtime composition.
There are no compatibility imports at the former `experiment`, `resources` or `simulate.ftf` locations.

## Framework boundary

`simulate` retains evolution, ADC sampling, baths, dephasing, numeric drift,
source-line response and component Hamiltonians. `experiment` retains environment
bundles, `white_noise`, readout planning and common drive-track conversion.
`resources.simulation_schema` and `resources.yaml` retain neutral records and YAML decoding.
The `framework-no-simulators` forbidden import contract covers every framework package.
`simulators` may consume those mechanisms; framework packages cannot import model code.
C9 also checks top-level import cycles. C16 prevents models from importing lab definitions.

Startup scripts and caller-owned model YAML live in
[`zcu_lab/simulation_envs`](../../../zcu_lab/simulation_envs/README.md).
The startup changes only select the new model imports.

## Private replay

`simulators.replay` owns opt-in private FTF and single-fluxonium input JSON,
content integrity, atomic file I/O and acquire/round reference resolution.
It retains detached waveform samples and settings, not measured I/Q.
Template arrays preserve dtype, shape and flat values, with typed float64, int64
and bool reconstruction. Acquire and round files use version 2 and reject version 1.
Request clocks retain positive integer num/den fields in MHz.
All time arrays retain int64 ticks, including ADC and envelope periods.
Numeric manifests remain opaque and are verified by their numeric owner.
Replay does not change normal waveform assets or data.h5 policies.

## Test owners

`tests/simulators/<model>/config/` owns YAML validation.
`tests/simulators/<model>/environment/` owns public planning and acquisition behavior.
`tests/simulators/ftf/physics/` owns joint-model physics and its simulation marker.
`tests/simulators/test_replay.py` owns private archive roundtrips and integrity errors.
Generic engine, ADC, bath and source-line tests stay with their framework owners.
