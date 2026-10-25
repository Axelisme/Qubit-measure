---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

# FTF acquisition with a shared environment

**Readers:** operator, lab-author. Run a finite FTF acquisition with a shared simulated environment and change DC bias between rounds.

This notebook uses the F1-SQUID-F2 environment with a finite charge-drive pulse, terminal readout and live DC bias changes between software rounds. It does not connect to hardware or save into a result directory. FTF supports static DC bias, charge drive and linearized small-signal flux drive. It does not implement large-amplitude fast-flux pulses.

## Load the environment

Copy `zcu_lab/simulation_envs/FTF/default.yaml` to `ftf.yaml` in the same directory before running. Run from the repository root, or supply an absolute YAML path below. This Notebook calls the public `ftf(...)` builder directly. It reserves two host CPUs and enables background numeric warmup.

For the GUI, use `scripts/app/run_measure_gui.py --script zcu_lab/simulation_envs/FTF/startup.py`. That trusted startup script uses one worker without warmup and installs resources through the GUI facets. See the [FTF startup instructions](../zcu_lab/simulation_envs/FTF/README.md) for project setup. Changing YAML requires restarting the GUI or rebuilding the Notebook environment. A bias change through the existing device controls does not reload YAML.

```python
import os
from pathlib import Path

from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils.device import set_flux
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

settings_path = Path("zcu_lab/simulation_envs/FTF/ftf.yaml").resolve()
env = ftf(
    settings_path,
    workers=max(1, (os.cpu_count() or 1) - 2),
    warmup=True,
)
nb_adapter = NotebookAdapter(
    soc=env.soc, soccfg=env.soccfg, device_manager=env.devices,
)
```

`nb_adapter` can wrap caller-selected experiment instances through its normal `run`, `analyze` and persistence APIs. It shares this exact manager; it does not create or reconnect instruments. The acquisition below calls the public program API directly so its waveform and timing are visible without a lab experiment dependency.

## Finite controls and terminal readout

Program lengths and delays are us, frequencies MHz and phases degrees. Program gain is signed normalized DAC amplitude. YAML microwave coefficients multiply the full charge operator in GHz/gain, not a hidden calibrated pi-pulse value. These illustrative pulses are not a gate calibration.

```python
microwave = PulseCfg(
    ch=2, nqz=1, freq=597.3, phase=90.0, gain=-0.1,
    waveform=ConstWaveformCfg(length=0.08),
)
readout = PulseReadoutCfg(
    gen_ch=0, ro_ch=0, nqz=2, freq=7190.0, gain=0.005,
    waveform=ConstWaveformCfg(length=1.0), ro_length=0.9,
    trig_offset=0.05,
)
program = ModularProgramV2(
    env.soccfg, ProgramV2Cfg(reps=8, rounds=2),
    modules=[microwave.build("microwave"), readout.build("readout")],
)
```

These channels match `default.yaml`: gen2 drives F1 charge, and gen0 with ADC0 reads the shared resonator line. Gen1 is unbound. The illustrative frequencies target the default initial bias and R1; changing the model or working point requires a new spectroscopy calibration.

The charge pulse evolves the joint state until readout starts. Readout uses the round's fixed DC model and energy-basis populations. Both conditional resonator fields include relaxation and thermal transitions. The receiver samples one joint initial state and adds I/Q noise for each shot. These are not quantum jump trajectories. Controls during or after readout do not contribute to this acquisition.

## Change the live DC value between rounds

The initial device value is applied only at startup. The next round reads the current value from the same manager. It does not restore YAML's initial value. The default coupler source uses A and allows -0.01 to 0.01 A. The example changes it to 0.0001 A, or 0.1 mA. The final average mixes the two conditions; this is not automatically a bias sweep.

```python
bias = env.devices.get_device("coupler_bias")
if not isinstance(bias, FakeDevice):
    raise TypeError("This example requires its explicitly built memory source")


def next_bias(round_count, raw, cancel_flag):
    if round_count == 1:
        set_flux(bias, 0.0001, unit="A", stop=StopSignal())


averaged_iq = program.acquire(env.soc, progress=False, round_hook=next_bias)
last_round_raw = program.get_raw()
decimated_iq = program.acquire_decimated(env.soc, progress=False)
```

Use I/Q and ordinary analysis to find working points. Normal GUI/MCP metadata exposes only the environment's public capabilities and opaque provenance ID. It does not expose populations, state labels or a calibrated working point. Environment authors may separately call `experiment.ftf.plan_template`, `Engine.evolve` and `populations` to inspect numeric states, outside the operator calibration path.

The YAML example supplies native/local/global dimensions and solver tolerances explicitly. Check larger fixed parent spaces, thermal-state coverage and segment refinement for your sample. This model uses weak dispersive readout, Markov/secular baths and exponential echo-characteristic dephasing. It does not guarantee arbitrary truncations or reproduce a full 1/f Ramsey/echo envelope.

## Close the shared manager

After synchronous work has finished, close the exact returned manager. In a longer notebook, put acquisition in a `try/finally` block so failures also release resources. The Notebook owns this manager throughout. The GUI startup instead transfers successfully registered drivers to the GUI and arranges cleanup with `on_close`.

```python
env.devices.close_all_devices()
```
