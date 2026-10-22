# Six-fluxonium demo

**Readers:** operator, lab-author. Start six independent fluxonium models and prepare mux measurements without hardware.

This environment has six uncoupled fluxonium qubits, six DC sources and six
readout cavities. Each qubit has its own microwave line and fast-flux line.
One mux generator serves all six readout tones. A separate pulse generator and
ADC serve single-tone scans. The environment has no coupler, qubit-qubit
interaction or DC crosstalk.

`default.yaml` holds physical settings. `entry-template.yaml` holds working
components, wiring and aliases. Neither file supplies calibrated pulses or a
module library. The [settings guide](field-guide.md) records units and model limits.

## Trusted startup

From the checkout root, launch the measure GUI with this trusted script:

```bash
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --script zcu_lab/simulation_envs/six_fluxonium/startup.py
```

For an agent's offline acceptance run, set `QT_QPA_PLATFORM=offscreen` before
launch. MCP can select the same script when it launches a new GUI:

```python
connect(launch="new", script="<checkout>/zcu_lab/simulation_envs/six_fluxonium/startup.py")
```

The script creates or reopens `six-fluxonium-demo/initial`. On first use it seeds
Q1 through Q6, R1 through R6, six `flux_bias_1` through `flux_bias_6` components
and the `MG123456` mux group. Each Qi references Ri and `flux_bias_i`.
Aliases bind `mux` to `MG123456`, `qubit` to `Q1`, `resonator` to
`$qubit.resonator` and `flux_dev` to `$qubit.flux_source`.

The script installs a mock SoC and six fixed-unit DC FakeDevices. It reads files
beside the script rather than relative to the shell cwd. Startup does not acquire,
run numeric warmup or access hardware. Existing entries keep their edited
components and aliases. To apply a changed template, use a new project root or
remove the demo directory under `result/` after saving any work you need.
Save outputs under `Database/` are independent; preserve or archive them before
reusing the demo name. Missing files, invalid settings and incomplete entries raise errors.
The script does not fall back to another environment.

The GUI closes each successfully registered driver after acquisition drains.
The script closes the backend, releases borrowed driver references and closes
any drivers that failed to transfer. Backend cleanup failure does not skip
device cleanup. Simultaneous cleanup failures retain both diagnostics.

## First measurements

Open `onetone/range_find` and scan 6900 to 7600 MHz with 300 points to locate
six resonator dips.
The scalar resonator alias supplies pulse generator 13 and ADC 6. All six
resonators share that pulse route. The mux group instead uses generator 0 and
ADCs 0 through 5. Do not copy the pulse route into a mux readout module.
Both routes assign each frequency to the nearest bare cavity; an exact tie
between cavities raises an error. Choose a grid that avoids those midpoints.
Use the measured centers to prepare `mux/onetone_fine`, then review its readout
module and writeback proposals. The mux group has no readout module slot until
you create or accept one. Missing-slot forms use group wiring for the generator
and ADCs; this does not supply a complete calibrated readout.

Qubit `freq` starts at null. Measure each transition before supplying frequencies
for `mux/twotone_spectrum` or calibrated excitation for `mux/t1`. Changing alias
`qubit` changes the scalar target, not the mux group's ordered six-qubit mapping.
Reopen or Reset a form after changing its source values or aliases.
Do not treat the physical YAML's parameters as a measured calibration.

Only terminal readout is supported. Earlier readout-generator pulses are outside
the single-cavity model's supported sequence. The six cavities do not form a
joint radiatively coupled readout model. This environment cannot validate
coupled-qubit gates or shared-line collective cavity effects.
