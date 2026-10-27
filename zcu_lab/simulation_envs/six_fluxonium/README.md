# Six-fluxonium demo

**Readers:** operator, lab-author. Start six independent fluxonium models and prepare mux measurements without hardware.

**Last updated:** 2026-10-27, shared single-fluxonium physics and component schema v2.

For existing entries, move resonator and mux-group `ext.time_of_fly` to top-level `time_of_fly`. The loader does not migrate it.

This environment has six uncoupled fluxonium qubits, seven DC current sources
and six readout cavities. Each qubit has its own microwave line and local
`Qn_bias` source. Changing a local current changes only that qubit's flux.
Changing `coil_bias` changes all six. Each qubit has separate local and coil
current offsets and periods.
There are no fast-flux lines. One mux generator drives all six readout tones.
A separate pulse generator and ADC handle single-tone scans. The environment
has no coupler, qubit-qubit interaction or crosstalk matrix.

`default.yaml` is the tracked physical-settings template. Each user edits
`six_fluxonium.yaml`, an untracked working copy beside the startup script.
`entry-template.yaml` holds working components, wiring and aliases. Neither
the physical settings nor the entry template supplies calibrated pulses or a
module library. The [settings guide](field-guide.md) records units and model limits.

Top-level `numerics` assigns the whole environment 768 MiB prepare, 64 MiB
result-memo and 4 MiB routing cache budgets, divided across six sources.
Each source receives 128 MiB for prepare and a separate 8 MiB ADC allowance.
These are residency caps, not process RSS limits. The [cache budget measurements](field-guide.md#cache-budget-measurements)
record a profiled old/new GUI peak reduction from 3239047168 to 2731724800 bytes.
With the earlier 256 MiB total prepare setting, a reconciled candidate without
the ownership profiler reached VmHWM 1559523328 bytes, below the 1.5 GiB
reference target. Its paired Rabi and T1 warm wall ratios were 0.999809 and
0.999689. The guide's [768 MiB recheck](field-guide.md#768-mib-prepare-recheck)
records the current setting's higher RSS peak and unchanged numeric bytes.
These observations do not attribute the remaining RSS to Qt.

Single-tone plateaus with carrier magnitude below 30 MHz use `slow_envelope`
instead of T6. It applies full-Hamiltonian midpoint exponentials with Strang
bath half-steps. Top-level `numerics.slow_envelope_segments_per_period` sets the
positive integer subdivisions per period, default 20, for all six qubits.
Dense-reference population tolerance is 1e-2; the 21-level Q1 calibration
measured about 1e-4 or less. Stronger drives, longer pulses and coherences need
separate convergence checks. Carriers at or above 30 MHz keep the existing route.

## Physical model and single-fluxonium consistency

`SixFluxoniumSource` delegates each cavity to `SingleFluxoniumSource`.
Both environments use `SingleFluxoniumModel`, including Purcell relaxation
for 0↔1 with detailed balance, capacitive, inductive and quasiparticle loss,
flux 1/f and photon dephasing, secular master-equation evolution and ADC readout.
Q1 has the same physical parameters as single_fluxonium. With matched settings,
its downward Purcell rate at reduced flux 0.5 is the same in both environments.

The differences are in the multi-qubit setup, not a separate physics model.
Q2 through Q6 have their own EJ, EC, EL, resonator frequencies and zero-current
flux. Each qubit receives coil and local DC contributions. There is no fast-flux
`flux_line`; cavity routing and derived RNG streams account for multiple qubits.
These differences do not change the shared physics used to validate coherence
strategies that do not use fast flux.

Before comparing runs, align physical constants in the two YAML working copies.
Both templates use g = 30 MHz, κ = 1.1 MHz and `thermal_occupation: 0`.
Those values do not define the validation range; edit the working copies for
the comparison rather than changing the tracked templates.

## Trusted startup

On first use, copy the tracked template from the checkout root:

```bash
cp zcu_lab/simulation_envs/six_fluxonium/default.yaml zcu_lab/simulation_envs/six_fluxonium/six_fluxonium.yaml
```

Startup reads `six_fluxonium.yaml` beside the script, not `default.yaml`.
If the working copy is missing, startup raises `FileNotFoundError` before
creating an entry. The error names the absolute path and gives the copy command.
It does not fall back to the template. Edit the working copy for local settings;
restart the GUI to load those changes.

Launch the measure GUI with this trusted script:

```bash
uv run --directory . --no-sync -- python -m apps.measure \
  --script zcu_lab/simulation_envs/six_fluxonium/startup.py
```

For an agent's offline acceptance run, set `QT_QPA_PLATFORM=offscreen` before
launch. Use
`--no-control` to disable the remote socket when no MCP session is needed.
Pass `--project-root <existing-directory>` and set `ZCU_TOOLS_CACHE_DIR` to a
separate absolute cache path to avoid reopening another session's entry or
layout. These options do not authorize hardware access.
MCP can select the same script when it launches a new GUI:

```python
connect(launch="new", script="<checkout>/zcu_lab/simulation_envs/six_fluxonium/startup.py")
```

The script creates or reopens `six-fluxonium-demo/initial`. On first use it seeds
Q1 through Q6, R1 through R6, local current-source components `Q1_bias` through
`Q6_bias`, shared `coil_bias` and the `MG123456` mux group. Each Qi references
Ri and its own `Qi_bias`. The group references `coil_bias`. Aliases bind `mux`
to `MG123456`, `qubit` to `Q1` and `resonator` to `$qubit.resonator`.
Scalar flux forms read `$qubit.flux_source` on Open/Reset without a device alias.
The template uses `schema_version: 2`. Each qubit owns
`flux.Qn_bias` and `flux.coil_bias`, with `half`, `integer` and signed `period`.
The local integer is its physical offset; the coil integer is its coil offset.
Each half is integer plus half the corresponding period. All currents use A.
Together these integers give zero total reduced flux.
These simulated-truth values are not measurement results. Resonators seed total
linewidth 1.1 MHz and pulse-route flight time 0.4 us from their physical settings.
Fluxdep writeback updates only the scanned entry, such as `Q1.flux.coil_bias`,
and appends a raw-fit record with the Run's original frame.
Existing v1 entries require the explicit [upgrade command](../../../docs/parameter-upgrades.md).
Startup never upgrades saved entries.

The script installs a mock SoC and seven DC FakeDevices with fixed unit A. It reads files
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
Use the measured centers to prepare `mux/onetone`, then review its readout
module and writeback proposals. The mux group has no readout module slot until
you create or accept one. Missing-slot forms use group wiring for the generator
and ADCs when known; this does not supply a complete calibrated readout.
Documents may keep `wiring: {}`. When group `ro_ch` is unknown, each tone ADC
stays blank and required in the form before Run.

Qubit `freq` starts at null. Measure each transition before supplying frequencies
for `mux/twotone_spectrum` or calibrated excitation for `mux/t1`. Changing alias
`qubit` changes the scalar target, not the mux group's ordered six-qubit mapping.
Changing the scalar form's selected local-source current affects that qubit only. The mux group's
`coil_bias` affects every qubit. Each circuit's reduced flux equals
`(I_coil - coil_offset)/coil_period + (I_local - offset)/period`, with all
currents, offsets and periods in A. The [settings guide](field-guide.md#dc-and-control-units)
lists both sets of physical coefficients. The lines have no crosstalk.
Reopen or Reset a form after changing its source values or aliases.
Do not treat the physical YAML's parameters as a measured calibration.

The template intentionally leaves calibration-dependent inputs incomplete.
For `range_find`, supply the absolute frequency sweep, readout waveform length
and trigger offset. The initial trigger expression has no measured
`resonator.time_of_fly` to read. Blank mux readout seeds its mixer from
`$mux.mixer_freq`, 7250 MHz in this template. Missing or null mixer stays required
and blank, independently of tone frequencies and kappa.
The group's `time_of_fly: 0.4` us corresponds to the receivers' 400 ns delay.
For `mux/onetone`, trigger starts at `$mux.time_of_fly + 5/(2π·min κ)`;
the ADC window starts at 1.0 us and pulse length at trigger plus that window.
Missing group TOF or any kappa leaves trigger and pulse length required and blank.
Other mux forms still need pulse length and ADC window. Review all six tones
and timing before Run. A readout slot keeps its mixer and existing timing rules.
A qubit experiment also needs valid inputs for every tone, including tones
excluded from Participants. Inspect form diagnostics before Run.

The [walkthrough settings and timings](field-guide.md#acceptance-timings) give
one acquired offscreen example. Its 64-rep qubit pilot was too noisy to establish
a response; the accepted qubit Runs used 16384 reps. Start with a short scan,
then choose averaging from the observed contrast and scatter. A finite curve
or nonzero span alone is not a calibration result.

Only terminal readout is supported. Earlier readout-generator pulses are outside
the single-cavity model's supported sequence. The six cavities do not form a
joint radiatively coupled readout model. This environment cannot validate
coupled-qubit gates or shared-line collective cavity effects.
