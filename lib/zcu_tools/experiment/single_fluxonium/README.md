# Single-fluxonium environment composition

**Last updated:** 2026-10-21

`single_fluxonium(path_or_settings)` assembles a `SimulationEnvironment` with
one fixed-unit DC FakeDevice, one microwave generator, one fast-flux generator
and one pulse-readout generator/ADC route. Startup applies the DC initial value
once. The caller owns device cleanup after return; builder failure closes the
devices it created. No real hardware or numeric warmup runs.

Acquisition is not connected. `SingleFluxoniumSource.open_acquire` and
`execute` raise `NotImplementedError` for both integrated and decimated modes.
No second cavity, placeholder IQ or alternate ADC path is used. Single-cavity
readout construction, sampling, acquire tests and MCP startup acceptance are
separate work.

## Template planning

`template.plan_template(settings, snapshot, template)` returns one
`SingleFluxoniumPlan`. It converts QICK us/MHz/degrees into ns/GHz/radians.
The shared `experiment.readout_plan.drive_track` preserves signed gains,
complex samples and the common carrier origin. Only microwave events enter
evolution tracks. Earlier readout pulses remain separate cavity-drive inputs.
They do not become microwave controls or trigger a cavity solve in this planner.

Fast flux uses `simulate.lines.FilteredFlux` with a one-loop calibration.
The planner samples the complete causal response until terminal readout.
A 2 ns grid, delayed event sides and the readout endpoint define the absolute
Phi0 waveform. The planner does not truncate passive tails or force a return
to DC. Ideal constant pulses use abrupt quantized levels without a waveform.
Bath levels use interval-midpoint flux on the configured bath grid, plus delayed
events, quantized to 1e-3 Phi0 relative to live DC. Equal adjacent levels merge.
The line owner computes filtering; evolution owns segmentation and propagation.
Every point remains in one complete `TimelineBatch`.

## Source state and caches

`SingleFluxoniumSource` retains the startup driver identity. `snapshot()`
reads its live A/V value and checks the fixed unit and inclusive limits.
Registry removal or renaming does not restore startup values.

`model_facts(snapshot)` lazily builds the fixed-DC `SingleFluxoniumModel`
and its `ModelSource`. The full-charge microwave calibration uses
`Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z)`. The model owner supplies Gibbs density,
directional baths, residual-photon dephasing and the existing 0↔1 Purcell model.
No ADC model or density propagation occurs here.

The source retains one snapshot's physical facts and one complete template plan.
Settings, the full snapshot and replay/numeric dependency versions key physical
facts. Plan keys also digest every current control/readout array and every
waveform sample and time. Borrowed buffers are reread on each lookup.
Neither noisy results nor sampling identities enter these caches.
`close()` releases residents and rejects further source calls. It does not
close caller-owned devices. Direct callers serialize access.

`replay.BACKEND_VERSION` is independent of FTF.
`replay_versions()` defines private compatibility requirements only.
This package does not yet archive inputs or execute replay rounds.
