# Single-fluxonium acquisition

**Last updated:** 2026-10-21

**Readers:** framework-dev, lab-author. Compose complete fluxonium timelines with one physical readout cavity.

`single_fluxonium(path_or_settings, *, private_replay=None)` returns the normal
`SimulationEnvironment`. It creates one fixed-unit DC FakeDevice, a mock SoC,
one microwave route, one fast-flux route and one readout generator/ADC route.
All generators bind to that ADC for transport admission. Only the configured
readout generator supplies physical readout pulses. No real hardware or numeric
warmup runs at construction.

The caller owns resources after return. After acquisition drains, call
`environment.close_backend()` when present, then clean up caller-owned devices.
GUI startup registers the backend cleanup through its normal on-close hook.
Transferring a driver to the GUI transfers only that driver's cleanup responsibility.

## Complete timelines

`plan_template(settings, snapshot, template)` returns a `SingleFluxoniumPlan`.
It converts QICK us/MHz/degrees into ns/GHz/radians. Shared
`experiment.readout_plan.drive_track` preserves signed gains, complex envelopes
and the common carrier origin. Each template remains one `TimelineBatch`.
`Engine.evolve` owns segmentation, kernel selection and propagation.

Fast flux uses `simulate.lines.FilteredFlux` with one loop. The planner samples
its complete causal response until terminal readout, including passive tails.
A 2 ns grid, delayed event sides and the endpoint define the continuous waveform.
Bath levels use interval-midpoint flux on the configured bath grid, quantized
relative to live DC in 1e-3 Phi0 steps. Equal adjacent levels merge.
Ideal constant pulses use abrupt schedule levels without a continuous waveform.
The planner never forces the endpoint back to DC.

Only terminal readout is supported. The planner identifies earlier readout
pulses as `cavity_drives`; execution raises `ValueError` before evolution when
any are present. It does not ignore those pulses or approximate their Stark
shifts, cavity memory or backaction. Terminal readout uses the DC model even
when flux has not returned to DC.

## Physical and ADC composition

`SingleFluxoniumSource(settings, devices, *, private_replay=None)` captures the
startup driver object. `snapshot()` reads its current A/V value and checks the
fixed unit and inclusive limits. Registry removal or renaming does not restore
startup values. `model_facts(snapshot)` builds the retained projection, Gibbs
initial density, full-charge drive, directional baths and one physical ADC model.
The charge calibration uses `Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z)`.

Evolution returns densities in DC energy coordinates. The source supplies
explicit `Basis` records to `batch_populations`, then calls
`normalize_populations` as the sole probability repair. Readout state order is
the retained DC energy order. Cavity frequencies include the model owner's
dressed shifts at that DC point. Receiver PSD converts to sample covariance
using the ADC sample interval. One scattering phase is always supplied explicitly.

Integrated rounds use ADC `round_statistics` and `PreparedRawShots`.
Decimated rounds use ADC `sample_traces`. No environment-owned random sampler,
synthetic second cavity or alternate IQ generator exists. QICK owns averaging,
round hooks, Stop checkpoints and successful-round pacing. The default pacing
floor is 0.2 s; settings may set it to zero. Stop does not interrupt an active
numeric round or a pacing sleep. The next program checkpoint observes it.

## Reuse and lifetime

`open_acquire(request)` invalidates the preceding session. `sample_round` and
`execute` propagate numeric, ADC and replay errors without partial results.
Invalidated sessions and closed sources raise `RuntimeError`. Direct callers
serialize source methods. `close()` clears model, plan, engine and ADC caches,
invalidates the session and leaves caller-owned drivers untouched.

One physical generation and one complete noiseless preparation remain resident.
Keys include settings, the full snapshot, numeric/dependency versions, layout,
all borrowed control/readout arrays and every waveform sample/time. Buffers are
reread on each lookup. Evolution validates reuse against its own routing policy.
Noise, counts, round identities and measured IQ are never cached.
Delivered raw sources retain their own immutable inputs after another acquire,
cache eviction or source close. Repeated materialization returns fresh arrays.

## Private replay

Pass an explicit absolute `private_replay` directory to the builder or source.
The resource owner archives detached settings, controls, envelopes, RNG identity
and the selected snapshot. It stores no measured IQ. Call
`replay_round(round_json_path)` to recompute only that round without live devices.
The single-fluxonium backend has its own version string. Wrong settings backend,
model identity, dependency versions or numeric references raise `ValueError`.
Integrity and I/O errors propagate. These author-only archives contain model
truth and do not belong in normal GUI/RPC metadata.
