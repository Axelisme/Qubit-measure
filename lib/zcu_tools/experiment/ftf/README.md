# FTF environment composition

**Last updated:** 2026-10-16

`ftf(path_or_settings)` builds the same `SimulationEnvironment` for GUI startup
and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, applies their initial values once and binds the selected channel
counts to the fixed mock QICK profile. The returned manager owns cleanup. A
builder failure closes its resources and preserves both startup and close errors.
The executable example is `notebook_md/simulation/environment.py` with adjacent
`ftf.yaml`. The Notebook walkthrough is `notebook_md/single_qubit.md`.

`FTFSource.snapshot` captures current device values immediately before each round.
It checks the configured source units and limits without restoring startup values.
`execute` uses only the detached `RoundRequest`. Identical physical executions
share deterministic preparations; repetitions add independent state/noise draws.
No density, source filter state or receiver noise is inherited by another shot.

`plan_point` is an author diagnostic interface. It converts each execution's
QICK us/MHz/degrees into ns/GHz/radians, preserving signed complex waveforms and
common carrier phase. The plan contains independent numeric inputs, not QICK
buffers. `FTFSource` explicitly calls `Evolution.evolve`, `populations`, then
`ADC.sample_windows`. The program adapter owns packing, rounding and averaging.

The receiver PSD is a two-sided white spectral density in ADC sample units²*ns.
Independent discrete samples have covariance PSD/dt_ns, where dt_ns comes from
that execution's ADC output sample rate. Receiver gain does not scale this noise.
Zero/singular PSD matrices are valid. Several windows share one joint-initial
state draw per shot; receiver noise remains independent between windows.

FTF supports one common terminal readout pulse with one generator and one ADC.
It rejects active gate/bias instructions during or after readout. It does not
reject a passive lowpass tail. Quantum evolution follows the actual tail up to
the common readout start. At that instant the plan builds Hro, its basis,
directional rates and all retained-state pulls at the actual flux, including
DC, offset and residual tail. The readout model freezes these values throughout
the window and ignores further tail changes. This is an approximation, not a
claim that the line has settled or that ADC supports time-dependent flux.
No settling tolerance, runtime warning or physical admission threshold is used.

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Native/local/M
truncations and numerical tolerances remain user choices. They do not certify
thermal-state coverage, dispersive accuracy or convergence. Conditional mean
fields include multilevel relaxation/thermal transitions, not random jump times.
Single-shot variance includes the joint-initial mixture and receiver noise only.

Plans, populations and state labels are author-only data. Normal GUI/MCP Run
metadata carries I/Q and the existing opaque provenance ID, not model truth.
There is no TLS fallback, single-qubit preset, runtime table build or GPU backend.
Private persistent replay and offline libraries have separate owners/delivery.
