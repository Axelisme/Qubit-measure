# Four-state joint readout

## Before run

Bind `f1` and `f2` to two different fluxonium components. Bind `coupler` to their transmon and `mux` to the MuxGroup. Open and Reset capture these scalar aliases, the point parameters and the module library. Changing a global alias does not retarget an open form. Reset to capture the new binding.

The MuxGroup needs ordered `qubits` and `resonators` for every tone. Each qubit must point to its matching resonator. F1 and F2 must be distinct members. Their tone positions come from `qubits`; neither tone labels nor ADC channel numbers identify a qubit. The form does not offer editable tone indices. It shows the captured roles and positions. The complete readout stays in its original tone order, including tones outside F1/F2.

Provide calibrated `$f1.module.x180` and `$f2.module.x180` pulses on different charge generators. Readout references `$mux.module.readout`, preserving the library's complete tones and scalar settings. If that slot is absent, fill the blank mux readout, including every bound tone. Confirm generator, mixer, ADC channels, integration window and trigger timing before running. These values are not hardware-safe defaults.

The two GE center pairs start from `$f1.ext.readout_cal.g_center` / `e_center` and the equivalent F2 leaves. Missing centers remain required fields. Nonfinite or coincident centers are invalid. There is no simulator-truth fallback. GE centers must use the same readout route, gain, normalization and timing as this run.

`reps` is the shots per round for each branch. `rounds` is the number of software rounds per branch. Initial and relaxation delays use µs. This experiment does not set a DC device or apply a reset pulse.

## Acquisition and saved data

B00, B01, B10 and B11 apply I or calibrated Xπ to the thermal initial state, then acquire one mux read. F1 is the first bit. There is no heralding, cooling assumption or shot selection. Nominal B11 is not a guarantee of pure |11〉 preparation.

Each completed round contributes its own raw complex IQ. The core uses `MuxReadout.collect_raw_iq`; it does not turn averaged IQ into shots or normalize a second time. The saved array has shape `4 × (rounds*reps) × full tones`. Shot order is round-major, then the program's original reps order. Every tone column at one shot index belongs to the same acquisition event. The native file also saves the complete cfg, role names, tone positions, GE centers and branch/tone/shot axes.

Cancel preserves completed round slots. Unacquired slots remain NaN. An acquire error is a failed operation, not a successful empty run. The measurement plot reports acquired shots per branch rather than a fidelity estimate.

## Load and analysis

Load uses the saved cfg and raw data. It does not replace missing historical settings with the current Form or library. A file with unavailable typed cfg can retain its raw data. Joint analysis refuses it because tone mapping and GE centers are required.

FIT performs nearest-center G/E classification separately on F1 and F2. Equidistant shots go to G. It forms each joint outcome from the same shot, then counts 00, 01, 10 and 11. It does not multiply marginal probabilities. Every shot in a complete branch enters exactly one cell. There is no radius exclusion, thermal filtering or leakage classifier.

A branch needs all expected rounds before analysis uses it. Completed branches can be analysed even if later branches were cancelled. Partial raw remains saved, but the corresponding matrix and counts row is missing, not an observed zero. The summary separates acquired shots from analysed shots and distinguishes incomplete from unstarted branches.

Rows are nominal B00/B01/B10/B11 preparations. Columns are observed joint outcomes. Each complete row sums to one. The figure explicitly states that it includes thermal population, preparation and readout effects. This is not a pure-state assignment calibration. Do not invert it to correct later populations, infer gate fidelity from its diagonal, or interpret the binary outputs as leakage populations. Low diagonal values do not block analysis. There are no writeback proposals.

## Delivery limit

The binding and normalized raw APIs are available. Offline synthetic transport verifies the acquisition, persistence and analysis chain, not FTF physics. Actual FTF mux acquisition and the two charge-line configuration still depend on their simulator owners. End-to-end FTF smoke and the other CZ calibration experiments are separate deliveries. See the [FTF bring-up guide](../guide.md).
