# Four-state joint readout

**Readers:** operator, agent. Acquire four nominal thermal branches through the GUI or MCP and interpret their same-shot joint readout matrix.

**Last updated:** 2026-10-21

## Before run

Bind `f1` and `f2` to two different fluxonium components. Bind `coupler` to their transmon and `mux` to the MuxGroup. Open and Reset capture these scalar aliases, the point parameters and the module library. An open form keeps its original bindings. Reset to capture changed aliases.

The MuxGroup needs ordered `qubits` and `resonators` for every tone. Each qubit must point to its matching resonator. F1 and F2 must be distinct members. Their tone positions come from `qubits`. The form shows the fixed roles and positions for review; use this mapping to identify the qubits. The complete readout stays in its original tone order, including tones outside F1/F2.

Provide calibrated `$f1.module.x180` and `$f2.module.x180` pulses on different charge generators. Readout references `$mux.module.readout`, preserving the library's complete tones and scalar settings. If that slot is absent, fill the blank mux readout, including every bound tone. Confirm generator, mixer, ADC channels, integration window and trigger timing before running. Review these editable values against the authorized hardware limits.

The two GE center pairs start from `$f1.ext.readout_cal.g_center` / `e_center` and the equivalent F2 leaves. Missing centers remain required fields. Nonfinite or coincident centers are invalid. Supply measured calibration for the actual readout conditions. GE centers must use the same readout route, gain, normalization and timing as this run.

`reps` is the shots per round for each branch. `rounds` is the number of software rounds per branch. Initial and relaxation delays use µs. Set DC devices externally before Run. Each branch starts from the thermal ensemble.

## Acquisition and saved data

B00, B01, B10 and B11 apply I or calibrated Xπ to the thermal initial state, then acquire one mux read. F1 is the first bit. Every thermal shot is retained, including the thermal admixture in nominal B11.

Each completed round contributes its own raw complex IQ. IQ retains the readout's ADC units/sample and acquired normalization. Each entry is an acquired shot. The saved array has shape `4 × (rounds*reps) × full tones`. Shot order is round-major, then the program's original reps order. Every tone column at one shot index belongs to the same acquisition event. The native file also saves the complete cfg, role names, tone positions, GE centers and branch/tone/shot axes.

Cancel preserves completed round slots. Unacquired slots remain NaN. An acquire error reports a failed operation. The measurement plot reports acquired shots per branch.

## Load and analysis

Load uses the saved cfg and raw data. Only that saved source supplies historical settings. A file with unavailable cfg can retain raw data; joint analysis then reports the missing source, since tone mapping and GE centers are required.

FIT performs nearest-center G/E classification separately on F1 and F2. Equidistant shots go to G. It forms each joint outcome from the same shot, then counts 00, 01, 10 and 11. The joint counts use the paired tones directly. Every shot in a complete branch enters exactly one cell. This binary partition retains the thermal ensemble and original readout effects; leakage remains unresolved.

A branch needs all expected rounds before analysis uses it. Completed branches can be analysed even if later branches were cancelled. Partial raw remains saved and the corresponding matrix and counts row is marked missing. The summary separates acquired shots from analysed shots and distinguishes incomplete from unstarted branches.

Rows are nominal B00/B01/B10/B11 preparations. Columns are observed joint outcomes. Each complete row sums to one. The figure explicitly states that it includes thermal population, preparation and readout effects. Interpret it as a thermal-ensemble diagnostic. Keep later populations in their original readout coordinates; matrix inversion, gate fidelity inference and leakage estimates require different calibration evidence. Low diagonal values remain valid diagnostic results. FIT provides figures and counts only.

## Continue calibration

Validate actual FTF mux acquisition, charge-line configuration and same-shot tone mapping on the control chain in use. Offline synthetic data checks the acquisition and analysis flow only. Continue with the separate CZ calibrations in the [FTF bring-up guide](../guide.md).
