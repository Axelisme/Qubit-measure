# Heralded CZ amplitude

**Readers:** operator, agent. Compare CZ returns by measured herald and fit the first return for gg.

## Set up the scan

Open `heralded/cz_amplitude` from the Heralded family. Bind `qubit1`, `qubit2`, `mux` and `coupler` before opening the tab or using Defaults. Qubit1 and Qubit2 use their positions in the full mux group's ordered qubit list, not ADC numbers. Use calibrated Xπ pulses with the same pre-delay and a full mux readout with both qubits' GE Post circle classifiers. The readout and its wait apply to both reads. See [position bindings and mux tone order](../../two_qubit/guide.md#綁定位置與讀出-tones).

The two CZ pulses start from the coupler's `cz_qubit1` and `cz_qubit2` slots. If either is missing, fill its frequency, signed gain, Nyquist zone and envelope length in the cosine template. The two pulses must share frequency, envelope length and pre-delay. Use independent generators. Each synchronized pulse pair waits for the later end, including any post-delay. The gain multiplier scales both signed gains and keeps their relative phase fixed. Enter a range starting at 0 and reaching beyond the first positive return. Use enough points to resolve the oscillation.

Leave Reset blank to omit it, or select an ordinary pulse-based reset. This experiment does not accept resets that discard shots by herald. At each multiplier, one program attempts `Reps × Rounds` shots for each nominal preparation, four times that number in total when all rounds complete. Stop retains completed rounds. Equal attempts do not guarantee equal measured herald populations.

## Read the four returns

Each shot runs **optional reset → nominal I/Xπ on each qubit → herald read → Xπ on both → CZ → Xπ on both → terminal read**. The program chooses the two preparation pulses by nominal preparation index. The herald class never controls a pulse.

The panels use measured classes in Qubit1, Qubit2 order. The first pair of Xπ pulses maps the measured labels gg, ge, eg and ee to inferred CZ inputs |11⟩, |10⟩, |01⟩ and |00⟩. The second pair maps an ideal return to a terminal class that matches the herald. These inferred inputs do not establish the true prepared state. The measured return does not establish gate fidelity, leakage or a π conditional phase.

Each panel pools counts across the four nominal preparations. Its numerator counts terminal results equal to the measured herald. Its denominator counts all nine terminal bins in that herald row, including Other. Other indicates classifier rejection, and is not a hit. Herald Other stays in the saved counts but does not get a panel. A herald row with no shots produces no point. A measured zero return is a point at zero. Error bars show one shot SEM, `√[p(1−p)/(N−1)]` for `N ≥ 2` shots in the row.

## Fit and save

Analyze fits only the measured gg return to a fixed-phase cosine. The fit uses the contiguous measured prefix from multiplier 0 and does not extrapolate a first return. A gap in gg ends the fit prefix. Gaps in the other panels do not affect it. A stopped scan can still yield a first return if its measured prefix covers one. The fit reports period uncertainty separately from shot SEM. This entry does not write back a calibration.

Save retains the four nominal preparations, gain scale and both measured classes as joint counts, not raw IQ. Load restores the typed scan and counts so Analyze can repeat the same calculation. The nominal preparation totals let you check the attempted allocation separately from the measured herald populations.

For common GUI actions, see the [operator workflow](../../operator-workflow.md).
