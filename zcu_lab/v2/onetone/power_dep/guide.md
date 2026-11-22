**Readers:** operator, agent. Track the resonator's drive-dependent frequency shift versus readout gain.

# One-tone power dependence

At each gain, sweep absolute readout RF. Each shot plays a readout pulse and integrates the demodulated IQ over the ADC window. Average IQ at each frequency before moving to the next gain. A notch or peak moving toward the bare resonator indicates a change in dispersive response.

## Set the sweep

- Bind the resonator. Use [lookback](../../lookback/guide.md) for flight time and [one-tone spectroscopy](../freq/guide.md) for low-gain center and linewidth. Verify Readout wiring and timing. The sweeps supply RF and gain instead of the module. **Trig offset** starts at flight time + `5/(2π·κ)` µs for κ in MHz, or flight time alone without κ. Pulse length covers trigger plus **RO length** (template length minus 0.03 µs), and is at least the template length. Without flight time, enter the required trigger and pulse length.
- **Post-delay** lets the cavity empty. Start at `5/(2π·κ)` µs; to calibrate this wait, compare herald → nominal Xπ with a longer wait.
- Set **Readout gain** to an approved range of DAC full-scale fractions, not watts or dBm. **Readout RF frequency** starts at the captured center ± 1.5 linewidths. Keep both flanks inside the absolute MHz window as gain changes.
- Blank **Early-stop SNR** uses all rounds unless stopped. A positive threshold ends averaging only at the current gain when the estimate reaches it; later gains still run and columns can contain different shot counts.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the map and refine the bracket

The upper map normalizes `|mean(I + iQ)|` separately at each gain: subtract the frequency median and divide by the frequency standard deviation. Compare feature positions, not raw amplitudes between columns. Blank cells are unmeasured. The lower panel shows up to ten cuts on that scale: latest plottable gain in blue, earlier gains in faint red. With early stopping active, Live shows the latest cut's SNR. Neither panel has pointwise shot SEM or gives state populations.

Refine a shift between the last gain on the low-drive branch and the first gain on the shifted branch. For low-power readout, choose a candidate below that bracket and measure its absolute center and g/e separation separately.

Analyze redraws the selected Run or Load without live SNR, fitting or writeback. Missing saved cfg does not block a valid map; it removes the configured Reps × Rounds line.

## Recipe inputs

`onetone_spectrum_over_power`: supply `center_mhz` and `span_mhz` together or omit both. Only one supplied returns `needs_parameters` before Run; if either is null with both supplied, both RF endpoints clear. Null `gain_range` clears both gain endpoints. Null `gain_points`, `freq_points`, `reps` or `rounds` goes to the Run Form. Unresolved required fields fail readiness after overrides, before Run. Null `readout_ref` is rejected selecting the required Readout, before Run.

The recipe returns the Run preview and saves the map, without Analyze or writeback. Check the save receipt; `map_acquired` is not a save receipt.
