**Readers:** operator, agent. Use a mux qubit spectrum to locate half- and integer-flux landmarks with a shared DC source.

# Mux two-tone flux map (global)

Track the qubit transition frequency against DC bias via the mean IQ response. Half- and integer-flux positions on that track give half a flux period in device units.

**Sequence:** write the shared DC source → wait **Settle time** (s) → for each frequency point, common reset → simultaneous drives → fixed-frequency mux readout. Run leaves the devices at their last written settings.

## Key settings

Bind `mux` to a `group/mux` with matching qubits and resonators in readout order. **Tones** chooses what to edit, not what to measure. **Participants** selects original zero-based positions in acquisition order; `[2,0]` means Q3 then Q1.

**Flux source** starts at the group's connected DC source. **DC bias** is one sweep for all participants. Its initial bounds are the union of the initial group's half-to-integer intervals, each expanded 10% at each end. The intervals use each qubit's flux offset and periods with the other sources held at their captured values; missing other device values count as zero. Missing a needed offset or swept period leaves the bounds blank. Changing Participants does not recompute them.

Choose **Flux unit** to match the connected device; the experiment does not convert values, change mode or enable output. Before Run confirm wiring, output state and safe bounds. Set the **Frequency** bounds in absolute MHz and choose a calibrated probe drive for each participant. The initial bounds use `qubit.freq ± 1.5 × qubit.kappa`; **Expts** is shared and drive frequencies advance together. Confirm the full **Readout**, including inactive tones' ADC identities and RFs. Without a slot, wiring and resonator frequencies seed the inline module but required pulse/ADC lengths are blank.

In `mux_twotone_flux_global`, omitted Form inputs keep their values. Null `participants`, `flux_unit`, `flux_n`, `flux`, `sweeps` or `n` clears the setting or bounds; the pre-Run readiness check rejects missing required values unless later overrides fill them. For the analysis line seeds, see the [mux flux recipe inputs](../../../recipes/mux-guide.md#mux-flux-maps).

## Read the plots and use the result

Live shows DC bias versus drive RF (MHz), colored by mean IQ magnitude, with the latest and previous complete spectra for comparison. There are no per-point shot SEM bars. An interrupted bias point has no complete spectrum; **Stop** stops the whole scan and leaves earlier complete bias points. Picking needs at least five complete bias points and two distinct frequencies.

In Analysis choose **Tone** by its original index (initially the first acquired one). The picker shows GHz and starts in phase-aware mode. Move the half- and integer-flux lines onto supported landmarks; **Done** produces a result and **Cancel** does not. If the feature or its landmarks are not resolved in the bias window, extend or redo the scan instead of writing back.

Writeback proposes that qubit's `flux.period.<scanned device>`, `flux.offset` and a record of the picked positions. Period magnitude is `2 × |half − integer|` and preserves the existing sign (positive if unset). Offset is `0.5 − half / swept period − Σ(other device value / other device period)`, using other values captured at Run start; absent device values count as zero. Review and repeat for each acquired tone.

Use native `data.h5` for Analyze and writeback. Version-4 cfg data can still be plotted when compatible, but its cfg is not restored and picking requires it. Without the Run/Load parameter capture there are no proposals. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
