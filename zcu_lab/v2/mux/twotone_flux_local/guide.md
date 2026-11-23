**Readers:** operator, agent. Use a mux qubit spectrum to locate half- and integer-flux landmarks with separate DC sources.

# Mux two-tone flux map (local)

Track the qubit transition frequency against DC bias via the mean IQ response. Half- and integer-flux positions on that track give half a flux period in device units.

**Sequence:** write the selected DC sources → wait **Settle time** (s) → for each frequency point, common reset → simultaneous drives → fixed-frequency mux readout. Run leaves the devices at their last written settings.

## Key settings

Bind `mux` to a `group/mux` with matching qubits and resonators in readout order. **Tones** chooses what to edit, not what to measure. **Participants** selects original zero-based positions in acquisition order; `[2,0]` means Q3 then Q1.

Each **Tones → Flux source** starts at that qubit's connected DC source. Use distinct sources. **DC bias** bounds are per tone; **Flux points** is shared and lines advance together, not as a grid of combinations. The initial bounds are `1.1 × integer − 0.1 × half` and `1.1 × half − 0.1 × integer`, projected from the qubit's offset and periods with other sources fixed to their captured values (absent values count as zero). Missing offset or swept period leaves bounds blank. Run writes only selected lines.

Choose **Flux unit** to match the connected devices; the experiment does not convert values, change mode or enable output. Before Run confirm wiring, output state and safe bounds. Set the **Frequency** bounds in absolute MHz and choose a calibrated probe drive for each participant. The initial bounds use `qubit.freq ± 1.5 × qubit.kappa`; **Expts** is shared and drive frequencies advance together. Confirm the full **Readout**, including inactive tones' ADC identities and RFs. Without a slot, wiring and resonator frequencies seed the inline module but required pulse/ADC lengths are blank.

In `mux_twotone_flux_local`, omitted Form inputs keep their values. Null `participants`, `flux_unit`, `flux_n`, `flux_sweeps`, `sweeps` or `n` clears the setting or bounds; the pre-Run readiness check rejects missing required values unless later overrides fill them. For the analysis line seeds, see the [mux flux recipe inputs](../../../recipes/mux-guide.md#mux-flux-maps).

## Read the plots and use the result

Live shows DC bias versus drive RF (MHz). At each bias, subtract the spectrum's median complex IQ before taking the magnitude (a.u.). The map and the latest and previous complete spectra show this same contrast. There are no per-point shot SEM bars. An interrupted bias point has no complete spectrum; **Stop** stops the whole scan and leaves earlier complete bias points. Picking needs at least five complete bias points and two distinct frequencies.

In Analysis choose **Tone** by its original index (initially the first acquired one). The picker shows GHz and starts in phase-aware mode. Move the half- and integer-flux lines onto supported landmarks; **Done** produces a result and **Cancel** does not. If the feature or its landmarks are not resolved in the bias window, extend or redo the scan instead of writing back.

Writeback proposes that qubit's `flux.period.<scanned device>`, `flux.offset` and a record of the picked positions. Period magnitude is `2 × |half − integer|` and preserves the existing sign (positive if unset). Offset is `0.5 − half / swept period − Σ(other device value / other device period)`, using other values captured at Run start; absent device values count as zero. Review and repeat for each acquired tone.

Use native `data.h5` for Analyze and writeback. Version-4 cfg data can still be plotted when compatible, but its cfg is not restored and picking requires it. Without the Run/Load parameter capture there are no proposals. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
