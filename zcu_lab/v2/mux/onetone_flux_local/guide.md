**Readers:** operator, agent. Use a mux resonator spectrum to locate half- and integer-flux landmarks with separate DC sources.

# Mux one-tone flux map (local)

Track the resonator frequency against DC bias via the mean IQ response. Half- and integer-flux positions on that track give half a flux period in device units.

**Sequence:** write the selected DC sources → wait **Settle time** (s) → at each shared RF offset, common reset → mux readout. Run leaves the devices at their last written settings.

## Key settings

Bind `mux` to a `group/mux` with matching qubits and resonators in readout order. **Tones** chooses what to edit, not what to measure. **Participants** selects original zero-based positions in acquisition order; `[2,0]` means Q3 then Q1.

Each **Tones → Flux source** starts at that qubit's connected DC source. Use distinct sources. **DC bias** bounds are per tone; **Flux points** is shared and lines advance together, not as a grid of combinations. The initial bounds are `1.1 × integer − 0.1 × half` and `1.1 × half − 0.1 × integer`, projected from the qubit's offset and periods with other sources fixed to their captured values (absent values count as zero). Missing offset or swept period leaves bounds blank. Run writes only selected lines.

Choose **Flux unit** to match the connected devices; the experiment does not convert values, change mode or enable output. Before Run confirm wiring, output state and safe bounds. **Frequency offset** is a shared MHz range added to each tone's base RF. Confirm the full **Readout**, including ADC identities and base RFs for inactive tones. Readout starts with the [mux onetone timing](../onetone/guide.md#無-readout-slot-的候選來源) candidates; verify mixer, gain, pulse and integration window before Run.

In `mux_onetone_flux_local`, omitted Form inputs keep their values. Null `participants`, `flux_unit`, `flux_n`, `flux_sweeps`, `df` or `n` clears the setting or bounds; the pre-Run readiness check rejects missing required values unless later overrides fill them. For the analysis line seeds, see the [mux flux recipe inputs](../../../recipes/mux-guide.md#mux-flux-maps).

## Read the plots and use the result

Live shows DC bias versus readout RF (MHz), colored by mean IQ magnitude, with the latest and previous complete spectra for comparison. There are no per-point shot SEM bars. An interrupted bias point has no complete spectrum; **Stop** stops the whole scan and leaves earlier complete bias points. Picking needs at least five complete bias points and two distinct frequencies.

In Analysis choose **Tone** by its original index (initially the first acquired one). The picker shows GHz and starts in magnitude mode. Move the half- and integer-flux lines onto supported landmarks; **Done** produces a result and **Cancel** does not. If the feature or its landmarks are not resolved in the bias window, extend or redo the scan instead of writing back.

Writeback proposes that qubit's `flux.period.<scanned device>`, `flux.offset` and a record of the picked positions. Period magnitude is `2 × |half − integer|` and preserves the existing sign (positive if unset). Offset is `0.5 − half / swept period − Σ(other device value / other device period)`, using other values captured at Run start; absent device values count as zero. Review and repeat for each acquired tone.

Use native `data.h5` for Analyze and writeback. Version-4 cfg data can still be plotted when compatible, but its cfg is not restored and picking requires it. Without the Run/Load parameter capture there are no proposals. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
