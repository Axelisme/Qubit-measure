# Single-fluxonium YAML values

**Last updated:** 2026-10-27

**讀者：** framework-dev。查閱 single-fluxonium YAML 的載入入口、欄位單位與驗證規則。

`load_single_fluxonium(path)` reads UTF-8 YAML. `parse_single_fluxonium(document)` validates a decoded document. Both return immutable `SingleFluxoniumSettings` and reject invalid values with dotted field paths. Neither creates devices or numerical models.

The format is `zcu.single-fluxonium`, integer version `1`. The schema has one fluxonium, one DC source, one microwave line, an optional fast-flux line and one pulse-readout cavity/ADC. Direct records enforce these counts. There is no coupler, second qubit or mux readout. The field contracts and units are on the types in [schema.py](schema.py).

- `chip.qubit` holds energies in H/h GHz and loss parameters. `chip.temperature_k` is the common temperature.
- `dc_bias` combines a reduced-flux offset with one named A/V source. Initial values apply at startup; live values determine each round's DC flux.
- `drive_line` calibrates the full charge operator in complex GHz/gain. `flux_line` calibrates reduced flux per gain and includes delay, a base response time and signed exponential distortion terms. Set it to null or omit it for DC-only flux. No pulsed flux generator is bound in that case.
- `readout` separates incident field, cavity and receiver calibrations. ADC noise is a two-sided spectral density, not sample covariance.
- Optional `drift` selects `q_cap`, `xqp` and/or `aphi` log-OU processes. Each takes finite `sigma >= 0` in ln(loss) and finite `tau_s > 0` seconds. A selected loss must be strictly positive, so disabled `q_cap: null` or zero `xqp`/`aphi` cannot drift. Omitted parameters stay constant. Omitted/null `drift` disables the feature without a format/version change. The stationary log mean is ln(the configured loss). Each environment starts at the configured losses, not at a random stationary draw.
- `numerics.truncation.native_cutoff` is the harmonic basis dimension. `levels` is the fixed space of lowest DC-reference eigenstates used away from DC as well. They are accuracy parameters, not convergence guarantees.

Source names must differ across DC, microwave and fast flux. All bound generator routes must differ. ADC channels use a separate namespace. Unknown fields, invalid dimensions, nonfinite numbers and non-PSD receiver noise fail before resource creation. YAML decoding uses the shared `resources.yaml.load_yaml`; common physical records come from `resources.simulation_schema`, also used by FTF.

The environment's `default.yaml` is a physical settings document. Its `entry-template.yaml` is a separate `zcu.entry-template` document, read by `resources.entry.load_entry_template` after caller registration. It contains Q1/R1/flux_bias working values and aliases, not the simulation model.
