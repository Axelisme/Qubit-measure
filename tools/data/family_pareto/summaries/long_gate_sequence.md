**Readers:** framework-dev. Read one scenario's measured outcomes and provenance.

# long_gate_sequence

Provenance: {"dirty":false,"exclusive":true,"head":"3545520dd8d5e8784be642246605f263654ce72b","started_at_utc":"2026-11-13T22:22:36.951585+00:00","tree":"c9f484253fae2cc0db5a7d80dc1dd21bbe808d77"}

Measured pinned families: t4, adiabatic_map, flux_dop853, slow_envelope, dyson, single_tone_rwa, floquet_period, t6, single_tone_gate, shaped_sambe, gate_cache, period_table, common_frame, multitone_gate, multitone_rwa, floquet, dop853.

Manifest: `00c7f0f4f947b23c1638528ee0a0af9b1408eba7d5ad7f1e1b376b765cb35455`. Physical: `23bda18781f18c084b6116f91056fa14c33804bd58dc4acdb1ca29f45fad76d1`. Oracle: `8704ce4867351f493aa19cd0d5c482b92f04005a36ff12c81e6de72cab1a0db6`.

Times describe this table only; differences between runs do not establish faster/slower families.

| Policy | Status | Final density error | Attribution | Whitelist | Companions | Refusal |
| --- | --- | ---: | --- | --- | --- | --- |
| t4 | not_candidate | None | target_and_companions | ('t4',) | () | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| adiabatic_map | not_candidate | None | target_and_companions | ('adiabatic_map', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| flux_dop853 | not_candidate | None | target_and_companions | ('flux_dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| slow_envelope | not_candidate | None | target_and_companions | ('slow_envelope', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dyson | guard_or_precondition | None | target_and_companions | ('dyson', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| single_tone_rwa | qualified | 0.0012470135795709194 | target_and_companions | ('single_tone_rwa', 't4') | ('t4',) | — |
| floquet_period | qualified | 1.6582348470837773e-07 | target_and_companions | ('floquet_period', 't4') | ('t4',) | — |
| t6 | qualified | 0.0002953807516523095 | target_and_companions | ('t6', 't4') | ('t4',) | — |
| single_tone_gate | not_candidate | None | target_and_companions | ('single_tone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| shaped_sambe | not_candidate | None | target_and_companions | ('shaped_sambe', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| gate_cache | not_candidate | None | target_and_companions | ('gate_cache', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| period_table | not_candidate | None | target_and_companions | ('period_table', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| common_frame | not_candidate | None | target_and_companions | ('common_frame', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_gate | not_candidate | None | target_and_companions | ('multitone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_rwa | not_candidate | None | target_and_companions | ('multitone_rwa', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| floquet | not_candidate | None | target_and_companions | ('floquet', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dop853 | not_candidate | None | target_and_companions | ('dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| default | qualified | 0.0012470135795709194 | default | None | () | — |
| fixed_default_group | qualified | 0.0012470135795709194 | fixed_family_group | ('single_tone_rwa', 't4') | () | — |

| Policy | Mode | Wall median s | Process CPU median s | Samples | Unmeasured reason |
| --- | --- | ---: | ---: | ---: | --- |
| t4 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| t4 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| t4 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| adiabatic_map | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| adiabatic_map | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| adiabatic_map | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| flux_dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| flux_dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| flux_dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| slow_envelope | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| slow_envelope | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| slow_envelope | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dyson | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dyson | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dyson | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| single_tone_rwa | first_evolve | 0.23440175200084923 | 0.23333753200000018 | 5 | — |
| single_tone_rwa | warm_clear_history | 0.23752572199737187 | 0.23646649799999997 | 5 | — |
| single_tone_rwa | warm_retain_history | 0.2375026000008802 | 0.23647035200000044 | 5 | — |
| floquet_period | first_evolve | 0.5784433049993822 | 0.5760267910000003 | 5 | — |
| floquet_period | warm_clear_history | 0.5752348829992115 | 0.5731678720000009 | 5 | — |
| floquet_period | warm_retain_history | 0.5758157320015016 | 0.5733352479999994 | 5 | — |
| t6 | first_evolve | 4.80565119899984 | 4.791610886000001 | 5 | — |
| t6 | warm_clear_history | 4.8062387550016865 | 4.791258450000001 | 5 | — |
| t6 | warm_retain_history | 4.828457240000716 | 4.814004940000004 | 5 | — |
| single_tone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| single_tone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| single_tone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| shaped_sambe | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| shaped_sambe | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| shaped_sambe | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| gate_cache | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| gate_cache | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| gate_cache | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| period_table | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| period_table | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| period_table | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| common_frame | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| common_frame | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| common_frame | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_rwa | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_rwa | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| multitone_rwa | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| floquet | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| floquet | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| floquet | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [0], segments [0] |
| default | first_evolve | 0.4139414940000279 | 0.41219878199999016 | 5 | — |
| default | warm_clear_history | 0.41774937200534623 | 0.41631362300000774 | 5 | — |
| default | warm_retain_history | 0.4159242150053615 | 0.41443610800000386 | 5 | — |
| fixed_default_group | first_evolve | 0.23648221800249303 | 0.23555099799999368 | 5 | — |
| fixed_default_group | warm_clear_history | 0.23616341700108023 | 0.23524319700000262 | 5 | — |
| fixed_default_group | warm_retain_history | 0.2359484970002086 | 0.23507480799999314 | 5 | — |
