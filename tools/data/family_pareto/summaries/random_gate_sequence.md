**Readers:** framework-dev. Read one scenario's measured outcomes and provenance.

# random_gate_sequence

Provenance: {"dirty":false,"exclusive":true,"head":"4e64d9abfcaf04f066c8d8f82672eaa2378c8e0f","started_at_utc":"2026-11-14T08:42:27.074498+00:00","tree":"5add8d093d14ae99097a7d839faad6fda57fa1aa"}

Measured pinned families: t4, adiabatic_map, flux_dop853, slow_envelope, dyson, single_tone_rwa, floquet_period, t6, single_tone_gate, shaped_sambe, gate_cache, period_table, common_frame, multitone_gate, multitone_rwa, floquet, dop853.

Manifest: `3dedc9760786ea38bdf9438992d3a3e2105c46d53c435bba5e2c6cfa46f8437f`. Physical: `ddcc582ad2d165cc6566f6c049108a5c98f17cb137e6b4e4a8ae87305dbbc7a5`. Oracle: `08e082c6ec65c9559826fd6c1a9a01968295972772613ec3d338ba1c53172a2a`.

Times describe this table only; differences between runs do not establish faster/slower families.

| Policy | Status | Final density error | Attribution | Whitelist | Companions | Refusal |
| --- | --- | ---: | --- | --- | --- | --- |
| t4 | not_candidate | None | target_and_companions | ('t4',) | () | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | not_candidate | None | target_and_companions | ('adiabatic_map', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | not_candidate | None | target_and_companions | ('flux_dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | not_candidate | None | target_and_companions | ('slow_envelope', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | guard_or_precondition | None | target_and_companions | ('dyson', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_rwa | over_limit | 0.004215103110585728 | target_and_companions | ('single_tone_rwa', 't4') | ('t4',) | — |
| floquet_period | qualified | 5.040631570812683e-07 | target_and_companions | ('floquet_period', 't4') | ('t4',) | — |
| t6 | qualified | 0.0003902167589396329 | target_and_companions | ('t6', 't4') | ('t4',) | — |
| single_tone_gate | not_candidate | None | target_and_companions | ('single_tone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | not_candidate | None | target_and_companions | ('shaped_sambe', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | not_candidate | None | target_and_companions | ('gate_cache', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | not_candidate | None | target_and_companions | ('period_table', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | not_candidate | None | target_and_companions | ('common_frame', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | not_candidate | None | target_and_companions | ('multitone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | not_candidate | None | target_and_companions | ('multitone_rwa', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | not_candidate | None | target_and_companions | ('floquet', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | not_candidate | None | target_and_companions | ('dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| default | over_limit | 0.004215103110585728 | default | None | () | — |
| fixed_default_group | over_limit | 0.004215103110585728 | fixed_family_group | ('single_tone_rwa', 't4') | () | — |

| Policy | Mode | Wall median s | Process CPU median s | Samples | Unmeasured reason |
| --- | --- | ---: | ---: | ---: | --- |
| t4 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| t4 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| t4 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_rwa | first_evolve | 0.08660033899650443 | 0.08639243899999993 | 5 | — |
| single_tone_rwa | warm_clear_history | 0.08534197900735307 | 0.08515281200000047 | 5 | — |
| single_tone_rwa | warm_retain_history | 0.08666851700400002 | 0.08647117699999995 | 5 | — |
| floquet_period | first_evolve | 0.09873235599661712 | 0.09849973600000084 | 5 | — |
| floquet_period | warm_clear_history | 0.09816451498772949 | 0.09793774200000094 | 5 | — |
| floquet_period | warm_retain_history | 0.09851432399591431 | 0.09827949000000302 | 5 | — |
| t6 | first_evolve | 5.656075024991878 | 5.644863569999998 | 5 | — |
| t6 | warm_clear_history | 5.609914052998647 | 5.600586559000007 | 5 | — |
| t6 | warm_retain_history | 5.629495108994888 | 5.621622045000009 | 5 | — |
| single_tone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| default | first_evolve | 0.09243483899626881 | 0.0921499380000057 | 5 | — |
| default | warm_clear_history | 0.09437789599178359 | 0.09414756500001431 | 5 | — |
| default | warm_retain_history | 0.09328657000150997 | 0.09311252899999545 | 5 | — |
| fixed_default_group | first_evolve | 0.08348491000651848 | 0.08316232199999263 | 5 | — |
| fixed_default_group | warm_clear_history | 0.08340066300297622 | 0.08319152100000338 | 5 | — |
| fixed_default_group | warm_retain_history | 0.08429123299720231 | 0.08400626199997419 | 5 | — |
